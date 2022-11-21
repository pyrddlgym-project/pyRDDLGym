import numpy as np
import jax
import jax.numpy as jnp
import optax

from pyRDDLGym.Core.Jax.JaxRDDLCompiler import JaxRDDLCompiler
from pyRDDLGym.Core.Jax.JaxRDDLSimulator import JaxRDDLSimulator
from pyRDDLGym.Core.Parser.rddl import RDDL


class JaxRDDLStraightlinePlanner:
    
    def __init__(self,
                 rddl: RDDL,
                 key: jax.random.PRNGKey,
                 n_steps: int,
                 n_batch: int,
                 initializer: jax.nn.initializers.Initializer=jax.nn.initializers.zeros,
                 optimizer: optax.GradientTransformation=optax.adam(0.1),
                 agg_fn=jnp.mean) -> None:
        self.sim = JaxRDDLSimulator(rddl, key)
        self.key = key
        self.init = initializer
        self.opt = optimizer
        
        subs = self.sim.subs
        cpfs = self.sim.cpfs
        invariants = self.sim.invariants
        reward_fn = self.sim.reward
        primed_unprimed = self.sim.state_unprimed
        
        # plan initialization
        action_types, self.action_shapes = {}, {}
        for pvar in rddl.domain.action_fluents.values():
            aname = pvar.name       
            action_types[aname] = JaxRDDLCompiler.RDDL_TO_JAX_TYPE[pvar.range]
            ashape = (n_steps,)
            avalue = subs[aname]
            if hasattr(avalue, 'shape'):
                ashape = ashape + avalue.shape
            self.action_shapes[aname] = ashape
    
        # perform one step of a roll-out
        ERR_CODE = JaxRDDLCompiler.ERROR_CODES['INVARIANT_NOT_SATISFIED']
        
        def _step(carry, actions):
            x, key, err = carry
            x.update(actions)
            
            # calculate all the CPFs (e.g., s') and r(s, a, s') in order
            for name, cpf_fn in cpfs.items():
                x[name], key, cpf_err = cpf_fn(x, key)
                err |= cpf_err
            reward, key, rew_err = reward_fn(x, key)
            err |= rew_err
            
            # check invariants
            valid_state = True
            for cpf_fn in invariants:
                satisfied, key, cpf_err = cpf_fn(x, key)
                err |= cpf_err
                valid_state &= satisfied            
            err |= jnp.logical_not(valid_state) * ERR_CODE
            
            # set s <- s'
            for primed, unprimed in primed_unprimed.items():
                x[unprimed] = x[primed]
                
            return (x, key, err), reward
        
        # perform a single roll-out
        def _process_action(action, action_type):
            if action_type == JaxRDDLCompiler.REAL:
                return action
            elif action_type == JaxRDDLCompiler.INT:
                return jnp.asarray(action, dtype=JaxRDDLCompiler.REAL)
            else:
                return jax.nn.sigmoid(action)
            
        def _rollout(plan, x0, key):
            x0 = x0.copy()
            
            # process action fluent by type
            for name, action_type in action_types.items():
                x0[name] = _process_action(x0[name], action_type)
                
            # set s' <- s            
            for primed, unprimed in primed_unprimed.items():
                x0[primed] = x0[unprimed]
                
            # unroll the time horizon
            err = JaxRDDLCompiler.ERROR_CODES['NORMAL']
            (x, key, err), rewards = jax.lax.scan(_step, (x0, key, err), plan)
            cuml_reward = jnp.sum(rewards)
            
            return cuml_reward, x, key, err
        
        # do a batch of roll-outs
        def _batched(value):
            value = jnp.asarray(value)
            batched_shape = (n_batch,) + value.shape
            return jnp.broadcast_to(value, shape=batched_shape) 
                
        def _batched_rollouts(plan, key):
            x_batch = jax.tree_map(_batched, subs)
            keys = jax.random.split(key, num=n_batch)
            returns, x_batch, keys, errs = jax.vmap(
                _rollout, in_axes=(None, 0, 0))(plan, x_batch, keys)
            key = keys[-1]
            return returns, x_batch, key, errs
        
        # aggregate all sampled returns
        def _loss(plan, key):
            returns, _, key, errs = _batched_rollouts(plan, key)
            loss_value = -agg_fn(returns)
            return loss_value, (key, errs)
        
        # gradient descent update
        def _update(plan, key, opt_state):
            grad, (key, errs) = jax.grad(_loss, has_aux=True)(plan, key)
            updates, opt_state = optimizer.update(grad, opt_state)
            plan = optax.apply_updates(plan, updates)       
            return plan, key, opt_state, errs
        
        self.loss = jax.jit(_loss)
        self.update = jax.jit(_update)
        
    def optimize(self, n_epochs: int):
        ''' Compute an optimal straight-line plan for the RDDL domain and instance.
        
        @param n_epochs: the maximum number of steps of gradient descent
        '''
        key = self.key
        plan = {action: self.init(key, ashape, dtype=JaxRDDLCompiler.REAL) 
                for action, ashape in self.action_shapes.items()}
        opt_state = self.opt.init(plan)

        for step in range(n_epochs):
            plan, key, opt_state, _ = self.update(plan, key, opt_state)
            loss_val, (key, errs) = self.loss(plan, key)
            errs = JaxRDDLCompiler.get_error_codes(np.bitwise_or.reduce(errs))
            yield step, plan, loss_val, errs
            
        yield n_epochs, plan, loss_val, errs
        
