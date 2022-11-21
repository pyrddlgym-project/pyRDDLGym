import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
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
                 aggregation=jnp.mean) -> None:
        self.key = key

        sim = JaxRDDLSimulator(rddl, key, enforce_diff=True)
        subs = sim.subs
        cpfs = sim.cpfs
        reward_fn = sim.reward
        primed_unprimed = sim.state_unprimed
        
        ERR_NORMAL = JaxRDDLCompiler.ERROR_CODES['NORMAL']
        
        # plan initialization
        action_info = {}
        for pvar in rddl.domain.action_fluents.values():
            aname = pvar.name       
            atype = JaxRDDLCompiler.RDDL_TO_JAX_TYPE[pvar.range]
            ashape = (n_steps,)
            avalue = subs[aname]
            if hasattr(avalue, 'shape'):
                ashape = ashape + avalue.shape
            action_info[aname] = (atype, ashape)
            if atype != JaxRDDLCompiler.REAL:
                subs[aname] = np.asarray(subs[aname], dtype=np.float32)
                
        # perform one step of a roll-out        
        def _step(carry, actions):
            x, key, err = carry            
            x.update(actions)
            
            # calculate all the CPFs (e.g., s') and r(s, a, s') in order
            for name, cpf_fn in cpfs.items():
                x[name], key, cpf_err = cpf_fn(x, key)
                err |= cpf_err
            reward, key, rew_err = reward_fn(x, key)
            err |= rew_err
            
            # set s <- s'
            for primed, unprimed in primed_unprimed.items():
                x[unprimed] = x[primed]
                
            return (x, key, err), reward
        
        # perform a single roll-out
        def _rollout(plan, x0, key):
            
            # set s' <- s at the first epoch
            x0 = x0.copy()               
            for primed, unprimed in primed_unprimed.items():
                x0[primed] = x0[unprimed]
            
            # generate roll-outs and cumulative reward
            (x, key, err), rewards = jax.lax.scan(_step, (x0, key, ERR_NORMAL), plan)
            cuml_reward = jnp.sum(rewards)
            
            return cuml_reward, x, key, err
        
        # force action ranges         
        def _force_action_ranges(plan):
            for name, value in plan.items():
                atype, _ = action_info[name]
                if atype == JaxRDDLCompiler.INT:
                    plan[name] = jnp.asarray(value, dtype=JaxRDDLCompiler.REAL)
                elif atype == bool:
                    plan[name] = jax.nn.sigmoid(value)
            return plan
        
        self.force_action_ranges = jax.jit(_force_action_ranges)
        
        # do a batch of roll-outs
        def _batched(value):
            value = jnp.asarray(value)
            batched_shape = (n_batch,) + value.shape
            return jnp.broadcast_to(value, shape=batched_shape) 
        
        def _batched_rollouts(plan, key):
            x_batch = jax.tree_map(_batched, subs)
            keys = jax.random.split(key, num=n_batch)
            plan = _force_action_ranges(plan)
            returns, x_batch, keys, errs = jax.vmap(
                _rollout, in_axes=(None, 0, 0))(plan, x_batch, keys)
            key = keys[-1]
            return returns, key, x_batch, errs
        
        # aggregate all sampled returns
        def _loss(plan, key):
            returns, key, x_batch, errs = _batched_rollouts(plan, key)
            loss_value = -aggregation(returns)
            errs = jax.lax.reduce(errs, ERR_NORMAL, jnp.bitwise_or, (0,))
            return loss_value, (key, x_batch, errs)
        
        # gradient descent update
        def _update(plan, opt_state, key):
            grad, (key, x_batch, errs) = jax.grad(_loss, has_aux=True)(plan, key)
            updates, opt_state = optimizer.update(grad, opt_state)
            plan = optax.apply_updates(plan, updates)       
            return plan, opt_state, key, x_batch, errs
        
        self.loss = jax.jit(_loss)
        self.update = jax.jit(_update)
        
        # initialization
        def _initialize(key):
            plan = {}
            for action, (_, ashape) in action_info.items():
                key, subkey = random.split(key)
                plan[action] = initializer(subkey, ashape, dtype=JaxRDDLCompiler.REAL)
            opt_state = optimizer.init(plan)
            return plan, opt_state, key
        
        self.initialize = jax.jit(_initialize)

    def optimize(self, n_epochs: int):
        ''' Compute an optimal straight-line plan for the RDDL domain and instance.
        
        @param n_epochs: the maximum number of steps of gradient descent
        '''
        plan, opt_state, self.key = self.initialize(self.key)

        for step in range(n_epochs):
            plan, opt_state, self.key, _, _ = self.update(plan, opt_state, self.key)
            loss_val, (self.key, x_batch, errs) = self.loss(plan, self.key)
            
            fixed_plan = self.force_action_ranges(plan)
            errs = JaxRDDLCompiler.get_error_codes(errs)
            yield step, fixed_plan, loss_val, x_batch, errs
        
