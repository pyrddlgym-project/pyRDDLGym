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
        reward_fn = self.sim.reward
        primed_unprimed = self.sim.state_unprimed
        
        # perform one step of a roll-out
        def _step(carry, actions):
            x, key = carry
            x.update(actions)
            for name, cpf_fn in cpfs.items():
                x[name], key, _ = cpf_fn(x, key)
            reward, key, _ = reward_fn(x, key)
            for primed, unprimed in primed_unprimed.items():
                x[unprimed] = x[primed]
            carry = (x, key)
            return carry, reward
                    
        # perform a single roll-out
        def _rollout(plan, x0, key):
            x0 = x0.copy()
            for primed, unprimed in primed_unprimed.items():
                x0[primed] = x0[unprimed]
            (x, key), rewards = jax.lax.scan(_step, (x0, key), plan)
            sum_reward = jnp.sum(rewards)
            return sum_reward, x, key
        
        # do a batch of roll-outs
        def _batched(value):
            value = jnp.asarray(value)
            batched_shape = (n_batch,) + value.shape
            return jnp.broadcast_to(value, shape=batched_shape) 
                
        def _batched_rollouts(plan, key):
            x_batch = jax.tree_map(_batched, subs)
            keys = jax.random.split(key, num=n_batch)
            sum_rewards, x_batch, keys = jax.vmap(
                _rollout, in_axes=(None, 0, 0))(plan, x_batch, keys)
            key = keys[-1]
            return sum_rewards, x_batch, key
        
        # aggregate all sampled returns
        def _loss(plan, key):
            sum_rewards, _, key = _batched_rollouts(plan, key)
            loss_value = -agg_fn(sum_rewards)
            return loss_value, key
        
        # gradient descent update
        def _update(plan, key, opt_state):
            grad, key = jax.grad(_loss, has_aux=True)(plan, key)
            updates, opt_state = optimizer.update(grad, opt_state)
            plan = optax.apply_updates(plan, updates)       
            return plan, key, opt_state
        
        self.loss = jax.jit(_loss)
        self.update = jax.jit(_update)
        
        # plan initialization
        self.action_shapes = {pvar.name: (n_steps,) + subs[pvar.name].shape
                              for pvar in rddl.domain.action_fluents.values()}
    
    def optimize(self, n_epochs: int):
        ''' Compute an optimal straight-line plan for the RDDL domain and instance.
        
        @param n_epochs: the maximum number of steps of gradient descent
        '''
        key = self.key
        plan = {action: self.init(key, shape, JaxRDDLCompiler.REAL) 
                for action, shape in self.action_shapes.items()}
        opt_state = self.opt.init(plan)
        
        for step in range(n_epochs):
            plan, key, opt_state = self.update(plan, key, opt_state)
            loss_val, key = self.loss(plan, key)
            yield step, plan, loss_val
            
        yield n_epochs, plan, loss_val
        
