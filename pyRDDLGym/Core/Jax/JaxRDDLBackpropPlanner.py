import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
import jax.nn.initializers as initializers
import optax
from typing import Dict, Generator

from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLTypeError

from pyRDDLGym.Core.Compiler.RDDLLiftedModel import RDDLLiftedModel
from pyRDDLGym.Core.Jax.JaxRDDLCompiler import JaxRDDLCompiler
from pyRDDLGym.Core.Jax.JaxRDDLCompilerWithGrad import JaxRDDLCompilerWithGrad

 
class JaxRDDLBackpropPlanner:
    
    def __init__(self, rddl: RDDLLiftedModel,
                 key: jax.random.PRNGKey,
                 batch_size_train: int,
                 batch_size_test: int=None,
                 action_bounds: Dict={},
                 max_concurrent_action: int=jnp.inf,
                 initializer: initializers.Initializer=initializers.zeros,
                 optimizer: optax.GradientTransformation=optax.rmsprop(0.1),
                 log_full_path: bool=False) -> None:
        self.rddl = rddl
        self.key = key
        self.max_concurrent_action = max_concurrent_action
        self.batch_size_train = batch_size_train
        if batch_size_test is None:
            batch_size_test = batch_size_train
        self.batch_size_test = batch_size_test
        self._action_bounds = action_bounds
        self.initializer = initializer
        self.optimizer = optimizer
        self.log_full_path = log_full_path
        
        self._compile_rddl()
        self._compile_action_info()        
        self._compile_backprop()
    
    # ===========================================================================
    # compilation of RDDL file to JAX
    # ===========================================================================
    
    def _compile_rddl(self):
        self.compiled = JaxRDDLCompilerWithGrad(rddl=self.rddl)
        self.compiled.compile()
        
        self.test_compiled = JaxRDDLCompiler(rddl=self.rddl)
        self.test_compiled.compile()

    def _compile_action_info(self):
        self.action_info, self.action_bounds = {}, {}
        for name, prange in self.rddl.variable_ranges.items():
            if self.rddl.variable_types[name] == 'action-fluent':
                shape = (self.rddl.horizon,)
                value = self.compiled.init_values[name]
                if hasattr(value, 'shape'):
                    shape = shape + value.shape
                if prange not in JaxRDDLCompiler.JAX_TYPES:
                    raise RDDLTypeError(
                        f'Invalid range {prange} of action-fluent <{name}>.')
                self.action_info[name] = (prange, shape)
                if prange == 'bool':
                    self.action_bounds[name] = (0.0, 1.0)
                else:
                    self.action_bounds[name] = self._action_bounds.get(
                        name, (-jnp.inf, +jnp.inf))
    
    # ===========================================================================
    # compilation of back-propagation info
    # ===========================================================================
    
    def _compile_backprop(self):
        
        # policy
        train_policy, test_policy = self._jax_predict()
        self.test_policy = jax.jit(test_policy)
        
        # roll-outs
        self.train_rollouts = self.compiled.compile_rollouts(
            policy=train_policy,
            n_steps=self.rddl.horizon,
            n_batch=self.batch_size_train)
        self.test_rollouts = self.test_compiled.compile_rollouts(
            policy=test_policy,
            n_steps=self.rddl.horizon,
            n_batch=self.batch_size_test)
        
        # losses
        self.train_loss = jax.jit(self._jax_loss(self.train_rollouts))
        self.test_loss = jax.jit(self._jax_loss(self.test_rollouts))
        
        # optimization
        self.initialize = jax.jit(self._jax_init(self.initializer, self.optimizer))
        self.update = jax.jit(self._jax_update(self.train_loss, self.optimizer))
        
    def _jax_predict(self):
        
        # TODO: use a one-hot for integer actions
        def _soft(_, step, params, key):
            plan = {}
            for name, param in params.items():
                prange, _ = self.action_info[name]
                if prange == 'real': 
                    plan[name] = param[step]
                else:
                    plan[name] = jnp.asarray(
                        param[step], dtype=JaxRDDLCompiler.REAL)
            return plan, key
        
        # def _hard_bool(params):
        #     values = [param 
        #               for name, param in params.items() 
        #               if self.action_info[name][0] == 'bool']
        #                 new_params = {}
        #     for name, param in params.items():
        #         prange, _ = self.action_info[name]
        #         if prange == 'bool':
        #             values.append(param)
        #         else:
        #             new_params[name] = param
        #     return new_params
                    
        def _hard(subs, step, params, key):
            soft, key = _soft(subs, step, params, key)
            hard = {}
            for name, param in soft.items():
                prange, _ = self.action_info[name]
                if prange == 'real':
                    hard[name] = param
                elif prange == 'int':
                    hard[name] = jnp.asarray(
                        jnp.round(param), dtype=JaxRDDLCompiler.INT)
                elif prange == 'bool':
                    hard[name] = param > 0.5
            return hard, key
        
        return _soft, _hard
             
    def _jax_init(self, initializer, optimizer):
        
        def _init(key):
            params = {}
            for action, (_, shape) in self.action_info.items():
                key, subkey = random.split(key)
                param = initializer(subkey, shape, dtype=JaxRDDLCompiler.REAL)
                param = jnp.clip(param, *self.action_bounds[action])
                params[action] = param
            opt_state = optimizer.init(params)
            return params, key, opt_state
        
        return _init
    
    def _jax_loss(self, rollouts):
        
        def _loss(params, key):
            logged, keys = rollouts(params, key)
            returns = jnp.sum(logged['reward'], axis=-1)
            logged['return'] = returns
            loss = -jnp.mean(returns)
            key = keys[-1]
            return loss, (key, logged)
        
        return _loss
    
    def _jax_update(self, loss, optimizer):
        
        def _clip(params):
            new_params = {}
            for name, param in params.items():
                new_params[name] = jnp.clip(param, *self.action_bounds[name])
            return new_params
        
        def _surplus(params):
            total, count = 0.0, 0
            for name, param in params.items():
                prange, _ = self.action_info[name]
                if prange == 'bool':
                    total += jnp.sum(param)
                    count += jnp.sum(param > 0)
            surplus = (total - self.max_concurrent_action) / count
            return (surplus, count)
        
        def _sogbofa_clip_condition(values):
            _, (surplus, count) = values
            return jnp.logical_and(count > 0, surplus > 0)
        
        def _sogbofa_clip_body(values):
            params, (surplus, _) = values
            new_params = {}
            for name, param in params.items():
                prange, _ = self.action_info[name]
                if prange == 'bool':
                    new_params[name] = jnp.maximum(param - surplus, 0.0)
                else:
                    new_params[name] = param
            new_surplus = _surplus(new_params)
            return (new_params, new_surplus)
        
        def _sogbofa_clip(params):
            surplus = _surplus(params)
            init_values = (params, surplus)
            params, _ = jax.lax.while_loop(
                _sogbofa_clip_condition, _sogbofa_clip_body, init_values)
            return params
        
        def _sogbofa_clip_batched(params):            
            params = _clip(params)
            params = jax.vmap(_sogbofa_clip, in_axes=0)(params)
            return params
            
        def _update(params, key, opt_state):
            grad, (key, logged) = jax.grad(loss, has_aux=True)(params, key)
            updates, opt_state = optimizer.update(grad, opt_state)
            params = optax.apply_updates(params, updates)
            params = _sogbofa_clip_batched(params)
            return params, key, opt_state, logged
        
        return _update
            
    # ===========================================================================
    # training
    # ===========================================================================
    
    def optimize(self, epochs: int, step: int=1) -> Generator[Dict, None, None]:
        ''' Compute an optimal straight-line plan.
        
        @param epochs: the maximum number of steps of gradient descent
        @param step: frequency the callback is provided back to the user
        '''
        params, key, opt_state = self.initialize(self.key)
        
        best_params = params
        best_loss = jnp.inf
        
        for it in range(epochs):
            params, key, opt_state, _ = self.update(params, key, opt_state)            
            train_loss, (key, _) = self.train_loss(params, key)            
            test_loss, (key, test_log) = self.test_loss(params, key)
            self.key = key
            
            if test_loss < best_loss:
                best_params = params
                best_loss = test_loss
            
            if it % step == 0:
                callback = {'iteration': it,
                            'train_loss': train_loss,
                            'test_loss': test_loss,
                            'best_loss': best_loss,
                            'best_params': best_params,
                            **test_log}
                yield callback
                
    def get_plan(self, params, key):
        plan = []
        for step in range(self.rddl.horizon):
            actions, key = self.test_policy(None, step, params, key)
            actions = jax.tree_map(np.ravel, actions)
            grounded_actions = {}
            for name, action in actions.items():
                grounded_action = self.compiled.tensors.expand(name, action)
                prange, _ = self.action_info[name]
                if prange == 'bool':
                    grounded_action = {var: value 
                                       for var, value in grounded_action.items()
                                       if value == True}
                grounded_actions.update(grounded_action)
            plan.append(grounded_actions)
        return plan, key
