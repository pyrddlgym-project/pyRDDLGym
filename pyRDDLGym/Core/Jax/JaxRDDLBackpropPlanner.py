import jax
import jax.numpy as jnp
import jax.random as random
import jax.nn.initializers as initializers
import optax
from typing import Dict, Generator

from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLTypeError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLValueOutOfRangeError
from pyRDDLGym.Core.Jax.JaxRDDLCompiler import JaxRDDLCompiler
from pyRDDLGym.Core.Jax.JaxRDDLCompilerWithGrad import JaxRDDLCompilerWithGrad
from pyRDDLGym.Core.Parser.rddl import RDDL

 
class JaxRDDLBackpropPlanner:
    
    def __init__(self,
                 rddl: RDDL,
                 key: jax.random.PRNGKey,
                 batch_size_train: int, batch_size_test: int=None,
                 action_bounds: Dict={},
                 initializer: initializers.Initializer=initializers.zeros,
                 optimizer: optax.GradientTransformation=optax.rmsprop(0.1),
                 log_full_path: bool=False) -> None:
        self.rddl = rddl
        self.key = key
        self.batch_size_train = batch_size_train
        if batch_size_test is None:
            batch_size_test = batch_size_train
        self.batch_size_test = batch_size_test
        self.action_bounds = action_bounds
        self.initializer = initializer
        self.optimizer = optimizer
        self.log_full_path = log_full_path
        
        self._compile_rddl_to_jax()
        self._compile_discount_and_horizon()
        self._compile_action_info()
        
        self._compile_jax_train_and_test()
    
    # ===========================================================================
    # compilation of RDDL file
    # ===========================================================================
    
    def _compile_rddl_to_jax(self):
        
        # graph for training
        compiled = JaxRDDLCompilerWithGrad(rddl=self.rddl)
        compiled.compile()
        self.compiled = compiled
        
        # graph for testing
        test_compiled = JaxRDDLCompiler(rddl=self.rddl)
        test_compiled.compile()
        self.test_compiled = test_compiled
        
    def _compile_discount_and_horizon(self):             
        horizon = self.rddl.instance.horizon
        if not (horizon > 0):
            raise RDDLValueOutOfRangeError(
                'Horizon {} in the instance is not > 0.'.format(horizon))
        self.horizon = horizon
            
        discount = self.rddl.instance.discount
        if not (0 <= discount <= 1):
            raise RDDLValueOutOfRangeError(
                'Discount {} in the instance is not in [0, 1].'.format(discount))
        self.discount = discount
    
    def _compile_action_info(self):
        self.action_info = {}
        self.action_bounds = {}
        for pvar in self.rddl.domain.action_fluents.values():
            name = pvar.name      
            prange = pvar.range
            value = self.compiled.init_values[name]
            shape = (self.horizon,)
            if hasattr(value, 'shape'):
                shape = shape + value.shape
            if prange not in JaxRDDLCompiler.JAX_TYPES:
                raise RDDLTypeError(
                    f'Invalid type {prange} of action fluent <{name}>.')
            self.action_info[name] = (prange, shape)            
            self.action_bounds[name] = self.action_bounds.get(
                name, (-jnp.inf, +jnp.inf))
    
    # ===========================================================================
    # compilation of training and testing graph
    # ===========================================================================
    
    def _compile_jax_train_and_test(self):
        
        # prediction
        predict_map, test_map = self._jax_predict()
        self.test_map = jax.jit(test_map)
        
        # training
        train_rollout = self._jax_rollout(self.compiled, self.batch_size_train)
        train_loss = self._jax_evaluate(predict_map, train_rollout)
        self.train_loss = jax.jit(train_loss)        
        self.initialize = jax.jit(self._jax_init(self.initializer, self.optimizer))
        self.update = jax.jit(self._jax_update(train_loss, self.optimizer))
        
        # testing
        test_rollout = self._jax_rollout(self.test_compiled, self.batch_size_test)
        test_loss = self._jax_evaluate(test_map, test_rollout)
        self.test_loss = jax.jit(test_loss)
        
    def _jax_predict(self):
        
        # TODO: use a one-hot for integer actions
        def _soft(params):
            plan = {}
            for name, param in params.items():
                prange, _ = self.action_info[name]                
                if prange == 'bool':
                    action = jax.nn.sigmoid(param)
                elif prange == 'int': 
                    action = jnp.asarray(param, dtype=JaxRDDLCompiler.REAL)
                else:
                    action = param           
                plan[name] = action
            return plan
    
        def _hard(params):
            plan = {}
            for name, value in _soft(params).items():
                prange, _ = self.action_info[name]
                if prange == 'bool':
                    plan[name] = value > 0.5
                elif prange == 'int':
                    plan[name] = jnp.asarray(
                        jnp.round(value), dtype=JaxRDDLCompiler.INT)
                else:
                    plan[name] = value
            return plan
        
        return _soft, _hard
    
    def _jax_rollout(self, compiled, batch_size):
        NORMAL = JaxRDDLCompiler.ERROR_CODES['NORMAL']        
        init_subs = compiled.init_values.copy()
        for next_state, state in compiled.next_states.items():
            init_subs[next_state] = init_subs[state]
        
        def _epoch(carried, action):
            subs, discount, key = carried  
            subs.update(action)
            
            error = NORMAL
            for name, cpf in compiled.cpfs.items():
                subs[name], key, cpf_err = cpf(subs, key)
                error |= cpf_err
                
            reward, key, reward_err = compiled.reward(subs, key)
            error |= reward_err
            
            for next_state, state in compiled.next_states.items():
                subs[state] = subs[next_state]
            
            discount *= self.discount
            
            carried = (subs, discount, key)
            
            logged = {'reward': reward, 'error': error}
            if self.log_full_path:
                logged['path'] = subs
                
            return carried, logged
        
        def _rollout(plan, subs, key): 
            carried = (subs, 1.0, key)
            carried, logged = jax.lax.scan(_epoch, carried, plan)
            * _, key = carried        
            return logged, key
        
        def _batched(x):
            value = jnp.asarray(x)
            shape = (batch_size,) + value.shape
            return jnp.broadcast_to(value, shape=shape) 
                
        def _rollouts(plan, key): 
            batched_init = jax.tree_map(_batched, init_subs)
            subkeys = jax.random.split(key, num=batch_size)
            logged, keys = jax.vmap(_rollout, in_axes=(None, 0, 0))(
                plan, batched_init, subkeys)
            logged['keys'] = subkeys
            return logged, keys
        
        return _rollouts
         
    def _jax_evaluate(self, predict, rollouts):
        
        def _loss(params, key):
            plan = predict(params)
            logged, keys = rollouts(plan, key)
            reward = logged['reward']
            returns = jnp.sum(reward, axis=-1)
            loss = -jnp.mean(returns)
            key = keys[-1]
            logged['return'] = returns
            aux = (logged, key)
            return loss, aux
        
        return _loss
    
    def _jax_init(self, initializer, optimizer):
        
        def _init(key):
            params = {}
            for action, (_, shape) in self.action_info.items():
                key, subkey = random.split(key)
                param = initializer(subkey, shape, dtype=JaxRDDLCompiler.REAL)
                param = jnp.clip(param, *self.action_bounds[action])
                params[action] = param
            opt_state = optimizer.init(params)
            return params, opt_state, key
        
        return _init
    
    def _jax_update(self, loss, optimizer):
        
        def _clip(params):
            clipped = {}
            for name, value in params.items():
                clipped[name] = jnp.clip(value, *self.action_bounds[name])
            return clipped
            
        def _update(params, opt_state, key):
            grad, (logged, key) = jax.grad(loss, has_aux=True)(params, key)
            updates, opt_state = optimizer.update(grad, opt_state)
            params = optax.apply_updates(params, updates)   
            params = _clip(params)    
            return params, opt_state, logged, key
        
        return _update
            
    def optimize(self, n_epochs: int) -> Generator[Dict, None, None]:
        ''' Compute an optimal straight-line plan for the RDDL domain and instance.
        
        @param n_epochs: the maximum number of steps of gradient descent
        '''
        params, opt_state, key = self.initialize(self.key)
        
        best_params = params
        best_plan = self.test_map(best_params)
        best_loss = jnp.inf
        
        for step in range(n_epochs):
            params, opt_state, _, key = self.update(params, opt_state, key)            
            train_loss, (_, key) = self.train_loss(params, key)
            
            test_loss, (test_log, key) = self.test_loss(params, key)
            self.key = key
            
            if test_loss < best_loss:
                best_params = params
                best_plan = self.test_map(params)
                best_loss = test_loss
            
            callback = {'step': step,
                        'train_loss': train_loss,
                        'test_loss': test_loss,
                        'best_plan': best_plan,
                        'best_loss': best_loss,
                        **test_log}
            yield callback
    
