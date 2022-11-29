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
        self._action_bounds = action_bounds
        self.initializer = initializer
        self.optimizer = optimizer
        self.log_full_path = log_full_path
        
        self._compile_rddl()
        self._compile_discount_and_horizon()
        self._compile_action_info()        
        self._compile_backprop()
    
    # ===========================================================================
    # compilation of RDDL file
    # ===========================================================================
    
    def _compile_rddl(self):
        self.compiled = JaxRDDLCompilerWithGrad(rddl=self.rddl)
        self.test_compiled = JaxRDDLCompiler(rddl=self.rddl)
        self.compiled.compile()
        self.test_compiled.compile()

    def _compile_discount_and_horizon(self): 
        horizon = self.rddl.instance.horizon
        if not (horizon > 0):
            raise RDDLValueOutOfRangeError(f'Horizon {horizon} not > 0.')
        self.horizon = horizon
            
        discount = self.rddl.instance.discount
        if not (0 <= discount <= 1):
            raise RDDLValueOutOfRangeError(f'Discount {discount} not in [0, 1].')
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
            n_steps=self.horizon,
            n_batch=self.batch_size_train)
        self.test_rollouts = self.test_compiled.compile_rollouts(
            policy=test_policy,
            n_steps=self.horizon,
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
                if prange == 'bool':
                    action = jax.nn.sigmoid(param[step])
                elif prange == 'int': 
                    action = jnp.asarray(param[step], dtype=JaxRDDLCompiler.REAL)
                else:
                    action = param[step]
                plan[name] = action
            return plan, key
    
        def _hard(subs, step, params, key):
            soft, key = _soft(subs, step, params, key)
            hard = {}
            for name, value in soft.items():
                prange, _ = self.action_info[name]
                if prange == 'bool':
                    hard[name] = value > 0.5
                elif prange == 'int':
                    hard[name] = jnp.asarray(
                        jnp.round(value), dtype=JaxRDDLCompiler.INT)
                else:
                    hard[name] = value
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
            clipped = {}
            for name, param in params.items():
                clipped[name] = jnp.clip(param, *self.action_bounds[name])
            return clipped
            
        def _update(params, key, opt_state):
            grad, (key, logged) = jax.grad(loss, has_aux=True)(params, key)
            updates, opt_state = optimizer.update(grad, opt_state)
            params = optax.apply_updates(params, updates)
            params = _clip(params)
            return params, key, opt_state, logged
        
        return _update
            
    def optimize(self, n_train_epochs: int) -> Generator[Dict, None, None]:
        ''' Compute an optimal straight-line plan for the RDDL domain and instance.
        
        @param n_train_epochs: the maximum number of steps of gradient descent
        '''
        params, key, opt_state = self.initialize(self.key)
        
        best_params = params
        best_loss = jnp.inf
        
        for it in range(n_train_epochs):
            params, key, opt_state, _ = self.update(params, key, opt_state)            
            train_loss, (key, _) = self.train_loss(params, key)            
            test_loss, (key, test_log) = self.test_loss(params, key)
            self.key = key
            
            if test_loss < best_loss:
                best_params = params
                best_loss = test_loss
            
            callback = {'iteration': it,
                        'train_loss': train_loss,
                        'test_loss': test_loss,
                        'best_loss': best_loss,
                        'best_params': best_params,
                        **test_log}
            yield callback
    
