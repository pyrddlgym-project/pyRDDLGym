import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
import jax.nn.initializers as initializers
import optax
from typing import Dict, Iterable

from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLNotImplementedError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLTypeError

from pyRDDLGym.Core.Compiler.RDDLLiftedModel import RDDLLiftedModel
from pyRDDLGym.Core.Jax.JaxRDDLCompiler import JaxRDDLCompiler
from pyRDDLGym.Core.Jax.JaxRDDLCompilerWithGrad import FuzzyLogic
from pyRDDLGym.Core.Jax.JaxRDDLCompilerWithGrad import JaxRDDLCompilerWithGrad
from pyRDDLGym.Core.Jax.JaxRDDLCompilerWithGrad import ProductLogic

 
class JaxRDDLBackpropPlanner:
    
    def __init__(self, rddl: RDDLLiftedModel,
                 key: jax.random.PRNGKey,
                 batch_size_train: int,
                 batch_size_test: int=None,
                 action_bounds: Dict={},
                 initializer: initializers.Initializer=initializers.zeros,
                 optimizer: optax.GradientTransformation=optax.rmsprop(0.1),
                 logic: FuzzyLogic=ProductLogic()) -> None:
        
        # jax compilation will only work on lifted domains for now
        if not isinstance(rddl, RDDLLiftedModel) or rddl.is_grounded:
            raise RDDLNotImplementedError(
                'Jax compilation only works on lifted domains for now.')
            
        self.rddl = rddl
        self.key = key
        self.batch_size_train = batch_size_train
        if batch_size_test is None:
            batch_size_test = batch_size_train
        self.batch_size_test = batch_size_test
        self._action_bounds = action_bounds
        self.initializer = initializer
        self.optimizer = optimizer
        self.logic = logic
        
        self._compile_rddl()
        self._compile_action_info()        
        self._compile_backprop()
    
    # ===========================================================================
    # compilation of RDDL file to JAX
    # ===========================================================================
    
    def _compile_rddl(self):
        self.compiled = JaxRDDLCompilerWithGrad(rddl=self.rddl, logic=self.logic)
        self.compiled.compile()
        
        self.test_compiled = JaxRDDLCompiler(rddl=self.rddl)
        self.test_compiled.compile()

    def _compile_action_info(self):
        self.action_shapes, self.action_bounds = {}, {}
        self.bool_actions = False
        for name, prange in self.rddl.variable_ranges.items():
            if self.rddl.variable_types[name] == 'action-fluent':
                value = self.compiled.init_values[name]
                shape = (self.rddl.horizon,)
                if type(value) is np.ndarray:
                    shape = shape + value.shape
                self.action_shapes[name] = shape
                    
                if prange not in JaxRDDLCompiler.JAX_TYPES:
                    raise RDDLTypeError(
                        f'Invalid range {prange} of action-fluent <{name}>, '
                        f'must be one of {set(JaxRDDLCompiler.JAX_TYPES.keys())}.')
                    
                if prange == 'bool':
                    self.action_bounds[name] = (0.0, 1.0)
                    self.bool_actions = True
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
            for var, param in params.items():
                if self.rddl.variable_ranges[var] == 'real': 
                    plan[var] = param[step]
                else:
                    plan[var] = jnp.asarray(
                        param[step], dtype=JaxRDDLCompiler.REAL)
            return plan, key
        
        def _hard(subs, step, params, key):
            soft, key = _soft(subs, step, params, key)
            hard = {}
            for var, param in soft.items():
                prange = self.rddl.variable_ranges[var]
                if prange == 'real':
                    hard[var] = param
                elif prange == 'int':
                    hard[var] = jnp.asarray(
                        jnp.round(param), dtype=JaxRDDLCompiler.INT)
                elif prange == 'bool':
                    hard[var] = param > 0.5
            return hard, key
        
        return _soft, _hard
             
    def _jax_init(self, initializer, optimizer):
        
        def _init(key):
            params = {}
            for var, shape in self.action_shapes.items():
                key, subkey = random.split(key)
                param = initializer(subkey, shape, dtype=JaxRDDLCompiler.REAL)
                param = jnp.clip(param, *self.action_bounds[var])
                params[var] = param
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
            for var, param in params.items():
                new_params[var] = jnp.clip(param, *self.action_bounds[var])
            return new_params
        
        use_sogbofa_clip_trick = self.bool_actions and \
            self.rddl.max_allowed_actions < len(self.rddl.actions)
        
        def _sogbofa_surplus(params):
            total, count = 0.0, 0
            for var, param in params.items():
                if self.rddl.variable_ranges[var] == 'bool':
                    total += jnp.sum(param)
                    count += jnp.sum(param > 0)
            surplus = (total - self.rddl.max_allowed_actions) / jnp.maximum(count, 1)
            return (surplus, count)
        
        def _sogbofa_clip_condition(values):
            _, (surplus, count) = values
            return jnp.logical_and(count > 0, surplus > 0)
        
        def _sogbofa_clip_body(values):
            params, (surplus, _) = values
            new_params = {}
            for var, param in params.items():
                if self.rddl.variable_ranges[var] == 'bool':
                    new_params[var] = jnp.maximum(param - surplus, 0.0)
                else:
                    new_params[var] = param
            new_surplus = _sogbofa_surplus(new_params)
            return (new_params, new_surplus)
        
        def _sogbofa_clip(params):
            surplus = _sogbofa_surplus(params)
            init_values = (params, surplus)
            params, _ = jax.lax.while_loop(
                _sogbofa_clip_condition, _sogbofa_clip_body, init_values)
            return params
        
        def _sogbofa_clip_batched(params):
            return jax.vmap(_sogbofa_clip, in_axes=0)(params)
            
        def _update(params, key, opt_state):
            grad, (key, logged) = jax.grad(loss, has_aux=True)(params, key)
            updates, opt_state = optimizer.update(grad, opt_state)
            params = optax.apply_updates(params, updates)
            params = _clip(params)
            if use_sogbofa_clip_trick:
                params = _sogbofa_clip_batched(params)
            return params, key, opt_state, logged
        
        return _update
            
    # ===========================================================================
    # training
    # ===========================================================================
    
    def optimize(self, epochs: int, step: int=1) -> Iterable[Dict[str, object]]:
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
                            'params': params,
                            'best_params': best_params,
                            **test_log}
                yield callback
                
    def get_plan(self, params, key):
        plan = []
        for step in range(self.rddl.horizon):
            actions, key = self.test_policy(None, step, params, key)
            actions = jax.tree_map(np.ravel, actions)
            grounded_actions = {}
            for var, action in actions.items():
                grounded_action = self.compiled.tensors.expand(var, action)
                if self.rddl.variable_ranges[var] == 'bool':
                    grounded_action = {gvar: value 
                                       for gvar, value in grounded_action
                                       if value == True}
                grounded_actions.update(grounded_action)
            plan.append(grounded_actions)
        return plan, key
