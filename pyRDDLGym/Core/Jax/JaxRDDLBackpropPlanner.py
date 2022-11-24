import jax
import jax.numpy as jnp
import jax.random as random
import optax
from typing import Dict, Generator
import warnings

from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLTypeError
from pyRDDLGym.Core.Jax.JaxRDDLCompiler import JaxRDDLCompiler
from pyRDDLGym.Core.Parser.rddl import RDDL


class FuzzyLogic:
    
    def __init__(self, If=jnp.where):
        self.If = If
    
    def And(self, a, b):
        raise NotImplementedError
    
    def Not(self, x):
        raise NotImplementedError
    
    def Or(self, a, b):
        return self.Not(self.And(self.Not(a), self.Not(b)))
    
    def xor(self, a, b):
        return self.And(self.Or(a, b), self.Not(self.And(a, b)))
    
    def implies(self, a, b):
        return self.Or(self.Not(a), b)
    
    def equiv(self, a, b):
        return self.And(self.implies(a, b), self.implies(b, a))
    
    def forall(self, x, axis=None):
        raise NotImplementedError
    
    def exists(self, x, axis=None):
        return self.Not(self.forall(self.Not(x), axis=axis))


class ProductLogic(FuzzyLogic):
    
    def And(self, a, b):
        return a * b
    
    def Not(self, x):
        return 1.0 - x
    
    def implies(self, a, b):
        return 1.0 - a * (1.0 - b)

    def forall(self, x, axis=None):
        return jnp.prod(x, axis=axis)
 

class JaxRDDLBackpropCompiler(JaxRDDLCompiler):
    
    def __init__(self, rddl: RDDL, logic: FuzzyLogic) -> None:
        super(JaxRDDLBackpropCompiler, self).__init__(rddl, allow_discrete=False)
        
        self.LOGICAL_OPS = {
            '^': logic.And,
            '|': logic.Or,
            '~': logic.xor,
            '=>': logic.implies,
            '<=>': logic.equiv
        }
        self.LOGICAL_NOT = logic.Not
        
        self.AGGREGATION_OPS = {
            'sum': jnp.sum,
            'avg': jnp.mean,
            'prod': jnp.prod,
            'min': jnp.min,
            'max': jnp.max,
            'forall': logic.forall,
            'exists': logic.exists
        }
        
        self.CONTROL_OPS = {
            'if': logic.If
        }
    
    def _jax_logical(self, expr, op, params):
        warnings.warn('Logical operator {} will be approximated.'.format(op),
                      FutureWarning, stacklevel=2)
        
        return super(JaxRDDLBackpropCompiler, self)._jax_logical(expr, op, params)
    
    def _jax_aggregation(self, expr, op, params):
        warnings.warn('Aggregation operator {} will be approximated.'.format(op),
                      FutureWarning, stacklevel=2)
        
        return super(JaxRDDLBackpropCompiler, self)._jax_aggregation(expr, op, params)
        
    def _jax_control(self, expr, op, params):
        warnings.warn('If statement will be be approximated.',
                      FutureWarning, stacklevel=2)
        
        return super(JaxRDDLBackpropCompiler, self)._jax_control(expr, op, params)

    def _jax_kron(self, expr, params):
        warnings.warn('KronDelta will be ignored.', FutureWarning, stacklevel=2)     
                       
        arg, = expr.args
        arg = self._jax(arg, params)
        return arg

 
class JaxRDDLBackpropPlanner:
    
    def __init__(self,
                 rddl: RDDL,
                 key: jax.random.PRNGKey,
                 batch_size_train: int,
                 batch_size_test: int=None,
                 action_bounds: Dict={},
                 initializer: jax.nn.initializers.Initializer=jax.nn.initializers.zeros,
                 optimizer: optax.GradientTransformation=optax.rmsprop(0.01),
                 logic: FuzzyLogic=ProductLogic(),
                 log_full_path: bool=False) -> None:
        self.key = key
        if batch_size_test is None:
            batch_size_test = batch_size_train
        self.log_full_path = log_full_path
        
        # compile training graph
        compiled = JaxRDDLBackpropCompiler(rddl, logic=logic)
        compiled.compile()
        self.compiled = compiled
        
        # compile testing graph
        test_compiled = JaxRDDLCompiler(rddl)
        test_compiled.compile()
        self.test_compiled = test_compiled
        
        # get information about action fluent
        self.action_info = {}
        self.action_bounds = {}
        for pvar in rddl.domain.action_fluents.values():
            name = pvar.name      
            atype = pvar.range
            value = compiled.init_values[name]
            shape = (compiled.horizon,)
            if hasattr(value, 'shape'):
                shape = shape + value.shape
            if atype not in JaxRDDLCompiler.RDDL_TO_JAX_TYPE:
                raise RDDLTypeError(
                    'Invalid type {} of action fluent <{}>.'.format(atype, name))
            self.action_info[name] = (atype, shape)
            
            self.action_bounds[name] = action_bounds.get(
                name, (-jnp.inf, +jnp.inf))
            
        # prediction
        predict_map, test_map = self._jax_predict()
        self.test_map = jax.jit(test_map)
        
        # training
        train_rollout = self._jax_rollout(compiled, batch_size_train)
        train_loss = self._jax_evaluate(predict_map, train_rollout)
        self.train_loss = jax.jit(train_loss)
        
        self.initialize = jax.jit(self._jax_init(initializer, optimizer))
        self.update = jax.jit(self._jax_update(train_loss, optimizer))
        
        # testing
        test_rollout = self._jax_rollout(test_compiled, batch_size_test)
        test_loss = self._jax_evaluate(test_map, test_rollout)
        self.test_loss = jax.jit(test_loss)
    
    def _jax_predict(self):
        
        # TODO: use a one-hot for integer actions
        def _soft(params):
            plan = {}
            for name, param in params.items():
                atype, _ = self.action_info[name]                
                if atype == 'bool':
                    action = jax.nn.sigmoid(param)
                elif atype == 'int': 
                    action = jnp.asarray(param, dtype=JaxRDDLCompiler.REAL)
                else:
                    action = param           
                plan[name] = action
            return plan
    
        def _hard(params):
            plan = {}
            for name, value in _soft(params).items():
                atype, _ = self.action_info[name]
                if atype == 'bool':
                    plan[name] = value > 0.5
                elif atype == 'int':
                    plan[name] = jnp.asarray(
                        jnp.round(value), dtype=JaxRDDLCompiler.INT)
                else:
                    plan[name] = value
            return plan
        
        return _soft, _hard
    
    def _jax_rollout(self, compiled, batch_size):
        NORMAL = JaxRDDLCompiler.ERROR_CODES['NORMAL']
        
        init_subs = compiled.init_values.copy()
        for next_state, state in compiled.states.items():
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
            
            for next_state, state in compiled.states.items():
                subs[state] = subs[next_state]
            
            discount *= compiled.discount
            
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
            logged, keys = jax.vmap(
                _rollout, in_axes=(None, 0, 0))(plan, batched_init, subkeys)
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
    
