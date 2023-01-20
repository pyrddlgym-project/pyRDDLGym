import jax
import jax.numpy as jnp
import jax.random as random
import jax.nn.initializers as initializers
import numpy as np
import optax
from typing import Dict, Iterable
import warnings

from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLTypeError

from pyRDDLGym.Core.Compiler.RDDLLiftedModel import RDDLLiftedModel
from pyRDDLGym.Core.Jax.JaxRDDLCompiler import JaxRDDLCompiler
from pyRDDLGym.Core.Jax.JaxRDDLLogic import FuzzyLogic, ProductLogic


class JaxRDDLCompilerWithGrad(JaxRDDLCompiler):
    '''Compiles a RDDL AST representation to an equivalent JAX representation. 
    Unlike its parent class, this class treats all fluents as real-valued, and
    replaces all mathematical operations by equivalent ones with a well defined 
    (e.g. non-zero) gradient where appropriate. 
    '''
    
    def __init__(self, *args, logic: FuzzyLogic=ProductLogic(), **kwargs) -> None:
        '''Creates a new RDDL to Jax compiler, where operations that are not
        differentiable are converted to approximate forms that have defined 
        gradients.
        
        :param *args: arguments to pass to base compiler
        :param logic: Fuzzy logic object that specifies how exact operations
        are converted to their approximate forms: this class may be subclassed
        to customize these operations
        :param *kwargs: keyword arguments to pass to base compiler
        '''
        super(JaxRDDLCompilerWithGrad, self).__init__(*args, **kwargs)
        self.logic = logic
        
        # actions and CPFs must be continuous
        warnings.warn(f'Initial values of CPFs and action-fluents '
                      f'will be cast to real.', stacklevel=2)   
        for (var, values) in self.init_values.items():
            if self.rddl.variable_types[var] != 'non-fluent':
                self.init_values[var] = np.asarray(values, dtype=JaxRDDLCompiler.REAL) 
        
        # overwrite basic operations with fuzzy ones
        self.RELATIONAL_OPS = {
            '>=': logic.greaterEqual,
            '<=': logic.lessEqual,
            '<': logic.less,
            '>': logic.greater,
            '==': logic.equal,
            '~=': logic.notEqual
        }
        self.LOGICAL_NOT = logic.Not  
        self.LOGICAL_OPS = {
            '^': logic.And,
            '&': logic.And,
            '|': logic.Or,
            '~': logic.xor,
            '=>': logic.implies,
            '<=>': logic.equiv
        }
        self.AGGREGATION_OPS = {
            'sum': jnp.sum,
            'avg': jnp.mean,
            'prod': jnp.prod,
            'minimum': jnp.min,
            'maximum': jnp.max,
            'forall': logic.forall,
            'exists': logic.exists,
            'argmin': logic.argmin,
            'argmax': logic.argmax
        }
        self.KNOWN_UNARY['sgn'] = logic.signum        
        self.CONTROL_OPS = {
            'if': logic.If,
            'switch': logic.Switch
        }
            
    def _compile_cpfs(self):
        warnings.warn('CPFs outputs will be cast to real.', stacklevel=2)      
        jax_cpfs = {}
        for (_, cpfs) in self.levels.items():
            for cpf in cpfs:
                _, expr = self.rddl.cpfs[cpf]
                jax_cpfs[cpf] = self._jax(expr, dtype=JaxRDDLCompiler.REAL)
        return jax_cpfs
    
    def _jax_kron(self, expr):
        warnings.warn('KronDelta will be ignored.', stacklevel=2)                       
        arg, = expr.args
        arg = self._jax(arg)
        return arg
    
    def _jax_bernoulli(self, expr): 
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_BERNOULLI']
        JaxRDDLCompiler._check_num_args(expr, 1)
        
        arg_prob, = expr.args
        jax_prob = self._jax(arg_prob)
        bernoulli = self.logic.bernoulli
        
        def _jax_wrapped_distribution_bernoulli(x, key):
            prob, key, err = jax_prob(x, key)
            key, subkey = random.split(key)
            sample = bernoulli(prob, subkey)
            out_of_bounds = jnp.logical_not(jnp.all((prob >= 0) & (prob <= 1)))
            err |= (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_bernoulli
    
    def _jax_discrete_helper(self):
        discrete = self.logic.discrete

        def _jax_discrete_calc_approx(prob, subkey):
            sample = discrete(prob, subkey)
            out_of_bounds = False
            return sample, out_of_bounds
        
        return _jax_discrete_calc_approx

 
class JaxRDDLBackpropPlanner:
    '''A class for optimizing an action sequence in the given RDDL MDP using 
    gradient descent.'''
    
    def __init__(self, rddl: RDDLLiftedModel,
                 key: jax.random.PRNGKey,
                 batch_size_train: int,
                 batch_size_test: int=None,
                 action_bounds: Dict={},
                 initializer: initializers.Initializer=initializers.zeros,
                 optimizer: optax.GradientTransformation=optax.rmsprop(0.1),
                 logic: FuzzyLogic=ProductLogic()) -> None:
        '''Creates a new gradient-based algorithm for optimizing action sequences
        (plan) in the given RDDL. Some operations will be converted to their
        differentiable counterparts; the specific operations can be customized
        by providing a subclass of FuzzyLogic.
        
        :param rddl: the RDDL domain to optimize
        :param key: a Jax PRNG for generating random numbers
        :param batch_size_train: how many rollouts to perform per optimization 
        step
        :param batch_size_test: how many rollouts to use to test the plan at each
        optimization step
        :param action_bounds: dict of valid ranges (min, max) for each action
        :param initializer: a Jax Initializer for setting the initial actions
        :param optimizer: an Optax algorithm that specifies how gradient updates
        are performed
        :param logic: a subclass of FuzzyLogic for mapping exact mathematical
        operations to their differentiable counterparts 
        '''
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
        
        # Jax compilation of the differentiable RDDL for training
        self.compiled = JaxRDDLCompilerWithGrad(rddl=self.rddl, logic=self.logic)
        self.compiled.compile()
        
        # Jax compilation of the exact RDDL for testing
        self.test_compiled = JaxRDDLCompiler(rddl=self.rddl)
        self.test_compiled.compile()

    def _compile_action_info(self):
        self.action_shapes, self.action_bounds = {}, {}
        for (name, prange) in self.rddl.variable_ranges.items():
            if self.rddl.variable_types[name] == 'action-fluent':
                
                # prepend the batch dimension to the action tensor shape
                value = self.compiled.init_values[name]
                self.action_shapes[name] = (self.rddl.horizon,) + np.shape(value)
                
                # the output type is valid for the action
                if prange not in JaxRDDLCompiler.JAX_TYPES:
                    raise RDDLTypeError(
                        f'Invalid range {prange} of action-fluent <{name}>, '
                        f'must be one of {set(JaxRDDLCompiler.JAX_TYPES.keys())}.')
                
                # clip boolean to (0, 1), otherwise try to use user action bounds
                if prange == 'bool':
                    self.action_bounds[name] = (0.0, 1.0)
                else:
                    self.action_bounds[name] = self._action_bounds.get(
                        name, (-np.inf, +np.inf))
    
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
        
        # convert actions that are not smooth to real-valued
        # TODO: use a one-hot for integer actions
        def _jax_wrapped_soft_action(_, step, params, key):
            plan = {}
            for (var, param) in params.items():
                if self.rddl.variable_ranges[var] == 'real': 
                    plan[var] = param[step]
                else:
                    plan[var] = jnp.asarray(
                        param[step], dtype=JaxRDDLCompiler.REAL)
            return plan, key
        
        # convert smooth actions back to discrete/boolean
        def _jax_wrapped_hard_action(subs, step, params, key):
            soft, key = _jax_wrapped_soft_action(subs, step, params, key)
            hard = {}
            for (var, param) in soft.items():
                prange = self.rddl.variable_ranges[var]
                if prange == 'real':
                    hard[var] = param
                elif prange == 'int':
                    hard[var] = jnp.asarray(
                        jnp.round(param), dtype=JaxRDDLCompiler.INT)
                elif prange == 'bool':
                    hard[var] = param > 0.5
            return hard, key
        
        return _jax_wrapped_soft_action, _jax_wrapped_hard_action
             
    def _jax_init(self, initializer, optimizer):
        
        # initialize the parameters and optimizer and clip parameters to bounds
        def _jax_wrapped_plan_init(key):
            params = {}
            for (var, shape) in self.action_shapes.items():
                key, subkey = random.split(key)
                param = initializer(subkey, shape, dtype=JaxRDDLCompiler.REAL)
                param = jnp.clip(param, *self.action_bounds[var])
                params[var] = param
            opt_state = optimizer.init(params)
            return params, key, opt_state
        
        return _jax_wrapped_plan_init
    
    def _jax_loss(self, rollouts):
        
        # the loss is the average return across all roll-outs
        def _jax_wrapped_plan_loss(params, key):
            logged, keys = rollouts(params, key)
            returns = jnp.sum(logged['reward'], axis=-1)
            logged['return'] = returns
            loss = -jnp.mean(returns)
            key = keys[-1]
            return loss, (key, logged)
        
        return _jax_wrapped_plan_loss
    
    def _jax_respect_max_nondef_actions(self):
        rddl = self.rddl
        
        # find if action clipping is required for max-definite-actions < pos-inf
        bool_action_count = sum(np.size(values)
                                for (var, values) in rddl.actions.items()
                                if rddl.variable_ranges[var] == 'bool')
        use_sogbofa_clip_trick = rddl.max_allowed_actions < bool_action_count
        
        # no clipping
        if not use_sogbofa_clip_trick:
            
            def _jax_wrapped_no_action_clip(params):
                return params
            
            return _jax_wrapped_no_action_clip
        
        # clipping is required
        warnings.warn(f'Using projected gradient trick to satisfy '
                      f'max_nondef_actions: total boolean actions '
                      f'{bool_action_count} > max_nondef_actions '
                      f'{rddl.max_allowed_actions}.', stacklevel=2)
        
        # calculates the surplus of actions above max-nondef-actions
        def _jax_wrapped_sogbofa_surplus(params):
            surplus, count = 0.0, 0
            for (var, param) in params.items():
                if rddl.variable_ranges[var] == 'bool':
                    surplus += jnp.sum(param)
                    count += jnp.sum(param > 0)
            return (surplus - rddl.max_allowed_actions) / jnp.maximum(count, 1)
            
        # returns whether the surplus is positive
        def _jax_wrapped_sogbofa_positive_surplus(values):
            _, surplus = values
            return surplus > 0
        
        # reduces all bool action values by the surplus clipping at zero
        def _jax_wrapped_sogbofa_subtract_surplus(values):
            params, surplus = values
            new_params = {}
            for (var, param) in params.items():
                if rddl.variable_ranges[var] == 'bool':
                    new_params[var] = jnp.maximum(param - surplus, 0.0)
                else:
                    new_params[var] = param
            new_surplus = _jax_wrapped_sogbofa_surplus(new_params)
            return new_params, new_surplus
        
        # continues to reduce bool action values by surplus until it becomes zero
        def _jax_wrapped_sogbofa_clip_actions(params):
            surplus = _jax_wrapped_sogbofa_surplus(params)
            params, _ = jax.lax.while_loop(
                cond_fun=_jax_wrapped_sogbofa_positive_surplus,
                body_fun=_jax_wrapped_sogbofa_subtract_surplus,
                init_val=(params, surplus))
            return params
        
        def _jax_wrapped_sogbofa_clip_actions_batched(params):
            return jax.vmap(_jax_wrapped_sogbofa_clip_actions, in_axes=0)(params)
        
        return _jax_wrapped_sogbofa_clip_actions_batched
    
    def _jax_update(self, loss, optimizer):
        _jax_wrapped_sogbofa_clip = self._jax_respect_max_nondef_actions()
        
        # the project gradient applies the action clipping to valid bounds
        # it also applies the SOGBOFA trick to satisfy constraint on concurrency
        def _jax_wrapped_plan_project_gradient(params):
            params = {var: jnp.clip(param, *self.action_bounds[var])
                      for (var, param) in params.items()}
            params = _jax_wrapped_sogbofa_clip(params)
            return params
        
        # calculates the plan gradient w.r.t. return loss and updates optimizer
        # clips actions to bounds and applies concurrent action trick
        def _jax_wrapped_plan_update(params, key, opt_state):
            grad, (key, logged) = jax.grad(loss, has_aux=True)(params, key)
            updates, opt_state = optimizer.update(grad, opt_state)
            params = optax.apply_updates(params, updates)
            params = _jax_wrapped_plan_project_gradient(params)
            return params, key, opt_state, logged
        
        return _jax_wrapped_plan_update
            
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
            
            # update the parameters of the plan
            params, key, opt_state, _ = self.update(params, key, opt_state)            
            train_loss, (key, _) = self.train_loss(params, key)            
            test_loss, (key, test_log) = self.test_loss(params, key)
            self.key = key
            
            # record the best plan so far
            if test_loss < best_loss:
                best_params = params
                best_loss = test_loss
            
            # periodically return a callback
            if it % step == 0:
                callback = {'iteration': it,
                            'train_return': -train_loss,
                            'test_return': -test_loss,
                            'best_return': -best_loss,
                            'params': params,
                            'best_params': best_params,
                            **test_log}
                yield callback
                
    def get_plan(self, params, key):
        plan = [None] * self.rddl.horizon
        for step in range(self.rddl.horizon):
            actions, key = self.test_policy(None, step, params, key)
            actions = jax.tree_map(np.ravel, actions)
            grounded_actions = {}
            for (var, action) in actions.items():
                grounded_action = dict(self.rddl.ground_values(var, action))
                if self.rddl.variable_ranges[var] == 'bool':
                    grounded_action = {gvar: value 
                                       for (gvar, value) in grounded_action.items()
                                       if value == True}
                grounded_actions.update(grounded_action)
            plan[step] = grounded_actions
        return plan, key
