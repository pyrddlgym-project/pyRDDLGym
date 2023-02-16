import jax
import jax.numpy as jnp
import jax.random as random
import jax.nn.initializers as initializers
import numpy as np
import optax
from typing import Dict, Iterable, Set, Tuple
import warnings

from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLTypeError

from pyRDDLGym.Core.Compiler.RDDLLiftedModel import RDDLLiftedModel
from pyRDDLGym.Core.Compiler.RDDLValueInitializer import RDDLValueInitializer
from pyRDDLGym.Core.Jax.JaxRDDLCompiler import JaxRDDLCompiler
from pyRDDLGym.Core.Jax.JaxRDDLLogic import FuzzyLogic, ProductLogic


class JaxRDDLCompilerWithGrad(JaxRDDLCompiler):
    '''Compiles a RDDL AST representation to an equivalent JAX representation. 
    Unlike its parent class, this class treats all fluents as real-valued, and
    replaces all mathematical operations by equivalent ones with a well defined 
    (e.g. non-zero) gradient where appropriate. 
    '''
    
    def __init__(self, *args,
                 logic: FuzzyLogic=ProductLogic(),
                 cpfs_without_grad: Set=set(),
                 **kwargs) -> None:
        '''Creates a new RDDL to Jax compiler, where operations that are not
        differentiable are converted to approximate forms that have defined 
        gradients.
        
        :param *args: arguments to pass to base compiler
        :param logic: Fuzzy logic object that specifies how exact operations
        are converted to their approximate forms: this class may be subclassed
        to customize these operations
        :param cpfs_without_grad: which CPFs do not have gradients (use straight
        through gradient trick)
        :param *kwargs: keyword arguments to pass to base compiler
        '''
        super(JaxRDDLCompilerWithGrad, self).__init__(*args, **kwargs)
        self.logic = logic
        self.cpfs_without_grad = cpfs_without_grad
        
        # actions and CPFs must be continuous
        warnings.warn(f'Initial values of pvariables will be cast to real.',
                      stacklevel=2)   
        for (var, values) in self.init_values.items():
            self.init_values[var] = np.asarray(values, dtype=RDDLValueInitializer.REAL) 
        
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
    
    def _jax_stop_grad(self, jax_expr):
        
        def _jax_wrapped_stop_grad(x, key):
            sample, key, error = jax_expr(x, key)
            sample = jax.lax.stop_gradient(sample)
            return sample, key, error
        
        return _jax_wrapped_stop_grad
        
    def _compile_cpfs(self):
        warnings.warn('CPFs outputs will be cast to real.', stacklevel=2)      
        jax_cpfs = {}
        for (_, cpfs) in self.levels.items():
            for cpf in cpfs:
                _, expr = self.rddl.cpfs[cpf]
                jax_cpfs[cpf] = self._jax(expr, dtype=JaxRDDLCompiler.REAL)
                if cpf in self.cpfs_without_grad:
                    warnings.warn(f'CPF {cpf} uses straight-through gradient.', 
                                  stacklevel=2)      
                    jax_cpfs[cpf] = self._jax_stop_grad(jax_cpfs[cpf])
        return jax_cpfs
    
    def _jax_if_helper(self):
        return self.logic.If
    
    def _jax_switch_helper(self):
        return self.logic.Switch
        
    def _jax_kron(self, expr):
        warnings.warn('KronDelta will be ignored.', stacklevel=2)                       
        arg, = expr.args
        arg = self._jax(arg)
        return arg
    
    def _jax_bernoulli_helper(self):
        return self.logic.bernoulli
    
    def _jax_discrete_helper(self):
        discrete = self.logic.discrete

        def _jax_discrete_calc_approx(key, prob):
            sample = discrete(key, prob)
            out_of_bounds = jnp.logical_not(jnp.logical_and(
                jnp.all(prob >= 0),
                jnp.allclose(jnp.sum(prob, axis=-1), 1.0)))
            return sample, out_of_bounds
        
        return _jax_discrete_calc_approx


class JaxPlan:
    
    def __init__(self) -> None:
        self._initializer = None
        self._train_policy = None
        self._test_policy = None
        self._projection = None
        
    def compile(self, compiled: JaxRDDLCompilerWithGrad,
                _bounds: Dict,
                horizon: int) -> None:
        raise NotImplementedError
    
    def guess_next_epoch(self, params: Dict) -> Dict:
        raise NotImplementedError
    
    @property
    def initializer(self):
        return self._initializer

    @initializer.setter
    def initializer(self, value):
        self._initializer = value
    
    @property
    def train_policy(self):
        return self._train_policy

    @train_policy.setter
    def train_policy(self, value):
        self._train_policy = value
        
    @property
    def test_policy(self):
        return self._test_policy

    @test_policy.setter
    def test_policy(self, value):
        self._test_policy = value
         
    @property
    def projection(self):
        return self._projection

    @projection.setter
    def projection(self, value):
        self._projection = value
        
    
class JaxStraightLinePlan(JaxPlan):
    '''A straight line plan implementation in JAX'''
    
    def __init__(self, initializer: initializers.Initializer=initializers.zeros) -> None:
        '''Creates a new straight line plan in JAX.
        
        :param initializer: a Jax Initializer for setting the initial actions
        '''
        super(JaxStraightLinePlan, self).__init__()
        self._initializer = initializer
        
    def compile(self, compiled: JaxRDDLCompilerWithGrad,
                _bounds: Dict,
                horizon: int) -> None:
        rddl = compiled.rddl
        init = self._initializer
        
        # calculate the correct action box bounds
        shapes, bounds = {}, {}
        for (name, prange) in rddl.variable_ranges.items():
            
            # make sure variable is an action fluent (what we are optimizing)
            if rddl.variable_types[name] != 'action-fluent':
                continue
            
            # prepend the rollout dimension to the action tensor shape
            shapes[name] = (horizon,) + np.shape(compiled.init_values[name])
                
            # the output type is valid for the action
            valid_types = JaxRDDLCompiler.JAX_TYPES
            if prange not in valid_types:
                raise RDDLTypeError(
                    f'Invalid range <{prange}. of action-fluent <{name}>, '
                    f'must be one of {set(valid_types.keys())}.')
                
            # clip boolean to (0, 1) otherwise use the user action bounds
            if prange == 'bool':
                bounds[name] = (0.0, 1.0)
            else:
                bounds[name] = _bounds.get(name, (-jnp.inf, jnp.inf))
                
        # initialize the parameters inside their valid ranges
        def _jax_wrapped_slp_init(key, subs):
            params = {}
            for (var, shape) in shapes.items():
                key, subkey = random.split(key)
                param = init(subkey, shape, dtype=JaxRDDLCompiler.REAL)
                param = jnp.clip(param, *bounds[var])
                params[var] = param
            return params
        
        self.initializer = _jax_wrapped_slp_init
    
        # convert actions that are not smooth to real-valued
        # TODO: use a one-hot for integer actions
        def _jax_wrapped_slp_predict_train(key, params, step, subs):
            actions = {}
            for (var, param) in params.items():
                action = jnp.asarray(param[step, ...], dtype=JaxRDDLCompiler.REAL)
                actions[var] = action
            return actions
        
        # convert smooth actions back to discrete/boolean
        def _jax_wrapped_slp_predict_test(key, params, step, subs):
            actions = {}
            for (var, param) in params.items():
                action = jnp.asarray(param[step, ...])
                prange = rddl.variable_ranges[var]
                if prange == 'int':
                    action = jnp.round(action).astype(JaxRDDLCompiler.INT)
                elif prange == 'bool':
                    action = action > 0.5
                actions[var] = action
            return actions
        
        self.train_policy = _jax_wrapped_slp_predict_train
        self.test_policy = _jax_wrapped_slp_predict_test
             
        # clip actions to valid box constraints
        def _jax_wrapped_slp_project_to_box(params):
            params = {var: jnp.clip(param, *bounds[var])
                      for (var, param) in params.items()}
            return params
            
        # find if action clipping is required for max-definite-actions < pos-inf
        bool_action_count = sum(np.size(values)
                                for (var, values) in rddl.actions.items()
                                if rddl.variable_ranges[var] == 'bool')
        use_sogbofa_clip_trick = rddl.max_allowed_actions < bool_action_count
        
        if use_sogbofa_clip_trick: 
            warnings.warn(f'Using projected gradient trick to satisfy '
                          f'max_nondef_actions: total boolean actions '
                          f'{bool_action_count} > max_nondef_actions '
                          f'{rddl.max_allowed_actions}.', stacklevel=2)
            
            # calculate the surplus of actions above max-nondef-actions
            def _jax_wrapped_sogbofa_surplus(params):
                surplus, count = 0.0, 0
                for (var, param) in params.items():
                    if rddl.variable_ranges[var] == 'bool':
                        surplus += jnp.sum(param)
                        count += jnp.sum(param > 0)
                count = jnp.maximum(count, 1)
                return (surplus - rddl.max_allowed_actions) / count
                
            # return whether the surplus is positive
            def _jax_wrapped_sogbofa_positive_surplus(values):
                _, surplus = values
                return surplus > 0
            
            # reduce all bool action values by the surplus clipping at zero
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
            
            # reduce bool action values by surplus until it becomes zero
            def _jax_wrapped_sogbofa_project(params):
                surplus = _jax_wrapped_sogbofa_surplus(params)
                params, _ = jax.lax.while_loop(
                    cond_fun=_jax_wrapped_sogbofa_positive_surplus,
                    body_fun=_jax_wrapped_sogbofa_subtract_surplus,
                    init_val=(params, surplus)
                )
                return params
            
            # clip actions to valid bounds and satisfy constraint on max actions
            def _jax_wrapped_slp_project_to_max_constraint(params):
                params = _jax_wrapped_slp_project_to_box(params)
                project_over_horizon = jax.vmap(
                    _jax_wrapped_sogbofa_project, in_axes=0
                )
                params = project_over_horizon(params)
                return params
            
            self.projection = _jax_wrapped_slp_project_to_max_constraint
            
        else:
            
            self.projection = _jax_wrapped_slp_project_to_box
    
    @staticmethod
    @jax.jit
    def _guess_next_epoch(param):
        # "progress" the plan one step forward and set last action to second-last
        return jnp.append(param[1:, ...], param[-1:, ...], axis=0)

    def guess_next_epoch(self, params: Dict) -> Dict:
        return jax.tree_map(JaxStraightLinePlan._guess_next_epoch, params)

            
class JaxRDDLBackpropPlanner:
    '''A class for optimizing an action sequence in the given RDDL MDP using 
    gradient descent.'''
    
    def __init__(self, rddl: RDDLLiftedModel,
                 plan: JaxPlan,
                 batch_size_train: int,
                 batch_size_test: int=None,
                 rollout_horizon: int=None,
                 action_bounds: Dict[str, Tuple[float, float]]={},
                 optimizer: optax.GradientTransformation=optax.rmsprop(0.1),
                 normalize_grad: bool=False,
                 logic: FuzzyLogic=ProductLogic(),
                 use_symlog_reward: bool=False,
                 utility=jnp.mean,
                 cpfs_without_grad: Set=set()) -> None:
        '''Creates a new gradient-based algorithm for optimizing action sequences
        (plan) in the given RDDL. Some operations will be converted to their
        differentiable counterparts; the specific operations can be customized
        by providing a subclass of FuzzyLogic.
        
        :param rddl: the RDDL domain to optimize
        :param plan: the policy/plan representation to optimize
        :param batch_size_train: how many rollouts to perform per optimization 
        step
        :param batch_size_test: how many rollouts to use to test the plan at each
        optimization step
        :param rollout_horizon: lookahead planning horizon: None uses the
        horizon parameter in the RDDL instance
        :param action_bounds: box constraints on actions
        :param optimizer: an Optax algorithm that specifies how gradient updates
        are performed
        :param normalize_grad: whether to normalize gradient during optimization
        :param logic: a subclass of FuzzyLogic for mapping exact mathematical
        operations to their differentiable counterparts 
        :param use_symlog_reward: whether to use the symlog transform on the 
        reward as a form of normalization
        :param utility: how to aggregate return observations to compute utility
        of a policy or plan
        :param cpfs_without_grad: which CPFs do not have gradients (use straight
        through gradient trick)
        '''
        self.rddl = rddl
        self.plan = plan
        self.batch_size_train = batch_size_train
        if batch_size_test is None:
            batch_size_test = batch_size_train
        self.batch_size_test = batch_size_test
        if rollout_horizon is None:
            rollout_horizon = rddl.horizon
        self.horizon = rollout_horizon
        self._action_bounds = action_bounds
        self.optimizer = optimizer
        self.normalize_grad = normalize_grad
        
        self.logic = logic
        self.use_symlog_reward = use_symlog_reward
        self.utility = utility
        self.cpfs_without_grad = cpfs_without_grad
        
        self._jax_compile_rddl()        
        self._jax_compile_optimizer()
        
    def _jax_compile_rddl(self):
        rddl = self.rddl
        
        # Jax compilation of the differentiable RDDL for training
        self.compiled = JaxRDDLCompilerWithGrad(
            rddl=rddl,
            logic=self.logic, 
            cpfs_without_grad=self.cpfs_without_grad)
        self.compiled.compile()
        
        # Jax compilation of the exact RDDL for testing
        self.test_compiled = JaxRDDLCompiler(rddl=rddl)
        self.test_compiled.compile()
    
        # calculate grounded no-op actions
        self.noop_actions = {}
        for (var, values) in self.test_compiled.init_values.items():
            if rddl.variable_types[var] == 'action-fluent':
                self.noop_actions.update(rddl.ground_values(var, values))
        
    def _jax_compile_optimizer(self):
        
        # policy
        self.plan.compile(self.compiled,
                          _bounds=self._action_bounds,
                          horizon=self.horizon)
        self.test_policy = jax.jit(self.plan.test_policy)
        
        # roll-outs
        train_rollouts = self.compiled.compile_rollouts(
            policy=self.plan.train_policy,
            n_steps=self.horizon,
            n_batch=self.batch_size_train)
        
        test_rollouts = self.test_compiled.compile_rollouts(
            policy=self.plan.test_policy,
            n_steps=self.horizon,
            n_batch=self.batch_size_test)
        
        # initialization
        def _jax_wrapped_init_policy(key, subs):
            params = self.plan.initializer(key, subs)
            opt_state = self.optimizer.init(params)
            return params, opt_state
        
        self.initialize = jax.jit(_jax_wrapped_init_policy)
        
        # losses
        train_loss = self._jax_loss(train_rollouts, use_symlog=self.use_symlog_reward)
        self.train_loss = jax.jit(train_loss)
        self.test_loss = jax.jit(self._jax_loss(test_rollouts, use_symlog=False))
        
        # optimization
        self.update = jax.jit(self._jax_update(
            train_loss, self.optimizer, self.plan.projection))
        
    def _jax_loss(self, rollouts, use_symlog=False):
        gamma = self.rddl.discount
        utility_fn = self.utility
        
        # symlog transform sign(x) * ln(|x| + 1) and discounting
        def _jax_wrapped_scale_reward(rewards):
            if use_symlog:
                rewards = jnp.sign(rewards) * jnp.log1p(jnp.abs(rewards))
            if gamma < 1:
                horizon = rewards.shape[1]
                discount = jnp.power(gamma, jnp.arange(horizon))
                discount = discount[jnp.newaxis, ...]
                rewards = rewards * discount
            return rewards
        
        # the loss is the average cumulative reward across all roll-outs
        def _jax_wrapped_plan_loss(key, params, subs):
            log = rollouts(key, params, subs)
            rewards = log['reward']
            rewards = _jax_wrapped_scale_reward(rewards)
            returns = jnp.sum(rewards, axis=1)
            utility = utility_fn(returns)
            loss = -utility
            return loss, log
        
        return _jax_wrapped_plan_loss
    
    def _jax_update(self, loss, optimizer, projection):
        normalize = self.normalize_grad
        
        # calculate the plan gradient w.r.t. return loss and update optimizer
        # optionally does gradient normalization
        # also perform a projection step to satisfy constraints on actions
        def _jax_wrapped_plan_update(key, params, subs, opt_state):
            grad, log = jax.grad(loss, argnums=1, has_aux=True)(key, params, subs)
            if normalize:
                leaves, _ = jax.tree_util.tree_flatten(grad)
                grad_norm = jnp.asarray([jnp.max(jnp.abs(leaf)) for leaf in leaves])
                grad_norm = jnp.max(grad_norm)
                nonzero_norm = jnp.where(grad_norm > 0, grad_norm, 1.0)
                grad = jax.tree_map(lambda g: g / nonzero_norm, grad)
            updates, opt_state = optimizer.update(grad, opt_state)
            params = optax.apply_updates(params, updates)
            params = projection(params)
            return params, opt_state, log
        
        return _jax_wrapped_plan_update
            
    def _batched_init_subs(self, subs): 
        rddl = self.rddl
        n_train, n_test = self.batch_size_train, self.batch_size_test
        
        # batched subs
        init_train, init_test = {}, {}
        for (name, value) in subs.items():
            value = np.asarray(value)[np.newaxis, ...]
            train_value = np.repeat(value, repeats=n_train, axis=0)
            train_value = train_value.astype(JaxRDDLCompiler.REAL)
            init_train[name] = train_value
            init_test[name] = np.repeat(value, repeats=n_test, axis=0)
        
        # make sure next-state fluents are also set
        for (state, next_state) in rddl.next_state.items():
            init_train[next_state] = init_train[state]
            init_test[next_state] = init_test[state]
                        
        return init_train, init_test
    
    def optimize(self, key: random.PRNGKey,
                 epochs: int,
                 step: int=1,
                 subs: Dict[str, object]=None,
                 guess: Dict[str, object]=None) -> Iterable[Dict[str, object]]:
        ''' Compute an optimal straight-line plan.
        
        :param key: JAX PRNG key
        :param epochs: the maximum number of steps of gradient descent
        :param step: frequency the callback is provided back to the user
        :param subs: dictionary mapping initial state and non-fluents to 
        their values: if None initializes all variables from the RDDL instance
        :param guess: initial policy parameters: if None will use the initializer
        specified in this instance
        '''
        
        # compute a batched version of the initial values
        if subs is None:
            subs = self.test_compiled.init_values
        train_subs, test_subs = self._batched_init_subs(subs)
        
        # initialize policy parameters
        if guess is None:
            key, subkey = random.split(key)
            params, opt_state = self.initialize(subkey, train_subs)
        else:
            params = guess
            opt_state = self.optimizer.init(params)
        best_params, best_loss = params, jnp.inf
        
        for it in range(epochs):
            
            # update the parameters of the plan
            key, subkey1, subkey2, subkey3 = random.split(key, num=4)
            params, opt_state, _ = self.update(subkey1, params, train_subs, opt_state)            
            train_loss, _ = self.train_loss(subkey2, params, train_subs)            
            test_loss, log = self.test_loss(subkey3, params, test_subs)
            
            # record the best plan so far
            if test_loss < best_loss:
                best_params, best_loss = params, test_loss
            
            # periodically return a callback
            if it % step == 0 or it == epochs - 1:
                callback = {
                    'iteration': it,
                    'train_return':-train_loss,
                    'test_return':-test_loss,
                    'best_return':-best_loss,
                    'params': params,
                    'best_params': best_params,
                    **log
                }
                yield callback
    
    def get_action(self, key: random.PRNGKey,
                   params: Dict,
                   step: int,
                   subs: Dict):
        '''Returns an action dictionary from the policy or plan with the given
        parameters.
        
        :param key: the JAX PRNG key
        :param params: the trainable parameter PyTree of the policy
        :param step: the time step at which decision is made
        :param subs: the dict of pvariables
        '''
        actions = self.test_policy(key, params, step, subs)
        grounded_actions = {}
        for (var, action) in actions.items():
            for (ground_var, ground_act) in self.rddl.ground_values(var, action):
                if ground_act != self.noop_actions[ground_var]:
                    grounded_actions[ground_var] = ground_act
        return grounded_actions
