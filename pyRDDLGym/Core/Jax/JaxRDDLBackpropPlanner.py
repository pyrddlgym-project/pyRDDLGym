import jax
import jax.numpy as jnp
import jax.random as random
import jax.nn.initializers as initializers
import numpy as np
np.seterr(all='raise')
import optax
from typing import Callable, Dict, Iterable, Set, Tuple, Optional
import warnings

from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLTypeError

from pyRDDLGym.Core.Compiler.RDDLLiftedModel import RDDLLiftedModel
from pyRDDLGym.Core.Jax.JaxRDDLCompiler import JaxRDDLCompiler
from pyRDDLGym.Core.Jax.JaxRDDLLogic import FuzzyLogic


class JaxRDDLCompilerWithGrad(JaxRDDLCompiler):
    '''Compiles a RDDL AST representation to an equivalent JAX representation. 
    Unlike its parent class, this class treats all fluents as real-valued, and
    replaces all mathematical operations by equivalent ones with a well defined 
    (e.g. non-zero) gradient where appropriate. 
    '''
    
    def __init__(self, *args,
                 logic: FuzzyLogic=FuzzyLogic(),
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
            self.init_values[var] = np.asarray(values, dtype=self.REAL) 
        
        # overwrite basic operations with fuzzy ones
        self.RELATIONAL_OPS = {
            '>=': logic.greaterEqual(),
            '<=': logic.lessEqual(),
            '<': logic.less(),
            '>': logic.greater(),
            '==': logic.equal(),
            '~=': logic.notEqual()
        }
        self.LOGICAL_NOT = logic.Not()
        self.LOGICAL_OPS = {
            '^': logic.And(),
            '&': logic.And(),
            '|': logic.Or(),
            '~': logic.xor(),
            '=>': logic.implies(),
            '<=>': logic.equiv()
        }
        self.AGGREGATION_OPS['forall'] = logic.forall()
        self.AGGREGATION_OPS['exists'] = logic.exists()
        self.AGGREGATION_OPS['argmin'] = logic.argmin()
        self.AGGREGATION_OPS['argmax'] = logic.argmax()
        self.KNOWN_UNARY['sgn'] = logic.signum()
        self.KNOWN_UNARY['floor'] = logic.floor()   
        self.KNOWN_UNARY['ceil'] = logic.ceil()   
        self.KNOWN_UNARY['round'] = logic.round()
        self.KNOWN_UNARY['sqrt'] = logic.sqrt()
        self.KNOWN_BINARY['div'] = logic.floorDiv()
        self.KNOWN_BINARY['mod'] = logic.mod()
    
    def _jax_stop_grad(self, jax_expr):
        
        def _jax_wrapped_stop_grad(x, params, key):
            sample, key, error = jax_expr(x, params, key)
            sample = jax.lax.stop_gradient(sample)
            return sample, key, error
        
        return _jax_wrapped_stop_grad
        
    def _compile_cpfs(self, info):
        warnings.warn('CPFs outputs will be cast to real.', stacklevel=2)      
        jax_cpfs = {}
        for (_, cpfs) in self.levels.items():
            for cpf in cpfs:
                _, expr = self.rddl.cpfs[cpf]
                jax_cpfs[cpf] = self._jax(expr, info, dtype=self.REAL)
                if cpf in self.cpfs_without_grad:
                    warnings.warn(f'CPF <{cpf}> stops gradient.', stacklevel=2)      
                    jax_cpfs[cpf] = self._jax_stop_grad(jax_cpfs[cpf])
        return jax_cpfs
    
    def _jax_if_helper(self):
        return self.logic.If()
    
    def _jax_switch_helper(self):
        return self.logic.Switch()
        
    def _jax_kron(self, expr, info):
        warnings.warn('KronDelta will be ignored.', stacklevel=2)                       
        arg, = expr.args
        arg = self._jax(arg, info)
        return arg
    
    def _jax_bernoulli_helper(self):
        return self.logic.bernoulli()
    
    def _jax_discrete_helper(self):
        jax_discrete, jax_param = self.logic.discrete()

        def _jax_wrapped_discrete_calc_approx(key, prob, params):
            sample = jax_discrete(key, prob, params)
            out_of_bounds = jnp.logical_not(jnp.logical_and(
                jnp.all(prob >= 0),
                jnp.allclose(jnp.sum(prob, axis=-1), 1.0)))
            return sample, out_of_bounds
        
        return _jax_wrapped_discrete_calc_approx, jax_param


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
    
    def __init__(self, initializer: initializers.Initializer=initializers.normal(),
                 wrap_sigmoid: bool=True,
                 min_action_prob: float=0.001,
                 wrap_non_bool: bool=False,
                 use_new_projection: bool=True,
                 max_constraint_iter: int=999) -> None:
        '''Creates a new straight line plan in JAX.
        
        :param initializer: a Jax Initializer for setting the initial actions
        :param wrap_sigmoid: wrap bool action parameters with sigmoid 
        (uses gradient clipping instead of sigmoid if None)
        :param min_action_prob: minimum value a soft boolean action can take
        (maximum is 1 - min_action_prob); required positive if wrap_sigmoid = True
        :param wrap_non_bool: whether to wrap real or int action fluent parameters
        with non-linearity (e.g. sigmoid or ELU) to satisfy box constraints
        :param use_new_projection: whether to use non-iterative (e.g. sort-based)
        projection method, or modified SOGBOFA projection method to satisfy
        action concurrency constraint
        :param max_constraint_iter: max iterations of projected 
        gradient for ensuring actions satisfy constraints, only required if 
        use_new_projection = True
        '''
        super(JaxStraightLinePlan, self).__init__()
        self._initializer = initializer
        self._wrap_sigmoid = wrap_sigmoid
        self._min_action_prob = min_action_prob
        self._wrap_non_bool = wrap_non_bool
        self._max_constraint_iter = max_constraint_iter
        self._use_new_projection = use_new_projection
        
    def compile(self, compiled: JaxRDDLCompilerWithGrad,
                _bounds: Dict,
                horizon: int) -> None:
        rddl = compiled.rddl
        init = self._initializer
        min_action = self._min_action_prob
        max_action = 1.0 - min_action
        wrap_sigmoid = self._wrap_sigmoid
        bool_threshold = 0.0 if wrap_sigmoid else 0.5
        wrap_non_bool = self._wrap_non_bool
        max_constraint_iter = self._max_constraint_iter
        
        # calculate the correct action box bounds
        shapes, bounds = {}, {}
        ranges = rddl.variable_ranges
        for (name, prange) in ranges.items():
            
            # make sure variable is an action fluent (what we are optimizing)
            if rddl.variable_types[name] != 'action-fluent':
                continue
            
            # prepend the rollout dimension to the action tensor shape
            shapes[name] = (horizon,) + np.shape(compiled.init_values[name])
                
            # the output type is valid for the action
            valid_types = compiled.JAX_TYPES
            if prange not in valid_types:
                raise RDDLTypeError(
                    f'Invalid range <{prange}. of action-fluent <{name}>, '
                    f'must be one of {set(valid_types.keys())}.')
                
            # clip boolean to (0, 1) otherwise use the user action bounds
            # for boolean, the bounds depend on min_action_prob and action weight
            # if wrap_sigmoid = True
            if prange == 'bool':
                bounds[name] = None
            else:
                lower, upper = _bounds.get(name, (-jnp.inf, jnp.inf))
                if lower is None: 
                    lower = -jnp.inf
                if upper is None: 
                    upper = jnp.inf
                bounds[name] = (lower, upper)
            warnings.warn(f'Bounds of action fluent <{name}> parameters '
                          f'set to {bounds[name]}', stacklevel=2)
        self.bounds = bounds
        
        # define the mapping between trainable parameter and action
        def _jax_bool_param_to_action(var, param, hyperparams):
            if wrap_sigmoid:
                weight = hyperparams[var]
                return jax.nn.sigmoid(weight * param)
            else:
                return param 
        
        def _jax_bool_action_to_param(var, action, hyperparams):
            if wrap_sigmoid:
                weight = hyperparams[var]
                return (-1.0 / weight) * jnp.log(1.0 / action - 1.0)
            else:
                return action
            
        def _jax_non_bool_param_to_action(var, param, hyperparams):
            if wrap_non_bool:
                lower, upper = bounds[var]
                if lower > -jnp.inf and upper < jnp.inf:
                    action = lower + (upper - lower) * jax.nn.sigmoid(param)
                elif lower > -jnp.inf:
                    action = lower + jax.nn.elu(param, alpha=1.0) + 1.0
                elif upper < jnp.inf:
                    action = upper - (jax.nn.elu(-param, alpha=1.0) + 1.0)
                else:
                    action = param
            else:
                action = param
            return action
        
        # handle box constraints    
        def _jax_project_bool_to_box(var, param, hyperparams):
            lower = _jax_bool_action_to_param(var, min_action, hyperparams)
            upper = _jax_bool_action_to_param(var, max_action, hyperparams)
            valid_param = jnp.clip(param, lower, upper)
            return valid_param
        
        def _jax_wrapped_slp_project_to_box(params, hyperparams):
            new_params = {}
            for (var, param) in params.items():
                if ranges[var] == 'bool':
                    new_params[var] = _jax_project_bool_to_box(var, param, hyperparams)
                elif wrap_non_bool:
                    new_params[var] = param
                else:
                    new_params[var] = jnp.clip(param, *bounds[var])
            return new_params, True
            
        # initialize the parameters inside their valid ranges
        def _jax_wrapped_slp_init(key, hyperparams, subs):
            params = {}
            for (var, shape) in shapes.items():
                key, subkey = random.split(key)
                param = init(subkey, shape, dtype=compiled.REAL)
                if ranges[var] == 'bool':
                    param += bool_threshold
                params[var] = param
            params, _ = _jax_wrapped_slp_project_to_box(params, hyperparams)
            return params
        
        self.initializer = _jax_wrapped_slp_init
    
        # convert actions that are not smooth to real-valued
        # TODO: use a one-hot for integer actions
        def _jax_wrapped_slp_predict_train(key, params, hyperparams, step, subs):
            actions = {}
            for (var, param) in params.items():
                action = jnp.asarray(param[step, ...], dtype=compiled.REAL)
                if ranges[var] == 'bool':
                    action = _jax_bool_param_to_action(var, action, hyperparams)
                else:
                    action = _jax_non_bool_param_to_action(var, action, hyperparams)
                actions[var] = action
            return actions
        
        # convert smooth actions back to discrete/boolean
        def _jax_wrapped_slp_predict_test(key, params, hyperparams, step, subs):
            actions = {}
            for (var, param) in params.items():
                action = jnp.asarray(param[step, ...])
                prange = ranges[var]
                if prange == 'bool':
                    action = action > bool_threshold
                else:
                    action = _jax_non_bool_param_to_action(var, action, hyperparams)
                    if prange == 'int':
                        action = jnp.round(action).astype(compiled.INT)
                actions[var] = action
            return actions
        
        self.train_policy = _jax_wrapped_slp_predict_train
        self.test_policy = _jax_wrapped_slp_predict_test
             
        # find if action clipping is required for max-definite-actions < pos-inf
        allowed_actions = rddl.max_allowed_actions
        bool_action_count = sum(np.size(values)
                                for (var, values) in rddl.actions.items()
                                if ranges[var] == 'bool')
        use_sogbofa_clip_trick = allowed_actions < bool_action_count
        
        if use_sogbofa_clip_trick: 
            warnings.warn(f'Using projected gradient trick to satisfy '
                          f'max_nondef_actions: total boolean actions '
                          f'{bool_action_count} > max_nondef_actions '
                          f'{allowed_actions}.', stacklevel=2)
            noop = {var: (values if isinstance(values, bool) else values[0])
                    for (var, values) in rddl.actions.items()}
        
        # use new projection method...
        if use_sogbofa_clip_trick and self._use_new_projection:
            
            # shift the boolean actions uniformly, clipping at the min/max values
            # the amount to move is such that only top allowed_actions actions
            # are still active (e.g. not equal to noop) after the shift
            def _jax_wrapped_sorting_project(params, hyperparams):
                
                # find the amount to shift action parameters
                # if noop is True pretend it is False and reflect the parameter
                scores = []
                for (var, param) in params.items():
                    if ranges[var] == 'bool':
                        param_flat = jnp.ravel(param)
                        if noop[var]:
                            param_flat = (-param_flat) if wrap_sigmoid else 1.0 - param_flat
                        scores.append(param_flat)
                scores = jnp.concatenate(scores)
                descending = jnp.sort(scores)[::-1]
                kplus1st_greatest = descending[allowed_actions]
                surplus = jnp.maximum(kplus1st_greatest - bool_threshold, 0.0)
                    
                # perform the shift
                new_params = {}
                for (var, param) in params.items():
                    if ranges[var] == 'bool':
                        new_param = param + (surplus if noop[var] else -surplus)
                        new_param = _jax_project_bool_to_box(var, new_param, hyperparams)
                    else:
                        new_param = param
                    new_params[var] = new_param
                return new_params, True
                
            # clip actions to valid bounds and satisfy constraint on max actions
            def _jax_wrapped_slp_project_to_max_constraint(params, hyperparams):
                params, _ = _jax_wrapped_slp_project_to_box(params, hyperparams)
                project_over_horizon = jax.vmap(
                    _jax_wrapped_sorting_project, in_axes=(0, None)
                )(params, hyperparams)
                return project_over_horizon
            
            self.projection = _jax_wrapped_slp_project_to_max_constraint
        
        # use SOGBOFA projection method...
        elif use_sogbofa_clip_trick and not self._use_new_projection:
            
            # calculate the surplus of actions above max-nondef-actions
            def _jax_wrapped_sogbofa_surplus(params, hyperparams):
                sum_action, count = 0.0, 0
                for (var, param) in params.items():
                    if ranges[var] == 'bool':
                        action = _jax_bool_param_to_action(var, param, hyperparams)                        
                        if noop[var]:
                            sum_action += jnp.size(action) - jnp.sum(action)
                            count += jnp.sum(action < 1)
                        else:
                            sum_action += jnp.sum(action)
                            count += jnp.sum(action > 0)
                surplus = jnp.maximum(sum_action - allowed_actions, 0.0)
                count = jnp.maximum(count, 1)
                return surplus / count
                
            # return whether the surplus is positive or reached compute limit
            def _jax_wrapped_sogbofa_continue(values):
                it, _, _, surplus = values
                return jnp.logical_and(it < max_constraint_iter, surplus > 0)
                
            # reduce all bool action values by the surplus clipping at minimum
            # for no-op = True, do the opposite, i.e. increase all
            # bool action values by surplus clipping at maximum
            def _jax_wrapped_sogbofa_subtract_surplus(values):
                it, params, hyperparams, surplus = values
                new_params = {}
                for (var, param) in params.items():
                    if ranges[var] == 'bool':
                        action = _jax_bool_param_to_action(var, param, hyperparams)
                        new_action = action + (surplus if noop[var] else -surplus)
                        new_action = jnp.clip(new_action, min_action, max_action)
                        new_param = _jax_bool_action_to_param(var, new_action, hyperparams)
                    else:
                        new_param = param
                    new_params[var] = new_param
                new_surplus = _jax_wrapped_sogbofa_surplus(new_params, hyperparams)
                new_it = it + 1
                return new_it, new_params, hyperparams, new_surplus
                
            # apply the surplus to the actions until it becomes zero
            def _jax_wrapped_sogbofa_project(params, hyperparams):
                surplus = _jax_wrapped_sogbofa_surplus(params, hyperparams)
                _, params, _, surplus = jax.lax.while_loop(
                    cond_fun=_jax_wrapped_sogbofa_continue,
                    body_fun=_jax_wrapped_sogbofa_subtract_surplus,
                    init_val=(0, params, hyperparams, surplus)
                )
                converged = jnp.logical_not(surplus > 0)
                return params, converged
                
            # clip actions to valid bounds and satisfy constraint on max actions
            def _jax_wrapped_slp_project_to_max_constraint(params, hyperparams):
                params, _ = _jax_wrapped_slp_project_to_box(params, hyperparams)
                project_over_horizon = jax.vmap(
                    _jax_wrapped_sogbofa_project, in_axes=(0, None)
                )(params, hyperparams)
                return project_over_horizon
            
            self.projection = _jax_wrapped_slp_project_to_max_constraint
        
        # just project to box constraints
        else: 
            self.projection = _jax_wrapped_slp_project_to_box
    
    @staticmethod
    @jax.jit
    def _guess_next_epoch(param):
        # "progress" the plan one step forward and set last action to second-last
        return jnp.append(param[1:, ...], param[-1:, ...], axis=0)

    def guess_next_epoch(self, params: Dict) -> Dict:
        next_fn = JaxStraightLinePlan._guess_next_epoch
        return jax.tree_map(next_fn, params)

            
class JaxRDDLBackpropPlanner:
    '''A class for optimizing an action sequence in the given RDDL MDP using 
    gradient descent.'''
    
    def __init__(self, rddl: RDDLLiftedModel,
                 plan: JaxPlan,
                 batch_size_train: int,
                 batch_size_test: int=None,
                 rollout_horizon: int=None,
                 use64bit: bool=False,
                 action_bounds: Dict[str, Tuple[float, float]]={},
                 optimizer: Callable[..., optax.GradientTransformation]=optax.rmsprop,
                 optimizer_kwargs: Dict[str, object]={'learning_rate': 0.1},
                 clip_grad: float=None,
                 logic: FuzzyLogic=FuzzyLogic(),
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
        :param use64bit: whether to perform arithmetic in 64 bit
        horizon parameter in the RDDL instance
        :param action_bounds: box constraints on actions
        :param optimizer: a factory for an optax SGD algorithm
        :param optimizer_kwargs: a dictionary of parameters to pass to the SGD
        factory (e.g. which parameters are controllable externally)
        :param clip_grad: maximum magnitude of gradient updates
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
        self.use64bit = use64bit
        self.clip_grad = clip_grad
        
        # set optimizer
        optimizer = optax.inject_hyperparams(optimizer)(**optimizer_kwargs)
        if clip_grad is None:
            self.optimizer = optimizer
        else:
            self.optimizer = optax.chain(
                optax.clip(clip_grad),
                optimizer
            )
            
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
            use64bit=self.use64bit,
            cpfs_without_grad=self.cpfs_without_grad)
        self.compiled.compile()
        
        # Jax compilation of the exact RDDL for testing
        self.test_compiled = JaxRDDLCompiler(
            rddl=rddl,
            use64bit=self.use64bit)
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
        self.initialize = jax.jit(self._jax_init())
        
        # losses
        train_loss = self._jax_loss(train_rollouts, use_symlog=self.use_symlog_reward)
        self.train_loss = jax.jit(train_loss)
        self.test_loss = jax.jit(self._jax_loss(test_rollouts, use_symlog=False))
        
        # optimization
        self.update = jax.jit(self._jax_update(train_loss))
    
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
        def _jax_wrapped_plan_loss(key, policy_params, hyperparams,
                                   subs, model_params):
            log = rollouts(key, policy_params, hyperparams, subs, model_params)
            rewards = log['reward']
            rewards = _jax_wrapped_scale_reward(rewards)
            returns = jnp.sum(rewards, axis=1)
            utility = utility_fn(returns)
            loss = -utility
            return loss, log
        
        return _jax_wrapped_plan_loss
    
    def _jax_init(self):
        init = self.plan.initializer
        optimizer = self.optimizer
        
        def _jax_wrapped_init_policy(key, hyperparams, subs):
            policy_params = init(key, hyperparams, subs)
            opt_state = optimizer.init(policy_params)
            return policy_params, opt_state
        
        return _jax_wrapped_init_policy
        
    def _jax_update(self, loss):
        optimizer = self.optimizer
        projection = self.plan.projection
        
        # calculate the plan gradient w.r.t. return loss and update optimizer
        # also perform a projection step to satisfy constraints on actions
        def _jax_wrapped_plan_update(key, policy_params, hyperparams,
                                     subs, model_params, opt_state):
            grad_fn = jax.grad(loss, argnums=1, has_aux=True)
            grad, log = grad_fn(key, policy_params, hyperparams, subs, model_params)  
            updates, opt_state = optimizer.update(grad, opt_state) 
            policy_params = optax.apply_updates(policy_params, updates)
            policy_params, converged = projection(policy_params, hyperparams)
            log['grad'] = grad
            log['updates'] = updates
            return policy_params, converged, opt_state, log
        
        return _jax_wrapped_plan_update
            
    def _batched_init_subs(self, subs): 
        rddl = self.rddl
        n_train, n_test = self.batch_size_train, self.batch_size_test
        
        # batched subs
        init_train, init_test = {}, {}
        for (name, value) in subs.items():
            value = np.asarray(value)[np.newaxis, ...]
            train_value = np.repeat(value, repeats=n_train, axis=0)
            train_value = train_value.astype(self.compiled.REAL)
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
                 policy_hyperparams: Dict[str, object]=None,
                 subs: Dict[str, object]=None,
                 guess: Dict[str, object]=None) -> Iterable[Dict[str, object]]:
        ''' Compute an optimal straight-line plan.
        
        :param key: JAX PRNG key
        :param epochs: the maximum number of steps of gradient descent
        :param step: frequency the callback is provided back to the user
        :param policy_hyperparams: hyper-parameters for the policy/plan, such as
        weights for sigmoid wrapping boolean actions
        :param subs: dictionary mapping initial state and non-fluents to 
        their values: if None initializes all variables from the RDDL instance
        :param guess: initial policy parameters: if None will use the initializer
        specified in this instance
        '''
        
        # compute a batched version of the initial values
        if subs is None:
            subs = self.test_compiled.init_values
        train_subs, test_subs = self._batched_init_subs(subs)
        
        # initialize, model parameters
        model_params = self.compiled.model_params
        model_params_test = self.test_compiled.model_params
        
        # initialize policy parameters
        if guess is None:
            key, subkey = random.split(key)
            policy_params, opt_state = self.initialize(
                subkey, policy_hyperparams, train_subs)
        else:
            policy_params = guess
            opt_state = self.optimizer.init(policy_params)
        best_params, best_loss = policy_params, jnp.inf
        last_iter_improve = 0
        
        for it in range(epochs):
            
            # update the parameters of the plan
            key, subkey1, subkey2, subkey3 = random.split(key, num=4)
            policy_params, converged, opt_state, train_log = self.update(
                subkey1, policy_params, policy_hyperparams,
                train_subs, model_params, opt_state)
            if not np.all(converged):
                warnings.warn(
                    f'Projected gradient method for satisfying action concurrency '
                    f'constraints reached the iteration limit: plan is possibly '
                    f'invalid for the current instance.', stacklevel=2)
            
            # evaluate losses
            train_loss, _ = self.train_loss(
                subkey2, policy_params, policy_hyperparams,
                train_subs, model_params)
            test_loss, log = self.test_loss(
                subkey3, policy_params, policy_hyperparams,
                test_subs, model_params_test)
            
            # record the best plan so far
            if test_loss < best_loss:
                best_params, best_loss = policy_params, test_loss
                last_iter_improve = it
            
            # periodically return a callback
            if it % step == 0 or it == epochs - 1:
                callback = {
                    'iteration': it,
                    'train_return':-train_loss,
                    'test_return':-test_loss,
                    'best_return':-best_loss,
                    'params': policy_params,
                    'best_params': best_params,
                    'last_iteration_improved': last_iter_improve,
                    'grad': train_log['grad'],
                    'updates': train_log['updates'],
                    **log
                }
                yield callback
    
    def get_action(self, key: random.PRNGKey,
                   params: Dict,
                   step: int,
                   subs: Dict,
                   policy_hyperparams: Dict[str, object]=None) -> Dict[str, object]:
        '''Returns an action dictionary from the policy or plan with the given
        parameters.
        
        :param key: the JAX PRNG key
        :param params: the trainable parameter PyTree of the policy
        :param step: the time step at which decision is made
        :param policy_hyperparams: hyper-parameters for the policy/plan, such as
        weights for sigmoid wrapping boolean actions
        :param subs: the dict of pvariables
        '''
        actions = self.test_policy(key, params, policy_hyperparams, step, subs)
        grounded_actions = {}
        for (var, action) in actions.items():
            for (ground_var, ground_act) in self.rddl.ground_values(var, action):
                if ground_act != self.noop_actions[ground_var]:
                    grounded_actions[ground_var] = ground_act
        return grounded_actions


class JaxRDDLBackpropPlannerMeta(JaxRDDLBackpropPlanner):
    
    def __init__(self, *args,
                 meta_optimizer: optax.GradientTransformation=optax.adam(0.01),
                 **kwargs):
        self.meta_optimizer = meta_optimizer
        super(JaxRDDLBackpropPlannerMeta, self).__init__(*args, **kwargs)
        
    def _jax_init(self):
        init = self.plan.initializer
        optimizer = self.optimizer
        meta_optimizer = self.meta_optimizer
        
        def _jax_wrapped_init_policy(key, hyperparams, subs, model_params):
            policy_params = init(key, hyperparams, subs)
            opt_state = optimizer.init(policy_params)
            meta_state = meta_optimizer.init((hyperparams, model_params))
            new_params = (policy_params, hyperparams, model_params)
            new_state = (opt_state, meta_state)
            return new_params, new_state
        
        return _jax_wrapped_init_policy
        
    def _jax_update(self, loss):
        plan_update_fn = super(JaxRDDLBackpropPlannerMeta, self)._jax_update(loss)
        meta_optimizer = self.meta_optimizer
        
        def _jax_wrapped_meta_loss(key, policy_params, hyperparams,
                                   subs, model_params, opt_state, 
                                   ref_model_params, ref_hyperparams):
            result = plan_update_fn(
                key, policy_params, hyperparams, subs, model_params, opt_state)
            policy_params, *_ = result
            key, subkey = random.split(key)
            meta_loss, _ = loss(
                subkey, policy_params, ref_hyperparams, subs, ref_model_params)
            return meta_loss, result
        
        def _jax_wrapped_meta_update(key, policy_params, hyperparams,
                                     subs, model_params, opt_state, meta_state,
                                     ref_model_params, ref_hyperparams):
            meta_grad_fn = jax.grad(
                _jax_wrapped_meta_loss, argnums=(2, 4), has_aux=True)
            meta_grad, grad_result = meta_grad_fn(
                key, policy_params, hyperparams, subs, model_params, opt_state, 
                ref_model_params, ref_hyperparams)
            meta_updates, meta_state = meta_optimizer.update(meta_grad, meta_state) 
            hyperparams, model_params = optax.apply_updates(
                (hyperparams, model_params), meta_updates)
            hyperparams = {name: jnp.clip(value, 1e-6) 
                           for (name, value) in hyperparams.items()}
            model_params = {name: jnp.clip(value, 1e-6) 
                            for (name, value) in model_params.items()}
            policy_params, converged, opt_state, log = grad_result
            new_params = (policy_params, hyperparams, model_params)
            new_state = (opt_state, meta_state)
            return new_params, converged, new_state, log
        
        return _jax_wrapped_meta_update

    def optimize(self, key: random.PRNGKey,
                 epochs: int,
                 step: int=1,
                 policy_hyperparams: Dict[str, object]=None,
                 subs: Dict[str, object]=None,
                 guess: Dict[str, object]=None) -> Iterable[Dict[str, object]]:
        
        # compute a batched version of the initial values
        if subs is None:
            subs = self.test_compiled.init_values
        train_subs, test_subs = self._batched_init_subs(subs)
        
        # initialize, model parameters
        model_params = self.compiled.model_params
        model_params_test = self.test_compiled.model_params
        ref_model_params = {name: 100.0 for name in model_params}
        ref_hyperparams = {name: 10.0 for name in policy_hyperparams}
        
        # initialize policy parameters
        if guess is None:
            key, subkey = random.split(key)
            (policy_params, hyperparams, model_params), \
            (opt_state, meta_state) = self.initialize(
                subkey, policy_hyperparams, train_subs, model_params)
        else:
            policy_params = guess
            hyperparams = policy_hyperparams
            opt_state = self.optimizer.init(policy_params)
            meta_state = self.meta_optimizer.init((hyperparams, model_params))
        best_params, best_loss = policy_params, jnp.inf
        
        for it in range(epochs):
            
            # update the parameters of the plan and the model_params
            key, subkey1, subkey2, subkey3 = random.split(key, num=4)
            (policy_params, hyperparams, model_params), converged, \
            (opt_state, meta_state), train_log = self.update(
                subkey1, policy_params, policy_hyperparams,
                train_subs, model_params, opt_state, meta_state, ref_model_params,
                ref_hyperparams)
            
            if not np.all(converged):
                warnings.warn(
                    f'Projected gradient method for satisfying action concurrency '
                    f'constraints reached the iteration limit: plan is possibly '
                    f'invalid for the current instance.', stacklevel=2)
            
            # evaluate losses
            train_loss, _ = self.train_loss(
                subkey2, policy_params, policy_hyperparams, 
                train_subs, model_params)
            test_loss, log = self.test_loss(
                subkey3, policy_params, policy_hyperparams,
                test_subs, model_params_test)
            
            # record the best plan so far
            if test_loss < best_loss:
                best_params, best_loss = policy_params, test_loss
            
            # periodically return a callback
            if it % step == 0 or it == epochs - 1:
                callback = {
                    'iteration': it,
                    'train_return':-train_loss,
                    'test_return':-test_loss,
                    'best_return':-best_loss,
                    'params': policy_params,
                    'best_params': best_params,
                    'hyperparams': hyperparams,
                    'model_params': model_params,
                    'grad': train_log['grad'],
                    'updates': train_log['updates'],
                    **log
                }
                yield callback




class JaxRDDLBackpropPlannerMetaWithRegularization(JaxRDDLBackpropPlanner):
    """ WIP """
    def __init__(self, *args,
                 meta_optimizer: optax.GradientTransformation=optax.adam(0.01),
                 reg_param: float=0.,
                 **kwargs):
        self.meta_optimizer = meta_optimizer
        self.reg_param = reg_param
        super(JaxRDDLBackpropPlannerMetaWithRegularization, self).__init__(*args, **kwargs)
        
    def _jax_init(self):
        init = self.plan.initializer
        optimizer = self.optimizer
        meta_optimizer = self.meta_optimizer
        
        def _jax_wrapped_init_policy(key, hyperparams, subs, model_params):
            policy_params = init(key, hyperparams, subs)
            opt_state = optimizer.init(policy_params)
            meta_state = meta_optimizer.init((hyperparams, model_params))
            new_params = (policy_params, hyperparams, model_params)
            new_state = (opt_state, meta_state)
            return new_params, new_state
        
        return _jax_wrapped_init_policy
        
    def _jax_update(self, loss):
        plan_update_fn = super(JaxRDDLBackpropPlannerMetaWithRegularization, self)._jax_update(loss)
        meta_optimizer = self.meta_optimizer
        
        def _jax_wrapped_meta_loss(key, policy_params, hyperparams,
                                   subs, model_params, opt_state, 
                                   ref_model_params, ref_hyperparams):
            result = plan_update_fn(
                key, policy_params, hyperparams, subs, model_params, opt_state)
            policy_params, *_ = result
            key, subkey = random.split(key)
            meta_loss, _ = loss(
                subkey, policy_params, ref_hyperparams, subs, ref_model_params)
            return meta_loss, result
        
        def _jax_wrapped_meta_update(key, policy_params, hyperparams,
                                     subs, model_params, opt_state, meta_state,
                                     ref_model_params, ref_hyperparams):
            meta_grad_fn = jax.grad(
                _jax_wrapped_meta_loss, argnums=(2, 4), has_aux=True)
            meta_grad, grad_result = meta_grad_fn(
                key, policy_params, hyperparams, subs, model_params, opt_state, 
                ref_model_params, ref_hyperparams)

            # Apply L2 regularization
            for k in hyperparams:
                meta_grad[0][k] -= self.reg_param * (1/hyperparams[k])
            for k in model_params:
                meta_grad[1][k] -= self.reg_param * (1/model_params[k])


            meta_updates, meta_state = meta_optimizer.update(meta_grad, meta_state) 
            hyperparams, model_params = optax.apply_updates(
                (hyperparams, model_params), meta_updates)
            hyperparams = {name: jnp.clip(value, 1e-6) 
                           for (name, value) in hyperparams.items()}
            model_params = {name: jnp.clip(value, 1e-6) 
                            for (name, value) in model_params.items()}

            policy_params, converged, opt_state, log = grad_result
            new_params = (policy_params, hyperparams, model_params)
            new_state = (opt_state, meta_state)
            return new_params, converged, new_state, log
        
        return _jax_wrapped_meta_update

    def optimize(self, key: random.PRNGKey,
                 epochs: int,
                 step: int=1,
                 policy_hyperparams: Dict[str, object]=None,
                 subs: Dict[str, object]=None,
                 guess: Dict[str, object]=None) -> Iterable[Dict[str, object]]:
        
        # compute a batched version of the initial values
        if subs is None:
            subs = self.test_compiled.init_values
        train_subs, test_subs = self._batched_init_subs(subs)
        
        # initialize model parameters
        model_params = self.compiled.model_params
        model_params_test = self.test_compiled.model_params
        ref_model_params = {name: 100.0 for name in model_params}
        ref_hyperparams = {name: 10.0 for name in policy_hyperparams}
        
        # initialize policy parameters
        if guess is None:
            key, subkey = random.split(key)
            (policy_params, hyperparams, model_params), \
            (opt_state, meta_state) = self.initialize(
                subkey, policy_hyperparams, train_subs, model_params)
        else:
            policy_params = guess
            hyperparams = policy_hyperparams
            opt_state = self.optimizer.init(policy_params)
            meta_state = self.meta_optimizer.init((hyperparams, model_params))
        best_params, best_loss = policy_params, jnp.inf
        
        for it in range(epochs):
            
            # update the parameters of the plan and the model_params
            key, subkey1, subkey2, subkey3 = random.split(key, num=4)
            (policy_params, hyperparams, model_params), converged, \
            (opt_state, meta_state), train_log = self.update(
                subkey1, policy_params, policy_hyperparams,
#                train_subs, model_params, opt_state, meta_state, ref_model_params,
#                ref_hyperparams)
                train_subs, model_params, opt_state, meta_state, model_params,
                hyperparams)
            
            if not np.all(converged):
                warnings.warn(
                    f'Projected gradient method for satisfying action concurrency '
                    f'constraints reached the iteration limit: plan is possibly '
                    f'invalid for the current instance.', stacklevel=2)
            
            # evaluate losses
            train_loss, _ = self.train_loss(
                subkey2, policy_params, policy_hyperparams, 
                train_subs, model_params)
            test_loss, log = self.test_loss(
                subkey3, policy_params, policy_hyperparams,
                test_subs, model_params_test)
            
            # record the best plan so far
            if test_loss < best_loss:
                best_params, best_loss = policy_params, test_loss
            
            # periodically return a callback
            if it % step == 0 or it == epochs - 1:
                callback = {
                    'iteration': it,
                    'train_return':-train_loss,
                    'test_return':-test_loss,
                    'best_return':-best_loss,
                    'params': policy_params,
                    'best_params': best_params,
                    'hyperparams': hyperparams,
                    'model_params': model_params,
                    'grad': train_log['grad'],
                    'updates': train_log['updates'],
                    **log
                }
                yield callback
