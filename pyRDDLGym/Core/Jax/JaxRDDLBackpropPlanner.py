from functools import partial
import haiku as hk
import jax
import jax.numpy as jnp
import jax.random as random
import jax.nn.initializers as initializers
import numpy as np
import optax
from typing import Dict, Iterable, List, Tuple
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
            
    def _compile_cpfs(self):
        warnings.warn('CPFs outputs will be cast to real.', stacklevel=2)      
        jax_cpfs = {}
        for (_, cpfs) in self.levels.items():
            for cpf in cpfs:
                _, expr = self.rddl.cpfs[cpf]
                jax_cpfs[cpf] = self._jax(expr, dtype=JaxRDDLCompiler.REAL)
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
        
    def compile(self, compiled: JaxRDDLCompilerWithGrad, **kwargs) -> None:
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
        
    def compile(self, compiled: JaxRDDLCompilerWithGrad, **kwargs) -> None:
        rddl = compiled.rddl
        init = self._initializer
        _bounds = kwargs['action_bounds']
        horizon = kwargs['horizon']
        num_train = kwargs['batch_size_train']
        num_test = kwargs['batch_size_test']
        
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
        def _jax_wrapped_slp_init(_, key):
            params = {}
            for (var, shape) in shapes.items():
                key, subkey = random.split(key)
                param = init(subkey, shape, dtype=JaxRDDLCompiler.REAL)
                param = jnp.clip(param, *bounds[var])
                params[var] = param
            return params, key
        
        self.initializer = _jax_wrapped_slp_init
    
        # convert actions that are not smooth to real-valued
        # TODO: use a one-hot for integer actions
        def _jax_wrapped_slp_predict_train(params, step, _, key):
            actions = {}
            for (var, param) in params.items():
                action = jnp.asarray(param[step], dtype=JaxRDDLCompiler.REAL)
                action = action[jnp.newaxis, ...]
                action = jnp.repeat(action, repeats=num_train, axis=0)
                actions[var] = action
            return actions, key
        
        # convert smooth actions back to discrete/boolean
        def _jax_wrapped_slp_predict_test(params, step, _, key):
            actions = {}
            for (var, param) in params.items():
                prange = rddl.variable_ranges[var]
                action = jnp.asarray(param[step])
                if prange == 'int':
                    action = jnp.round(action).astype(JaxRDDLCompiler.INT)
                elif prange == 'bool':
                    action = action > 0.5
                action = action[jnp.newaxis, ...]
                action = jnp.repeat(action, repeats=num_test, axis=0)
                actions[var] = action
            return actions, key
        
        self.train_policy = _jax_wrapped_slp_predict_train
        self.test_policy = _jax_wrapped_slp_predict_test
             
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
                    init_val=(params, surplus))
                return params
            
            # clip actions to valid bounds and satisfy constraint on max actions
            def _jax_wrapped_slp_project_to_max_constraint(params):
                params = {var: jnp.clip(param, *bounds[var])
                          for (var, param) in params.items()}
                params = jax.vmap(_jax_wrapped_sogbofa_project, in_axes=0)(params)
                return params
            
            self.projection = _jax_wrapped_slp_project_to_max_constraint
            
        else:
            
            # just clip actions to valid bounds
            def _jax_wrapped_slp_project_to_box(params):
                params = {var: jnp.clip(param, *bounds[var])
                          for (var, param) in params.items()}
                return params
            
            self.projection = _jax_wrapped_slp_project_to_box
        

class JaxDeepReactivePolicy(JaxPlan):
    '''A deep reactive policy network implementation in JAX'''
    
    def __init__(self, neurons: List[int], activation: str) -> None:
        '''Creates a new deep reactive policy in JAX.
        
        :param neurons: number of neurons per layer
        :param activation: activation function between hidden layers (a string 
        name of a function from jax.nn)
        '''
        super(JaxDeepReactivePolicy, self).__init__()
        self._neurons = neurons
        self._activation = getattr(jax.nn, activation)
        
    def compile(self, compiled: JaxRDDLCompilerWithGrad, **kwargs) -> None:
        rddl = compiled.rddl
        neurons, activation = self._neurons, self._activation
        _bounds = kwargs['action_bounds']
        
        # calculate the correct action box bounds
        shapes, bounds = {}, {}
        for (name, prange) in rddl.variable_ranges.items():
            
            # make sure variable is an action fluent (what we are optimizing)
            if rddl.variable_types[name] != 'action-fluent':
                continue
            
            # get each action tensor shape
            shapes[name] = np.shape(compiled.init_values[name])
                
            # the output type is valid for the action
            valid_types = JaxRDDLCompiler.JAX_TYPES
            if prange not in valid_types:
                raise RDDLTypeError(
                    f'Invalid range {prange} of action-fluent <{name}>, '
                    f'must be one of {set(valid_types.keys())}.')
                
            # clip boolean to (0, 1) otherwise use the user action bounds
            if prange == 'bool':
                bounds[name] = (0.0, 1.0)
            else:
                bounds[name] = _bounds.get(name, (-jnp.inf, jnp.inf))
        
        # transform the state dictionary into a flattened array
        def _jax_wrapped_drp_inputs(subs):
            states = []
            for (var, value) in subs.items():
                if rddl.variable_types[var] == 'state-fluent':
                    state = jnp.atleast_1d(value)
                    states.append(state)
            states_1d = jnp.concatenate(
                states, axis=None, dtype=JaxRDDLCompiler.REAL)
            return states_1d
        
        # construct hidden layers of DRP network
        def _jax_wrapped_drp_hidden(inputs):
            layers = []
            for (i, size) in enumerate(neurons): 
                layer = hk.Linear(size, name=f'hidden_layer_{i + 1}')
                layers.extend([layer, activation])
            mlp = hk.Sequential(layers)
            hidden = mlp(inputs)
            return hidden
            
        # project actions to box constraints
        def _jax_wrapped_drp_constrain_action_to_box(action, bound):
            low, high = bound
            if low > -jnp.inf and high < jnp.inf:
                return low + (high - low) * jax.nn.sigmoid(action)
            elif low > -jnp.inf and high == jnp.inf:
                return low + jnp.exp(action)
            elif low == -jnp.inf and high < jnp.inf:
                return high - jnp.exp(-action)
            else:
                return action
        
        # create an output layer in the policy net for the given action
        def _jax_wrapped_drp_action_layer(var, hidden):
            shape = shapes[var]
            num_actions = np.prod(shape, dtype=int)
            name = var.replace('-', '_')
            layer = hk.Linear(num_actions, name=f'action_layer_{name}')
            action = layer(hidden)
            action = jax.vmap(partial(jnp.reshape, newshape=shape), in_axes=0)(action)
            action = _jax_wrapped_drp_constrain_action_to_box(action, bounds[var])
            return action
        
        # calculate prediction of policy network given state
        def _jax_wrapped_drp_predict(batch):
            inputs = jax.vmap(_jax_wrapped_drp_inputs, in_axes=0)(batch)
            hidden = _jax_wrapped_drp_hidden(inputs)
            actions = {var: _jax_wrapped_drp_action_layer(var, hidden)
                       for var in shapes}
            return actions
        
        drp = hk.transform(_jax_wrapped_drp_predict)
        
        # initialize the parameters inside their valid ranges
        def _jax_wrapped_drp_init(batch, key):
            params = drp.init(key, batch)
            return params, key
        
        self.initializer = _jax_wrapped_drp_init
            
        # extract the predictions of the DRP network
        def _jax_wrapped_drp_predict_train(params, _, batch, key): 
            actions = drp.apply(params, key, batch)
            return actions, key                            
        
        # convert smooth actions back to discrete/boolean
        def _jax_wrapped_drp_predict_test(params, _, batch, key):
            train, key = _jax_wrapped_drp_predict_train(params, None, batch, key)
            actions = {}
            for (var, action) in train.items():
                prange = rddl.variable_ranges[var]
                if prange == 'int':
                    actions[var] = jnp.round(action).astype(JaxRDDLCompiler.INT)
                elif prange == 'bool':
                    actions[var] = action > 0.5
                else:
                    actions[var] = action
            return actions, key
        
        self.train_policy = _jax_wrapped_drp_predict_train
        self.test_policy = _jax_wrapped_drp_predict_test
        
        # projection is dummy
        def _jax_wrapped_slp_project_to_max_constraint(params):
            return params
        
        self.projection = _jax_wrapped_slp_project_to_max_constraint
        
    
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
                 logic: FuzzyLogic=ProductLogic(),
                 use_symlog_reward: bool=False,
                 utility=jnp.mean) -> None:
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
        :param logic: a subclass of FuzzyLogic for mapping exact mathematical
        operations to their differentiable counterparts 
        :param use_symlog_reward: whether to use the symlog transform on the 
        reward as a form of normalization
        :param utility: how to aggregate return observations to compute utility
        of a policy or plan
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
        self.logic = logic
        self.use_symlog_reward = use_symlog_reward
        self.utility = utility
        
        self._jax_compile_rddl()        
        self._jax_compile_optimizer()
        
    def _jax_compile_rddl(self):
        rddl = self.rddl
        
        # Jax compilation of the differentiable RDDL for training
        self.compiled = JaxRDDLCompilerWithGrad(rddl=rddl, logic=self.logic)
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
                          action_bounds=self._action_bounds, 
                          horizon=self.horizon,
                          batch_size_train=self.batch_size_train,
                          batch_size_test=self.batch_size_test)
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
        def _jax_wrapped_init(batch, key):
            params, key = self.plan.initializer(batch, key)
            opt_state = self.optimizer.init(params)
            return params, key, opt_state
        
        self.initialize = jax.jit(_jax_wrapped_init)
        
        # losses
        train_loss = self._jax_loss(
            train_rollouts, use_symlog=self.use_symlog_reward)
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
        def _jax_wrapped_plan_loss(params, batch, key):
            log, key = rollouts(params, batch, key)
            rewards = log['reward']
            rewards = _jax_wrapped_scale_reward(rewards)
            returns = jnp.sum(rewards, axis=-1)
            log['return'] = returns
            utility = utility_fn(returns)
            loss = -utility
            aux = (key, log)
            return loss, aux
        
        return _jax_wrapped_plan_loss
    
    def _jax_update(self, loss, optimizer, projection):
            
        # calculate the plan gradient w.r.t. return loss and update optimizer
        # also perform a projection step to satisfy constraints on actions
        def _jax_wrapped_plan_update(params, batch, key, opt_state):
            trainable = jax.tree_map(jnp.shape, params)
            warnings.warn(f'trainable parameters:\n{trainable}', stacklevel=2)
            grad, (key, log) = jax.grad(loss, has_aux=True)(params, batch, key)
            updates, opt_state = optimizer.update(grad, opt_state)
            params = optax.apply_updates(params, updates)
            params = projection(params)
            return params, key, opt_state, log
        
        return _jax_wrapped_plan_update
            
    def _batched_init_state(self, subs): 
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
    
    def optimize(self, key: jax.random.PRNGKey,
                 epochs: int,
                 step: int=1,
                 init_subs: Dict[str, object]=None) -> Iterable[Dict[str, object]]:
        ''' Compute an optimal straight-line plan.
        
        @param key: JAX PRNG key
        @param epochs: the maximum number of steps of gradient descent
        @param step: frequency the callback is provided back to the user
        @param init_subs: dictionary mapping initial state and non-fluents to 
        their values: if None initializes all variables from the RDDL instance
        '''
        
        # compute a batched version of the initial values
        if init_subs is None:
            init_subs = self.test_compiled.init_values
        train_batch, test_batch = self._batched_init_state(init_subs)
        
        # initialize policy parameters
        params, key, opt_state = self.initialize(train_batch, key)        
        best_params, best_loss = params, jnp.inf
        
        for it in range(epochs):
            
            # update the parameters of the plan
            params, key, opt_state, _ = self.update(params, train_batch, key, opt_state)            
            train_loss, (key, _) = self.train_loss(params, train_batch, key)            
            test_loss, (key, log) = self.test_loss(params, test_batch, key)
            
            # record the best plan so far
            if test_loss < best_loss:
                best_params, best_loss = params, test_loss
            
            # periodically return a callback
            if it % step == 0:
                callback = {'iteration': it,
                            'train_return':-train_loss,
                            'test_return':-test_loss,
                            'best_return':-best_loss,
                            'params': params,
                            'best_params': best_params,
                            **log}
                yield callback
    
    def get_action(self, params, step, subs, key):
        rddl = self.rddl
        noop = self.noop_actions
        
        # predict raw action tensors from plan or policy
        batch = jax.tree_map(partial(np.expand_dims, axis=0), subs)
        actions, key = self.test_policy(params, step, batch, key)
        
        # record only those actions that differ from no-op
        grounded_actions = {}
        for (var, action) in actions.items():
            for (ground_var, ground_act) in rddl.ground_values(var, action[0]):
                if ground_act != noop[ground_var]:
                    grounded_actions[ground_var] = ground_act                    
        return grounded_actions, key
    
