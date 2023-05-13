import haiku as hk
import jax
import jax.nn.initializers as initializers
import jax.numpy as jnp
import jax.random as random
import numpy as np
import optax
from typing import Callable, Dict, Sequence

from pyRDDLGym.Core.Compiler.RDDLLiftedModel import RDDLLiftedModel
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxRDDLCompiler
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxRDDLCompilerWithGrad
from pyRDDLGym.Core.Jax.JaxRDDLLogic import FuzzyLogic


class JaxStraightLinePlan:
    
    def __init__(self, initializer: initializers.Initializer=initializers.normal()):
        self._initializer = initializer
        
    def compile(self, compiled: JaxRDDLCompilerWithGrad,
                _bounds: Dict, horizon: int) -> None:
        rddl = compiled.rddl
        
        # calculate the correct action shapes and bounds
        shapes = {var: (horizon,) + np.shape(compiled.init_values[var])
                  for var in rddl.actions}
        action_sizes = {var: np.prod(shape[1:], dtype=int) 
                        for (var, shape) in shapes.items()}
        
        # sample from soft policy
        def _jax_wrapped_stochastic_policy(key, params, hyperparams, step, subs):
            
            # sample one-hot actions from action distribution
            logits = params['action'][step, ...]
            sample = random.categorical(key=key, logits=logits, axis=-1)
            sample = sample[jnp.newaxis, ...]
            onehot = jax.nn.one_hot(sample, num_classes=logits.shape[-1], dtype=bool)
            
            # split one-hot action vector into RDDL action dict
            actions = {}
            start = 0
            for (name, size) in action_sizes.items():
                action = onehot[..., start:start + size]
                action = jnp.reshape(action, newshape=shapes[name][1:])
                actions[name] = action
                start += size
            return actions
        
        self.stochastic_policy = _jax_wrapped_stochastic_policy
        
        # compute action probability of soft policy
        def _jax_wrapped_action_prob(params, states, actions):
            
            # flatten RDDL action dict into one-hot action vector
            flat_actions = []
            for name in action_sizes:
                action = actions[name]
                action = jnp.reshape(action, newshape=action.shape[:2] + (-1,))
                flat_actions.append(action)
            flat_actions = jnp.concatenate(flat_actions, axis=-1, dtype=compiled.REAL)
            
            # add option for setting all actions to false
            noop = 1.0 - jnp.sum(flat_actions, axis=-1, keepdims=True)
            flat_actions = jnp.concatenate([flat_actions, noop], axis=-1)
            
            # compute probability of action sequence
            action_prob = jax.nn.softmax(params['action'], axis=-1)
            action_prob = action_prob[jnp.newaxis, ...]
            return jnp.sum(action_prob * flat_actions, axis=-1)
        
        self.action_prob = jax.jit(_jax_wrapped_action_prob)
        
        # initialize policy parameters
        init = self._initializer
        num_actions = sum(action_sizes.values()) + 1

        def _jax_wrapped_init_policy(key, hyperparams, subs):
            shape = (horizon, num_actions)
            params = init(key, shape, dtype=compiled.REAL)
            params = {'action': params}
            return params
        
        self.initializer = _jax_wrapped_init_policy
            

class JaxDeepReactivePolicy:
    
    def __init__(self, topology: Sequence[int],
                 activation: Callable=jax.nn.relu,
                 initializer: hk.initializers.Initializer=hk.initializers.VarianceScaling(),
                 normalize: bool=True):
        self._topology = topology
        self._activations = [activation for _ in topology]
        self._initializer = initializer
        self._normalize = normalize
        self._eps = 1e-15
    
    def compile(self, compiled: JaxRDDLCompilerWithGrad,
                _bounds: Dict, horizon: int) -> None:
        rddl = compiled.rddl
        
        # calculate the correct action shapes and bounds
        shapes = {var: (horizon,) + np.shape(compiled.init_values[var])
                  for var in rddl.actions}
        action_sizes = {var: np.prod(shape[1:], dtype=int) 
                        for (var, shape) in shapes.items()}
        num_actions = sum(action_sizes.values()) + 1
        
        # predict from policy network
        normalize = self._normalize
        init = self._initializer
        layers = list(enumerate(zip(self._topology, self._activations)))
        
        def _jax_wrapped_policy_network_predict(state):
            
            # apply layer norm
            if normalize:
                normalizer = hk.LayerNorm(
                    axis=-1, param_axis=-1,
                    create_offset=True, create_scale=True,
                    name='input_norm')
                state = normalizer(state)
            
            # compute network prediction
            hidden = state
            for (i, (num, activation)) in layers:
                linear = hk.Linear(num, name=f'hidden_{i + 1}', w_init=init)
                hidden = activation(linear(hidden))
            linear = hk.Linear(num_actions, name='output', w_init=init)
            return jax.nn.softmax(linear(hidden))
        
        predict_fn = hk.transform(_jax_wrapped_policy_network_predict)
        predict_fn = hk.without_apply_rng(predict_fn)
        
        # compute probability distribution over actions
        def _jax_wrapped_action_dist(params, states):
            states = [jnp.reshape(value, newshape=value.shape[:2] + (-1,)) 
                      for (var, value) in states.items() 
                      if var in rddl.states]
            states = jnp.concatenate(states, axis=-1)
            return predict_fn.apply(params, states)
        
        self.action_dist = jax.jit(_jax_wrapped_action_dist)
        
        # compute probability of a trajectory
        def _jax_wrapped_action_prob(params, states, actions):
            
            # flatten RDDL action dict into one-hot action vector
            flat_actions = []
            for name in action_sizes:
                action = actions[name]
                action = jnp.reshape(action, newshape=action.shape[:2] + (-1,))
                flat_actions.append(action)
            flat_actions = jnp.concatenate(
                flat_actions, axis=-1, dtype=compiled.REAL)
            
            # add option for setting all actions to false
            noop = 1.0 - jnp.sum(flat_actions, axis=-1, keepdims=True)
            flat_actions = jnp.concatenate([flat_actions, noop], axis=-1)
            
            # compute probability of action sequence
            action_dist = _jax_wrapped_action_dist(params, states)
            return jnp.sum(action_dist * flat_actions, axis=-1)
        
        self.action_prob = jax.jit(_jax_wrapped_action_prob)
        
        # convert RDDL state dict to vector
        def _jax_subs_to_vec(subs):
            subs = {var: value
                    for (var, value) in subs.items()
                    if var in rddl.states}
            subs = jax.tree_map(jnp.ravel, subs) 
            return jnp.concatenate(list(subs.values()))
        
        # sample from soft policy
        def _jax_wrapped_stochastic_policy(key, params, hyperparams, step, subs):
            
            # compute action log-probability
            states = _jax_subs_to_vec(subs)
            action_prob = predict_fn.apply(params, states)
            logits = jnp.log(action_prob + self._eps)
            
            # sample one-hot actions from action distribution
            sample = random.categorical(key=key, logits=logits, axis=-1)
            sample = sample[jnp.newaxis, ...]
            onehot = jax.nn.one_hot(sample, num_classes=logits.shape[-1], dtype=bool)
            
            # split one-hot action vector into RDDL action dict
            actions = {}
            start = 0
            for (name, size) in action_sizes.items():
                action = onehot[..., start:start + size]
                action = jnp.reshape(action, newshape=shapes[name][1:])
                actions[name] = action
                start += size
            return actions
        
        self.stochastic_policy = _jax_wrapped_stochastic_policy
        
        # initialize policy network
        def _jax_wrapped_init_policy(key, hyperparams, subs):
            subs = {var: value[0, ...] 
                    for (var, value) in subs.items()
                    if var in rddl.states}
            state = _jax_subs_to_vec(subs)
            return predict_fn.init(key, state)
        
        self.initializer = _jax_wrapped_init_policy

        
class JaxRDDLPolicyGradient:
    
    def __init__(self, rddl: RDDLLiftedModel,
                 policy: JaxStraightLinePlan,
                 batch_size: int,
                 covariate_coeff: float=0.0,
                 rollout_horizon: int=None,
                 use64bit: bool=False,
                 optimizer: optax.GradientTransformation=optax.adam(0.001),
                 clip_grad: float=None,
                 logic: FuzzyLogic=FuzzyLogic()):
        self.rddl = rddl
        self.policy = policy
        self.batch_size = batch_size
        self.covariate_coeff = covariate_coeff
        if rollout_horizon is None:
            rollout_horizon = rddl.horizon
        self.horizon = rollout_horizon
        self.use64bit = use64bit
        self.logic = logic
        self.eps = 1e-15
        jax.config.update('jax_log_compiles', True)  # for testing ONLY
        
        if clip_grad is None:
            self.optimizer = optimizer
        else:
            self.optimizer = optax.chain(
                optax.clip(clip_grad),
                optimizer
            )
            
        self._jax_compile()
        
        # calculate grounded no-op actions
        self.noop_actions = {}
        for (var, values) in self.exact_model.init_values.items():
            if rddl.variable_types[var] == 'action-fluent':
                self.noop_actions.update(rddl.ground_values(var, values))
        
    def _jax_compile(self):
        rddl = self.rddl
        policy = self.policy
        
        # compile the approximate RDDL
        self.approx_model = JaxRDDLCompilerWithGrad(
            rddl=rddl, logic=self.logic, use64bit=self.use64bit)
        self.approx_model.compile()
        
        # compile the exact RDDL
        self.exact_model = JaxRDDLCompiler(rddl=rddl, use64bit=self.use64bit)
        self.exact_model.compile()
    
        # compile the stochastic policy
        policy.compile(self.exact_model, _bounds={}, horizon=self.horizon)
        self.stochastic_policy = jax.jit(policy.stochastic_policy)
        
        # compile rollouts
        exact_rollouts = self.exact_model.compile_rollouts(
            policy=policy.stochastic_policy,
            n_steps=self.horizon,
            n_batch=self.batch_size)
        self.exact_rollouts = jax.jit(exact_rollouts)
        
        approx_rollouts = self.approx_model.compile_rollouts(
            policy=policy.stochastic_policy,
            n_steps=self.horizon,
            n_batch=self.batch_size)
        
        # compute trajectory from the stochastic policy
        gamma = rddl.discount
        
        def _jax_wrapped_returns(rewards):
            if gamma != 1:
                discount = gamma ** jnp.arange(rewards.shape[1])
                rewards = rewards * discount[jnp.newaxis, ...]
            return jnp.cumsum(rewards[:,::-1], axis=1)[:,::-1]
            
        def _jax_wrapped_exact_trajectory(key, params, hyperparams, subs, model_params):
            log = exact_rollouts(key, params, hyperparams, subs, model_params)
            states, actions, rewards = log['pvar'], log['action'], log['reward']
            returns = _jax_wrapped_returns(rewards)
            return states, actions, returns
        
        self.exact_trajectory = jax.jit(_jax_wrapped_exact_trajectory)
        
        def _jax_wrapped_approx_trajectory(key, params, hyperparams, subs, model_params):
            log = approx_rollouts(key, params, hyperparams, subs, model_params)
            states, actions, rewards = log['pvar'], log['action'], log['reward']
            returns = _jax_wrapped_returns(rewards)
            return states, actions, returns
        
        self.approx_trajectory = jax.jit(_jax_wrapped_approx_trajectory)
        
        # compute policy gradient
        action_prob_fn = policy.action_prob
        
        def _jax_wrapped_loss(params, trajectory):
            states, actions, returns = trajectory
            action_prob = action_prob_fn(params, states, actions)
            logits = jnp.log(action_prob + self.eps)
            return jnp.mean(-returns * logits, axis=(0, 1))
            
        # optimization
        optimizer = self.optimizer
        
        def _jax_wrapped_policy_update(params, exact_tr, approx_tr, opt_state):
            exact_pg = jax.jacobian(_jax_wrapped_loss)(params, exact_tr)
            # TODO: put control variate code here
            # approx_pg = jax.jacobian(_jax_wrapped_loss)(params, approx_tr)
            updates, opt_state = optimizer.update(exact_pg, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state
        
        self.policy_update = jax.jit(_jax_wrapped_policy_update)
        
        # initialization
        initializer = policy.initializer
        
        def _jax_wrapped_init_policy(key, hyperparams, subs):
            params = initializer(key, hyperparams, subs)
            opt_state = optimizer.init(params)
            return params, opt_state
        
        self.initializer = jax.jit(_jax_wrapped_init_policy)
        
    def _batched_init_subs(self, subs): 
        init_approx, init_exact = {}, {}
        for (name, value) in subs.items():
            value = np.asarray(value)[np.newaxis, ...]
            train_value = np.repeat(value, repeats=self.batch_size, axis=0)
            if name not in self.rddl.actions:
                train_value = train_value.astype(self.approx_model.REAL)
            init_approx[name] = train_value
            init_exact[name] = np.repeat(value, repeats=self.batch_size, axis=0)
        
        # make sure next-state fluents are also set
        for (state, next_state) in self.rddl.next_state.items():
            init_approx[next_state] = init_approx[state]
            init_exact[next_state] = init_exact[state]
        
        return init_approx, init_exact
    
    def optimize(self, key: random.PRNGKey, epochs: int, step: int=1,
                 test_batches: int=20, hyperparams: Dict[str, object]=None):
        model_params = self.approx_model.model_params
        
        # compute a batched version of the initial values
        subs = self.exact_model.init_values
        approx_subs, exact_subs = self._batched_init_subs(subs)

        # initialize policy parameters
        key, subkey = random.split(key)
        params, opt_state = self.initializer(subkey, hyperparams, approx_subs)
        best_params, best_return = params, -jnp.inf

        # main training loop
        for it in range(epochs):
            
            # sample a batch of trajectories
            key, subkey = random.split(key)
            exact_trajectory = self.exact_trajectory(
                subkey, params, hyperparams, exact_subs, model_params)          
            approx_trajectory = self.approx_trajectory(
                subkey, params, hyperparams, approx_subs, model_params)
            
            # do policy gradient updates
            params, opt_state = self.policy_update(
                params, exact_trajectory, approx_trajectory, opt_state)
            
            # compute testing return
            if it % step == 0:
                avg_return = 0.
                for _ in range(test_batches):
                    key, subkey = random.split(key)
                    _, _, returns = self.exact_trajectory(
                        subkey, params, hyperparams, exact_subs, model_params)
                    avg_return += np.mean(returns[:, 0]) / test_batches
                    
                # record the best plan so far
                if avg_return > best_return:
                    best_params, best_return = params, avg_return
                
                yield {'params': params,
                       'avg_return': avg_return,
                       'best_params': best_params,
                       'best_return': best_return}
                
    def get_action(self, key: random.PRNGKey,
                   params: Dict,
                   step: int,
                   subs: Dict,
                   policy_hyperparams: Dict[str, object]=None) -> Dict[str, object]:
        actions = self.stochastic_policy(key, params, policy_hyperparams, step, subs)
        grounded_actions = {}
        for (var, action) in actions.items():
            for (ground_var, ground_act) in self.rddl.ground_values(var, action):
                if ground_act != self.noop_actions[ground_var]:
                    grounded_actions[ground_var] = ground_act
        return grounded_actions
