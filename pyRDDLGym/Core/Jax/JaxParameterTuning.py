import GPyOpt
import jax
import numpy as np
import optax
import time
from typing import Dict, List, Tuple, Union

from pyRDDLGym import ExampleManager
from pyRDDLGym import RDDLEnv
from pyRDDLGym.Core.Jax.JaxRDDLLogic import FuzzyLogic
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxRDDLBackpropPlanner
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxStraightLinePlan
np.seterr(all='warn')


class JaxParameterTuning:
    '''A general-purpose class for tuning a Jax planner.'''
    
    def __init__(self, env: RDDLEnv,
                 action_bounds: Dict[str, Tuple[float, float]],
                 max_train_epochs: int,
                 timeout_episode: float,
                 timeout_epoch: float,
                 gp_iterations: int,
                 initial_stddevs: Union[Tuple[float, float], List[float]],
                 learning_rates: Union[Tuple[float, float], List[float]],
                 model_weights: Union[Tuple[float, float], List[float]],
                 planning_horizons: Union[Tuple[float, float], List[float]],
                 key: int=42,
                 eval_horizon: int=None, 
                 eval_trials: int=5,
                 init_num_points: int=1,
                 batch_size_train: int=32,
                 print_step: int=None,
                 verbose: bool=True,
                 wrap_sigmoid: bool=True,
                 **planner_kwargs) -> None:
        '''Creates a new instance for tuning hyper-parameters for Jax planners
        on the given RDDL domain and instance.
        
        :param env: the RDDLEnv describing the MDP to optimize
        :param action_bounds: the bounds on the actions
        :param max_train_epochs: the maximum number of iterations of SGD per 
        step or trial
        :param timeout_episode: the maximum amount of time to spend training per
        trial (in seconds)
        :param timeout_epoch: the maximum amount of time to spend training per
        decision epoch (in seconds, for MPC only)
        :param gp_iterations: maximum number of steps of Bayesian optimization
        :param initial_stddevs: set of initializer standard deviations for the 
        policy or plan parameters (uses Gaussian noise) to tune: this can either
        be specified as a list enumerating all choices, or a range of possible
        continuous values as a tuple, e.g. (a, b)
        :param learning_rates: set of learning rates for SGD (uses rmsprop by 
        default) to tune
        :param model_weights: set of model weight parameters for continuous 
        relaxation to tune
        :param planning_horizons: set of possible lookahead horizons for MPC to
        tune
        :param key: seed for JAX PRNG key
        :param eval_horizon: maximum number of decision epochs to evaluate (also
        applies for training if using straight-line planning)
        :param eval_trials: how many trials to perform for MPC (batch size has
        the same effect for non-MPC)
        :param init_num_points: number of iterations to seed Bayesian optimizer
        (these are used as initial guess to build loss surrogate model)
        :param batch_size_train: training batch size for planner
        :param print_step: how often to print training callback
        :param verbose: whether to print intermediate results of tuning
        :param wrap_sigmoid: whether to wrap bool action-fluents with sigmoid
        :param planner_kwargs: additional arguments to feed to the planner
        '''
        self.env = env
        self.action_bounds = action_bounds
        self.max_train_epochs = max_train_epochs
        self.timeout_episode = timeout_episode
        self.timeout_epoch = timeout_epoch
        self.key = key
        self.evaluation_horizon = eval_horizon
        if eval_horizon is None:
            self.evaluation_horizon = env.horizon
        self.evaluation_trials = eval_trials
        self.gp_iterations = gp_iterations
        self.init_num_points = init_num_points
        self.batch_size_train = batch_size_train
        self.print_step = print_step
        self.verbose = verbose
        self.wrap_sigmoid = wrap_sigmoid
        self.planner_kwargs = planner_kwargs
        
        # set up bounds for parameter search
        self.bounds = self._get_bounds(
            initial_stddevs, learning_rates, model_weights, planning_horizons)
            
    def _get_bounds(self, initial_stddevs, learning_rates, model_weights,
                    planning_horizons=None):
        ranges = [initial_stddevs, learning_rates, model_weights]
        names = ['init', 'lr', 'w']
        if planning_horizons is not None:
            ranges.append(planning_horizons)
            names.append('T')
        bounds = []
        for name, bound in zip(names, ranges):
            if isinstance(bound, tuple) and len(bound) == 2:
                vtype = 'continuous'
            else:
                vtype = 'discrete'
            bounds.append({'name': name, 'type': vtype, 'domain': bound})
        return bounds
    
    def _train_epoch(self, key, policy_hyperparams, subs, planner, timeout): 
        starttime = None
        for (it, callback) in enumerate(planner.optimize(
            key=key,
            epochs=self.max_train_epochs,
            step=1,
            policy_hyperparams=policy_hyperparams,
            subs=subs
        )):
            if starttime is None:
                starttime = time.time()
            currtime = time.time()  
            elapsed = currtime - starttime    
            if self.print_step is not None and it % self.print_step == 0:
                print('[{:.4f} s] step={} train_return={:.6f} test_return={:.6f}'.format(
                    elapsed,
                    str(callback['iteration']).rjust(4),
                    callback['train_return'],
                    callback['test_return']))
            if not np.isfinite(callback['train_return']):
                if self.verbose:
                    print('aborting due to NaN or inf value')
                break
            if elapsed >= timeout:
                break
        return callback
    
    def _bayes_optimize(self, objective, name):
        myBopt = GPyOpt.methods.BayesianOptimization(
            f=objective,
            domain=self.bounds,
            initial_design_numdata=self.init_num_points,
            maximize=True)
        myBopt.run_optimization(
            self.gp_iterations,
            verbosity=self.verbose,
            save_models_parameters=True,
            report_file=f'{name}_report.txt',
            evaluations_file=f'{name}_eval.txt',
            models_file=f'{name}_models.txt')
        myBopt.plot_acquisition(f'{name}_acq.pdf')
        myBopt.plot_convergence(f'{name}_conv.pdf')

    def tune_slp(self):
        '''Tunes the hyper-parameters for Jax planner using straight-line 
        planning approach.'''
        env = self.env
        timeout_episode = self.timeout_episode

        def objective(x):
            x = np.ravel(x)
            std, lr, w = x[0], x[1], x[2]
            if self.verbose:
                print('\n' + f'optimizing SLP with std={std}, lr={lr}, w={w}...')
            
            # initialize planner
            key = jax.random.PRNGKey(self.key)
            planner = JaxRDDLBackpropPlanner(
                rddl=env.model,
                plan=JaxStraightLinePlan(
                    initializer=jax.nn.initializers.normal(std),
                    wrap_sigmoid=self.wrap_sigmoid),
                batch_size_train=self.batch_size_train,
                rollout_horizon=self.evaluation_horizon,
                action_bounds=self.action_bounds,
                optimizer=optax.rmsprop(lr),
                logic=FuzzyLogic(weight=w),
                **self.planner_kwargs)
            policy_hyperparams = {name: w for name in self.action_bounds}
            
            # perform training
            callback = self._train_epoch(
                key, policy_hyperparams, None, planner, timeout_episode)
            total_reward = callback['best_return']
            if self.verbose:
                print(f'total reward={total_reward}\n')
            return total_reward
        
        # run Bayesian optimization
        self._bayes_optimize(objective, 'gp_slp')
        
    def tune_mpc(self):
        '''Tunes the hyper-parameters for Jax planner using MPC/receding horizon
        planning approach.'''
        env = self.env
        timeout_episode = self.timeout_episode
        timeout_epoch = self.timeout_epoch
        
        def objective(x):
            x = np.ravel(x)
            std, lr, w, T = x[0], x[1], x[2], x[3]
            T = int(T)
            if self.verbose:
                print('\n' + f'optimizing MPC with std={std}, lr={lr}, w={w}, T={T}...')
            
            # initialize planner
            key = jax.random.PRNGKey(self.key)
            planner = JaxRDDLBackpropPlanner(
                rddl=env.model,
                plan=JaxStraightLinePlan(
                    initializer=jax.nn.initializers.normal(std),
                    wrap_sigmoid=self.wrap_sigmoid),
                batch_size_train=self.batch_size_train,
                rollout_horizon=T,
                action_bounds=self.action_bounds,
                optimizer=optax.rmsprop(lr),
                logic=FuzzyLogic(weight=w),
                **self.planner_kwargs)
            policy_hyperparams = {name: w for name in self.action_bounds}
            
            # perform training and collect rewards
            average_reward = 0.0
            for trial in range(self.evaluation_trials):
                if self.verbose:
                    print('\n' + f'starting trial {trial + 1}...')
                total_reward = 0.0
                env.reset() 
                starttime = time.time()
                for step in range(self.evaluation_horizon):
                    currtime = time.time()
                    elapsed = currtime - starttime            
                    if elapsed < timeout_episode:
                        subs = env.sampler.subs
                        timeout = min(timeout_episode - elapsed, timeout_epoch)
                        callback = self._train_epoch(
                            key, policy_hyperparams, subs, planner, timeout)
                        params = callback['best_params']
                        key, subkey = jax.random.split(key)
                        action = planner.get_action(subkey, params, 0, subs)
                    else:
                        action = {}            
                    _, reward, done, _ = env.step(action)
                    total_reward += reward 
                    if self.verbose:
                        print(f'step={step}, reward={reward}')
                    if done: 
                        break  
                if self.verbose:
                    print(f'total reward={total_reward}\n')
                average_reward += total_reward / self.evaluation_trials
            return average_reward
        
        # run Bayesian optimization
        self._bayes_optimize(objective, 'gp_mpc')


if __name__ == '__main__':
    EnvInfo = ExampleManager.GetEnvInfo('Traffic')
    env = RDDLEnv.RDDLEnv(EnvInfo.get_domain(), EnvInfo.get_instance(0))
    tuning = JaxParameterTuning(
        env=env,
        action_bounds={'advance': (0., 1.)},
        max_train_epochs=1000,
        timeout_episode=300,
        timeout_epoch=None,
        gp_iterations=20,
        initial_stddevs=(0., 1.),
        learning_rates=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1],
        model_weights=(0., 200.),
        planning_horizons=None,
        eval_horizon=300,
        batch_size_train=8,
        print_step=100,
        use64bit=True,
        clip_grad=1.0)
    tuning.tune_slp()
