import GPyOpt
import jax
import numpy as np
import optax
import time

from pyRDDLGym import ExampleManager
from pyRDDLGym import RDDLEnv
from pyRDDLGym.Core.Jax.JaxRDDLLogic import FuzzyLogic
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxRDDLBackpropPlanner
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxStraightLinePlan


class JaxParameterTuning:
    '''A general-purpose class for tuning a Jax planner.'''
    
    def __init__(self, env, action_bounds,
                 max_train_epochs, timeout_episode, timeout_epoch,
                 gp_iterations, initial_stddevs, learning_rates, model_weights,
                 planning_horizons, eval_horizon=None,
                 init_num_points=1, batch_size_train=32, print_step=None,
                 verbose=False) -> None:
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
        :param eval_horizon: maximum number of decision epochs to evaluate (also
        applies for training if using straight-line planning)
        :param init_num_points: number of iterations to seed Bayesian optimizer
        (these are used as initial guess to build loss surrogate model)
        :param batch_size_train: training batch size for planner
        :param print_step: how often to print training callback
        :parma verbose: whether to print intermediate results of tuning
        '''
        self.env = env
        self.action_bounds = action_bounds
        self.max_train_epochs = max_train_epochs
        self.timeout_episode = timeout_episode
        self.timeout_epoch = timeout_epoch
        self.evaluation_horizon = eval_horizon
        if eval_horizon is None:
            self.evaluation_horizon = env.horizon
        self.gp_iterations = gp_iterations
        self.init_num_points = init_num_points
        self.batch_size_train = batch_size_train
        self.print_step = print_step
        self.verbose = verbose
        
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
            if elapsed >= timeout:
                break
        return callback
    
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
            key = jax.random.PRNGKey(42)
            planner = JaxRDDLBackpropPlanner(
                rddl=env.model,
                plan=JaxStraightLinePlan(
                    initializer=jax.nn.initializers.normal(std)),
                batch_size_train=self.batch_size_train,
                rollout_horizon=self.evaluation_horizon,
                action_bounds=self.action_bounds,
                optimizer=optax.rmsprop(lr),
                logic=FuzzyLogic(weight=w))
            policy_hyperparams = {name: w for name in self.action_bounds}
            np.seterr(all='warn')
            
            # perform training
            callback = self._train_epoch(
                key, policy_hyperparams, None, planner, timeout_episode)
            total_reward = callback['test_return']
            if self.verbose:
                print(f'total reward={total_reward}\n')
            return total_reward
        
        # run Bayesian optimization
        myBopt = GPyOpt.methods.BayesianOptimization(
            f=objective,
            domain=self.bounds,
            initial_design_numdata=self.init_num_points,
            maximize=True)
        myBopt.run_optimization(self.gp_iterations,
                                verbosity=self.verbose,
                                save_models_parameters=True,
                                report_file='gp_slp_report.txt',
                                evaluations_file='gp_slp_eval.txt',
                                models_file='gp_slp_models.txt')
        myBopt.plot_acquisition('gp_slp_acq.pdf')
        myBopt.plot_convergence('gp_slp_conv.pdf')
        
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
            key = jax.random.PRNGKey(42)
            planner = JaxRDDLBackpropPlanner(
                rddl=env.model,
                plan=JaxStraightLinePlan(
                    initializer=jax.nn.initializers.normal(std)),
                batch_size_train=self.batch_size_train,
                rollout_horizon=T,
                action_bounds=self.action_bounds,
                optimizer=optax.rmsprop(lr),
                logic=FuzzyLogic(weight=w))
            policy_hyperparams = {name: w for name in self.action_bounds}
            np.seterr(all='warn')
            
            # perform training and collect rewards
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
            return total_reward
        
        # run Bayesian optimization
        myBopt = GPyOpt.methods.BayesianOptimization(
            f=objective,
            domain=self.bounds,
            initial_design_numdata=self.init_num_points,
            maximize=True)
        myBopt.run_optimization(self.gp_iterations,
                                verbosity=self.verbose,
                                save_models_parameters=True,
                                report_file='gp_mpc_report.txt',
                                evaluations_file='gp_mpc_eval.txt',
                                models_file='gp_mpc_models.txt')
        myBopt.plot_acquisition('gp_mpc_acq.pdf')
        myBopt.plot_convergence('gp_mpc_conv.pdf')


if __name__ == '__main__':
    EnvInfo = ExampleManager.GetEnvInfo('HVAC')
    env = RDDLEnv.RDDLEnv(EnvInfo.get_domain(), EnvInfo.get_instance(1))
    tuning = JaxParameterTuning(
        env=env,
        action_bounds={'fan-in': (0.05, None), 'heat-input': (0.0, None)},
        max_train_epochs=10000,
        timeout_episode=60,
        timeout_epoch=None,
        gp_iterations=20,
        initial_stddevs=(0., 1.),
        learning_rates=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1],
        model_weights=(0., 200.),
        planning_horizons=None,
        batch_size_train=8,
        eval_horizon=40,
        print_step=500,
        verbose=True)
    tuning.tune_slp()
