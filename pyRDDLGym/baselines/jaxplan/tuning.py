from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction
from copy import deepcopy
import csv
import datetime
import jax
from multiprocessing import get_context
import numpy as np
import os
import time
from typing import Callable, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

from pyRDDLGym.Core.Env.RDDLEnv import RDDLEnv
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxOfflineController
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxOnlineController
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxRDDLBackpropPlanner
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxStraightLinePlan
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxDeepReactivePolicy

# do this after imports to prevent it from being overwritten
np.seterr(all='warn')


# ===============================================================================
# 
# GENERIC TUNING MODULE
# 
# Currently contains three implementations:
# 1. straight line plan
# 2. re-planning
# 3. deep reactive policies
# 
# ===============================================================================
class JaxParameterTuning:
    '''A general-purpose class for tuning a Jax planner.'''
    
    def __init__(self, env: RDDLEnv,
                 hyperparams_dict: Dict[str, Tuple[float, float, Callable]],
                 train_epochs: int,
                 timeout_training: float,
                 timeout_tuning: float=np.inf,
                 eval_trials: int=5,
                 verbose: bool=True,
                 planner_kwargs: Dict={},
                 plan_kwargs: Dict={},
                 pool_context: str='spawn',
                 num_workers: int=1,
                 poll_frequency: float=0.2,
                 gp_iters: int=25,
                 acquisition=None,
                 gp_init_kwargs: Dict={},
                 gp_params: Dict={'n_restarts_optimizer': 10}) -> None:
        '''Creates a new instance for tuning hyper-parameters for Jax planners
        on the given RDDL domain and instance.
        
        :param env: the RDDLEnv describing the MDP to optimize
        :param hyperparams_dict: dictionary mapping name of each hyperparameter
        to a triple, where the first two elements are lower/upper bounds on the
        parameter value, and the last is a callable mapping the parameter to its
        RDDL equivalent
        :param train_epochs: the maximum number of iterations of SGD per 
        step or trial
        :param timeout_training: the maximum amount of time to spend training per
        trial/decision step (in seconds)
        :param timeout_tuning: the maximum amount of time to spend tuning 
        hyperparameters in general (in seconds)
        :param eval_trials: how many trials to perform independent training
        in order to estimate the return for each set of hyper-parameters
        :param verbose: whether to print intermediate results of tuning
        :param planner_kwargs: additional arguments to feed to the planner
        :param plan_kwargs: additional arguments to feed to the plan/policy
        :param pool_context: context for multiprocessing pool (defaults to 
        "spawn")
        :param num_workers: how many points to evaluate in parallel
        :param poll_frequency: how often (in seconds) to poll for completed
        jobs, necessary if num_workers > 1
        :param gp_iters: number of iterations of optimization
        :param acquisition: acquisition function for Bayesian optimizer
        :parm gp_init_kwargs: additional parameters to feed to Bayesian 
        during initialization  
        :param gp_params: additional parameters to feed to Bayesian optimizer 
        after initialization optimization
        '''
        
        self.env = env
        self.hyperparams_dict = hyperparams_dict
        self.train_epochs = train_epochs
        self.timeout_training = timeout_training
        self.timeout_tuning = timeout_tuning
        self.eval_trials = eval_trials
        self.verbose = verbose
        self.planner_kwargs = planner_kwargs
        self.plan_kwargs = plan_kwargs
        self.pool_context = pool_context
        self.num_workers = num_workers
        self.poll_frequency = poll_frequency
        self.gp_iters = gp_iters
        self.gp_init_kwargs = gp_init_kwargs
        self.gp_params = gp_params
        
        # create acquisition function
        self.acq_args = None
        if acquisition is None:
            num_samples = self.gp_iters * self.num_workers
            acquisition, self.acq_args = JaxParameterTuning._annealing_utility(num_samples)
        self.acquisition = acquisition
    
    def summarize_hyperparameters(self):
        print(f'hyperparameter optimizer parameters:\n'
              f'    tuned_hyper_parameters    ={self.hyperparams_dict}\n'
              f'    initialization_args       ={self.gp_init_kwargs}\n'
              f'    additional_args           ={self.gp_params}\n'
              f'    tuning_iterations         ={self.gp_iters}\n'
              f'    tuning_timeout            ={self.timeout_tuning}\n'
              f'    tuning_batch_size         ={self.num_workers}\n'
              f'    mp_pool_context_type      ={self.pool_context}\n'
              f'    mp_pool_poll_frequency    ={self.poll_frequency}\n'
              f'meta-objective parameters:\n'
              f'    planning_trials_per_iter  ={self.eval_trials}\n'
              f'    planning_iters_per_trial  ={self.train_epochs}\n'
              f'    planning_timeout_per_trial={self.timeout_training}\n'
              f'    acquisition_fn            ={type(self.acquisition).__name__}')
        if self.acq_args is not None:
            print(f'using default acquisition function:\n'
                  f'    utility_kind ={self.acq_args[0]}\n'
                  f'    initial_kappa={self.acq_args[1]}\n'
                  f'    kappa_decay  ={self.acq_args[2]}')
        
    @staticmethod
    def _annealing_utility(n_samples, n_delay_samples=0, kappa1=10.0, kappa2=1.0):
        kappa_decay = (kappa2 / kappa1) ** (1.0 / (n_samples - n_delay_samples))
        utility_fn = UtilityFunction(
            kind='ucb',
            kappa=kappa1,
            kappa_decay=kappa_decay,
            kappa_decay_delay=n_delay_samples)
        utility_args = ['ucb', kappa1, kappa_decay]
        return utility_fn, utility_args
    
    def _pickleable_objective_with_kwargs(self):
        raise NotImplementedError
    
    @staticmethod
    def _wrapped_evaluate(index, params, key, func, kwargs):
        target = func(params=params, kwargs=kwargs, key=key, index=index)
        pid = os.getpid()
        return index, pid, params, target

    def tune(self, key: jax.random.PRNGKey, filename: str,
             save_plot: bool=False) -> Dict[str, object]:
        '''Tunes the hyper-parameters for Jax planner, returns the best found.'''
        self.summarize_hyperparameters()
        
        start_time = time.time()
        
        # objective function
        objective = self._pickleable_objective_with_kwargs()
        evaluate = JaxParameterTuning._wrapped_evaluate
            
        # create optimizer
        hyperparams_bounds = {
            name: hparam[:2] 
            for (name, hparam) in self.hyperparams_dict.items()
        }
        optimizer = BayesianOptimization(
            f=None,  # probe() is not called
            pbounds=hyperparams_bounds,
            allow_duplicate_points=True,  # to avoid crash
            random_state=np.random.RandomState(key),
            **self.gp_init_kwargs
        )
        optimizer.set_gp_params(**self.gp_params)
        utility = self.acquisition
        
        # suggest initial parameters to evaluate
        num_workers = self.num_workers
        suggested, kappas = [], []
        for _ in range(num_workers):
            utility.update_params()
            probe = optimizer.suggest(utility)
            suggested.append(probe)  
            kappas.append(utility.kappa)
        
        # clear and prepare output file
        filename = self._filename(filename, 'csv')
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                ['pid', 'worker', 'iteration', 'target', 'best_target', 'kappa'] + \
                list(hyperparams_bounds.keys())
            )
                
        # start multiprocess evaluation
        worker_ids = list(range(num_workers))
        best_params, best_target = None, -np.inf
        
        for it in range(self.gp_iters): 
            
            # check if there is enough time left for another iteration
            elapsed = time.time() - start_time
            if elapsed >= self.timeout_tuning:
                print(f'global time limit reached at iteration {it}, aborting')
                break
            
            # continue with next iteration
            print('\n' + '*' * 25 + 
                  '\n' + f'[{datetime.timedelta(seconds=elapsed)}] ' + 
                  f'starting iteration {it}' + 
                  '\n' + '*' * 25)
            key, *subkeys = jax.random.split(key, num=num_workers + 1)
            rows = [None] * num_workers
            
            # create worker pool: note each iteration must wait for all workers
            # to finish before moving to the next
            with get_context(self.pool_context).Pool(processes=num_workers) as pool:
                
                # assign jobs to worker pool
                # - each trains on suggested parameters from the last iteration
                # - this way, since each job finishes asynchronously, these
                # parameters usually differ across jobs
                results = [
                    pool.apply_async(evaluate, worker_args + objective)
                    for worker_args in zip(worker_ids, suggested, subkeys)
                ]
            
                # wait for all workers to complete
                while results:
                    time.sleep(self.poll_frequency)
                    
                    # determine which jobs have completed
                    jobs_done = []
                    for (i, candidate) in enumerate(results):
                        if candidate.ready():
                            jobs_done.append(i)
                    
                    # get result from completed jobs
                    for i in jobs_done[::-1]:
                        
                        # extract and register the new evaluation
                        index, pid, params, target = results.pop(i).get()
                        optimizer.register(params, target)
                        
                        # update acquisition function and suggest a new point
                        utility.update_params()  
                        suggested[index] = optimizer.suggest(utility)
                        old_kappa = kappas[index]
                        kappas[index] = utility.kappa
                        
                        # transform suggestion back to natural space
                        rddl_params = {
                            name: pf(params[name])
                            for (name, (*_, pf)) in self.hyperparams_dict.items()
                        }
                        
                        # update the best suggestion so far
                        if target > best_target:
                            best_params, best_target = rddl_params, target
                        
                        # write progress to file in real time
                        rows[index] = [pid, index, it, target, best_target, old_kappa] + \
                                      list(rddl_params.values())
                        
            # write results of all processes in current iteration to file
            with open(filename, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(rows)
        
        # print summary of results
        elapsed = time.time() - start_time
        print(f'summary of hyper-parameter optimization:\n'
              f'    time_elapsed         ={datetime.timedelta(seconds=elapsed)}\n'
              f'    iterations           ={it + 1}\n'
              f'    best_hyper_parameters={best_params}\n'
              f'    best_meta_objective  ={best_target}\n')
        
        if save_plot:
            self._save_plot(filename)
        return best_params

    def _filename(self, name, ext):
        domainName = self.env.model.domainName()
        instName = self.env.model.instanceName()
        domainName = ''.join(c for c in domainName if c.isalnum() or c == '_')
        instName = ''.join(c for c in instName if c.isalnum() or c == '_')
        filename = f'{name}_{domainName}_{instName}.{ext}'
        return filename
    
    def _save_plot(self, filename):
        try:
            import matplotlib.pyplot as plt
            from sklearn.manifold import MDS
        except Exception as e:
            warnings.warn(f'failed to import packages matplotlib or sklearn, '
                          f'aborting plot of search space\n'
                          f'{e}', stacklevel=2)
        else:
            data = np.loadtxt(filename, delimiter=',', dtype=object)
            data, target = data[1:, 3:], data[1:, 2]
            data = data.astype(np.float64)
            target = target.astype(np.float64)
            target = (target - np.min(target)) / (np.max(target) - np.min(target))
            embedding = MDS(n_components=2, normalized_stress='auto')
            data1 = embedding.fit_transform(data)
            sc = plt.scatter(data1[:, 0], data1[:, 1], c=target, s=4.,
                             cmap='seismic', edgecolor='gray',
                             linewidth=0.01, alpha=0.4)
            plt.colorbar(sc)
            plt.savefig(self._filename('gp_points', 'pdf'))
            plt.clf()
            plt.close()


# ===============================================================================
# 
# STRAIGHT LINE PLANNING
#
# ===============================================================================
def objective_slp(params, kwargs, key, index):
                    
    # transform hyper-parameters to natural space
    param_values = [
        pmap(params[name])
        for (name, (*_, pmap)) in kwargs['hyperparams_dict'].items()
    ]
    
    # unpack hyper-parameters
    if kwargs['wrapped_bool_actions']:
        std, lr, w, wa = param_values
    else:
        std, lr, w = param_values
        wa = None                      
    if kwargs['verbose']:
        print(f'[{index}] key={key}, std={std}, lr={lr}, w={w}, wa={wa}...', flush=True)
        
    # initialize planning algorithm
    planner = JaxRDDLBackpropPlanner(
        rddl=deepcopy(kwargs['rddl']),
        plan=JaxStraightLinePlan(
            initializer=jax.nn.initializers.normal(std),
            **kwargs['plan_kwargs']),
        optimizer_kwargs={'learning_rate': lr},
        **kwargs['planner_kwargs'])
    policy_hparams = {name: wa for name in kwargs['wrapped_bool_actions']}  
    model_params = {name: w for name in planner.compiled.model_params}
    
    # initialize policy
    key, subkey = jax.random.split(key)
    policy = JaxOfflineController(
        planner=planner,
        key=subkey,
        eval_hyperparams=policy_hparams,
        train_on_reset=True,
        epochs=kwargs['train_epochs'],
        train_seconds=kwargs['timeout_training'],
        model_params=model_params,
        policy_hyperparams=policy_hparams,
        verbose=0,
        tqdm_position=index)
    
    # initialize env for evaluation (need fresh copy to avoid concurrency)
    env = RDDLEnv(domain=kwargs['domain'],
                  instance=kwargs['instance'],
                  vectorized=True,
                  enforce_action_constraints=True)

    # perform training
    average_reward = 0.0
    for trial in range(kwargs['eval_trials']):
        key, subkey = jax.random.split(key)
        total_reward = policy.evaluate(env, seed=np.array(subkey)[0])['mean']
        if kwargs['verbose']:
            print(f'    [{index}] trial {trial + 1} key={subkey}, '
                  f'reward={total_reward}', flush=True)
        average_reward += total_reward / kwargs['eval_trials']        
    if kwargs['verbose']:
        print(f'[{index}] average reward={average_reward}', flush=True)
    return average_reward

        
def power_ten(x):
    return 10.0 ** x

    
class JaxParameterTuningSLP(JaxParameterTuning):
    
    def __init__(self, *args,
                 hyperparams_dict: Dict[str, Tuple[float, float, Callable]]={
                    'std': (-5., 2., power_ten),
                    'lr': (-5., 2., power_ten),
                    'w': (0., 5., power_ten),
                    'wa': (0., 5., power_ten)
                 },
                 **kwargs) -> None:
        '''Creates a new tuning class for straight line planners.
        
        :param *args: arguments to pass to parent class
        :param hyperparams_dict: same as parent class, but here must contain
        weight initialization (std), learning rate (lr), model weight (w), and
        action weight (wa) if wrap_sigmoid and boolean action fluents exist
        :param **kwargs: keyword arguments to pass to parent class
        '''
        
        super(JaxParameterTuningSLP, self).__init__(
            *args, hyperparams_dict=hyperparams_dict, **kwargs)
        
        # action parameters required if wrap_sigmoid and boolean action exists
        self.wrapped_bool_actions = []
        if self.plan_kwargs.get('wrap_sigmoid', True):
            for var in self.env.model.actions:
                if self.env.model.variable_ranges[var] == 'bool':
                    self.wrapped_bool_actions.append(var)
        if not self.wrapped_bool_actions:
            self.hyperparams_dict.pop('wa', None)
        
    def _pickleable_objective_with_kwargs(self):
        objective_fn = objective_slp
        
        # duplicate planner and plan keyword arguments must be removed
        plan_kwargs = self.plan_kwargs.copy()
        plan_kwargs.pop('initializer', None) 
               
        planner_kwargs = self.planner_kwargs.copy()
        planner_kwargs.pop('rddl', None)
        planner_kwargs.pop('plan', None)
        planner_kwargs.pop('optimizer_kwargs', None)
                    
        kwargs = {
            'rddl': self.env.model,
            'domain': self.env.domain_text,
            'instance': self.env.instance_text,
            'hyperparams_dict': self.hyperparams_dict,
            'timeout_training': self.timeout_training,
            'train_epochs': self.train_epochs,
            'planner_kwargs': planner_kwargs,
            'plan_kwargs': plan_kwargs,
            'verbose': self.verbose,
            'wrapped_bool_actions': self.wrapped_bool_actions,
            'eval_trials': self.eval_trials
        }
        return objective_fn, kwargs


# ===============================================================================
# 
# REPLANNING
#
# ===============================================================================
def objective_replan(params, kwargs, key, index):

    # transform hyper-parameters to natural space
    param_values = [
        pmap(params[name])
        for (name, (*_, pmap)) in kwargs['hyperparams_dict'].items()
    ]
    
    # unpack hyper-parameters
    if kwargs['wrapped_bool_actions']:
        std, lr, w, wa, T = param_values
    else:
        std, lr, w, T = param_values
        wa = None        
    if kwargs['verbose']:
        print(f'[{index}] key={key}, std={std}, lr={lr}, w={w}, wa={wa}, T={T}...', flush=True)

    # initialize planning algorithm
    planner = JaxRDDLBackpropPlanner(
        rddl=deepcopy(kwargs['rddl']),
        plan=JaxStraightLinePlan(
            initializer=jax.nn.initializers.normal(std),
            **kwargs['plan_kwargs']),
        rollout_horizon=T,
        optimizer_kwargs={'learning_rate': lr},
        **kwargs['planner_kwargs'])
    policy_hparams = {name: wa for name in kwargs['wrapped_bool_actions']}
    model_params = {name: w for name in planner.compiled.model_params}
    
    # initialize controller
    key, subkey = jax.random.split(key)
    policy = JaxOnlineController(
        planner=planner,
        key=subkey,
        eval_hyperparams=policy_hparams,
        warm_start=kwargs['use_guess_last_epoch'],
        epochs=kwargs['train_epochs'],
        train_seconds=kwargs['timeout_training'],
        model_params=model_params,
        policy_hyperparams=policy_hparams,
        verbose=0,
        tqdm_position=index)
    
    # initialize env for evaluation (need fresh copy to avoid concurrency)
    env = RDDLEnv(domain=kwargs['domain'],
                  instance=kwargs['instance'],
                  vectorized=True,
                  enforce_action_constraints=True)

    # perform training
    average_reward = 0.0
    for trial in range(kwargs['eval_trials']):
        key, subkey = jax.random.split(key)
        total_reward = policy.evaluate(env, seed=np.array(subkey)[0])['mean']
        if kwargs['verbose']:
            print(f'    [{index}] trial {trial + 1} key={subkey}, '
                  f'reward={total_reward}', flush=True)
        average_reward += total_reward / kwargs['eval_trials']        
    if kwargs['verbose']:
        print(f'[{index}] average reward={average_reward}', flush=True)
    return average_reward

    
class JaxParameterTuningSLPReplan(JaxParameterTuningSLP):
    
    def __init__(self,
                 *args,
                 hyperparams_dict: Dict[str, Tuple[float, float, Callable]]={
                    'std': (-5., 2., power_ten),
                    'lr': (-5., 2., power_ten),
                    'w': (0., 5., power_ten),
                    'wa': (0., 5., power_ten),
                    'T': (1, None, int)
                 },
                 use_guess_last_epoch: bool=True,
                 **kwargs) -> None:
        '''Creates a new tuning class for straight line planners.
        
        :param *args: arguments to pass to parent class
        :param hyperparams_dict: same as parent class, but here must contain
        weight initialization (std), learning rate (lr), model weight (w), 
        action weight (wa) if wrap_sigmoid and boolean action fluents exist, and
        lookahead horizon (T)
        :param use_guess_last_epoch: use the trained parameters from previous 
        decision to warm-start next decision
        :param **kwargs: keyword arguments to pass to parent class
        '''
        
        super(JaxParameterTuningSLPReplan, self).__init__(
            *args, hyperparams_dict=hyperparams_dict, **kwargs)
        
        self.use_guess_last_epoch = use_guess_last_epoch
        
        # set upper range of lookahead horizon to environment horizon
        if self.hyperparams_dict['T'][1] is None:
            self.hyperparams_dict['T'] = (1, self.env.horizon, int)
            
    def _pickleable_objective_with_kwargs(self):
        objective_fn = objective_replan
            
        # duplicate planner and plan keyword arguments must be removed
        plan_kwargs = self.plan_kwargs.copy()
        plan_kwargs.pop('initializer', None)
        
        planner_kwargs = self.planner_kwargs.copy()
        planner_kwargs.pop('rddl', None)
        planner_kwargs.pop('plan', None)
        planner_kwargs.pop('rollout_horizon', None)
        planner_kwargs.pop('optimizer_kwargs', None)
                        
        kwargs = {
            'rddl': self.env.model,
            'domain': self.env.domain_text,
            'instance': self.env.instance_text,
            'hyperparams_dict': self.hyperparams_dict,
            'timeout_training': self.timeout_training,
            'train_epochs': self.train_epochs,
            'planner_kwargs': planner_kwargs,
            'plan_kwargs': plan_kwargs,
            'verbose': self.verbose,
            'wrapped_bool_actions': self.wrapped_bool_actions,
            'eval_trials': self.eval_trials,
            'use_guess_last_epoch': self.use_guess_last_epoch
        }
        return objective_fn, kwargs


# ===============================================================================
# 
# DEEP REACTIVE POLICIES
#
# ===============================================================================
def objective_drp(params, kwargs, key, index):
                    
    # transform hyper-parameters to natural space
    param_values = [
        pmap(params[name])
        for (name, (*_, pmap)) in kwargs['hyperparams_dict'].items()
    ]
    
    # unpack hyper-parameters
    lr, w, layers, neurons = param_values                      
    if kwargs['verbose']:
        print(f'[{index}] key={key}, lr={lr}, w={w}, layers={layers}, neurons={neurons}...', flush=True)
           
    # initialize planning algorithm
    planner = JaxRDDLBackpropPlanner(
        rddl=deepcopy(kwargs['rddl']),
        plan=JaxDeepReactivePolicy(
            topology=[neurons] * layers,
            **kwargs['plan_kwargs']),
        optimizer_kwargs={'learning_rate': lr},
        **kwargs['planner_kwargs'])
    policy_hparams = {name: None for name in planner._action_bounds}
    model_params = {name: w for name in planner.compiled.model_params}
    
    # initialize policy
    key, subkey = jax.random.split(key)
    policy = JaxOfflineController(
        planner=planner,
        key=subkey,
        eval_hyperparams=policy_hparams,
        train_on_reset=True,
        epochs=kwargs['train_epochs'],
        train_seconds=kwargs['timeout_training'],
        model_params=model_params,
        policy_hyperparams=policy_hparams,
        verbose=0,
        tqdm_position=index)
    
    # initialize env for evaluation (need fresh copy to avoid concurrency)
    env = RDDLEnv(domain=kwargs['domain'],
                  instance=kwargs['instance'],
                  vectorized=True,
                  enforce_action_constraints=True)
    
    # perform training
    average_reward = 0.0
    for trial in range(kwargs['eval_trials']):
        key, subkey = jax.random.split(key)
        total_reward = policy.evaluate(env, seed=np.array(subkey)[0])['mean']
        if kwargs['verbose']:
            print(f'    [{index}] trial {trial + 1} key={subkey}, '
                  f'reward={total_reward}', flush=True)
        average_reward += total_reward / kwargs['eval_trials']
    if kwargs['verbose']:
        print(f'[{index}] average reward={average_reward}', flush=True)
    return average_reward


def power_two_int(x):
    return 2 ** int(x)


class JaxParameterTuningDRP(JaxParameterTuning):
    
    def __init__(self, *args,
                 hyperparams_dict: Dict[str, Tuple[float, float, Callable]]={
                    'lr': (-7., 2., power_ten),
                    'w': (0., 5., power_ten),
                    'layers': (1., 3., int),
                    'neurons': (2., 9., power_two_int)
                 },
                 **kwargs) -> None:
        '''Creates a new tuning class for deep reactive policies.
        
        :param *args: arguments to pass to parent class
        :param hyperparams_dict: same as parent class, but here must contain
        learning rate (lr), model weight (w), number of hidden layers (layers) 
        and number of neurons per hidden layer (neurons)
        :param **kwargs: keyword arguments to pass to parent class
        '''
        
        super(JaxParameterTuningDRP, self).__init__(
            *args, hyperparams_dict=hyperparams_dict, **kwargs)
    
    def _pickleable_objective_with_kwargs(self):
        objective_fn = objective_drp
        
        # duplicate planner and plan keyword arguments must be removed
        plan_kwargs = self.plan_kwargs.copy()
        plan_kwargs.pop('topology', None)
        
        planner_kwargs = self.planner_kwargs.copy()
        planner_kwargs.pop('rddl', None)
        planner_kwargs.pop('plan', None)
        planner_kwargs.pop('optimizer_kwargs', None)
                     
        kwargs = {
            'rddl': self.env.model,
            'domain': self.env.domain_text,
            'instance': self.env.instance_text,
            'hyperparams_dict': self.hyperparams_dict,
            'timeout_training': self.timeout_training,
            'train_epochs': self.train_epochs,
            'planner_kwargs': planner_kwargs,
            'plan_kwargs': plan_kwargs,
            'verbose': self.verbose,
            'eval_trials': self.eval_trials
        }
        return objective_fn, kwargs
