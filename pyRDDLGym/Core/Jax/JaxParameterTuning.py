from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction
from colorama import init as colorama_init, Back, Fore, Style
colorama_init()
from concurrent.futures import as_completed, ProcessPoolExecutor      
import csv
import jax
import multiprocessing
import numpy as np
import os
import time
from typing import Callable, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

from pyRDDLGym.Core.Env.RDDLEnv import RDDLEnv
from pyRDDLGym.Core.Jax.JaxRDDLLogic import FuzzyLogic
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxRDDLBackpropPlanner
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxStraightLinePlan
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxDeepReactivePolicy
np.seterr(all='warn')

       
class JaxParameterTuning:
    '''A general-purpose class for tuning a Jax planner.'''
    
    def __init__(self, env: RDDLEnv,
                 hyperparams_dict: Dict[str, Tuple[float, float, Callable]],
                 max_train_epochs: int,
                 timeout_episode: float,
                 verbose: bool=True,
                 print_step: int=None,
                 planner_kwargs: Dict={},
                 num_workers: int=1,
                 gp_iters: int=25,
                 acquisition=None,
                 gp_params: Dict={'n_restarts_optimizer': 10}) -> None:
        '''Creates a new instance for tuning hyper-parameters for Jax planners
        on the given RDDL domain and instance.
        
        :param env: the RDDLEnv describing the MDP to optimize
        :param hyperparams_dict: dictionary mapping name of each hyperparameter
        to a triple, where the first two elements are lower/upper bounds on the
        parameter value, and the last is a callable mapping the parameter to its
        RDDL equivalent
        :param max_train_epochs: the maximum number of iterations of SGD per 
        step or trial
        :param timeout_episode: the maximum amount of time to spend training per
        trial (in seconds)
        :param verbose: whether to print intermediate results of tuning
        :param print_step: how often to print training callback
        :param planner_kwargs: additional arguments to feed to the planner
        :param num_workers: how many points to evaluate in parallel
        :param gp_iters: number of iterations of optimization
        :param acquisition: acquisition function for Bayesian optimizer
        :param gp_params: additional parameters to feed to Bayesian optimizer        
        '''
        
        self.env = env
        self.hyperparams_dict = hyperparams_dict
        self.max_train_epochs = max_train_epochs
        self.timeout_episode = timeout_episode
        self.verbose = verbose
        self.print_step = print_step
        self.planner_kwargs = planner_kwargs
        self.num_workers = num_workers
        self.gp_iters = gp_iters
        self.gp_params = gp_params
        
        # map parameters to RDDL equivalent
        self.hyperparams_bounds = {
            name: hparam[:2] 
            for (name, hparam) in self.hyperparams_dict.items()
        }
        self.hyperparams_map = (lambda params: [
            hparam[2](params[name])
            for (name, hparam) in self.hyperparams_dict.items()
        ])
        
        # create acquisition function
        if acquisition is None:
            num_samples = self.gp_iters * self.num_workers
            acquisition = JaxParameterTuning.annealing_utility(num_samples)
        self.acquisition = acquisition
        
        # create valid color variations for multiprocess output
        self.colors = JaxParameterTuning.color_variations()
        self.num_workers = min(num_workers, len(self.colors))

    @staticmethod
    def color_variations():
        foreground = [Fore.BLUE, Fore.CYAN, Fore.GREEN,
                      Fore.MAGENTA, Fore.RED, Fore.YELLOW]
        background = [Back.RESET, Back.BLUE, Back.CYAN, Back.GREEN,
                      Back.MAGENTA, Back.RED, Back.YELLOW]
        return [(fore, back) 
                for back in background
                for fore in foreground
                if int(back[2:-1]) - int(fore[2:-1]) != 10]

    @staticmethod
    def annealing_utility(n_samples, n_delay_samples=0, kappa1=10.0, kappa2=1.0):
        return UtilityFunction(
            kind='ucb',
            kappa=kappa1,
            kappa_decay=(kappa2 / kappa1) ** (1.0 / (n_samples - n_delay_samples)),
            kappa_decay_delay=n_delay_samples)
    
    def _pickleable_objective_with_kwargs(self):
        raise NotImplementedError
    
    @staticmethod
    def wrapped_evaluate(args):
        index, params, key, color, func, kwargs = args
        target = func(params=params, kwargs=kwargs, key=key, color=color)
        pid = os.getpid()
        return index, pid, params, target

    def tune(self, key: jax.random.PRNGKey, filename: str) -> None:
        '''Tunes the hyper-parameters for Jax planner'''
            
        # create optimizer
        optimizer = BayesianOptimization(
            f=None,  # probe() is not called
            pbounds=self.hyperparams_bounds,
            allow_duplicate_points=True,  # to avoid crash
            random_state=np.random.RandomState(key)
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
        
        # saving to file
        filename = self._filename(filename, 'csv')
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            row = ['pid', 'worker', 'iteration',
                   'target', 'best_target', 'kappa'] + \
                   list(self.hyperparams_bounds.keys())
            writer.writerow(row)
        file = open(filename, 'a', newline='')
        writer = csv.writer(file)        
        lock = multiprocessing.Manager().Lock()
        
        # objective function
        objective = self._pickleable_objective_with_kwargs()
        best_target = -np.inf
        colors = self.colors[:num_workers]
        evaluate = JaxParameterTuning.wrapped_evaluate
                
        # start multi-process evaluation
        for it in range(self.gp_iters): 
            print('\n' + '*' * 25 + 
                  '\n' + f'starting iteration {it}' + 
                  '\n' + '*' * 25)
            key, *subkeys = jax.random.split(key, num=num_workers + 1)
            
            # create worker pool: note each iteration must wait for all workers
            # to finish before moving to the next
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                
                # each worker evaluates planner for a suggested parameter dict
                worker_id = list(range(num_workers))
                jobs = [
                    executor.submit(evaluate, worker_args + objective) 
                    for worker_args in zip(worker_id, suggested, subkeys, colors)
                ]
                
                # on completion of each job...
                for job in as_completed(jobs):
                    
                    # extract and register the new evaluation
                    index, pid, params, target = job.result()
                    optimizer.register(params, target)
                    
                    # update acquisition function and suggest a new point
                    utility.update_params()  
                    suggested[index] = optimizer.suggest(utility)
                    old_kappa = kappas[index]
                    kappas[index] = utility.kappa
                    best_target = max(best_target, target)
                    
                    # write progress to file in real time
                    rddl_params = [
                        pmap(params[name])
                        for (name, (*_, pmap)) in self.hyperparams_dict.items()
                    ]
                    with lock:
                        writer.writerow(
                            [pid, index, it, target, best_target, old_kappa] + \
                            rddl_params
                        )
                
        # close file stream, save plot
        file.close()
        self._save_plot(filename)

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
            print('failed to import packages for plotting search space:')
            print(e)
            print('aborting plot of search space')
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


def train_epoch(key, policy_hyperparams, subs, planner, timeout,
                 max_train_epochs, verbose, print_step, color, guess=None): 
    colorstr = f'{color[0]}{color[1]}'
    starttime = None
    for (it, callback) in enumerate(planner.optimize(
        key=key,
        epochs=max_train_epochs,
        step=1,
        policy_hyperparams=policy_hyperparams,
        subs=subs,
        guess=guess
    )):
        if starttime is None:
            starttime = time.time()
        currtime = time.time()  
        elapsed = currtime - starttime    
        if verbose and print_step is not None and it > 0 and it % print_step == 0:
            print(f'|------ {colorstr}' 
                  '[{:.4f} s] step={} train_return={:.6f} test_return={:.6f}'.format(
                      elapsed,
                      str(callback['iteration']).rjust(4),
                      callback['train_return'],
                      callback['test_return']) + 
                  f'{Style.RESET_ALL}')
        if not np.isfinite(callback['train_return']):
            if verbose:
                print(f'|------ {colorstr}'
                      f'aborting due to NaN or inf value!'
                      f'{Style.RESET_ALL}')
            break
        if elapsed >= timeout:
            break
    return callback


def objective_slp(params, kwargs, key, color=(Fore.RESET, Back.RESET)):
                    
    # transform hyper-parameters to natural space
    param_values = [
        pmap(params[name])
        for (name, (*_, pmap)) in kwargs['hyperparams_dict'].items()
    ]
    
    # unpack hyper-parameters
    if kwargs['use_wa_param']:
        std, lr, w, wa = param_values
    else:
        std, lr, w = param_values
        wa = None
                      
    if kwargs['verbose']:
        print(f'| {color[0]}{color[1]}'
                f'optimizing SLP with PRNG key={key}, ' 
                f'std={std}, lr={lr}, w={w}, wa={wa}...{Style.RESET_ALL}')
                    
    # initialize planner
    planner = JaxRDDLBackpropPlanner(
        rddl=kwargs['rddl'],
        plan=JaxStraightLinePlan(
            initializer=jax.nn.initializers.normal(std),
            wrap_sigmoid=kwargs['wrap_sigmoid']),
        optimizer_kwargs={'learning_rate': lr},
        logic=FuzzyLogic(weight=w),
        **kwargs['planner_kwargs'])
                    
    # perform training
    callback = train_epoch(
        key=key,
        policy_hyperparams={name: wa for name in planner._action_bounds},
        subs=None,
        planner=planner,
        timeout=kwargs['timeout_episode'],
        max_train_epochs=kwargs['max_train_epochs'],
        verbose=kwargs['verbose'],
        print_step=kwargs['print_step'],
        color=color,
        guess=None)
    total_reward = float(callback['best_return'])
            
    if kwargs['verbose']:
        print(f'| {color[0]}{color[1]}'
                f'done optimizing SLP, '
                f'total reward={total_reward}{Style.RESET_ALL}')
    return total_reward

        
def power_ten(x):
    return 10.0 ** x

    
class JaxParameterTuningSLP(JaxParameterTuning):
    
    def __init__(self, *args,
                 hyperparams_dict: Dict[str, Tuple[float, float, Callable]]={
                    'std': (-5., 0., power_ten),
                    'lr': (-5., 0., power_ten),
                    'w': (0., 5., power_ten),
                    'wa': (0., 5., power_ten)
                 },
                 wrap_sigmoid: bool=True,
                 **kwargs):
        '''Creates a new tuning class for straight line planners.
        
        :param *args: arguments to pass to parent class
        :param hyperparams_dict: same as parent class, but here must contain
        weight initialization (std), learning rate (lr), model weight (w), and
        action weight (wa) if wrap_sigmoid and boolean action fluents exist
        :param wrap_sigmoid: whether to wrap bool action-fluents with sigmoid
        :param **kwargs: keyword arguments to pass to parent class
        '''
        
        # action parameters required if wrap_sigmoid and boolean action exists
        self.wrap_sigmoid = wrap_sigmoid
        self.use_wa_param = False
        if self.wrap_sigmoid:
            env = kwargs.get('env', None)
            if env is None:
                env = args[0]
            for var in env.model.actions:
                if env.model.variable_ranges[var] == 'bool':
                    self.use_wa_param = True
                    break
        if not self.use_wa_param:
            hyperparams_dict.pop('wa', None)
        
        super(JaxParameterTuningSLP, self).__init__(
            *args, hyperparams_dict=hyperparams_dict, **kwargs)
        
    def _pickleable_objective_with_kwargs(self):
        objective_fn = objective_slp
        kwargs = {
            'rddl': self.env.model,
            'hyperparams_dict': self.hyperparams_dict,
            'timeout_episode': self.timeout_episode,
            'max_train_epochs': self.max_train_epochs,
            'planner_kwargs': self.planner_kwargs,
            'verbose': self.verbose,
            'print_step': self.print_step,
            'wrap_sigmoid': self.wrap_sigmoid,
            'use_wa_param': self.use_wa_param
        }
        return objective_fn, kwargs


def objective_replan(params, kwargs, key, color=(Fore.RESET, Back.RESET)):

    # transform hyper-parameters to natural space
    param_values = [
        pmap(params[name])
        for (name, (*_, pmap)) in kwargs['hyperparams_dict'].items()
    ]
    
    # unpack hyper-parameters
    if kwargs['use_wa_param']:
        std, lr, w, wa, T = param_values
    else:
        std, lr, w, T = param_values
        wa = None
        
    if kwargs['verbose']:
        print(f'| {color[0]}{color[1]}'
              f'optimizing MPC with PRNG key={key}, ' 
              f'std={std}, lr={lr}, w={w}, wa={wa}, T={T}...'
              f'{Style.RESET_ALL}')

    # initialize planner
    planner = JaxRDDLBackpropPlanner(
        rddl=kwargs['rddl'],
        plan=JaxStraightLinePlan(
            initializer=jax.nn.initializers.normal(std),
            wrap_sigmoid=kwargs['wrap_sigmoid']),
        rollout_horizon=T,
        optimizer_kwargs={'learning_rate': lr},
        logic=FuzzyLogic(weight=w),
        **kwargs['planner_kwargs'])
    policy_hyperparams = {name: wa for name in planner._action_bounds}

    # initialize env for evaluation (need fresh copy to avoid concurrency)
    env = RDDLEnv(domain=kwargs['domain'],
                  instance=kwargs['instance'],
                  enforce_action_constraints=True)

    # perform training
    average_reward = 0.0
    for trial in range(kwargs['eval_trials']):
        
        # start the next trial
        if kwargs['verbose']:
            print(f'|--- {color[0]}{color[1]}'
                  f'starting trial {trial + 1} '
                  f'with PRNG key={key}...{Style.RESET_ALL}')
            
        total_reward = 0.0
        guess = None
        env.reset() 
        starttime = time.time()
        for _ in range(kwargs['eval_horizon']):
            currtime = time.time()
            elapsed = currtime - starttime            
            if elapsed < kwargs['timeout_episode']:
                subs = env.sampler.subs
                timeout = min(kwargs['timeout_episode'] - elapsed,
                              kwargs['timeout_epoch'])
                key, subkey1, subkey2 = jax.random.split(key, num=3)
                callback = train_epoch(
                    key=subkey1,
                    policy_hyperparams=policy_hyperparams,
                    subs=subs,
                    planner=planner,
                    timeout=timeout,
                    max_train_epochs=kwargs['max_train_epochs'],
                    verbose=kwargs['verbose'],
                    print_step=None,
                    color=color,
                    guess=guess)
                params = callback['best_params']
                action = planner.get_action(subkey2, params, 0, subs)
                if kwargs['use_guess_last_epoch']:
                    guess = planner.plan.guess_next_epoch(params)
            else:
                action = {}            
            _, reward, done, _ = env.step(action)
            total_reward += reward 
            if done: 
                break  
            
        # update average reward across trials
        if kwargs['verbose']:
            print(f'|--- {color[0]}{color[1]}'
                  f'done trial {trial + 1}, '
                  f'total reward={total_reward}{Style.RESET_ALL}')
        average_reward += total_reward / kwargs['eval_trials']
        
    if kwargs['verbose']:
        print(f'| {color[0]}{color[1]}'
              f'done optimizing MPC, '
              f'average reward={average_reward}{Style.RESET_ALL}')
    return average_reward

    
class JaxParameterTuningSLPReplan(JaxParameterTuningSLP):
    
    def __init__(self, timeout_epoch: float,
                 *args,
                 hyperparams_dict: Dict[str, Tuple[float, float, Callable]]={
                    'std': (-5., 0., power_ten),
                    'lr': (-5., 0., power_ten),
                    'w': (0., 5., power_ten),
                    'wa': (0., 5., power_ten),
                    'T': (1, 100, int)
                 },
                 eval_trials: int=5,
                 use_guess_last_epoch: bool=True,
                 **kwargs):
        '''Creates a new tuning class for straight line planners.
        
        :param timeout_epoch: the maximum amount of time to spend training per
        decision time step
        :param *args: arguments to pass to parent class
        :param hyperparams_dict: same as parent class, but here must contain
        weight initialization (std), learning rate (lr), model weight (w), 
        action weight (wa) if wrap_sigmoid and boolean action fluents exist, and
        lookahead horizon (T)
        :param eval_trials: how many trials to perform independent training
        in order to estimate the return for each set of hyper-parameters
        :param use_guess_last_epoch: use the trained parameters from previous 
        decision to warm-start next decision
        :param **kwargs: keyword arguments to pass to parent class
        '''
        
        super(JaxParameterTuningSLPReplan, self).__init__(
            *args, hyperparams_dict=hyperparams_dict, **kwargs)
        
        self.timeout_epoch = timeout_epoch
        self.eval_trials = eval_trials
        self.use_guess_last_epoch = use_guess_last_epoch
        
    def _pickleable_objective_with_kwargs(self):
        objective_fn = objective_replan
        kwargs = {
            'rddl': self.env.model,
            'domain': self.env.domain_text,
            'instance': self.env.instance_text,
            'hyperparams_dict': self.hyperparams_dict,
            'timeout_episode': self.timeout_episode,
            'max_train_epochs': self.max_train_epochs,
            'planner_kwargs': self.planner_kwargs,
            'verbose': self.verbose,
            'print_step': self.print_step,
            'wrap_sigmoid': self.wrap_sigmoid,
            'use_wa_param': self.use_wa_param,
            'timeout_epoch': self.timeout_epoch,
            'eval_trials': self.eval_trials,
            'eval_horizon': self.env.horizon,
            'use_guess_last_epoch': self.use_guess_last_epoch
        }
        return objective_fn, kwargs


def objective_drp(params, kwargs, key, color=(Fore.RESET, Back.RESET)):
                    
    # transform hyper-parameters to natural space
    param_values = [
        pmap(params[name])
        for (name, (*_, pmap)) in kwargs['hyperparams_dict'].items()
    ]
    
    # unpack hyper-parameters
    lr, w, layers, neurons = param_values
                      
    if kwargs['verbose']:
        print(f'| {color[0]}{color[1]}'
                f'optimizing DRP with PRNG key={key}, ' 
                f'lr={lr}, w={w}, layers={layers}, neurons={neurons}...{Style.RESET_ALL}')
                    
    # initialize planner
    planner = JaxRDDLBackpropPlanner(
        rddl=kwargs['rddl'],
        plan=JaxDeepReactivePolicy(
            topology=[neurons] * layers),
        optimizer_kwargs={'learning_rate': lr},
        logic=FuzzyLogic(weight=w),
        **kwargs['planner_kwargs'])
                    
    # perform training
    callback = train_epoch(
        key=key,
        policy_hyperparams={name: None for name in planner._action_bounds},
        subs=None,
        planner=planner,
        timeout=kwargs['timeout_episode'],
        max_train_epochs=kwargs['max_train_epochs'],
        verbose=kwargs['verbose'],
        print_step=kwargs['print_step'],
        color=color,
        guess=None)
    total_reward = float(callback['best_return'])
            
    if kwargs['verbose']:
        print(f'| {color[0]}{color[1]}'
                f'done optimizing DRP, '
                f'total reward={total_reward}{Style.RESET_ALL}')
    return total_reward


def power_two_int(x):
    return 2 ** int(x)


class JaxParameterTuningDRP(JaxParameterTuning):
    
    def __init__(self, *args,
                 hyperparams_dict: Dict[str, Tuple[float, float, Callable]]={
                    'lr': (-6., 0., power_ten),
                    'w': (0., 5., power_ten),
                    'layers': (1., 3., int),
                    'neurons': (1., 9., power_two_int)
                 },
                 **kwargs):
        super(JaxParameterTuningDRP, self).__init__(
            *args, hyperparams_dict=hyperparams_dict, **kwargs)
    
    def _pickleable_objective_with_kwargs(self):
        objective_fn = objective_drp
        kwargs = {
            'rddl': self.env.model,
            'hyperparams_dict': self.hyperparams_dict,
            'timeout_episode': self.timeout_episode,
            'max_train_epochs': self.max_train_epochs,
            'planner_kwargs': self.planner_kwargs,
            'verbose': self.verbose,
            'print_step': self.print_step
        }
        return objective_fn, kwargs
