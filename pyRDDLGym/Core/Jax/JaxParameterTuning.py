from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction
from colorama import init as colorama_init, Back, Fore, Style
colorama_init()
from concurrent.futures import as_completed, ProcessPoolExecutor      
import csv
import jax
import multiprocessing
import numpy as np
import optax
import os
import time
from typing import Callable, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

from pyRDDLGym.Core.Env.RDDLEnv import RDDLEnv
from pyRDDLGym.Core.Jax.JaxRDDLLogic import FuzzyLogic
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxRDDLBackpropPlanner
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxStraightLinePlan
np.seterr(all='warn')


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

       
def objective_slp(std, lr, w, wa, key, std_space, lr_space, w_space, wa_space,
                  model, wrap_sigmoid, eval_horizon, action_bounds, planner_kwargs,
                  timeout_episode, max_train_epochs, verbose, print_step,
                  color=(Fore.RESET, Back.RESET)):
            
    # transform hyper-parameters to natural space
    std = std_space[2](std)
    lr = lr_space[2](lr)
    w = w_space[2](w)
    if wa is not None:
        wa = wa_space[2](wa)            
    if verbose:
        colorstr = f'{color[0]}{color[1]}'
        print(f'| {colorstr}'
                f'optimizing SLP with PRNG key={key}, ' 
                f'std={std}, lr={lr}, w={w}, wa={wa}...{Style.RESET_ALL}')
            
    # initialize planner
    planner = JaxRDDLBackpropPlanner(
        rddl=model,
        plan=JaxStraightLinePlan(
            initializer=jax.nn.initializers.normal(std),
            wrap_sigmoid=wrap_sigmoid),
        rollout_horizon=eval_horizon,
        action_bounds=action_bounds,
        optimizer=optax.rmsprop,
        optimizer_kwargs={'learning_rate': lr},
        logic=FuzzyLogic(weight=w),
        **planner_kwargs)
    policy_hyperparams = {name: wa for name in action_bounds}
            
    # perform training
    callback = train_epoch(
        key=key,
        policy_hyperparams=policy_hyperparams,
        subs=None,
        planner=planner,
        timeout=timeout_episode,
        max_train_epochs=max_train_epochs,
        verbose=verbose,
        print_step=print_step,
        color=color,
        guess=None)
    total_reward = float(callback['best_return'])
    if verbose:
        print(f'| {colorstr}'
                f'done optimizing SLP, '
                f'total reward={total_reward}{Style.RESET_ALL}')
    return total_reward


def objective_mpc(std, lr, w, wa, T, key, std_space, lr_space, w_space, wa_space,
                  T_space, domain, instance, model, wrap_sigmoid,
                  eval_horizon, eval_trials, use_guess_last_epoch,
                  action_bounds, planner_kwargs,
                  timeout_episode, timeout_epoch, max_train_epochs,
                  verbose, print_step, color=(Fore.RESET, Back.RESET)):
    
    # transform hyper-parameters to natural space
    std = std_space[2](std)
    lr = lr_space[2](lr)
    w = w_space[2](w)
    if wa is not None:
        wa = wa_space[2](wa)
    T = T_space[2](T)        
        
    if verbose:
        colorstr = f'{color[0]}{color[1]}'
        print(f'| {colorstr}'
              f'optimizing MPC with PRNG key={key}, ' 
              f'std={std}, lr={lr}, w={w}, wa={wa}, T={T}...'
              f'{Style.RESET_ALL}')
    
    # initialize planner
    planner = JaxRDDLBackpropPlanner(
        rddl=model,
        plan=JaxStraightLinePlan(
            initializer=jax.nn.initializers.normal(std),
            wrap_sigmoid=wrap_sigmoid),
        rollout_horizon=T,
        action_bounds=action_bounds,
        optimizer=optax.rmsprop,
        optimizer_kwargs={'learning_rate': lr},
        logic=FuzzyLogic(weight=w),
        **planner_kwargs)
    policy_hyperparams = {name: wa for name in action_bounds}
    
    # initialize env for evaluation (need fresh copy to avoid concurrency)
    env = RDDLEnv(domain=domain,
                  instance=instance,
                  enforce_action_constraints=True)
    
    # perform training and collect rewards
    average_reward = 0.0
    for trial in range(eval_trials):
        if verbose:
            print(f'|--- {colorstr}'
                  f'starting trial {trial + 1} '
                  f'with PRNG key={key}...{Style.RESET_ALL}')
        total_reward = 0.0
        guess = None
        env.reset() 
        starttime = time.time()
        for _ in range(eval_horizon):
            currtime = time.time()
            elapsed = currtime - starttime            
            if elapsed < timeout_episode:
                subs = env.sampler.subs
                timeout = min(timeout_episode - elapsed, timeout_epoch)
                key, subkey1, subkey2 = jax.random.split(key, num=3)
                callback = train_epoch(
                    key=subkey1,
                    policy_hyperparams=policy_hyperparams,
                    subs=subs,
                    planner=planner,
                    timeout=timeout,
                    max_train_epochs=max_train_epochs,
                    verbose=verbose,
                    print_step=None,
                    color=color,
                    guess=guess)
                params = callback['best_params']
                action = planner.get_action(subkey2, params, 0, subs)
                if use_guess_last_epoch:
                    guess = planner.plan.guess_next_epoch(params)
            else:
                action = {}            
            _, reward, done, _ = env.step(action)
            total_reward += reward 
            if done: 
                break  
        if verbose:
            print(f'|--- {colorstr}'
                  f'done trial {trial + 1}, '
                  f'total reward={total_reward}{Style.RESET_ALL}')
        average_reward += total_reward / eval_trials
    if verbose:
        print(f'| {colorstr}'
              f'done optimizing MPC, '
              f'average reward={average_reward}{Style.RESET_ALL}')
    return average_reward
    

def evaluate(args):
    params, index, key, color, func, kwargs = args
    func_kwargs = {**params, **kwargs}
    target = func(key=key, color=color, **func_kwargs)
    pid = os.getpid()
    return pid, index, params, target


def power_ten(x):
    return 10.0 ** x


def annealing_utility(n_samples, n_delay_samples=0, kappa1=10.0, kappa2=1.0):
    return UtilityFunction(
        kind='ucb',
        kappa=kappa1,
        kappa_decay=(kappa2 / kappa1) ** (1.0 / (n_samples - n_delay_samples)),
        kappa_decay_delay=n_delay_samples
    )
    
    
class JaxParameterTuning:
    '''A general-purpose class for tuning a Jax planner.'''
    
    def __init__(self, env: RDDLEnv,
                 action_bounds: Dict[str, Tuple[float, float]],
                 max_train_epochs: int,
                 timeout_episode: float,
                 timeout_epoch: float,
                 stddev_space: Tuple[float, float, Callable]=(-5., 0., power_ten),
                 lr_space: Tuple[float, float, Callable]=(-5., 0., power_ten),
                 model_weight_space: Tuple[float, float, Callable]=(0., 5., power_ten),
                 action_weight_space: Tuple[float, float, Callable]=(0., 5., power_ten),
                 lookahead_space: Tuple[float, float]=(1, 100, int),
                 num_workers: int=1,
                 gp_iters: int=25,
                 acquisition=None,
                 gp_params: Dict={'n_restarts_optimizer': 10},
                 eval_horizon: int=None,
                 eval_trials: int=5,
                 print_step: int=None,
                 verbose: bool=True,
                 wrap_sigmoid: bool=True,
                 use_guess_last_epoch: bool=True,
                 planner_kwargs: Dict={}) -> None:
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
        :param stddev_space: set of initializer standard deviations for the 
        policy or plan parameters (uses Gaussian noise) to tune
        :param lr_space: set of learning rates for SGD (uses rmsprop by 
        default) to tune
        :param model_weight_space: set of model weight parameters for continuous 
        relaxation to tune
        :param action_weight_space: sigmoid weights for boolean actions to tune
        :param lookahead_space: set of possible lookahead horizons for MPC to
        tune
        :param num_workers: how many points to evaluate in parallel
        :param gp_iters: number of iterations of optimization
        :param acquisition: acquisition function for Bayesian optimizer
        :param gp_params: additional parameters to feed to Bayesian optimizer
        :param eval_horizon: maximum number of decision epochs to evaluate (also
        applies for training if using straight-line planning)
        :param eval_trials: how many trials to perform for MPC (batch size has
        the same effect for non-MPC)
        :param print_step: how often to print training callback
        :param verbose: whether to print intermediate results of tuning
        :param wrap_sigmoid: whether to wrap bool action-fluents with sigmoid
        :param use_guess_last_epoch: for MPC approach, use the trained parameters
        from previous epoch to seed next epoch actions
        :param planner_kwargs: additional arguments to feed to the planner
        '''
        self.env = env
        self.action_bounds = action_bounds
        self.max_train_epochs = max_train_epochs
        self.timeout_episode = timeout_episode
        self.timeout_epoch = timeout_epoch
        self.evaluation_horizon = eval_horizon
        if eval_horizon is None:
            self.evaluation_horizon = env.horizon
        self.evaluation_trials = eval_trials
        self.stddev_space = stddev_space
        self.lr_space = lr_space
        self.model_weight_space = model_weight_space
        self.action_weight_space = action_weight_space
        self.lookahead_space = lookahead_space
        self.gp_iters = gp_iters
        self.gp_params = gp_params
        self.print_step = print_step
        self.verbose = verbose
        self.wrap_sigmoid = wrap_sigmoid
        self.use_guess_last_epoch = use_guess_last_epoch
        self.planner_kwargs = planner_kwargs
        
        # check if action parameters are required
        self.use_wa_param = False
        if self.wrap_sigmoid:
            for var in self.env.model.actions:
                if self.env.model.variable_ranges[var] == 'bool':
                    self.use_wa_param = True
                    break
        
        # project search space back to parameter space
        self._map = {'std': stddev_space[2],
                     'lr': lr_space[2],
                     'w': model_weight_space[2],
                     'wa': action_weight_space[2],
                     'T': lookahead_space[2]}
        
        # yields 36 valid format variations
        foreground = [Fore.BLUE, Fore.CYAN, Fore.GREEN,
                      Fore.MAGENTA, Fore.RED, Fore.YELLOW]
        background = [Back.RESET, Back.BLUE, Back.CYAN, Back.GREEN,
                      Back.MAGENTA, Back.RED, Back.YELLOW]
        self.colors = []
        for back in background:
            for fore in foreground:
                if int(back[2:-1]) - int(fore[2:-1]) != 10:
                    self.colors.append((fore, back))
        self.num_workers = min(num_workers, len(self.colors))
    
        if acquisition is None:
            acquisition = annealing_utility(self.gp_iters * self.num_workers)
        self.acquisition = acquisition

    def _filename(self, name, ext):
        domainName = self.env.model.domainName()
        instName = self.env.model.instanceName()
        domainName = ''.join(c for c in domainName if c.isalnum() or c == '_')
        instName = ''.join(c for c in instName if c.isalnum() or c == '_')
        filename = f'{name}_{domainName}_{instName}.{ext}'
        return filename
    
    def _save_plot(self, name):
        try:
            import matplotlib.pyplot as plt
            from sklearn.manifold import MDS
            filename = self._filename(name, 'csv')
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
       
    def _bayes_optimize(self, key, name, mpc): 
        
        # set the bounds on variables
        pbounds = {'std': self.stddev_space[:2],
                   'lr': self.lr_space[:2],
                   'w': self.model_weight_space[:2]}
        if self.use_wa_param:
            pbounds['wa'] = self.action_weight_space[:2]
        if mpc:
            pbounds['T'] = self.lookahead_space[:2]
            
        # set non changing variables
        kwargs = {'std_space': self.stddev_space,
                  'lr_space': self.lr_space,
                  'w_space': self.model_weight_space,
                  'wa_space': self.action_weight_space,
                  'model': self.env.model,
                  'wrap_sigmoid': self.wrap_sigmoid,
                  'eval_horizon': self.evaluation_horizon,
                  'action_bounds': self.action_bounds,
                  'planner_kwargs': self.planner_kwargs,
                  'timeout_episode': self.timeout_episode,
                  'max_train_epochs': self.max_train_epochs,
                  'verbose': self.verbose,
                  'print_step': self.print_step}  
        if not self.use_wa_param:
            kwargs['wa'] = None
        if mpc:
            kwargs['T_space'] = self.lookahead_space
            kwargs['domain'] = self.env.domain_text
            kwargs['instance'] = self.env.instance_text
            kwargs['eval_trials'] = self.evaluation_trials
            kwargs['use_guess_last_epoch'] = self.use_guess_last_epoch
            kwargs['timeout_epoch'] = self.timeout_epoch
                
        # create optimizer
        optimizer = BayesianOptimization(
            f=None,  # probe() is not called
            pbounds=pbounds,
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
            suggested.append(optimizer.suggest(utility))  
            kappas.append(utility.kappa)          
        
        # saving to file
        keys = list(pbounds.keys())
        filename = self._filename(name, 'csv')
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                ['pid', 'worker', 'iteration',
                 'target', 'best_target', 'kappa'] + keys)
        file = open(filename, 'a', newline='')
        writer = csv.writer(file)        
        lock = multiprocessing.Manager().Lock()
        
        # continue with multi-process evaluation
        objective = objective_mpc if mpc else objective_slp
        best_target = -np.inf
        colors = self.colors[:num_workers]
        
        for it in range(self.gp_iters): 
            print('\n' + '*' * 25 + 
                  '\n' + f'starting iteration {it}' + 
                  '\n' + '*' * 25)
            key, *subkeys = jax.random.split(key, num=num_workers + 1)
            
            # create worker pool: note each iteration must wait for all workers
            # to finish before moving to the next
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                
                # each worker evaluates planner for a suggested parameter dict
                jobs = [
                    executor.submit(
                        evaluate, tuple(worker_args) + (objective, kwargs)) 
                    for worker_args in zip(
                        suggested, range(num_workers), subkeys, colors)
                ]
                
                # on completion of each job...
                for job in as_completed(jobs):
                    
                    # extract and register the new evaluation
                    pid, index, params, target = job.result()
                    optimizer.register(params, target)
                    
                    # update acquisition function and suggest a new point
                    utility.update_params()  
                    suggested[index] = optimizer.suggest(utility)
                    old_kappa = kappas[index]
                    kappas[index] = utility.kappa
                    best_target = max(best_target, target)
                    
                    # write progress to file in real time
                    with lock:
                        writer.writerow(
                            [pid, index, it, target, best_target, old_kappa] + \
                            [self._map[k](params[k]) for k in keys]
                        )
                
        # close file stream, save plot
        file.close()
        self._save_plot(name)

    def tune_slp(self, key: jax.random.PRNGKey) -> None:
        '''Tunes the hyper-parameters for Jax planner using straight-line 
        planning approach.'''
        self._bayes_optimize(key, 'gp_slp', False)
    
    def tune_mpc(self, key: jax.random.PRNGKey) -> None:
        '''Tunes the hyper-parameters for Jax planner using MPC/receding horizon
        planning approach.'''
        self._bayes_optimize(key, 'gp_mpc', True)
    
