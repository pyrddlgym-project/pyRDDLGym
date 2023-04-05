import asyncio
from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction
from colorama import init as colorama_init, Back, Fore, Style
colorama_init()
import csv
import jax
import json
import numpy as np
import optax
import requests
from threading import Thread, Lock
import time
import tornado.ioloop
import tornado.httpserver
from tornado.web import RequestHandler
from typing import Callable, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

from pyRDDLGym.Core.Env.RDDLEnv import RDDLEnv
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
                 stddev_space: Tuple[float, float, Callable]=(-5., 0., (lambda x: 10 ** x)),
                 lr_space: Tuple[float, float, Callable]=(-5., 0., (lambda x: 10 ** x)),
                 model_weight_space: Tuple[float, float, Callable]=(0., 5., (lambda x: 10 ** x)),
                 action_weight_space: Tuple[float, float, Callable]=(0., 5., (lambda x: 10 ** x)),
                 lookahead_space: Tuple[float, float]=(1, 100, (lambda x: int(x))),
                 gp_kwargs: Dict={'n_iter': 25},
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
        :param gp_kwargs: dictionary of parameters to pass to Bayesian optimizer
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
        self.gp_kwargs = gp_kwargs
        self.print_step = print_step
        self.verbose = verbose
        self.wrap_sigmoid = wrap_sigmoid
        self.use_guess_last_epoch = use_guess_last_epoch
        self.planner_kwargs = planner_kwargs
        
        self._map = {'std': stddev_space[2],
                     'lr': lr_space[2],
                     'w': model_weight_space[2],
                     'wa': action_weight_space[2],
                     'T': lookahead_space[2]}
        
    def _train_epoch(self, key, policy_hyperparams, subs, planner, timeout, color,
                     guess=None): 
        colorstr = f'{color[0]}{color[1]}'
        starttime = None
        for (it, callback) in enumerate(planner.optimize(
            key=key,
            epochs=self.max_train_epochs,
            step=1,
            policy_hyperparams=policy_hyperparams,
            subs=subs,
            guess=guess
        )):
            if starttime is None:
                starttime = time.time()
            currtime = time.time()  
            elapsed = currtime - starttime    
            if self.verbose and self.print_step is not None \
            and it > 0 and it % self.print_step == 0:
                print(f'|------ {colorstr}' 
                      '[{:.4f} s] step={} train_return={:.6f} test_return={:.6f}'.format(
                          elapsed,
                          str(callback['iteration']).rjust(4),
                          callback['train_return'],
                          callback['test_return']) + 
                      f'{Style.RESET_ALL}')
            if not np.isfinite(callback['train_return']):
                if self.verbose:
                    print(f'|------ {colorstr}'
                          f'aborting due to NaN or inf value!'
                          f'{Style.RESET_ALL}')
                break
            if elapsed >= timeout:
                break
        return callback
    
    def _filename(self, name, ext):
        domainName = self.env.model.domainName()
        instName = self.env.model.instanceName()
        domainName = ''.join(c for c in domainName if c.isalnum() or c == '_')
        instName = ''.join(c for c in instName if c.isalnum() or c == '_')
        filename = f'{name}_{domainName}_{instName}.{ext}'
        return filename
    
    def _save_results(self, optimizer, name): 
        with open(self._filename(name, 'csv'), 'w', newline='') as file:
            writer = csv.writer(file)
            has_header = False
            keys = None
            for (i, res) in enumerate(optimizer.res):
                if 'params' in res and 'target' in res:
                    target, params = res['target'], res['params']
                    if not has_header:
                        keys = list(params.keys())
                        writer.writerow(['iteration', 'target'] + keys)
                        has_header = True
                    writer.writerow(
                        [i, target] + [self._map[k](params[k]) for k in keys])  
        self._save_plot(name)
        
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
        
    def _bayes_optimize(self, key, objective, name, read_T):
        self.key = key
        
        # set the bounds on variables
        pbounds = {'std': self.stddev_space[:2],
                   'lr': self.lr_space[:2],
                   'w': self.model_weight_space[:2],
                   'wa': self.action_weight_space[:2]}
        if read_T:
            pbounds['T'] = self.lookahead_space[:2]
        
        # run optimizer
        optimizer = BayesianOptimization(
            f=objective,
            pbounds=pbounds,
            verbose=0
        )
        optimizer.maximize(**self.gp_kwargs)
        self._save_results(optimizer, name)
    
    def _objective_slp(self):
        
        def objective(std, lr, w, wa, color=(Fore.RESET, Back.RESET)):
            
            # transform hyper-parameters to natural space
            std = self.stddev_space[2](std)
            lr = self.lr_space[2](lr)
            w = self.model_weight_space[2](w)
            wa = self.action_weight_space[2](wa)            
            if self.verbose:
                colorstr = f'{color[0]}{color[1]}'
                print(f'| {colorstr}'
                      f'optimizing SLP with PRNG key={self.key}, ' 
                      f'std={std}, lr={lr}, w={w}, wa={wa}...{Style.RESET_ALL}')
            
            # initialize planner
            planner = JaxRDDLBackpropPlanner(
                rddl=self.env.model,
                plan=JaxStraightLinePlan(
                    initializer=jax.nn.initializers.normal(std),
                    wrap_sigmoid=self.wrap_sigmoid),
                rollout_horizon=self.evaluation_horizon,
                action_bounds=self.action_bounds,
                optimizer=optax.rmsprop(lr),
                logic=FuzzyLogic(weight=w),
                **self.planner_kwargs)
            policy_hyperparams = {name: wa for name in self.action_bounds}
            
            # perform training
            self.key, subkey = jax.random.split(self.key)
            callback = self._train_epoch(
                subkey, policy_hyperparams, None, planner, self.timeout_episode,
                color)
            total_reward = float(callback['best_return'])
            if self.verbose:
                print(f'| {colorstr}'
                      f'done optimizing SLP, '
                      f'total reward={total_reward}{Style.RESET_ALL}')
            return total_reward
        
        return objective
        
    def tune_slp(self, key: jax.random.PRNGKey) -> None:
        '''Tunes the hyper-parameters for Jax planner using straight-line 
        planning approach.'''
        self._bayes_optimize(key, self._objective_slp(), 'gp_slp', False)
    
    def _objective_mpc(self):
        
        def objective(std, lr, w, wa, T, color=(Fore.RESET, Back.RESET)):
            
            # transform hyper-parameters to natural space
            std = self.stddev_space[2](std)
            lr = self.lr_space[2](lr)
            w = self.model_weight_space[2](w)
            wa = self.action_weight_space[2](wa)
            T = self.lookahead_space[2](T)            
            if self.verbose:
                colorstr = f'{color[0]}{color[1]}'
                print(f'| {colorstr}'
                      f'optimizing MPC with PRNG key={self.key}, ' 
                      f'std={std}, lr={lr}, w={w}, wa={wa}, T={T}...'
                      f'{Style.RESET_ALL}')
            
            # initialize planner
            planner = JaxRDDLBackpropPlanner(
                rddl=self.env.model,
                plan=JaxStraightLinePlan(
                    initializer=jax.nn.initializers.normal(std),
                    wrap_sigmoid=self.wrap_sigmoid),
                rollout_horizon=T,
                action_bounds=self.action_bounds,
                optimizer=optax.rmsprop(lr),
                logic=FuzzyLogic(weight=w),
                **self.planner_kwargs)
            policy_hyperparams = {name: wa for name in self.action_bounds}
            
            # initialize env for evaluation (need fresh copy to avoid concurrency)
            env = RDDLEnv(domain=self.env.domain_text,
                          instance=self.env.instance_text,
                          enforce_action_constraints=True)
            print_each_step = not issubclass(type(self), JaxParameterTuning)
            
            # perform training and collect rewards
            average_reward = 0.0
            for trial in range(self.evaluation_trials):
                if self.verbose:
                    print(f'|--- {colorstr}'
                          f'starting trial {trial + 1} '
                          f'with PRNG key={self.key}...{Style.RESET_ALL}')
                total_reward = 0.0
                guess = None
                env.reset() 
                starttime = time.time()
                for step in range(self.evaluation_horizon):
                    currtime = time.time()
                    elapsed = currtime - starttime            
                    if elapsed < self.timeout_episode:
                        subs = env.sampler.subs
                        timeout = min(self.timeout_episode - elapsed,
                                      self.timeout_epoch)
                        self.key, subkey1, subkey2 = jax.random.split(self.key, num=3)
                        callback = self._train_epoch(
                            subkey1, policy_hyperparams, subs, planner, timeout,
                            color, guess)
                        params = callback['best_params']
                        action = planner.get_action(subkey2, params, 0, subs)
                        if self.use_guess_last_epoch:
                            guess = planner.plan.guess_next_epoch(params)
                    else:
                        action = {}            
                    _, reward, done, _ = env.step(action)
                    total_reward += reward 
                    if self.verbose and print_each_step:
                        print(f'|------ {colorstr}'
                              f'step={step}, '
                              f'reward={reward}{Style.RESET_ALL}')
                    if done: 
                        break  
                if self.verbose:
                    print(f'|--- {colorstr}'
                          f'done trial {trial + 1}, '
                          f'total reward={total_reward}{Style.RESET_ALL}')
                average_reward += total_reward / self.evaluation_trials
            if self.verbose:
                print(f'| {colorstr}'
                      f'done optimizing MPC, '
                      f'average reward={average_reward}{Style.RESET_ALL}')
            return average_reward
        
        return objective
        
    def tune_mpc(self, key: jax.random.PRNGKey) -> None:
        '''Tunes the hyper-parameters for Jax planner using MPC/receding horizon
        planning approach.'''
        self._bayes_optimize(key, self._objective_mpc(), 'gp_mpc', True)


class JaxParameterTuningParallel(JaxParameterTuning):
    
    def __init__(self, *args, num_workers: int, **kwargs): 
        super(JaxParameterTuningParallel, self).__init__(*args, **kwargs)
        
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
        
    def _bayes_optimize(self, key, objective, name, read_T): 
        self.key = key
        
        # set the bounds on variables
        pbounds = {'std': self.stddev_space[:2],
                   'lr': self.lr_space[:2],
                   'w': self.model_weight_space[:2],
                   'wa': self.action_weight_space[:2]}
        if read_T:
            pbounds['T'] = self.lookahead_space[:2]
        
        # create optimizer
        optimizer = BayesianOptimization(
            f=objective,
            pbounds=pbounds,
            allow_duplicate_points=True
        )
        utility = self.gp_kwargs.get(
            'acquisition_function', UtilityFunction(kind='ucb', kappa=3, xi=1))
        
        # lock for saving to file
        keys = list(pbounds.keys())
        filename = self._filename(name, 'csv')
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['worker', 'iteration', 'target'] + keys)
        lock = Lock()
        file = open(filename, 'a', newline='')
        writer = csv.writer(file)
        params_map = self._map
        
        # class for processing requests (i.e. hyper-parameters)
        class BayesianOptimizationHandler(RequestHandler):
            
            def post(self):
                body = tornado.escape.json_decode(self.request.body)        
                try: 
                    worker, it = body['worker'], body['iter']
                    target, params = body['target'], body['params']
                    optimizer.register(params=params, target=target)  
                    with lock:
                        writer.writerow([worker, it, target] + \
                                        [params_map[k](params[k]) for k in keys])
                except KeyError: 
                    pass
                finally: 
                    params = optimizer.suggest(utility)
                self.write(json.dumps(params))
        
        # manages threads
        port = 9009

        def run_optimization_app():
            asyncio.set_event_loop(asyncio.new_event_loop())
            handlers = [(r"/jax_tuning", BayesianOptimizationHandler)]
            server = tornado.httpserver.HTTPServer(tornado.web.Application(handlers))
            server.listen(port)
            tornado.ioloop.IOLoop.instance().start()
        
        optimizers_config = [
            {'name': f'optimizer {i + 1}', 'color': self.colors[i]}
            for i in range(self.num_workers)
        ]
        
        # worker
        def run_optimizer():
            config = optimizers_config.pop()
            color = config['color']
            register_data = {}
            max_target = None
            for it in range(self.gp_kwargs.get('n_iter', 25)): 
                params = requests.post(
                    url=f'http://localhost:{port}/jax_tuning',
                    json=register_data,
                ).json()
                target = objective(**params, color=color)
                if max_target is None or target > max_target:
                    max_target = target
                register_data = {'worker': config['name'], 'iter': it,
                                 'params': params, 'target': target}
        
        # create multi-threading instance, assign jobs, run etc.
        ioloop = tornado.ioloop.IOLoop.instance()        
        app_thread = Thread(target=run_optimization_app)
        app_thread.daemon = True
        app_thread.start()
        
        optimizer_threads = []
        for _ in range(self.num_workers):
            optimizer_threads.append(Thread(target=run_optimizer))
            optimizer_threads[-1].daemon = True
            optimizer_threads[-1].start()    
        for optimizer_thread in optimizer_threads:
            optimizer_thread.join()    
        ioloop.stop()
        
        # close file stream, save plot
        file.close()
        self._save_plot(name)

