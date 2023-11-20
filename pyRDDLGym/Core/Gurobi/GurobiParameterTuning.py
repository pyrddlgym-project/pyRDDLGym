from copy import deepcopy
import csv
import datetime
import gurobipy
import jax
from multiprocessing import get_context
import numpy as np
import os
import time
from typing import Dict, List

from pyRDDLGym.Core.Env.RDDLEnv import RDDLEnv
from pyRDDLGym.Core.Gurobi.GurobiRDDLPlanner import GurobiStraightLinePlan
from pyRDDLGym.Core.Gurobi.GurobiRDDLPlanner import GurobiOnlineController

# use a global instance
GLOBAL_ENV = gurobipy.Env()


# ===============================================================================
# 
# REPLANNING
#
# ===============================================================================
def objective_replan(T, kwargs, key, index):

    # transform hyper-parameters to natural space
    if kwargs['verbose']:
        print(f'[{index}] key={key}, T={T}...', flush=True)

    # initialize policy
    policy = GurobiOnlineController(
        rddl=deepcopy(kwargs['rddl']),
        plan=GurobiStraightLinePlan(**kwargs['plan_kwargs']),
        env=GLOBAL_ENV,
        rollout_horizon=T,
        **kwargs['planner_kwargs'])

    # initialize env for evaluation (need fresh copy to avoid concurrency)
    env = RDDLEnv(domain=kwargs['domain'],
                  instance=kwargs['instance'],
                  enforce_action_constraints=False)
    
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
    del policy.env
    return average_reward


class GurobiParameterTuningReplan:
    
    def __init__(self, env: RDDLEnv,
                 lookahead_range: List[int]=list(range(1, 41)),
                 timeout_training: float=None,
                 timeout_tuning: float=np.inf,
                 eval_trials: int=5,
                 verbose: bool=True,
                 planner_kwargs: Dict={
                     'model_params': {'NonConvex': 2, 'OutputFlag': 0},
                     'verbose': 0,
                 },
                 plan_kwargs: Dict={},
                 pool_context: str='spawn',
                 num_workers: int=4,
                 poll_frequency: float=0.2) -> None:
        '''Creates a new instance for tuning the lookahead horizon for Gurobi
        planners.
        
        :param env: the RDDLEnv describing the MDP to optimize
        :param lookahead_range: list of lookahead horizons to validate from
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
        '''
        # timeout for training must be set in Gurobi variables
        if timeout_training is not None:
            if 'model_params' not in planner_kwargs:
                planner_kwargs['model_params'] = {}
            planner_kwargs['model_params']['TimeLimit'] = timeout_training
        
        self.env = env
        self.lookahead_range = [t for t in lookahead_range 
                                if t <= self.env.horizon]
        self.timeout_training = timeout_training
        self.timeout_tuning = timeout_tuning
        self.eval_trials = eval_trials
        self.verbose = verbose
        self.planner_kwargs = planner_kwargs
        self.plan_kwargs = plan_kwargs
        self.pool_context = pool_context
        self.num_workers = num_workers
        self.poll_frequency = poll_frequency
        
    def summarize_hyperparameters(self):
        print(f'hyperparameter optimizer parameters:\n'
              f'    tuned_hyper_parameters    =horizon\n'
              f'    tuning_batch_size         ={self.num_workers}\n'
              f'    tuning_timeout            ={self.timeout_tuning}\n'
              f'    mp_pool_context_type      ={self.pool_context}\n'
              f'    mp_pool_poll_frequency    ={self.poll_frequency}\n'
              f'meta-objective parameters:\n'
              f'    planning_trials_per_iter  ={self.eval_trials}\n'
              f'    planning_timeout_per_trial={self.timeout_training}')
        
    def _pickleable_objective_with_kwargs(self):
        objective_fn = objective_replan
            
        # duplicate planner and plan keyword arguments must be removed
        plan_kwargs = self.plan_kwargs.copy()        
        planner_kwargs = self.planner_kwargs.copy()
        planner_kwargs.pop('rollout_horizon', None)
                        
        kwargs = {
            'rddl': self.env.model,
            'domain': self.env.domain_text,
            'instance': self.env.instance_text,
            'planner_kwargs': planner_kwargs,
            'plan_kwargs': plan_kwargs,
            'verbose': self.verbose,
            'eval_trials': self.eval_trials
        }
        return objective_fn, kwargs
    
    @staticmethod
    def _wrapped_evaluate(index, T, key, func, kwargs):
        target = func(T=T, kwargs=kwargs, key=key, index=index)
        pid = os.getpid()
        return index, pid, T, target
    
    def tune(self, key: int, filename: str) -> Dict[str, object]:
        '''Tunes the hyper-parameters for Jax planner, returns the best found.'''
        key = jax.random.PRNGKey(key)
        self.summarize_hyperparameters()
        
        start_time = time.time()
        
        # objective function
        objective = self._pickleable_objective_with_kwargs()
        evaluate = GurobiParameterTuningReplan._wrapped_evaluate
        
        # clear and prepare output file
        filename = self._filename(filename, 'csv')
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                ['pid', 'worker', 'iteration', 'target', 'best_target', 'T']
            )
                
        # start multiprocess evaluation
        best_T, best_target = None, -np.inf
        
        for (it, i) in enumerate(range(
            0, len(self.lookahead_range), self.num_workers)):
            batch = self.lookahead_range[i:i + self.num_workers]
            
            # check if there is enough time left for another iteration
            elapsed = time.time() - start_time
            if elapsed >= self.timeout_tuning:
                print(f'global time limit reached at iteration {it}, aborting')
                break
            
            # continue with next iteration
            print('\n' + '*' * 25 + 
                  '\n' + f'[{datetime.timedelta(seconds=elapsed)}] ' + 
                  f'starting iteration {it} with batch {batch}' + 
                  '\n' + '*' * 25)
            worker_ids = list(range(len(batch)))
            key, *subkeys = jax.random.split(key, num=len(batch) + 1)
            rows = [None] * len(batch)
            
            # create worker pool: note each iteration must wait for all workers
            # to finish before moving to the next            
            with get_context(self.pool_context).Pool(processes=len(batch)) as pool:
                
                # assign jobs to worker pool
                results = [
                    pool.apply_async(evaluate, worker_args + objective)
                    for worker_args in zip(worker_ids, batch, subkeys)
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
                        index, pid, T, target = results.pop(i).get()
                        if target > best_target:
                            best_T, best_target = T, target
                        rows[index] = [pid, index, it, target, best_target, T]
                        
            # write results of all processes in current iteration to file
            with open(filename, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(rows)
        
        # print summary of results
        elapsed = time.time() - start_time
        print(f'summary of hyper-parameter optimization:\n'
              f'    time_elapsed       ={datetime.timedelta(seconds=elapsed)}\n'
              f'    iterations         ={it + 1}\n'
              f'    best_T             ={best_T}\n'
              f'    best_meta_objective={best_target}\n')
        
        return best_T

    def _filename(self, name, ext):
        domainName = self.env.model.domainName()
        instName = self.env.model.instanceName()
        domainName = ''.join(c for c in domainName if c.isalnum() or c == '_')
        instName = ''.join(c for c in instName if c.isalnum() or c == '_')
        filename = f'{name}_{domainName}_{instName}.{ext}'
        return filename
    
