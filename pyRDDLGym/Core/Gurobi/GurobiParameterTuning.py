import jax
import numpy as np
from typing import Callable, Dict, Tuple

from pyRDDLGym.Core.Env.RDDLEnv import RDDLEnv
from pyRDDLGym.Core.Gurobi.GurobiRDDLPlanner import GurobiStraightLinePlan
from pyRDDLGym.Core.Gurobi.GurobiRDDLPlanner import GurobiOnlineController
from pyRDDLGym.Core.Jax.JaxParameterTuning import JaxParameterTuning


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
    T, = param_values
    if kwargs['verbose']:
        print(f'[{index}] key={key}, T={T}...', flush=True)

    # initialize policy
    policy = GurobiOnlineController(
        rddl=kwargs['rddl'],
        plan=GurobiStraightLinePlan(**kwargs['plan_kwargs']),
        rollout_horizon=T,
        **kwargs['planner_kwargs'])

    # initialize env for evaluation (need fresh copy to avoid concurrency)
    env = RDDLEnv(domain=kwargs['domain'],
                  instance=kwargs['instance'],
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
    del policy.env
    return average_reward


class GurobiParameterTuningReplan(JaxParameterTuning):
    
    def __init__(self, *args,
                 hyperparams_dict: Dict[str, Tuple[float, float, Callable]]={
                     'T': (1, None, int)
                 },
                 timeout_training: float=None,
                 planner_kwargs: Dict={
                     'model_params': {'NonConvex': 2, 'OutputFlag': 0},
                     'verbose': 0,
                 },
                 **kwargs) -> None:
        
        # timeout for training must be set in Gurobi variables
        if timeout_training is not None:
            if 'model_params' not in planner_kwargs:
                planner_kwargs['model_params'] = {}
            planner_kwargs['model_params']['TimeLimit'] = timeout_training
            
        super(GurobiParameterTuningReplan, self).__init__(
            *args, 
            hyperparams_dict=hyperparams_dict,
            train_epochs=None,
            timeout_training=timeout_training,
            planner_kwargs=planner_kwargs,
            **kwargs)
        
        # set upper range of lookahead horizon to environment horizon
        if self.hyperparams_dict['T'][1] is None:
            self.hyperparams_dict['T'] = (1, self.env.horizon, int)
    
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
            'hyperparams_dict': self.hyperparams_dict,
            'planner_kwargs': planner_kwargs,
            'plan_kwargs': plan_kwargs,
            'verbose': self.verbose,
            'eval_trials': self.eval_trials
        }
        return objective_fn, kwargs
    
    def tune(self, key: int, filename: str, save_plot: bool=False) -> Dict[str, object]:
        key = jax.random.PRNGKey(key)
        return super(GurobiParameterTuningReplan, self).tune(key, filename, save_plot)
