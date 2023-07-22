import json 
import numpy as np
import time
from typing import Dict

from pyRDDLGym.Core.Env.RDDLEnv import RDDLEnv
from pyRDDLGym.Core.Grounder.RDDLGrounder import RDDLGrounder
from pyRDDLGym.Core.Gurobi.GurobiRDDLBilevelOptimizer import GurobiRDDLBilevelOptimizer
from pyRDDLGym.Core.Gurobi.GurobiRDDLPlan import GurobiRDDLPlan
from pyRDDLGym.Core.Simulator.RDDLSimulator import RDDLSimulator

from pyRDDLGym.Examples.ExampleManager import ExampleManager


class GurobiExperiment:
    
    def __init__(self, model_params: Dict={'Presolve': 2, 'OutputFlag': 1},
                 iters: int=10, rollouts: int=100, seed: int=None, 
                 **compiler_kwargs):
        if seed is None:
            seed = GurobiExperiment.seed_from_time()
        model_params['Seed'] = seed
        self.seed = seed
        self.model_params = model_params
        self.iters = iters
        self.rollouts = rollouts
        self.compiler_kwargs = compiler_kwargs
        
    @staticmethod
    def _evaluate(world, policy, planner, n_steps, n_episodes):
        returns = []
        for _ in range(n_episodes):
            world.reset()
            total_reward = 0.0
            for t in range(n_steps):
                subs = world.subs
                if policy is None:
                    actions = {}
                else:
                    actions = policy.evaluate(
                        planner.compiler, planner.params, t, subs)
                _, reward, done = world.step(actions)
                total_reward += reward 
                if done: 
                    break
            returns.append(total_reward)
        return returns
    
    @staticmethod
    def _to_dict(d) -> Dict:
        if isinstance(d, (list, tuple)):
            res = {}
            for (i, v) in enumerate(d):
                res[i] = GurobiExperiment._to_dict(v)
        elif isinstance(d, dict):
            res = {}
            for (k, v) in d.items():
                res[k] = GurobiExperiment._to_dict(v)
        else:
            res = d
        return res
    
    @staticmethod
    def seed_from_time():
        t = int(time.time() * 1000.0)
        seed = ((t & 0xff000000) >> 24) + ((t & 0x00ff0000) >> 8) + \
               ((t & 0x0000ff00) << 8) + ((t & 0x000000ff) << 24)
        seed = seed % 2000000000
        return seed    
         
    def get_policy(self, model: RDDLEnv) -> GurobiRDDLPlan:
        raise NotImplementedError
    
    def get_state_init_bounds(self, model: RDDLEnv) -> Dict:
        raise NotImplementedError
    
    def get_experiment_id_str(self) -> str:
        raise NotImplementedError
    
    def run(self, domain: str, inst: int, horizon: int) -> None:
        
        # build the model of the environment
        EnvInfo = ExampleManager.GetEnvInfo(domain)    
        model = RDDLEnv(domain=EnvInfo.get_domain(),
                        instance=EnvInfo.get_instance(inst)).model
        
        # build the policy
        policy = self.get_policy(model)
        
        # build the bi-level planner
        planner = GurobiRDDLBilevelOptimizer(
            model, policy,
            state_bounds=self.get_state_init_bounds(model),
            rollout_horizon=horizon,
            use_cc=True,
            model_params=self.model_params,
            **self.compiler_kwargs)
        
        # build the evaluation environment
        world = RDDLGrounder(model._AST).Ground()
        world = RDDLSimulator(world, rng=np.random.default_rng(seed=self.seed))    
        
        # evaluate the baseline (i.e. no-op) policy
        log_dict = {}
        returns = GurobiExperiment._evaluate(
            world, None, planner, horizon, self.rollouts)
        print(f'\naverage return: {np.mean(returns)}\n')
        log_dict[-1] = {'returns': returns, 'mean_return': np.mean(returns), 
                        'std_return': np.std(returns)}
        
        # run the bi-level planner and evaluate at each iteration
        for callback in planner.solve(self.iters, float('nan')): 
            returns = GurobiExperiment._evaluate(
                world, policy, planner, horizon, self.rollouts)
            print(f'\naverage return: {np.mean(returns)}\n')  
            print('\nfinal policy:\n' + callback['policy_string']) 
            callback['returns'] = returns
            callback['mean_return'] = np.mean(returns)
            callback['std_return'] = np.std(returns)
            log_dict[callback['it']] = GurobiExperiment._to_dict(callback)
            
        # save log to file
        idstr = self.get_experiment_id_str()
        with open(f'{domain}_{inst}_{horizon}_{self.seed}_{idstr}.log', 'w') as file:
            json.dump(log_dict, file, indent=2)
