from typing import Dict

from pyRDDLGym.Core.Env.RDDLEnv import RDDLEnv
from pyRDDLGym.Core.Grounder.RDDLGrounder import RDDLGrounder
from pyRDDLGym.Core.Gurobi.GurobiRDDLBilevelOptimizer import GurobiRDDLBilevelOptimizer
from pyRDDLGym.Core.Gurobi.GurobiRDDLPlan import GurobiRDDLPlan
from pyRDDLGym.Core.Simulator.RDDLSimulator import RDDLSimulator

from pyRDDLGym.Examples.ExampleManager import ExampleManager


class GurobiExperiment:
    
    def __init__(self, model_params: Dict={'Presolve': 2, 'OutputFlag': 1},
                 iters: int=10, rollouts: int=500):
        self.model_params = model_params
        self.iters = iters
        self.rollouts = rollouts
        
    @staticmethod
    def _evaluate(world, policy, planner, n_steps, n_episodes):
        avg_reward = 0.
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
            avg_reward += total_reward / n_episodes
        return avg_reward
    
    @staticmethod
    def _pretty(d, indent=0) -> str:
        res = ''
        if isinstance(d, (list, tuple)):
            for (i, v) in enumerate(d):
                res += '\n' + '\t' * indent + str(i) + ':' + \
                GurobiExperiment._pretty(v, indent + 1)
        elif isinstance(d, dict):
            for (k, v) in d.items():
                res += '\n' + '\t' * indent + str(k) + ':' + \
                GurobiExperiment._pretty(v, indent + 1)
        else:
            res += '\n' + '\t' * indent + str(d)
        return res
    
    def get_policy(self, model: RDDLEnv) -> GurobiRDDLPlan:
        raise NotImplementedError
    
    def get_state_init_bounds(self, model: RDDLEnv) -> Dict:
        raise NotImplementedError
    
    def get_experiment_id_str(self) -> str:
        raise NotImplementedError
    
    def run(self, domain, inst, horizon):
        
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
            model_params=self.model_params)
        
        # build the evaluation environment
        world = RDDLGrounder(model._AST).Ground()
        world = RDDLSimulator(world)    
        
        # evaluate the baseline (i.e. no-op) policy
        reward_hist, error_hist = [], []
        avg_reward = GurobiExperiment._evaluate(
            world, None, planner, horizon, self.rollouts)
        print(f'\naverage reward achieved: {avg_reward}\n')
        reward_hist.append(avg_reward)
        
        # run the bi-level planner and evaluate at each iteration
        log = ''        
        for callback in planner.solve(self.iters, float('nan')): 
            avg_reward = GurobiExperiment._evaluate(
                world, policy, planner, horizon, self.rollouts)
            print(f'\naverage reward achieved: {avg_reward}\n')
            reward_hist.append(avg_reward)
            error_hist.append(callback['error'])    
            print('\nfinal policy:\n' + callback['policy_string'])
            log += GurobiExperiment._pretty(callback) + '\n\n'
        
        # append the reward and error history to the log
        log += 'reward history:\n' + '\n'.join(map(str, reward_hist)) + '\n\n' 
        log += 'error history:\n' + '\n'.join(map(str, error_hist))
        
        # save log to file
        idstr = self.get_experiment_id_str()
        with open(f'{domain}_{inst}_{horizon}_{idstr}.log', 'w') as file:
            file.write(log)
