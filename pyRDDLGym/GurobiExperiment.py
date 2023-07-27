import glob
import json 
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from typing import Dict

from pyRDDLGym.Core.Env.RDDLEnv import RDDLEnv
from pyRDDLGym.Core.Grounder.RDDLGrounder import RDDLGrounder
from pyRDDLGym.Core.Gurobi.GurobiRDDLBilevelOptimizer import GurobiRDDLBilevelOptimizer
from pyRDDLGym.Core.Gurobi.GurobiRDDLPlan import GurobiPiecewisePolicy
from pyRDDLGym.Core.Gurobi.GurobiRDDLPlan import GurobiQuadraticPolicy
from pyRDDLGym.Core.Gurobi.GurobiRDDLPlan import GurobiRDDLPlan
from pyRDDLGym.Core.Simulator.RDDLSimulator import RDDLSimulator

from pyRDDLGym.Examples.ExampleManager import ExampleManager

# settings for pyplot
SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams['text.usetex'] = True


class GurobiExperiment:
    
    def __init__(self, constr: str='S', value: str='C', cases: int=1,
                 model_params: Dict={'Presolve': 2,
                                     'PreSparsify': 1,
                                     'NumericFocus': 2,
                                     'MIPFocus': 3,
                                     'MIPGap': 0.05,
                                     'OutputFlag': 1},
                 iters: int=10,
                 rollouts: int=500,
                 chance: float=0.995,
                 seed: int=None):
        self.policy_class = GurobiExperiment._get_policy_class(constr, value, cases)
        self.constr = constr
        self.value = value
        self.cases = cases
        if value in {'S', 'L', 'Q'} or constr in {'L'}:
            model_params['NonConvex'] = 2
        
        if seed is None:
            seed = GurobiExperiment.seed_from_time()            
        model_params['Seed'] = seed
        
        self.model_params = model_params
        self.iters = iters
        self.rollouts = rollouts
        self.chance = chance
        self.seed = seed
    
    @staticmethod
    def _get_policy_class(constr, value, cases):
        assert constr in {'S', 'L'}
        assert value in {'C', 'S', 'L', 'Q'}
        if value == 'Q':
            assert cases == 0
        if cases == 0:
            return f'{value}'
        else:
            return f'PW{constr}{cases}-{value}'
        
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
    def seed_from_time():
        t = int(time.time() * 1000.0)
        seed = ((t & 0xff000000) >> 24) + ((t & 0x00ff0000) >> 8) + \
               ((t & 0x0000ff00) << 8) + ((t & 0x000000ff) << 24)
        seed = seed % 2000000000
        return seed    
         
    @staticmethod
    def load_json(domain: str, inst: int, horizon: int, policy: str, chance: str):
        values = []
        print(f'{domain}_{inst}_{horizon}_*_{policy}_{chance}.log')
        for filepath in glob.glob(os.path.join(
            'gurobi_results',
            f'{domain}_{inst}_{horizon}_*_{policy}_{chance}.log')):
            print(f'loading {filepath}')
            with open(filepath) as file:
                values.append(json.load(file, strict=False))
        return values
    
    @staticmethod
    def simulation_plots(domain: str, inst: str, horizon: int, chance: str,
                         policies, label='return', 
                         legend=True, 
                         legend_args={'loc': 'lower right', 'ncol': 2,
                                      'columnspacing': 0.2, 'borderpad': 0.2,
                                      'labelspacing': 0.2}): 
        
        # load log files
        logs = [GurobiExperiment.load_json(domain, inst, horizon, policy, chance) 
                for policy in policies]
        
        # read attributes from log files
        means, errors = [], []
        for pol_logs in logs:
            pol_values = []
            for log in pol_logs:
                log_values = []
                its = -1 if label == 'return' else 0
                for it in range(its, len(log) - 1):
                    log_it = log[str(it)]
                    if label == 'return':
                        value = log_it['rollouts']['mean']
                    elif label == 'epsilon':
                        value = log_it['inner_value']['epsilon']
                    log_values.append(value)
                pol_values.append(log_values)
            pol_values_mean = np.mean(pol_values, axis=0)
            pol_values_se = np.std(pol_values, axis=0) / len(pol_values)
            means.append(pol_values_mean)
            errors.append(pol_values_se)
        
        # plotting
        plt.figure(figsize=(6.4, 3.2))
        for curve, bars, policy in zip(means, errors, policies):
            x = np.arange(curve.size) + int(label == 'epsilon')
            plt.errorbar(x, curve, yerr=bars, label=policy)
            plt.xlabel('$\\mathrm{iteration}$')
            plt.ylabel('$\\mathrm{' + label + '}$')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        if legend:
            plt.legend(**legend_args)
        plt.tight_layout()
        plt.savefig(os.path.join(
            'gurobi_results',
            f'{domain}_{inst}_{horizon}_{chance}_{label}.pdf'))
        plt.clf()
        plt.close()
            
    def get_state_bounds(self, model) -> Dict:
        raise NotImplementedError
    
    def get_action_bounds(self, model) -> Dict:
        raise NotImplementedError
    
    def get_state_init_bounds(self, model) -> Dict:
        raise NotImplementedError
    
    def get_state_dependencies_S(self, model) -> Dict:
        raise NotImplementedError
    
    def get_state_dependencies_L(self, model) -> Dict:
        states = sorted(model.states.keys())
        return {action: states for action in model.actions}
    
    def get_policy(self, model) -> GurobiRDDLPlan:
        
        # state variables in constraints
        if self.constr == 'S':
            deps_constr = self.get_state_dependencies_S(model)
        elif self.constr == 'L':
            deps_constr = self.get_state_dependencies_L(model)
        
        # state variables in values
        if self.value == 'C':
            deps_values = {} 
        elif self.value == 'S':
            deps_values = self.get_state_dependencies_S(model)
        elif self.value == 'L':
            deps_values = self.get_state_dependencies_L(model)
        
        # policy
        if self.value == 'Q':
            policy = GurobiQuadraticPolicy(
                action_bounds=self.get_action_bounds(model)
            )
        else:
            policy = GurobiPiecewisePolicy(
                state_bounds=self.get_state_bounds(model),
                action_bounds=self.get_action_bounds(model),
                dependencies_constr=deps_constr,
                dependencies_values=deps_values,
                num_cases=self.cases
            )
        return policy
            
    def run(self, domain: str, inst: int, horizon: int, domain_test: str) -> None:
        
        # build the model of the environment
        EnvInfo = ExampleManager.GetEnvInfo(domain)    
        model = RDDLEnv(domain=EnvInfo.get_domain(),
                        instance=EnvInfo.get_instance(inst)).model
        
        # build the evaluation environment
        EnvInfo = ExampleManager.GetEnvInfo(domain_test) 
        world_model = RDDLEnv(domain=EnvInfo.get_domain(),
                              instance=EnvInfo.get_instance(inst)).model
        world_model = RDDLGrounder(world_model._AST).Ground()
        world = RDDLSimulator(world_model, rng=np.random.default_rng(seed=self.seed))    
        
        # build the policy
        policy = self.get_policy(world_model)
        
        # build the bi-level planner
        planner = GurobiRDDLBilevelOptimizer(
            model, policy,
            state_bounds=self.get_state_init_bounds(model),
            rollout_horizon=horizon,
            use_cc=True,
            model_params=self.model_params,
            chance=self.chance)
        
        # evaluate the baseline (i.e. no-op) policy
        log_dict = {}
        returns = GurobiExperiment._evaluate(
            world, None, planner, horizon, self.rollouts)
        log_dict[-1] = {'rollouts': {'mean': np.mean(returns),
                                     'std': np.std(returns),
                                     'min': np.min(returns),
                                     'max': np.max(returns),
                                     'values': ','.join(map(str, returns))}}
        
        print(f'\nPOLICY SUMMARY [{self.policy_class}]:'
              f'\nmean return: {np.mean(returns)}'
              f'\nstd return:  {np.std(returns)}')
        
        # run the bi-level planner and evaluate at each iteration
        for callback in planner.solve(self.iters, float('nan')): 
            returns = GurobiExperiment._evaluate(
                world, policy, planner, horizon, self.rollouts)
            callback['rollouts'] = {'mean': np.mean(returns),
                                    'std': np.std(returns),
                                    'min': np.min(returns),
                                    'max': np.max(returns),
                                    'values': ','.join(map(str, returns))}
            log_dict[callback['iteration']] = callback
            
            print(f'\nPOLICY SUMMARY [{self.policy_class}]:'
                  f'\nmean return: {np.mean(returns)}'
                  f'\nstd return:  {np.std(returns)}'
                  f'\npolicy:      {callback["policy"]}'
                  f'\nerror:       {callback["inner_value"]["epsilon"]}'
                  f'\nouter value: {callback["outer_value"]["policy"]}')
            
        # save log to file
        with open(os.path.join(
            'gurobi_results',
            f'{domain}_{inst}_{horizon}_{self.seed}_{self.policy_class}_{self.chance}.log'), 'w') as file:
            json.dump(log_dict, file, indent=2)
    
