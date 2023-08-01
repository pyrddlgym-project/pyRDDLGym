import glob
import json 
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
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
plt.style.use('ggplot')
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
                 model_params: Dict={'PreSparsify': 1,
                                     'Presolve': 2,
                                     'NumericFocus': 2,
                                     'MIPGap': 0.05,
                                     'OutputFlag': 1},
                 iters: int=10,
                 rollouts: int=500,
                 chance: float=0.995,
                 seed: int=None,
                 log: bool=False):
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
        self.log = log
        
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
        discount = world.rddl.discount
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
                total_reward += reward * (discount ** t) 
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
        filename = GurobiExperiment.filename(
            domain, inst, horizon, policy, chance, '*') + '.json'
        print(filename)
        for filepath in glob.glob(os.path.join('gurobi_results', filename)):
            print(f'loading {filepath}')
            with open(filepath) as file:
                values.append(json.load(file, strict=False))
        return values
    
    @staticmethod
    def filename(domain: str, inst: int, horizon: int, policy: str, 
                 chance: str, seed: int):
        return f'{domain}_{inst}_{horizon}_{policy}_{chance}_{seed}'
    
    @staticmethod
    def summarize(logs, seq_keys):
        values = []
        for log in logs:
            log_values = []
            for it in range(-1, len(log) - 1):
                log_it = log.get(str(it), None)
                if log_it is not None:
                    value = log_it
                    for key in seq_keys:
                        value = value.get(key, None)
                        if value is None:
                            break
                    if value is not None:
                        log_values.append(value)
            values.append(log_values)
        values_mean = np.mean(values, axis=0)
        values_se = 2 * np.std(values, axis=0) / len(values)
        return values_mean, values_se
        
    @staticmethod
    def simulation_plots(domain: str, inst: str, horizon: int, chance: str,
                         policies, label='return', 
                         legend=True, 
                         legend_args={'loc': 'lower right', 'ncol': 2,
                                      'columnspacing': 0.2, 'borderpad': 0.2,
                                      'labelspacing': 0.2},
                         ylim=None): 
        
        # read attributes from log files
        logs = [GurobiExperiment.load_json(domain, inst, horizon, policy, chance) 
                for policy in policies]
        means, errors = [], []
        for pol_logs in logs:
            if label == 'return':
                seq_keys = ['rollouts', 'mean']
            elif label == 'epsilon':
                seq_keys = ['inner_value', 'epsilon']
            values_mean, values_se = GurobiExperiment.summarize(pol_logs, seq_keys)
            means.append(values_mean)
            errors.append(values_se)
        
        # plotting
        plt.figure(figsize=(6.4, 3.2))
        for curve, bars, policy in zip(means, errors, policies):
            x = np.arange(curve.size) + int(label == 'epsilon')
            plt.errorbar(x, curve, yerr=bars, label=policy)
            plt.xlabel('$\\mathrm{iteration}$')
            plt.ylabel('$\\mathrm{' + label + '}$')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        if ylim is not None:
            plt.ylim(ylim)
        plt.gca().set_yscale('log')

        if legend:
            plt.legend(**legend_args)
        plt.tight_layout()
        plt.savefig(os.path.join(
            'gurobi_results',
            f'{domain}_{inst}_{horizon}_{chance}_{label}.pdf'))
        plt.clf()
        plt.close()
    
    @staticmethod
    def size_plots(domain: str, inst: str, horizon: int, chance: str,
                   policies, model='outer'):
        for policy in policies:
            log = GurobiExperiment.load_json(
                domain, inst, horizon, policy, chance)[0]
            vars_data = {'iteration': [], 'binary': [], 'integer': [], 'real': []}
            constrs_data = {'iteration': [], 'linear': [], 'quadratic': [], 'general': []}
            for it in range(len(log) - 1):
                log_it = log[str(it)]
                log_vars = log_it['model_stats'][model]['variables']
                total = log_vars['total']
                discrete = log_vars['integer']
                binary = log_vars['binary']
                pw = log_vars['piecewise']
                integer = discrete - binary
                real = total - discrete - pw
                vars_data['iteration'].append(it)
                vars_data['binary'].append(binary)
                vars_data['integer'].append(integer)
                vars_data['real'].append(real)
                
                log_constrs = log_it['model_stats'][model]['constraints']
                linear = log_constrs['linear']
                quadratic = log_constrs['quadratic']
                general = log_constrs['general'] + log_constrs['SOS']
                constrs_data['iteration'].append(it)
                constrs_data['linear'].append(linear)
                constrs_data['quadratic'].append(quadratic)
                constrs_data['general'].append(general)
            
            vars_data = pd.DataFrame(vars_data)
            constrs_data = pd.DataFrame(constrs_data)
            
            fig, ax = plt.subplots(figsize=(6.4, 3.2))
            vars_data.plot(ax=ax, x='iteration', y=['binary', 'integer', 'real'], kind='bar')
            plt.xlabel('$\\mathrm{iteration}$')
            plt.ylabel('$\\mathrm{variables}$')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(
                'gurobi_results',
                f'{domain}_{inst}_{horizon}_{chance}_{policy}_{model}_variables.pdf'))
            plt.clf()
            plt.close()
            
            fig, ax = plt.subplots(figsize=(6.4, 3.2))
            constrs_data.plot(ax=ax, x='iteration', y=['linear', 'quadratic', 'general'], kind='bar')
            plt.xlabel('$\\mathrm{iteration}$')
            plt.ylabel('$\\mathrm{constraints}$')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(
                'gurobi_results',
                f'{domain}_{inst}_{horizon}_{chance}_{policy}_{model}_constraints.pdf'))
            plt.clf()
            plt.close()
               
    @staticmethod
    def plot_worst_case(domain: str, inst: str, horizon: int, chance: str,
                        policy: str, state_name: str, action_name: str, 
                        noise_name: str, n_noise_vars_per_step: int=2, 
                        noise: str='normal', state_bounds=[(20, 80), (30, 180)]):
        log = GurobiExperiment.load_json(domain, inst, horizon, policy, chance)[0]
        
        for it in range(len(log) - 1):
            data = log[str(it)]['inner_sol']
            worst_state = {k: v[0] for k, v in data['init_state'].items()}
            worst_states = [list(v.values()) for v in data['states']['policy']]
            worst_states = [list(worst_state.values())] + worst_states
            worst_states = np.asarray(worst_states)
            n_iters, n_cols = worst_states.shape
            worst_states = pd.DataFrame(
                {f'$\\mathrm{{{state_name}}} {i + 1}$': worst_states[:, i]
                 for i in range(n_cols)})
            worst_states['$\\mathrm{epoch}$'] = list(range(n_iters))
            
            _, ax = plt.subplots(figsize=(6.4, 3.2))
            worst_states.plot(ax=ax, x='$\\mathrm{epoch}$', 
                              y=[f'$\\mathrm{{{state_name}}} {i + 1}$' for i in range(n_cols)], 
                              kind='bar')
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            for (i, (sx, sy)) in enumerate(state_bounds):
                plt.axhline(y=sx, linestyle='dotted', color=colors[i])
                plt.axhline(y=sy, linestyle='dotted', color=colors[i])
            plt.xlabel('$\\mathrm{epoch}$')
            plt.ylabel(f'$\\mathrm{{{state_name}}}$')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(
                'gurobi_results',
                f'{domain}_{inst}_{horizon}_{chance}_{policy}_{it}_worst_states.pdf'))
            plt.clf()
            plt.close()
            
            worst_actions = [list(v.values()) for v in data['actions']['policy']]
            worst_actions = np.asarray(worst_actions)
            n_iters, n_cols = worst_actions.shape
            worst_actions = pd.DataFrame(
                {f'$\\mathrm{{{action_name}}} {i + 1}$': worst_actions[:, i]
                 for i in range(n_cols)})
            worst_actions['$\\mathrm{epoch}$'] = list(range(1, n_iters + 1))
            
            _, ax = plt.subplots(figsize=(6.4, 3.2))
            worst_actions.plot(ax=ax, x='$\\mathrm{epoch}$', 
                               y=[f'$\\mathrm{{{action_name}}} {i + 1}$' for i in range(n_cols)], 
                               kind='bar')
            plt.xlabel('$\\mathrm{epoch}$')
            plt.ylabel(f'$\\mathrm{{{action_name}}}$')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(
                'gurobi_results',
                f'{domain}_{inst}_{horizon}_{chance}_{policy}_{it}_worst_actions.pdf'))
            plt.clf()
            plt.close()
            
            if noise is not None:
                worst_noise = [v[0] for v in data['noises'][noise].values()]
                worst_noise = [tuple(worst_noise[i:i + n_noise_vars_per_step]) 
                               for i in range(0, len(worst_noise), n_noise_vars_per_step)]
                worst_noise = np.asarray(worst_noise)
                n_iters, n_cols = worst_noise.shape
                worst_noise = pd.DataFrame(
                    {f'$\\mathrm{{{noise_name}}} {i + 1}$': worst_noise[:, i]
                     for i in range(n_cols)})
                worst_noise['$\\mathrm{epoch}$'] = list(range(1, n_iters + 1))
                
                _, ax = plt.subplots(figsize=(6.4, 3.2))
                worst_noise.plot(ax=ax, x='$\\mathrm{epoch}$', 
                                 y=[f'$\\mathrm{{{noise_name}}} {i + 1}$' for i in range(n_cols)], 
                                 kind='bar')
                plt.xlabel('$\\mathrm{epoch}$')
                plt.ylabel(f'$\\mathrm{{{noise_name}}}$')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(
                    'gurobi_results',
                    f'{domain}_{inst}_{horizon}_{chance}_{policy}_{it}_worst_noise.pdf'))
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
        
        # update Gurobi parameters with log file path
        filename = GurobiExperiment.filename(
            domain, inst, horizon, self.policy_class, self.chance, self.seed)
        if self.log:
            model_params = self.model_params.copy()
            model_params['LogFile'] = f'gurobi_results\\{filename}.log'
        else:
            model_params = self.model_params
            
        # build the bi-level planner
        planner = GurobiRDDLBilevelOptimizer(
            model, policy,
            state_bounds=self.get_state_init_bounds(model),
            rollout_horizon=horizon,
            use_cc=True,
            model_params=model_params,
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
        with open(os.path.join('gurobi_results', f'{filename}.json'), 'w') as file:
            json.dump(log_dict, file, indent=2)
    
