import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from pyRDDLGym.Core.Gurobi.GurobiRDDLPlan import GurobiPiecewisePolicy
from pyRDDLGym.GurobiExperiment import GurobiExperiment

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


class GurobiReservoirExperiment(GurobiExperiment):
    
    def __init__(self, *args, cases: int=1, linear_value: bool=False,
                 factored: bool=True, **kwargs):
        super(GurobiReservoirExperiment, self).__init__(*args, **kwargs)
        if linear_value:
            self.model_params['NonConvex'] = 2
        self.cases = cases
        self.linear_value = linear_value
        self.factored = factored
        self._chance = kwargs['chance']
        
    def get_policy(self, model):
        state_bounds = {'rlevel___t1': (0, 100),
                        'rlevel___t2': (0, 200),
                        'rlevel___t3': (0, 400),
                        'rlevel___t4': (0, 500)}
        action_bounds = {'release___t1': (0, 100),
                         'release___t2': (0, 200),
                         'release___t3': (0, 400),
                         'release___t4': (0, 500)}
        if self.factored:
            dependencies = {'release___t1': ['rlevel___t1'],
                            'release___t2': ['rlevel___t2'],
                            'release___t3': ['rlevel___t3'],
                            'release___t4': ['rlevel___t4']}
        else:
            dependencies = None
    
        policy = GurobiPiecewisePolicy(
            action_bounds=action_bounds,
            state_bounds=state_bounds,
            dependencies=dependencies,
            num_cases=self.cases,
            linear_value=self.linear_value
        )
        return policy
    
    def get_state_init_bounds(self, model):
        state_init_bounds = {'rlevel___t1': (50, 100),
                             'rlevel___t2': (100, 200),
                             'rlevel___t3': (250, 350),
                             'rlevel___t4': (350, 450)}
        return state_init_bounds
    
    def get_experiment_id_str(self):
        return f'{self.cases}_{self.linear_value}_{self.factored}_{self._chance}' 
    
    def prepare_simulation_plots(self, domain, inst, horizon, start_it=1, error=False): 
        id_strs = {'$\\mathrm{S}$': f'0_True_True_{self._chance}',
                   '$\\mathrm{PWS-C}$': f'1_False_True_{self._chance}',
                   '$\\mathrm{PWS-S}$': f'1_True_True_{self._chance}',
                   '$\\mathrm{PWL-L}$': f'1_True_False_{self._chance}'}
        datas = {k: GurobiExperiment.load_json(domain, inst, horizon, v) 
                 for (k, v) in id_strs.items()}
        
        # return curves vs iteration with error bars
        key = 'error' if error else 'mean_return'
        label = 'error' if error else 'return'
        plt.figure(figsize=(6.4, 3.2))
        for st in id_strs:
            values = []
            for data in datas[st]:
                values.append([it_data.get(key, np.nan) for it_data in data.values()])
            if values:
                values = np.asarray(values)
                return_curve = np.mean(values, axis=0)[start_it:]
                return_std = np.std(values, axis=0)[start_it:] / np.sqrt(values.shape[0])
                x = np.arange(1, return_curve.size + 1)
                plt.errorbar(x, return_curve, yerr=return_std, label=st)
        plt.xlabel('$\\mathrm{iteration}$')
        plt.ylabel('$\\mathrm{' + label + '}$')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        if error:
            plt.legend(loc='upper right')
        else:
            plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(
             'gurobi_results', f'{domain}_{inst}_{horizon}_{label}.pdf'))
        plt.clf()
        plt.close()

    def prepare_policy_plot(self, domain, inst, horizon):
        
        def pws_s(level):
            r1 = 2.442770810130966 + 0.0 * level if level >= 17.66596188384031 and level <= 69.88017050753558 else 43.19908300223227 + 0.0006996429874287747 * level
            r2 = 0.5615733378292518 + 0.11519666638587732 * level if level >= 27.665961883840197 and level <= 100.00000000000004 else 82.29421914403906 + 0.0 * level
            r3 = 0.0 + 0.32473268362836244 * level if level >= 37.66596188384028 and level <= 150.16130570243052 else 142.51100357900523 + 0.18827389288118004 * level
            return (r1, r2, r3)
            
        policies = {'pwss': pws_s}
        levels = list(range(40, 400))
        
        for key, policy in policies.items():
            rs = [[] for _ in range(3)]
            for stock in levels:
                r = policy(stock)
                for ri, rsi in zip(r, rs):
                    rsi.append(ri)
            plt.figure(figsize=(6.4, 3.2))
            for i, rsi in enumerate(rs):
                plt.plot(levels, rsi, label='$\mathrm{reservoir_' + str(i + 1) + '}$') 
            plt.xlabel('$\\mathrm{level}_i$')
            plt.ylabel('$\\mathrm{release}_i$')
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.legend(loc='upper right')
            plt.tight_layout()
            plt.savefig(os.path.join(
                'gurobi_results', f'{domain}_{inst}_{horizon}_{key}_policy.pdf'))
    
    def prepare_worst_case_analysis(self, domain, inst, horizon):
        id_str = self.get_experiment_id_str()
        data = GurobiExperiment.load_json(domain, inst, horizon, id_str)[0]
        data = data['9']
        ws0 = data['worst_state']
        wss = data['worst_next_states']
        wa = data['worst_action']
        noise = data['worst_noise']['normal']
        noise = [max(xi[0], 0) for xi in noise.values()]
        ris, ria = [], []
        rir1, rir2, rir3 = [], [], []
        for res in range(1, 4):
            ri = [ws0[f'rlevel___t{res}'][0]] + [d[f'rlevel___t{res}\''] for d in wss]
            ris.append(ri)
            ria.append([d[f'release___t{res}'] for d in wa])
        for it in range(0, 30, 3):
            rir1.append(noise[it])
            rir2.append(noise[it + 1])
            rir3.append(noise[it + 2])
        
        # plot state trajectory
        plt.figure(figsize=(6.4, 3.2))
        xs = np.arange(len(ris[0]))
        plt.bar(xs - 0.2, ris[0], width=0.2, label='$\mathrm{reservoir_1}$')
        plt.bar(xs + 0.0, ris[1], width=0.2, label='$\mathrm{reservoir_2}$')
        plt.bar(xs + 0.2, ris[2], width=0.2, label='$\mathrm{reservoir_3}$')
        plt.axhline(y=20, color='blue', linestyle='dotted', linewidth=1)
        plt.axhline(y=80, color='blue', linestyle='dotted', linewidth=1)
        plt.axhline(y=30, color='orange', linestyle='dotted', linewidth=1)
        plt.axhline(y=180, color='orange', linestyle='dotted', linewidth=1)
        plt.axhline(y=40, color='green', linestyle='dotted', linewidth=1)
        plt.axhline(y=380, color='green', linestyle='dotted', linewidth=1)
        plt.xlabel('$\\mathrm{epoch}$')
        plt.ylabel('$\\mathrm{level}_i$')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(
            'gurobi_results', f'{domain}_{inst}_{horizon}_worst_states.pdf'))
        plt.clf()
        plt.close()
        
        # plot worst rainfall
        plt.figure(figsize=(6.4, 3.2))
        plt.bar(xs[:-1] - 0.2, rir1, width=0.2, label='$r1$')
        plt.bar(xs[:-1] + 0.0, rir2, width=0.2, label='$r2$')
        plt.bar(xs[:-1] + 0.2, rir3, width=0.2, label='$r3$')
        plt.xlabel('$\\mathrm{epoch}$')
        plt.ylabel('$\\mathrm{rainfall}_i$')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(
            'gurobi_results', f'{domain}_{inst}_{horizon}_worst_rainfall.pdf'))
        plt.clf()
        plt.close()
        
        # plot control sequence
        plt.figure(figsize=(6.4, 3.2))
        plt.bar(xs[:-1] - 0.2, ria[0], width=0.2, label='$\mathrm{reservoir_1}$')
        plt.bar(xs[:-1] + 0.0, ria[1], width=0.2, label='$\mathrm{reservoir_2}$')
        plt.bar(xs[:-1] + 0.2, ria[2], width=0.2, label='$\mathrm{reservoir_3}$')
        plt.axhline(y=0, color='blue', linestyle='dotted', linewidth=1)
        plt.axhline(y=100, color='blue', linestyle='dotted', linewidth=1)
        plt.axhline(y=0, color='orange', linestyle='dotted', linewidth=1)
        plt.axhline(y=200, color='orange', linestyle='dotted', linewidth=1)
        plt.axhline(y=0, color='green', linestyle='dotted', linewidth=1)
        plt.axhline(y=400, color='green', linestyle='dotted', linewidth=1)
        plt.xlabel('$\\mathrm{epoch}$')
        plt.ylabel('$\\mathrm{release}_i$')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(
            'gurobi_results', f'{domain}_{inst}_{horizon}_worst_controls.pdf'))
        plt.clf()
        plt.close()


if __name__ == "__main__":
    dom = 'Reservoir linear'
    linear_value = False
    factored = True
    if len(sys.argv) < 5:
        inst, horizon, cases, chance = 1, 10, 1, 0.995
    else:
        inst, horizon, cases, chance = sys.argv[1:5]
        horizon, cases, chance = int(horizon), int(cases), float(chance)   
    for _ in range(5): 
        experiment = GurobiReservoirExperiment(
            cases=cases,
            linear_value=linear_value,
            factored=factored,
            chance=chance
        )
        experiment.run(dom, inst, horizon)

