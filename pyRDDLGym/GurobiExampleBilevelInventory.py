import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from pyRDDLGym.Core.Gurobi.GurobiRDDLPlan import GurobiPiecewisePolicy
from pyRDDLGym.GurobiExperiment import GurobiExperiment

# settings for pyplot
SMALL_SIZE = 18
MEDIUM_SIZE = 24
BIGGER_SIZE = 28

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams['text.usetex'] = True


class GurobiInventoryExperiment(GurobiExperiment):
    
    def __init__(self, *args, cases: int=1, linear_value: bool=False, 
                 factored: bool=True, **kwargs):
        super(GurobiInventoryExperiment, self).__init__(*args, **kwargs)
        if linear_value:
            self.model_params['NonConvex'] = 2
        self.cases = cases
        self.linear_value = linear_value
        self.factored = factored
        self._chance = kwargs['chance']
        
    def get_policy(self, model):
        MAX_ORDER = model.nonfluents['MAX-ITEMS']
        action_bounds = {'order___i1': (0, MAX_ORDER),
                         'order___i2': (0, MAX_ORDER),
                         'order___i3': (0, MAX_ORDER),
                         'order___i4': (0, MAX_ORDER)}    
        state_bounds = {'stock___i1': (0, MAX_ORDER),
                        'stock___i2': (0, MAX_ORDER),
                        'stock___i3': (0, MAX_ORDER),
                        'stock___i4': (0, MAX_ORDER)}  
        if self.factored:
            dependencies = {'order___i1': ['stock___i1'],
                            'order___i2': ['stock___i2'],
                            'order___i3': ['stock___i3'],
                            'order___i4': ['stock___i4']}
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
        state_init_bounds = {'stock___i1': (0, 3),
                             'stock___i2': (0, 3),
                             'stock___i3': (0, 3),
                             'stock___i4': (0, 3)}
        return state_init_bounds
    
    def get_experiment_id_str(self):
        return f'{self.cases}_{self.linear_value}_{self._chance}'

    def prepare_simulation_plots(self, domain, inst, horizon, start_it=1, error=False): 
        id_strs = {'$\\mathrm{L}$': f'0_True_{self._chance}',
                   '$\\mathrm{PWS-C}$': f'1_False_{self._chance}',
                   '$\\mathrm{PWS-L}$': f'1_True_{self._chance}'}
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
        
        def pws_c(level):
            r1 = 6.0 if level >= 0.0 and level <= 0.135 else 0.0
            r2 = 8.0 if level >= 0.0 and level <= 0.135 else 2.0
            return (r1, r2)
        
        def pws_l(level):
            r1 = 6.0 + 0.0 * level if level >= 0.0 and level <= 0.135 else 0.0 + 0.0 * level
            r2 = 8.0 + 0.0 * level if level >= 0.0 and level <= 0.134 else 2.0 + 0.0 * level
            return (r1, r2)
        
        policies = {'pwsc': pws_c, 'pwsl': pws_l}
        stocks = list(range(0, 20))
        
        for key, policy in policies.items():
            rs = [[] for _ in range(2)]
            for stock in stocks:
                r = policy(stock)
                for ri, rsi in zip(r, rs):
                    rsi.append(ri)
            plt.figure(figsize=(6.4, 3.2))
            for i, rsi in enumerate(rs):
                plt.plot(stocks, rsi, label='$\mathrm{product_' + str(i + 1) + '}$') 
            plt.xlabel('$\\mathrm{inventory}_i$')
            plt.ylabel('$\\mathrm{reorder}_i$')
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.legend(loc='upper right')
            plt.tight_layout()
            plt.savefig(os.path.join(
                'gurobi_results', f'{domain}_{inst}_{horizon}_{key}_policy.pdf'))

if __name__ == "__main__":
    dom = 'Inventory continuous'
    linear_value = False
    factored = True
    if len(sys.argv) < 5:
        inst, horizon, cases, chance = 1, 10, 1, 0.995
    else:
        inst, horizon, cases, chance = sys.argv[1:5]
        horizon, cases, chance = int(horizon), int(cases), float(chance)    
    for _ in range(5):
        experiment = GurobiInventoryExperiment(
            cases=cases, 
            linear_value=linear_value, 
            chance=chance,
            factored=factored)
        experiment.run(dom, inst, horizon)

