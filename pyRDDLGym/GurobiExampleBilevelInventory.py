import matplotlib.pyplot as plt
import numpy as np
import os
import sys

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


class GurobiInventoryExperiment(GurobiExperiment):
    
    def get_action_bounds(self, model):
        MAX_ORDER = model.nonfluents['MAX-ITEMS']
        return {'order___i1': (0, MAX_ORDER),
                'order___i2': (0, MAX_ORDER),
                'order___i3': (0, MAX_ORDER)} 
    
    def get_state_bounds(self, model):
        return {'stock___i1': (0, 100),
                'stock___i2': (0, 100),
                'stock___i3': (0, 100)}  
    
    def get_state_init_bounds(self, model):
        return {'stock___i1': (0, 2),
                'stock___i2': (0, 2),
                'stock___i3': (0, 2)}
    
    def get_state_dependencies_S(self, model):
        return {'order___i1': ['stock___i1'],
                'order___i2': ['stock___i2'],
                'order___i3': ['stock___i3']} 
        
    def prepare_simulation_plots(self, domain, inst, horizon, start_it=1, error=False): 
        id_strs1 = {'$\\mathrm{S}$': f'0_True_True_{self._chance}',
                   '$\\mathrm{PWS-C}$': f'1_False_True_{self._chance}',
                   '$\\mathrm{PWS-S}$': f'1_True_True_{self._chance}'}
        datas1 = {k: GurobiExperiment.load_json(domain, inst, horizon, v) 
                 for (k, v) in id_strs1.items()}
        
        id_strs2 = {'$\\mathrm{K=0}$': f'0_False_True_{self._chance}',
                    '$\\mathrm{K=1}$': f'1_False_True_{self._chance}',
                    '$\\mathrm{K=2}$': f'2_False_True_{self._chance}',
                    '$\\mathrm{K=3}$': f'3_False_True_{self._chance}'}
        datas2 = {k: GurobiExperiment.load_json(domain, inst, horizon, v) 
                  for (k, v) in id_strs2.items()}
        
        all_idstrs = [id_strs1, id_strs2]
        all_datas = [datas1, datas2]
        labels = ['', 'pwsc_']
        
        # return curves vs iteration with error bars
        for lbl, id_strs, datas in zip(labels, all_idstrs, all_datas):
            label = 'error' if error else 'return'
            plt.figure(figsize=(6.4, 3.2))
            for st in id_strs:
                values = []
                for data in datas[st]:
                    if error:
                        values.append([it_data.get('worst_value_inner', {}).get('slp', np.nan) - \
                                       it_data.get('worst_value_inner', {}).get('policy', np.nan)
                                       for it_data in data.values()])
                    else:
                        values.append([it_data.get('mean_return', np.nan) 
                                       for it_data in data.values()])                        
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
            plt.tight_layout()
            plt.savefig(os.path.join(
                 'gurobi_results', f'{domain}_{inst}_{horizon}_{lbl + label}.pdf'))
            plt.clf()
            plt.close()
    
    def prepare_policy_plot(self, domain, inst, horizon):
        
        def pws_c(level):
            r1 = 6.0 if level >= 0.0 and level <= 0.135 else 0.0
            r2 = 8.0 if level >= 0.0 and level <= 0.135 else 2.0
            return (r1, r2)
        
        policies = {'pwsc': pws_c}
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
    if len(sys.argv) < 7:
        inst, horizon, constr, value, cases, chance = 1, 10, 'S', 'C', 1, 0.995
    else:
        inst, horizon, constr, value, cases, chance = sys.argv[1:7]
        horizon, cases, chance = int(horizon), int(cases), float(chance)
        
    dom = 'Inventory deterministic'
    dom_test = 'Inventory randomized'
    
    for _ in range(1):
        experiment = GurobiInventoryExperiment(
            constr=constr, value=value, cases=cases, chance=chance)
        experiment.run(dom, inst, horizon, dom_test)
