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


class GurobiReservoirExperiment(GurobiExperiment):
    
    def get_action_bounds(self, model):
        return {'release___t1': (0, 100),
                'release___t2': (0, 200),
                'release___t3': (0, 400)}
    
    def get_state_bounds(self, model):
        return {'rlevel___t1': (0, 100),
                'rlevel___t2': (0, 200),
                'rlevel___t3': (0, 400)}
    
    def get_state_init_bounds(self, model):
        return {'rlevel___t1': (50, 100),
                'rlevel___t2': (100, 200),
                'rlevel___t3': (250, 350)}
    
    def get_state_dependencies_S(self, model):
        return {'release___t1': ['rlevel___t1'],
                'release___t2': ['rlevel___t2'],
                'release___t3': ['rlevel___t3']}
    
    def prepare_simulation_plots(self, domain, inst, horizon, start_it=1, error=False): 
        id_strs1 = {'$\\mathrm{S}$': f'0_True_True_{self._chance}',
                   '$\\mathrm{PWS-C}$': f'1_False_True_{self._chance}',
                   '$\\mathrm{PWS-S}$': f'1_True_True_{self._chance}'}
        datas1 = {k: GurobiExperiment.load_json(domain, inst, horizon, v) 
                 for (k, v) in id_strs1.items()}
        
        id_strs2 = {'$\\mathrm{0}$': f'0_False_True_{self._chance}',
                    '$\\mathrm{1}$': f'1_False_True_{self._chance}',
                    '$\\mathrm{2}$': f'2_False_True_{self._chance}',
                    '$\\mathrm{3}$': f'3_False_True_{self._chance}'}
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
            if error:
                plt.legend(loc='upper right', ncol=4)
            plt.tight_layout()
            plt.savefig(os.path.join(
                 'gurobi_results', f'{domain}_{inst}_{horizon}_{lbl + label}.pdf'))
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
    

if __name__ == "__main__":
    if len(sys.argv) < 7:
        inst, horizon, constr, value, cases, chance = 1, 10, 'S', 'C', 1, 0.995
    else:
        inst, horizon, constr, value, cases, chance = sys.argv[1:7]
        horizon, cases, chance = int(horizon), int(cases), float(chance)
        
    dom = 'Reservoir linear'
    dom_test = dom
    for _ in range(1):
        experiment = GurobiReservoirExperiment(
            constr=constr, value=value, cases=cases, chance=chance)
        experiment.run(dom, inst, horizon, dom_test)
