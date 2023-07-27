import matplotlib.pyplot as plt
import os
import sys

from pyRDDLGym.GurobiExperiment import GurobiExperiment


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
    for _ in range(5):
        experiment = GurobiReservoirExperiment(
            constr=constr, value=value, cases=cases, chance=chance)
        experiment.run(dom, inst, horizon, dom_test)
