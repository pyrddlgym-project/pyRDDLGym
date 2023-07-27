import matplotlib.pyplot as plt
import os
import sys

from pyRDDLGym.GurobiExperiment import GurobiExperiment


class GurobiInventoryExperiment(GurobiExperiment):
    
    def get_action_bounds(self, model):
        return {'order___i1': (0, 15),
                'order___i2': (0, 15),
                'order___i3': (0, 15)} 
    
    def get_state_bounds(self, model):
        return {'stock___i1': (-100, 100),
                'stock___i2': (-100, 100),
                'stock___i3': (-100, 100)}  
    
    def get_state_init_bounds(self, model):
        return {'stock___i1': (0, 2),
                'stock___i2': (0, 2),
                'stock___i3': (0, 2)}
    
    def get_state_dependencies_S(self, model):
        return {'order___i1': ['stock___i1'],
                'order___i2': ['stock___i2'],
                'order___i3': ['stock___i3']} 
        
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
        inst, horizon, constr, value, cases, chance = 1, 8, 'S', 'C', 1, 0.995
    else:
        inst, horizon, constr, value, cases, chance = sys.argv[1:7]
        horizon, cases, chance = int(horizon), int(cases), float(chance)
        
    dom = 'Inventory linear'
    dom_test = dom
    
    for _ in range(5):
        experiment = GurobiInventoryExperiment(
            constr=constr, value=value, cases=cases, chance=chance, log=True)
        experiment.run(dom, inst, horizon, dom_test)
