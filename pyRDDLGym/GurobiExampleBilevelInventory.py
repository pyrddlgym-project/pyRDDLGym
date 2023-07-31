import sys

from pyRDDLGym.GurobiExperiment import GurobiExperiment


class GurobiInventoryExperiment(GurobiExperiment):
    
    def get_action_bounds(self, model):
        return {'order___i1': (0, 10),
                'order___i2': (0, 10),
                'order___i3': (0, 10)} 
    
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
        

if __name__ == "__main__":
    if len(sys.argv) < 7:
        inst, horizon, constr, value, cases, chance = 0, 8, 'S', 'C', 1, 0.995
    else:
        inst, horizon, constr, value, cases, chance = sys.argv[1:7]
        horizon, cases, chance = int(horizon), int(cases), float(chance)
        
    dom = 'Inventory linear'
    dom_test = dom
    
    for _ in range(5):
        experiment = GurobiInventoryExperiment(
            constr=constr, value=value, cases=cases, chance=chance,
            iters=12)
        experiment.run(dom, inst, horizon, dom_test)
