import sys

from pyRDDLGym.Core.Gurobi.GurobiRDDLPlan import GurobiPiecewisePolicy
from pyRDDLGym.GurobiExperiment import GurobiExperiment


class GurobiInventoryExperiment(GurobiExperiment):
    
    def __init__(self, *args, cases: int=1, **kwargs):
        super(GurobiInventoryExperiment, self).__init__(*args, **kwargs)
        self.cases = cases
        
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
        dependencies = {'order___i1': ['stock___i1'],
                        'order___i2': ['stock___i2'],
                        'order___i3': ['stock___i3'],
                        'order___i4': ['stock___i4']}
        
        policy = GurobiPiecewisePolicy(
            action_bounds=action_bounds,
            state_bounds=state_bounds,
            dependencies=dependencies,
            num_cases=self.cases
        )
        return policy

    def get_state_init_bounds(self, model):
        state_init_bounds = {'stock___i1': (0, 3),
                             'stock___i2': (0, 3),
                             'stock___i3': (0, 3),
                             'stock___i4': (0, 3)}
        return state_init_bounds
    
    def get_experiment_id_str(self):
        return str(self.cases)

            
if __name__ == "__main__":
    if len(sys.argv) < 5:
        dom, inst, horizon, cases = 'Inventory continuous', 1, 10, 1
    else:
        dom, inst, horizon, cases = sys.argv[1:5]
        horizon, cases = int(horizon), int(cases)
        
    experiment = GurobiInventoryExperiment(cases=cases)
    experiment.run(dom, inst, horizon)
    

