import sys

from pyRDDLGym.Core.Gurobi.GurobiRDDLPlan import GurobiPiecewisePolicy
from pyRDDLGym.GurobiExperiment import GurobiExperiment


class GurobiInventoryExperiment(GurobiExperiment):
    
    def __init__(self, *args, cases: int=1, linear_value: bool=False, **kwargs):
        model_params = {'Presolve': 2, 'OutputFlag': 1}
        if linear_value:
            model_params['NonConvex'] = 2
        super(GurobiInventoryExperiment, self).__init__(
            *args, model_params=model_params, **kwargs)
        self.cases = cases
        self.linear_value = linear_value
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
        dependencies = {'order___i1': ['stock___i1'],
                        'order___i2': ['stock___i2'],
                        'order___i3': ['stock___i3'],
                        'order___i4': ['stock___i4']}
        
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

            
if __name__ == "__main__":
    dom = 'Inventory continuous'
    linear_value = False
    if len(sys.argv) < 5:
        inst, horizon, cases, chance = 1, 10, 1, 0.995
    else:
        inst, horizon, cases, chance = sys.argv[1:5]
        horizon, cases, chance = int(horizon), int(cases), float(chance)    
    experiment = GurobiInventoryExperiment(
        cases=cases, linear_value=linear_value, chance=chance)
    experiment.run(dom, inst, horizon)
    

