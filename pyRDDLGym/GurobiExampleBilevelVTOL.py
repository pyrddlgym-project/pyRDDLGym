import sys

from pyRDDLGym.Core.Gurobi.GurobiRDDLPlan import GurobiQuadraticPolicy
from pyRDDLGym.GurobiExperiment import GurobiExperiment


class GurobiVTOLExperiment(GurobiExperiment):
    
    def __init__(self, *args, **kwargs):
        model_params = {'Presolve': 2, 'NonConvex': 2, 'OutputFlag': 1}
        super(GurobiVTOLExperiment, self).__init__(
            *args, iters=5, rollouts=1, model_params=model_params, **kwargs)
        self._chance = kwargs['chance']
        
    def get_policy(self, model):
        action_bounds = {'F': (-1, 1)}
        state_bounds = {'theta': (-0.38942, 0.71736),
                        'omega': (-20.0, 20.0)}
        
        policy = GurobiQuadraticPolicy(
            action_bounds=action_bounds,
            state_bounds=state_bounds
        )
        return policy
    
    def get_state_init_bounds(self, model):
        state_bounds_init = {'theta': (0.1, 0.1),
                             'omega': (0.0, 0.0)}
        return state_bounds_init
    
    def get_experiment_id_str(self):
        return f'{self._chance}'


if __name__ == "__main__":
    dom = 'VTOL'
    if len(sys.argv) < 3:
        horizon, chance = 6, 0.995
    else:
        horizon, chance = sys.argv[1:3]
        horizon, chance = int(horizon), float(chance)    
    for _ in range(5): 
        experiment = GurobiVTOLExperiment(chance=chance)
        experiment.run(dom, 0, horizon)
    
