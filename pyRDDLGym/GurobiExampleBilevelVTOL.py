import sys

from pyRDDLGym.Core.Gurobi.GurobiRDDLPlan import GurobiPiecewisePolicy
from pyRDDLGym.GurobiExperiment import GurobiExperiment


class GurobiVTOLExperiment(GurobiExperiment):
    
    def __init__(self, *args, cases: int=1, **kwargs):
        model_params = {'Presolve': 2, 'NonConvex': 2, 'OutputFlag': 1, 'MIPGap': 0.02}
        super(GurobiVTOLExperiment, self).__init__(
            *args, rollouts=1, model_params=model_params, **kwargs)
        self.cases = cases
        
    def get_policy(self, model):
        action_bounds = {'F': (-1, 1)}
        state_bounds = {'theta': (-0.38942, 0.71736),
                        'omega': (-20.0, 20.0)}
        dependencies = {'F': ['theta']}
        
        policy = GurobiPiecewisePolicy(
            action_bounds=action_bounds,
            state_bounds=state_bounds,
            dependencies=dependencies,
            linear_value=True,
            num_cases=self.cases
        )
        return policy
    
    def get_state_init_bounds(self, model):
        state_bounds_init = {'theta': (0.1, 0.1),
                             'omega': (0.0, 0.0)}
        return state_bounds_init
    
    def get_experiment_id_str(self):
        return str(self.cases)


if __name__ == "__main__":
    if len(sys.argv) < 5:
        dom, inst, horizon, cases = 'VTOL', 0, 6, 4
    else:
        dom, inst, horizon, cases = sys.argv[1:5]
        horizon, cases = int(horizon), int(cases)
        
    experiment = GurobiVTOLExperiment(cases=cases)
    experiment.run(dom, inst, horizon)
    
