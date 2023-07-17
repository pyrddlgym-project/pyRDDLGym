from pyRDDLGym.Core.Gurobi.GurobiRDDLPlan import GurobiPiecewisePolicy

from pyRDDLGym.GurobiExperiment import GurobiExperiment


class GurobiReservoirExperiment(GurobiExperiment):
    
    def __init__(self, *args, cases: int=1, **kwargs):
        super(GurobiReservoirExperiment, self).__init__(*args, **kwargs)
        self.cases = cases
        
    def get_policy(self, model):
        state_bounds = {'rlevel___t1': (0, 100),
                        'rlevel___t2': (0, 200),
                        'rlevel___t3': (0, 400),
                        'rlevel___t4': (0, 500)}
        action_bounds = {'release___t1': (0, 100),
                         'release___t2': (0, 200),
                         'release___t3': (0, 400),
                         'release___t4': (0, 500)}
        dependencies = {'release___t1': ['rlevel___t1'],
                        'release___t2': ['rlevel___t2'],
                        'release___t3': ['rlevel___t3'],
                        'release___t4': ['rlevel___t4']}
    
        policy = GurobiPiecewisePolicy(
            action_bounds=action_bounds,
            state_bounds=state_bounds,
            dependencies=dependencies,
            num_cases=self.cases
        )
        return policy
    
    def get_state_init_bounds(self, model):
        state_init_bounds = {'rlevel___t1': (50, 100),
                             'rlevel___t2': (100, 200),
                             'rlevel___t3': (200, 400),
                             'rlevel___t4': (300, 500)}
        return state_init_bounds
    
    def get_experiment_id_str(self):
        return str(self.cases)

            
if __name__ == "__main__":
    experiment = GurobiReservoirExperiment(cases=1)
    experiment.run('Reservoir linear', 1, 10)
    
