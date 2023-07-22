import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from pyRDDLGym.Core.Gurobi.GurobiRDDLPlan import GurobiPiecewisePolicy
from pyRDDLGym.GurobiExperiment import GurobiExperiment


class GurobiReservoirExperiment(GurobiExperiment):
    
    def __init__(self, *args, cases: int=1, linear_value: bool=False, **kwargs):
        super(GurobiReservoirExperiment, self).__init__(*args, **kwargs)
        if linear_value:
            self.model_params['NonConvex'] = 2
        self.cases = cases
        self.linear_value = linear_value
        self._chance = kwargs['chance']
        
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
            num_cases=self.cases,
            linear_value=self.linear_value
        )
        return policy
    
    def get_state_init_bounds(self, model):
        state_init_bounds = {'rlevel___t1': (50, 100),
                             'rlevel___t2': (100, 200),
                             'rlevel___t3': (250, 350),
                             'rlevel___t4': (350, 450)}
        return state_init_bounds
    
    def get_experiment_id_str(self):
        return f'{self.cases}_{self.linear_value}_{self._chance}' 

            
if __name__ == "__main__":
    dom = 'Reservoir linear'
    linear_value = False
    if len(sys.argv) < 5:
        inst, horizon, cases, chance = 1, 10, 1, 0.995
    else:
        inst, horizon, cases, chance = sys.argv[1:5]
        horizon, cases, chance = int(horizon), int(cases), float(chance)   
    for _ in range(5):   
        experiment = GurobiReservoirExperiment(
            cases=cases, linear_value=linear_value, chance=chance)
        experiment.run(dom, inst, horizon)
    
