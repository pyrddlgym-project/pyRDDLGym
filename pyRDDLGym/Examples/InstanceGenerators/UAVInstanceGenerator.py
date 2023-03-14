import os
from typing import Dict

from pyRDDLGym.Examples.InstanceGenerator import InstanceGenerator


class UAVInstanceGenerator(InstanceGenerator):
    
    def get_env_path(self) -> str:
        return os.path.join('UAV', 'Continuous')
    
    def get_domain_name(self) -> str:
        return 'kinematic_UAVs_con'
    
    def generate_rddl_variables(self, params: Dict[str, object]) -> Dict[str, object]: 
        aircraft = [f'a{i + 1}' for i in range(params['num_aircraft'])]
        
        nonfluents = {}
        if 'CONTROLLABLE' in params:
            for i in params['CONTROLLABLE']:
                nonfluents[f'CONTROLLABLE({aircraft[i - 1]})'] = True
        if 'GRAVITY' in params:
            nonfluents['GRAVITY'] = params['GRAVITY']
        if 'GOAL' in params:
            for (i, (gx, gy, gz)) in enumerate(params['GOAL']):
                nonfluents[f'GOAL-X({aircraft[i]})'] = gx
                nonfluents[f'GOAL-Y({aircraft[i]})'] = gy
                nonfluents[f'GOAL-Z({aircraft[i]})'] = gz
        
        states = {}
        if 'pos' in params:
            for (i, (x, y, z)) in enumerate(params['pos']):
                states[f'pos-x({aircraft[i]})'] = x
                states[f'pos-y({aircraft[i]})'] = y
                states[f'pos-z({aircraft[i]})'] = z
        if 'vel' in params:
            for (i, v) in enumerate(params['vel']):
                states[f'vel({aircraft[i]})'] = v
        if 'angle' in params:
            for (i, (a1, a2, a3)) in enumerate(params['angle']):
                states[f'psi({aircraft[i]})'] = a1
                states[f'phi({aircraft[i]})'] = a2
                states[f'theta({aircraft[i]})'] = a3
                
        return {
            'objects': {'aircraft': aircraft},
            'non-fluents': nonfluents,
            'init-states': states,
            'horizon': 300,
            'discount': 1.0,
            'max-nondef-actions': 'pos-inf'
        }
            
    
params = [
    
    # one aircraft
    {'num_aircraft': 1, 'CONTROLLABLE': [1],
     'GOAL': [(50., 50., 50.)],
     'pos': [(0., 0., 0.)]},
    
    # three aircraft, one controllable
    {'num_aircraft': 3, 'CONTROLLABLE': [1],
     'GOAL': [(50., 50., 50.)] * 3,
     'pos': [(0., 0., 0.), (-5., -5., 0.), (5., 5., 0.)]},
    
    # two aircraft, two controllable
    {'num_aircraft': 2, 'CONTROLLABLE': [1, 2],
     'GOAL': [(50., 50., 50.), (-50., -50., 50.)],
     'pos': [(0., 0., 0.), (5., 5., 0.)]},
    
    # five aircraft, three controllable
    {'num_aircraft': 5, 'CONTROLLABLE': [1, 2, 4],
     'GOAL': [(100., 25., 25.), (25., 100., 25.), (50., 50., 50.),
              (25., 25., 100.), (50., 50., 50.)],
     'pos': [(0., 0., 0.), (5., 5., 0.), (-5., 5., 0.), (-5., -5., 0.), (5., -5., 0.)]},
    
    # ten aircraft, ten controllable
    {'num_aircraft': 10, 'CONTROLLABLE': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
     'GOAL': [(100., 100., 100.)] * 10,
     'pos': [(0., 0., 0.), (5., 5., 0.), (-5., 5., 0.), (-5., -5., 0.), (5., -5., 0.),
             (10., 10., 0.), (-10., 10., 0.), (-10., -10., 0.), (10., -10., 0.)]},
]

inst = UAVInstanceGenerator()
inst.save_instances(params)
        
