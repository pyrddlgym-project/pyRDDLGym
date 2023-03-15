import random
from typing import Dict

from pyRDDLGym.Examples.InstanceGenerator import InstanceGenerator


class MarsRoverInstanceGenerator(InstanceGenerator):
    
    def get_env_path(self) -> str:
        return 'MarsRover'
    
    def get_domain_name(self) -> str:
        return 'mars_rover_science_mission'
    
    
    def sample_instance(self, params: Dict[str, object]) -> Dict[str, object]:
        nm = params['num_minerals']
        nr = params['num_rovers']
        x1, x2 = params['location_bounds']
        y1, y2 = x1, x2
        a1, a2 = params['area_bounds']
        v1, v2 = params['value_bounds']
        
        minerals = [f'm{i + 1}' for i in range(nm)]
        rovers = [f'd{i + 1}' for i in range(nr)]
        
        nonfluents = {}
        for m in minerals:
            nonfluents[f'MINERAL-POS-X({m})'] = random.uniform(x1, x2)
            nonfluents[f'MINERAL-POS-X({m})'] = random.uniform(y1, y2)
            nonfluents[f'MINERAL-AREA({m})'] = random.uniform(a1, a2)
            nonfluents[f'MINERAL-VALUE({m})'] = random.uniform(v1, v2)
        
        states = {}
        for r in rovers:
            states[f'pos-x({r})'] = random.uniform(x1, x2)
            states[f'pos-y({r})'] = random.uniform(y1, y2)
            states[f'vel-x({r})'] = 0.
            states[f'vel-y({r})'] = 0.
        
        return {
            'objects': {'mineral': minerals, 'rover': rovers},
            'non-fluents': nonfluents,
            'init-states': states,
            'horizon': params['horizon'],
            'discount': params['discount'],
            'max-nondef-actions': 'pos-inf'
        }
            
    
params = [
    
    # 2 rover and 2 mineral
    {'num_minerals': 2, 'num_rovers': 2, 'location_bounds': (-10., 10.),
     'area_bounds': (1., 7.), 'value_bounds': (1., 10.), 
     'horizon': 200, 'discount': 1.0},
    
    # 5 rover and 6 mineral
    {'num_minerals': 6, 'num_rovers': 5, 'location_bounds': (-10., 10.),
     'area_bounds': (1., 7.), 'value_bounds': (1., 10.), 
     'horizon': 200, 'discount': 1.0},
    
    # 10 rover and 15 mineral
    {'num_minerals': 15, 'num_rovers': 10, 'location_bounds': (-10., 10.),
     'area_bounds': (1., 7.), 'value_bounds': (1., 10.), 
     'horizon': 200, 'discount': 1.0},
    
    # 25 rover and 50 mineral
    {'num_minerals': 50, 'num_rovers': 25, 'location_bounds': (-10., 10.),
     'area_bounds': (1., 7.), 'value_bounds': (1., 10.), 
     'horizon': 200, 'discount': 1.0},
    
    # 50 rover and 100 mineral
    {'num_minerals': 100, 'num_rovers': 50, 'location_bounds': (-10., 10.),
     'area_bounds': (1., 7.), 'value_bounds': (1., 10.), 
     'horizon': 200, 'discount': 1.0}
]

inst = MarsRoverInstanceGenerator()
for i, param in enumerate(params):
    inst.save_instance(i + 1, param)
        
