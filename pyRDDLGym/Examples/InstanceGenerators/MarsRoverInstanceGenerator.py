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
            nonfluents[f'MINERAL-POS-Y({m})'] = random.uniform(y1, y2)
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
    
    # the difficulty here is modulated by increasing number of minerals relative
    # to the number of rovers, and making some minerals much more inferior to 
    # others, the agent is forced to delay some small immediate reward for taking
    # some additional time steps and fuel to accept a much larger reward in future.
    {'num_minerals': 2, 'num_rovers': 4, 'location_bounds': (-10., 10.),
     'area_bounds': (3., 6.), 'value_bounds': (0., 20.), 
     'horizon': 100, 'discount': 1.0},
    
    {'num_minerals': 5, 'num_rovers': 6, 'location_bounds': (-10., 10.),
     'area_bounds': (2., 6.), 'value_bounds': (0., 20.), 
     'horizon': 100, 'discount': 1.0},
    
    {'num_minerals': 8, 'num_rovers': 8, 'location_bounds': (-10., 10.),
     'area_bounds': (2., 5.), 'value_bounds': (0., 20.), 
     'horizon': 100, 'discount': 1.0},
    
    {'num_minerals': 12, 'num_rovers': 9, 'location_bounds': (-10., 10.),
     'area_bounds': (1., 4.), 'value_bounds': (0., 20.), 
     'horizon': 100, 'discount': 1.0},
    
    {'num_minerals': 20, 'num_rovers': 10, 'location_bounds': (-10., 10.),
     'area_bounds': (1., 3.), 'value_bounds': (0., 20.), 
     'horizon': 100, 'discount': 1.0}
]


inst = MarsRoverInstanceGenerator()
for i, param in enumerate(params):
    inst.save_instance(i + 1, param)
        

