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
