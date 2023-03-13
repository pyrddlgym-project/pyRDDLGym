from typing import Dict

from pyRDDLGym.Examples.InstanceGenerator import InstanceGenerator


class MarsRoverInstanceGenerator(InstanceGenerator):
    
    def get_env_path(self) -> str:
        return 'MarsRover'
    
    def get_domain_name(self) -> str:
        return 'mars_rover_science_mission'
    
    def generate_rddl_variables(self, params: Dict[str, object]) -> Dict[str, object]: 
        minerals = [f'm{i + 1}' for i in range(len(params['minerals']))]
        rovers = [f'd{i + 1}' for i in range(params['num_rovers'])]
        
        nonfluents = {}
        for (m, mineral_args) in zip(minerals, params['minerals']):
            x, y = mineral_args['pos']
            area = mineral_args['area']
            value = mineral_args['value']
            nonfluents[f'MINERAL-POS-X({m})'] = x
            nonfluents[f'MINERAL-POS-Y({m})'] = y
            nonfluents[f'MINERAL-AREA({m})'] = area
            nonfluents[f'MINERAL-VALUE({m})'] = value
            
        states = {}
        if 'pos' in params:
            for (r, (x, y)) in zip(rovers, params['pos']):
                states[f'pos-x({r})'] = x
                states[f'pos-y({r})'] = y
        if 'vel' in params:
            for (r, (x, y)) in zip(rovers, params['vel']):
                states[f'vel-x({r})'] = x
                states[f'vel-y({r})'] = y
        
        return {
            'objects': {'mineral': minerals, 'rover': rovers},
            'non-fluents': nonfluents,
            'init-states': states,
            'horizon': 100,
            'discount': 1.0,
            'max-nondef-actions': 'pos-inf'
        }
            
    
params = [
    
    # one rover and one mineral
    {'minerals': [
        {'pos': (5., 5.), 'area': 6., 'value': 8.}
    ],
     'num_rovers': 1,
     'pos': [(0., 0.)],
     'vel': [(1., 1.)]},
    
    # one rover and two mineral
    {'minerals': [
        {'pos': (7., 7.), 'area': 4., 'value': 8.},
        {'pos': (-5., -5.), 'area': 4., 'value': 1.}
    ],
     'num_rovers': 1,
     'pos': [(0., 0.)],
     'vel': [(1., 0.)]},
    
    # two rover and two mineral
    {'minerals': [
        {'pos': (5., 5.), 'area': 6., 'value': 8.},
        {'pos': (-8., -8.), 'area': 8., 'value': 5.}
    ],
     'num_rovers': 2,
     'pos': [(0., 0.), (3., 3.)],
     'vel': [(1., 1.), (2., 2.)]},
    
    # three rover and five mineral
    {'minerals': [
        {'pos': (10., 3.), 'area': 3., 'value': 9.},
        {'pos': (3., 10.), 'area': 5., 'value': 5.},
        {'pos': (0., 0.), 'area': 4., 'value': 2.},
        {'pos': (-8., -8.), 'area': 3., 'value': 10.},
        {'pos': (3., -8.), 'area': 4., 'value': 9.}
    ],
     'num_rovers': 3,
     'pos': [(-3., 0.), (-1., 0.), (1., 1.)],
     'vel': [(0., 0.), (0., 0.), (0., 0.)]}
]

inst = MarsRoverInstanceGenerator()
inst.save_instances(params)
        
