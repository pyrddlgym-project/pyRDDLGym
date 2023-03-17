import os
import random
from typing import Dict

from pyRDDLGym.Examples.InstanceGenerator import InstanceGenerator


class UAVInstanceGenerator(InstanceGenerator):
    
    def get_env_path(self) -> str:
        return os.path.join('UAV', 'Continuous')
    
    def get_domain_name(self) -> str:
        return 'kinematic_UAVs_con'
    
    def sample_instance(self, params: Dict[str, object]) -> Dict[str, object]:
        aircraft = [f'a{i + 1}' for i in range(params['num_aircraft'])]
        
        nonfluents = {}
        xrange = params['xrange']
        yrange = params['yrange']
        zrange = params['zrange']
        for ac in aircraft:
            nonfluents[f'GOAL-X({ac})'] = random.uniform(*xrange)
            nonfluents[f'GOAL-Y({ac})'] = random.uniform(*yrange)
            nonfluents[f'GOAL-Z({ac})'] = random.uniform(*zrange)
        nonfluents['RANDOM-WALK-COEFF'] = params['variance']
        
        crafts = list(range(len(aircraft)))
        random.shuffle(crafts)
        controlled = crafts[:params['num_control']]
        controlled = sorted(controlled)
        for ac in controlled:
            nonfluents[f'CONTROLLABLE({aircraft[ac]})'] = True
        
        states = {}
        for ac in aircraft:
            states[f'pos-x({ac})'] = random.uniform(*xrange)
            states[f'pos-y({ac})'] = random.uniform(*yrange)
            states[f'pos-z({ac})'] = zrange[0]
                        
        return {
            'objects': {'aircraft': aircraft},
            'non-fluents': nonfluents,
            'init-states': states,
            'horizon': params['horizon'],
            'discount': params['discount'],
            'max-nondef-actions': 'pos-inf'
        }
            
    
params = [
    
    # the difficulty of this problem is the noise of uncontrollable aircraft,
    # so the difficulty is controlled by number of such craft (relative to controlled),
    # the total number of craft (scale) and the variance of the uncontrolled state
    {'num_aircraft': 2, 'num_control': 2, 'variance': 0.1,
     'xrange': (-100., 100.), 'yrange': (-100., 100.), 'zrange': (0., 200.),
     'horizon': 300, 'discount': 1.0},
    
    {'num_aircraft': 4, 'num_control': 3, 'variance': 0.15,
     'xrange': (-100., 100.), 'yrange': (-100., 100.), 'zrange': (0., 200.),
     'horizon': 300, 'discount': 1.0},
    
    {'num_aircraft': 9, 'num_control': 6, 'variance': 0.2,
     'xrange': (-100., 100.), 'yrange': (-100., 100.), 'zrange': (0., 200.),
     'horizon': 300, 'discount': 1.0},
    
    {'num_aircraft': 20, 'num_control': 10, 'variance': 0.25,
     'xrange': (-100., 100.), 'yrange': (-100., 100.), 'zrange': (0., 200.),
     'horizon': 300, 'discount': 1.0},
    
    {'num_aircraft': 40, 'num_control': 10, 'variance': 0.3,
     'xrange': (-100., 100.), 'yrange': (-100., 100.), 'zrange': (0., 200.),
     'horizon': 300, 'discount': 1.0}
]

inst = UAVInstanceGenerator()
for i, param in enumerate(params):
    inst.save_instance(i + 1, param)
        
