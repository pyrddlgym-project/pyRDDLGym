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
            nonfluents[f'MIN-ACC({ac})'] = -10.0
            nonfluents[f'MAX-ACC({ac})'] = 10.0            
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

