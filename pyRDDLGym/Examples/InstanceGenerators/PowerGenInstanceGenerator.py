import os
import random
from typing import Dict

from pyRDDLGym.Examples.InstanceGenerator import InstanceGenerator


class PowerGenInstanceGenerator(InstanceGenerator):
    
    def get_env_path(self) -> str:
        return os.path.join('PowerGen', 'Continuous')
    
    def get_domain_name(self) -> str:
        return 'power_gen'
    
    def sample_instance(self, params: Dict[str, object]) -> Dict[str, object]:
        plants = [f'p{i + 1}' for i in range(params['num_plants'])]
        
        nonfluent_keys = ['TEMP-VARIANCE', 'DEMAND-EXP-COEF', 
                          'MIN-DEMAND-TEMP', 'MIN-CONSUMPTION']
        nonfluents = {key: params[key] for key in nonfluent_keys if key in params}
        for p in plants:
            nonfluents[f'PROD-UNITS-MIN({p})'] = 0.0
            nonfluents[f'PROD-UNITS-MAX({p})'] = 20.0
        
        states = {'temperature': random.uniform(
            params['temp_min'], params['temp_max'])}
        
        return {
            'objects': {'plant': plants},
            'non-fluents': nonfluents,
            'init-states': states,
            'horizon': params['horizon'],
            'discount': params['discount'],
            'max-nondef-actions': 'pos-inf'
        }


params = [
    
    # 3 generators, lower variance
    {'num_plants': 3, 
     'TEMP-VARIANCE': 4.0, 'DEMAND-EXP-COEF': 0.01, 
     'MIN-DEMAND-TEMP': 11.7, 'MIN-CONSUMPTION': 2.0,
     'temp_min': 0.0, 'temp_max': 30.0,
     'horizon': 100, 'discount': 1.0},
    
    # 5 generators, lower variance
    {'num_plants': 5, 
     'TEMP-VARIANCE': 6.0, 'DEMAND-EXP-COEF': 0.015, 
     'MIN-DEMAND-TEMP': 11.7, 'MIN-CONSUMPTION': 2.5,
     'temp_min': 0.0, 'temp_max': 30.0,
     'horizon': 100, 'discount': 1.0},
    
    # 10 generators, mid variance
    {'num_plants': 10, 
     'TEMP-VARIANCE': 8.0, 'DEMAND-EXP-COEF': 0.02, 
     'MIN-DEMAND-TEMP': 11.7, 'MIN-CONSUMPTION': 3.0,
     'temp_min': 0.0, 'temp_max': 30.0,
     'horizon': 100, 'discount': 1.0},
    
    # 25 generators, mid variance
    {'num_plants': 25, 
     'TEMP-VARIANCE': 10.0, 'DEMAND-EXP-COEF': 0.025, 
     'MIN-DEMAND-TEMP': 11.7, 'MIN-CONSUMPTION': 3.5,
     'temp_min': 0.0, 'temp_max': 30.0,
     'horizon': 100, 'discount': 1.0},
    
    # 50 generators, high variance
    {'num_plants': 50, 
     'TEMP-VARIANCE': 12.0, 'DEMAND-EXP-COEF': 0.03, 
     'MIN-DEMAND-TEMP': 11.7, 'MIN-CONSUMPTION': 4.0,
     'temp_min': 0.0, 'temp_max': 30.0,
     'horizon': 100, 'discount': 1.0}
]
              
inst = PowerGenInstanceGenerator()
for i, param in enumerate(params):
    inst.save_instance(i + 1, param)
