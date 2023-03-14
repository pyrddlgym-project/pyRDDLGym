import os
from typing import Dict

from pyRDDLGym.Examples.InstanceGenerator import InstanceGenerator


class PowerGenInstanceGenerator(InstanceGenerator):
    
    def get_env_path(self) -> str:
        return os.path.join('PowerGen', 'Continuous')
    
    def get_domain_name(self) -> str:
        return 'power_gen'
    
    def generate_rddl_variables(self, params: Dict[str, object]) -> Dict[str, object]:
        plants = [f'p{i + 1}' for i in range(params['plants'])]
        
        nonfluent_keys = ['TEMP-VARIANCE', 'DEMAND-EXP-COEF',
                          'MIN-DEMAND-TEMP', 'MIN-CONSUMPTION']
        nonfluents = {key: params[key] for key in nonfluent_keys if key in params}
        if 'PROD-UNITS-MIN' in params:
            for (i, p) in enumerate(plants):
                nonfluents[f'PROD-UNITS-MIN({p})'] = params['PROD-UNITS-MIN'][i]
        if 'PROD-UNITS-MAX' in params:
            for (i, p) in enumerate(plants):
                nonfluents[f'PROD-UNITS-MAX({p})'] = params['PROD-UNITS-MAX'][i]
                
        state_keys = ['temperature']
        
        return {
            'objects': {'plant': plants},
            'non-fluents': nonfluents,
            'init-states': {key: params[key] for key in state_keys if key in params},
            'horizon': 40,
            'discount': 1.0,
            'max-nondef-actions': 'pos-inf'
        }


params = [
    
    # three generators, lower variance
    {'plants': 3, 'TEMP-VARIANCE': 4.0,
     'PROD-UNITS-MIN': [0.0] * 3, 'PROD-UNITS-MAX': [10.0] * 3,
     'DEMAND-EXP-COEF': 0.01, 'MIN-DEMAND-TEMP': 11.7, 'MIN-CONSUMPTION': 2.0},
    
    # five generators, lower variance
    {'plants': 5, 'TEMP-VARIANCE': 6.0,
     'PROD-UNITS-MIN': [0.0] * 5, 'PROD-UNITS-MAX': [15.0] * 5,
     'DEMAND-EXP-COEF': 0.015, 'MIN-DEMAND-TEMP': 11.7, 'MIN-CONSUMPTION': 2.5},
    
    # ten generators, mid variance
    {'plants': 10, 'TEMP-VARIANCE': 8.0,
     'PROD-UNITS-MIN': [0.0] * 10, 'PROD-UNITS-MAX': [20.0] * 10,
     'DEMAND-EXP-COEF': 0.02, 'MIN-DEMAND-TEMP': 11.7, 'MIN-CONSUMPTION': 3.0},
    
    # twenty generators, mid variance
    {'plants': 20, 'TEMP-VARIANCE': 10.0,
     'PROD-UNITS-MIN': [0.0] * 20, 'PROD-UNITS-MAX': [25.0] * 20,
     'DEMAND-EXP-COEF': 0.025, 'MIN-DEMAND-TEMP': 11.7, 'MIN-CONSUMPTION': 3.5},
    
    # thirty generators, high variance
    {'plants': 30, 'TEMP-VARIANCE': 12.0,
     'PROD-UNITS-MIN': [0.0] * 30, 'PROD-UNITS-MAX': [30.0] * 30,
     'DEMAND-EXP-COEF': 0.03, 'MIN-DEMAND-TEMP': 11.7, 'MIN-CONSUMPTION': 4.0}
]
              
inst = PowerGenInstanceGenerator()
inst.save_instances(params)
