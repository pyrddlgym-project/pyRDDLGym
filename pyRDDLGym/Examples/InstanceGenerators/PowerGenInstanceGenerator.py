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
        num_gas = params['num_gas']
        num_nuclear = params['num_nuclear']
        num_solar = params['num_solar']
        num_plants = num_gas + num_nuclear + num_solar
        
        plants = [f'p{i + 1}' for i in range(num_plants)]        
        plant_types = ['gas'] * num_gas + ['nuclear'] * num_nuclear + ['solar'] * num_solar
        plant_params = [self._sample_plant(t) for t in plant_types]
        
        nonfluents = {}
        nonfluents[f'MIN-CONSUMPTION'] = 2.0 * params['demand_scale']
        nonfluents[f'DEMAND-EXP-COEF'] = 0.01 * params['demand_scale']
        
        able_demand = 0.0
        nonfluents[f'TEMP-VARIANCE'] = params['temp_variance']
        for plant, param in zip(plants, plant_params):
            able_demand += param['PROD-UNITS-MAX']
            for key, value in param.items():
                nonfluents[f'{key}({plant})'] = value
        
        def demand(t):
            return nonfluents[f'MIN-CONSUMPTION'] + nonfluents[f'DEMAND-EXP-COEF'] * (t - 11.7) ** 2
        
        max_demand = max(demand(-30.0), demand(+40.0))
        print(f'max demand = {max_demand}, capacity = {able_demand}')
        
        states = {}
        states[f'temperature'] = random.uniform(*params['temp_range'])
        
        return {
            'objects': {'plant': plants},
            'non-fluents': nonfluents,
            'init-states': states,
            'horizon': params['horizon'],
            'discount': params['discount'],
            'max-nondef-actions': 'pos-inf'
        }
    
    def _sample_plant(self, plant_type):
        if plant_type == 'gas':
            return self._sample_gas_fired_plant()
        elif plant_type == 'nuclear':
            return self._sample_nuclear_plant()
        elif plant_type == 'solar':
            return self._sample_solar_plant()
        else:
            raise Exception(f'Invalid plant type {plant_type}.')
        
    def _sample_gas_fired_plant(self):
        return {
            'PROD-UNITS-MIN': 1.0,
            'PROD-UNITS-MAX': 6.0,
            'TURN-ON-COST': 6.0,
            'PROD-CHANGE-PENALTY': 1.0,
            'COST-PER-UNIT': 4.0,
            'PROD-SHAPE': 1.0,
            'PROD-SCALE': 1e-7
        }
    
    def _sample_nuclear_plant(self):
        return {
            'PROD-UNITS-MIN': 2.0,
            'PROD-UNITS-MAX': 20.0,
            'TURN-ON-COST': 60.0,
            'PROD-CHANGE-PENALTY': 2.0,
            'COST-PER-UNIT': 1.0,
            'PROD-SHAPE': 1.0,
            'PROD-SCALE': 1e-3
        }
        
    def _sample_solar_plant(self):
        return {
            'PROD-UNITS-MIN': 1.0,
            'PROD-UNITS-MAX': 4.0,
            'TURN-ON-COST': 4.0,
            'PROD-CHANGE-PENALTY': 0.5,
            'COST-PER-UNIT': 3.0,
            'PROD-SHAPE': 1.0,
            'PROD-SCALE': 1.0
        }


