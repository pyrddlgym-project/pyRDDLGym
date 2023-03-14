from typing import Dict

from pyRDDLGym.Examples.InstanceGenerator import InstanceGenerator


class HVACInstanceGenerator(InstanceGenerator):
    
    def get_env_path(self) -> str:
        return 'HVAC'
    
    def get_domain_name(self) -> str:
        return 'hvac'
    
    def generate_rddl_variables(self, params: Dict[str, object]) -> Dict[str, object]: 
        zones, heaters, adj_zone, adj_heater = self.generate_dag(
            params['ADJ-ZONES'], params['ADJ-HEATER'])
        
        nonfluents = {}
        for (z1, z2) in adj_zone:
            nonfluents[f'ADJ-ZONES({z1}, {z2})'] = True
        for (h1, z2) in adj_heater:
            nonfluents[f'ADJ-HEATER({h1}, {z2})'] = True
        
        states = {}
        if 'temp-zone' in params:
            for (z, val) in zip(zones, params['temp-zone']):
                states[f'temp-zone({z})'] = val
        if 'temp-heater' in params:
            for (h, val) in zip(heaters, params['temp-heater']):
                states[f'temp-heater({h})'] = val
        
        return {
            'objects': {'zone': zones, 'heater': heaters},
            'non-fluents': nonfluents,
            'init-states': states,
            'horizon': 120,
            'discount': 1.0,
            'max-nondef-actions': 'pos-inf'
        }
            
    def generate_dag(self, zone_connected, heater_connected):
        zones = sorted({i for tup in zone_connected for i in tup}.union(
                       {i for (_, i) in heater_connected}))
        zones = [f'z{zone}' for zone in zones]
        heaters = sorted({i for (i, _) in heater_connected})
        heaters = [f'h{heater}' for heater in heaters]
        
        adj_zone = [(f'z{z1}', f'z{z2}') for (z1, z2) in zone_connected]
        adj_heater = [(f'h{h1}', f'z{z2}') for (h1, z2) in heater_connected]
        return zones, heaters, adj_zone, adj_heater

    
params = [
    
    # one zone, one heater
    {'ADJ-ZONES': [],
     'ADJ-HEATER': [(1, 1)],
     'temp-zone': [5.], 'temp-heater': [0.]
    },
    
    # two zone, two heater
    {'ADJ-ZONES': [],
     'ADJ-HEATER': [(1, 1), (2, 2)],
     'temp-zone': [5., 1.], 'temp-heater': [0., 1.]
    },
    
    # three zone, two heater
    {'ADJ-ZONES': [(1, 3)],
     'ADJ-HEATER': [(1, 1), (2, 2), (2, 3)],
     'temp-zone': [5., 1., 1.], 'temp-heater': [0., 1., 1.]
    }
]

inst = HVACInstanceGenerator()
inst.save_instances(params)
        
