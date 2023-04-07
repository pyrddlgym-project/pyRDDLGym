import random
from typing import Dict

from pyRDDLGym.Examples.InstanceGenerator import InstanceGenerator


class HVACInstanceGenerator(InstanceGenerator):
    
    def get_env_path(self) -> str:
        return 'HVAC'
    
    def get_domain_name(self) -> str:
        return 'hvac'
    
    def sample_instance(self, params: Dict[str, object]) -> Dict[str, object]:
        nh = params['num_heaters']
        nz = params['num_zones']
        heaters, zones = self._generate_layout(nh, nz, params['density'])
        
        obj_heaters = [f'h{i + 1}' for i in range(nh)]
        obj_zones = [f'z{i + 1}' for i in range(nz)]
        
        nonfluents = {}
        nonfluents['TEMP-ZONE-MIN'] = params['TEMP-ZONE-MIN']
        nonfluents['TEMP-ZONE-MAX'] = params['TEMP-ZONE-MAX']
        zones_to_switch = list(range(nz))
        random.shuffle(zones_to_switch)
        for i in zones_to_switch[:params['p-switch-number']]:
            z = obj_zones[i]
            nonfluents[f'P-SWITCH({z})'] = params['p-switch-prob']
        for h, h2z in zip(obj_heaters, heaters):
            for z in h2z:
                nonfluents[f'ADJ-HEATER({h}, {obj_zones[z]})'] = True
        for z, z2z in zip(obj_zones, zones):
            for z2 in z2z:
                nonfluents[f'ADJ-ZONES({z}, {obj_zones[z2]})'] = True
        
        states = {}
        for i, z in enumerate(obj_zones):
            states[f'temp-zone({z})'] = random.uniform(*params['temp-zone-range-init'])
        for i, h in enumerate(obj_heaters):
            states[f'temp-heater({h})'] = random.uniform(*params['temp-heater-range-init'])
                
        return {
            'objects': {'zone': obj_zones, 'heater': obj_heaters},
            'non-fluents': nonfluents,
            'init-states': states,
            'horizon': params['horizon'],
            'discount': params['discount'],
            'max-nondef-actions': 'pos-inf'
        }
            
    def _generate_layout(self, nh, nz, density):
        heaters = [set() for _ in range(nh)]
        zones = [set() for _ in range(nz)]
    
        # Each heater must be connected to at least one zone
        for h in range(nh):
            z = random.randrange(nz)
            heaters[h].add(z)
    
        # Each zone must be connected to at least one heater
        for z in range(nz):
            if not any((z in htz) for htz in heaters):
                h = random.randrange(nh)
                heaters[h].add(z)
    
        # Zones can be interconnected
        for z1 in range(nz):
            for z2 in range(z1 + 1, nz):
                if random.random() < density:
                    zones[z1].add(z2)
        return heaters, zones
    
