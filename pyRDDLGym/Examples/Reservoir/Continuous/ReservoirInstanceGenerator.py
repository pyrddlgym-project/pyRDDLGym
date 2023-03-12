import os
from typing import Dict

from pyRDDLGym.Examples.InstanceGenerator import InstanceGenerator


class ReservoirInstanceGenerator(InstanceGenerator):
    
    def get_env_path(self) -> str:
        return os.path.join('Reservoir', 'Continuous')
    
    def get_domain_name(self) -> str:
        return 'reservoir_control_dis'
    
    def generate_rddl_variables(self, params: Dict[str, object]) -> Dict[str, object]: 
        nodes, edges = self.generate_dag(params['connected'])
        reservoirs = [f't{i + 1}' for i in range(len(nodes))]
        
        nonfluents = {}
        for (e1, e2) in edges:
            nonfluents[f'RES_CONNECT({e1}, {e2})'] = True
        nonfluents[f'CONNECTED_TO_SEA({nodes[-1]})'] = True
        if 'MIN_LEVEL' in params:
            for (res, val) in zip(reservoirs, params['MIN_LEVEL']):
                nonfluents[f'MIN_LEVEL({res})'] = val
        if 'MAX_LEVEL' in params:
            for (res, val) in zip(reservoirs, params['MAX_LEVEL']):
                nonfluents[f'MAX_LEVEL({res})'] = val
        if 'TOP_RES' in params:
            for (res, val) in zip(reservoirs, params['TOP_RES']):
                nonfluents[f'TOP_RES({res})'] = val
        
        states = {}
        if 'rlevel' in params:
            for (res, val) in zip(reservoirs, params['rlevel']):
                states[f'rlevel({res})'] = val
        
        return {
            'objects': {'reservoir': reservoirs},
            'non-fluents': nonfluents,
            'init-states': states,
            'horizon': 100,
            'discount': 1.0,
            'max-nondef-actions': 'pos-inf'
        }
            
    def generate_dag(self, edges):
        nodes = sorted({i for tup in edges for i in tup})
        nodes = [f't{node}' for node in nodes]
        edges = [(f't{e1}', f't{e2}') for (e1, e2) in edges]
        return nodes, edges

    
params = [
    
    # two reservoirs
    {'connected': [(1, 2)],
     'MIN_LEVEL': [20.] * 2, 'MAX_LEVEL': [80.] * 2, 'TOP_RES': [100.] * 2,
     'rlevel': [50., 60.]},
    
    # three reservoirs
    {'connected': [(1, 3), (2, 3)],
     'MIN_LEVEL': [20.] * 3, 'MAX_LEVEL': [80.] * 3, 'TOP_RES': [100.] * 3,
     'rlevel': [45., 50., 50.]},
    
    # five reservoirs
    {'connected': [(1, 3), (2, 3), (3, 5), (4, 5)],
     'MIN_LEVEL': [20.] * 5, 'MAX_LEVEL': [80.] * 5, 'TOP_RES': [100.] * 5,
     'rlevel': [45., 50., 50., 45., 60.]},
    
    # ten reservoirs
    {'connected': [(1, 5), (2, 5), (2, 6), (3, 6),
                   (3, 7), (4, 7), (5, 8), (6, 8),
                   (6, 9), (7, 9), (8, 10), (9, 10)],
     'MIN_LEVEL': [20.] * 10, 'MAX_LEVEL': [80.] * 10, 'TOP_RES': [100.] * 10,
     'rlevel': [45., 50., 50., 60., 50., 50., 50., 40., 50., 95.]},
    
    # twenty reservoirs
    {'connected': [(1, 2), (2, 3), (3, 4), (4, 5),
                   (5, 6), (6, 7), (7, 8), (8, 9),
                   (9, 10), (10, 11), (11, 12), (12, 13),
                   (13, 14), (14, 15), (15, 16), (16, 17),
                   (17, 18), (18, 19), (19, 20)],
     'MIN_LEVEL': [70.] * 20, 'MAX_LEVEL': [140.] * 20, 'TOP_RES': [200.] * 20,
     'rlevel': [80.] * 20},
]

inst = ReservoirInstanceGenerator()
inst.save_instances(params)
        
