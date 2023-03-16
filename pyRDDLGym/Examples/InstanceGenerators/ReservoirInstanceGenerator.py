import os
import random
from typing import Dict

from pyRDDLGym.Examples.InstanceGenerator import InstanceGenerator


class ReservoirInstanceGenerator(InstanceGenerator):
    
    def get_env_path(self) -> str:
        return os.path.join('Reservoir', 'Continuous')
    
    def get_domain_name(self) -> str:
        return 'reservoir_control_cont'
    
    def _generate_random_dag(self, n, maxedges):
        nodes = list(range(n))
        random.shuffle(nodes)
        graph = {node: set() for node in nodes}
        for i in range(n - 2, -1, -1):
            upstream = nodes[i + 1:]
            numedges = random.randint(1, min(maxedges, len(upstream)))
            children = random.sample(upstream, numedges)
            graph[nodes[i]].update(children)
        connected_to_sea = nodes[-1]
        return graph, connected_to_sea

    def sample_instance(self, params: Dict[str, object]) -> Dict[str, object]:
        n = params['num_reservoirs']
        graph, connected_to_sea = self._generate_random_dag(n, params['max_edges'])
        
        reservoirs = [f't{i + 1}' for i in range(n)]
        
        nonfluents = {}
        for t1, out in graph.items():
            for t2 in out:
                nonfluents[f'RES_CONNECT({reservoirs[t1]}, {reservoirs[t2]})'] = True
        nonfluents[f'CONNECTED_TO_SEA({reservoirs[connected_to_sea]})'] = True
        
        states = {}
        for i, res in enumerate(reservoirs):
            top = random.uniform(*params['top_range'])
            range_prop = random.uniform(*params['target_range'])
            range_top = range_prop * top
            lower = random.uniform(0., top - range_top)
            upper = lower + range_top
            nonfluents[f'MIN_LEVEL({res})'] = lower
            nonfluents[f'MAX_LEVEL({res})'] = upper
            nonfluents[f'TOP_RES({res})'] = top
            nonfluents[f'RAIN_VAR({res})'] = params['rain_var']
            states[f'rlevel({res})'] = random.uniform(0., top * 0.9)
        
        return {
            'objects': {'reservoir': reservoirs},
            'non-fluents': nonfluents,
            'init-states': states,
            'horizon': params['horizon'],
            'discount': params['discount'],
            'max-nondef-actions': 'pos-inf'
        }
            
    
params = [
    
    # 2 reservoirs
    {'num_reservoirs': 2, 'max_edges': 1,
     'top_range': (100., 200.), 'target_range': (0.4, 0.8), 'rain_var': 5.,
     'horizon': 100, 'discount': 1.0},
    
    # 5 reservoirs
    {'num_reservoirs': 5, 'max_edges': 2,
     'top_range': (200., 400.), 'target_range': (0.3, 0.7), 'rain_var': 8.,
     'horizon': 100, 'discount': 1.0},
    
    # 10 reservoirs
    {'num_reservoirs': 10, 'max_edges': 4,
     'top_range': (300., 500.), 'target_range': (0.3, 0.6), 'rain_var': 10.,
     'horizon': 100, 'discount': 1.0},
    
    # 25 reservoirs
    {'num_reservoirs': 25, 'max_edges': 10,
     'top_range': (400., 600.), 'target_range': (0.2, 0.5), 'rain_var': 20.,
     'horizon': 100, 'discount': 1.0},
    
    # 50 reservoirs
    {'num_reservoirs': 50, 'max_edges': 30,
     'top_range': (500., 700.), 'target_range': (0.1, 0.3), 'rain_var': 30.,
     'horizon': 100, 'discount': 1.0}
]

inst = ReservoirInstanceGenerator()
for i, param in enumerate(params):
    inst.save_instance(i + 1, param)
        
