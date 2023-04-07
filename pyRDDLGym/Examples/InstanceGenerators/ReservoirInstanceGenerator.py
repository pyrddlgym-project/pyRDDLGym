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
            
    
