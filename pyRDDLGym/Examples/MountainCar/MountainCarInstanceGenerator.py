import numpy as np
from typing import Dict

from pyRDDLGym.Examples.InstanceGenerator import InstanceGenerator


class MountainCarInstanceGenerator(InstanceGenerator):
    
    def get_env_path(self) -> str:
        return 'MountainCar'
    
    def get_domain_name(self) -> str:
        return 'mountain_car'
    
    def generate_rddl_variables(self, params: Dict[str, object]) -> Dict[str, object]:
        points = self.generate_terrain_points(params['terrain'])
        
        segments = [f's{i + 1}' for i in range(len(points))]
        
        nonfluents = {}
        for (i, (x1, y1, x2, y2)) in enumerate(points):
            nonfluents[f'X-START({segments[i]})'] = x1
            nonfluents[f'Y-START({segments[i]})'] = y1
            nonfluents[f'X-END({segments[i]})'] = x2
            nonfluents[f'Y-END({segments[i]})'] = y2
        if 'FORCE-NOISE-VAR' in params:
            nonfluents['FORCE-NOISE-VAR'] = params['FORCE-NOISE-VAR']
        
        states = {}
        if 'pos' in params:
            states['pos'] = params['pos']
        if 'vel' in params:
            states['vel'] = params['vel']
        
        return {
            'objects': {'segment': segments},
            'non-fluents': nonfluents,
            'init-states': states,
            'horizon': 200,
            'discount': 1.0,
            'max-nondef-actions': 'pos-inf'
        }
    
    def generate_terrain_points(self, fx, num_points=80, xmin=-1.2, xmax=0.6):
        xs = np.linspace(xmin, xmax, num_points)
        ys = np.asarray([fx(x) for x in xs])
        xstart = xs[:-1]
        ystart = ys[:-1]
        xend = xs[1:]
        yend = ys[1:]
        return list(zip(xstart, ystart, xend, yend))
        

params = [
    
    # easy terrain
    {'terrain': (lambda x: np.sin(3 * x) * 0.3 + 0.4),
     'FORCE-NOISE-VAR': 0.0},
    
    # normal terrain
    {'terrain': (lambda x: np.sin(3 * x) * 0.45 + 0.55),
     'FORCE-NOISE-VAR': 0.0},
    
    # hard terrain
    {'terrain': (lambda x: np.sin(3 * x) * 0.6 + 0.7),
     'FORCE-NOISE-VAR': 0.0},
    
    # normal terrain with small noise
    {'terrain': (lambda x: np.sin(3 * x) * 0.45 + 0.55),
     'FORCE-NOISE-VAR': 0.05},
    
    # hard terrain with large noise
    {'terrain': (lambda x: np.sin(3 * x) * 0.6 + 0.7),
     'FORCE-NOISE-VAR': 0.1}
]
              
inst = MountainCarInstanceGenerator()
inst.save_instances(params)
