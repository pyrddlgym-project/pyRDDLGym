import numpy as np
from typing import Dict

from pyRDDLGym.Examples.InstanceGenerator import InstanceGenerator


class RaceCarInstanceGenerator(InstanceGenerator):
    
    def get_env_path(self) -> str:
        return 'RaceCar'
    
    def get_domain_name(self) -> str:
        return 'racecar'
    
    def generate_rddl_variables(self, params: Dict[str, object]) -> Dict[str, object]: 
        nonfluent_keys = ['RADIUS', 'X0', 'Y0', 'GX', 'GY']
        nonfluents = {key: params[key] for key in nonfluent_keys if key in params}
        boundaries = []
        for (obs_type, obs_kwargs) in params['obstacles']:
            if obs_type == 'circle':
                boundaries += self._draw_circle(**obs_kwargs)
            elif obs_type == 'box':
                boundaries += self._draw_box(**obs_kwargs)
        
        bounds = [f'b{i + 1}' for i in range(len(boundaries))]
        
        for (i, (X1, Y1, X2, Y2)) in enumerate(boundaries):
            nonfluents[f'X1({bounds[i]})'] = X1
            nonfluents[f'Y1({bounds[i]})'] = Y1
            nonfluents[f'X2({bounds[i]})'] = X2
            nonfluents[f'Y2({bounds[i]})'] = Y2
        
        states = {}
        if 'X0' in params:
            states['x'] = params['X0']
        if 'Y0' in params:
            states['y'] = params['Y0']
        states['vx'] = 0.0
        states['vy'] = 0.0
        
        return {
            'objects': {'b': bounds},
            'non-fluents': nonfluents,
            'init-states': states,
            'horizon': 200,
            'discount': 1.0,
            'max-nondef-actions': 'pos-inf'
        }
            
    def _draw_circle(self, center, radius, num_points):
        cx, cy = center
        angles = np.linspace(0.0, 2.0 * np.pi, num_points)
        xs = radius * np.cos(angles) + cx
        ys = radius * np.sin(angles) + cy
        X1, Y1 = xs[:-1], ys[:-1]
        X2, Y2 = xs[1:], ys[1:]
        return list(zip(X1, Y1, X2, Y2))
    
    def _draw_box(self, top_left, bottom_right):
        x1, y1 = top_left
        x2, y2 = bottom_right
        return [(x1, y1, x2, y1), (x2, y1, x2, y2),
                (x2, y2, x1, y2), (x1, y2, x1, y1)]

          
params = [
    
    # square track with no obstacles
    {'RADIUS': 0.05, 'X0':-0.8, 'Y0':-0.8, 'GX': 0.8, 'GY': 0.8,
     'obstacles': [
         ('box', {'top_left': (-1., -1.), 'bottom_right': (1., 1.)})
     ]
    },
    
    # circular track with no obstacles
    {'RADIUS': 0.05, 'X0':-0.7, 'Y0': 0.0, 'GX': 0.7, 'GY': 0.0,
     'obstacles': [
         ('circle', {'center': (0., 0.), 'radius': 1.0, 'num_points': 36})
     ]
    },
    
    # circular track with central circular obstacle
    {'RADIUS': 0.05, 'X0':-0.7, 'Y0': 0.0, 'GX': 0.7, 'GY': 0.0,
     'obstacles': [
         ('circle', {'center': (0., 0.), 'radius': 1.0, 'num_points': 36}),
         ('circle', {'center': (0., 0.), 'radius': 0.3, 'num_points': 18})
     ]
    },
    
    # large square track with two obstacles
    {'RADIUS': 0.05, 'X0':-1.7, 'Y0':-1.7, 'GX': 1.7, 'GY': 1.7,
     'obstacles': [
         ('box', {'top_left': (-2., -2.), 'bottom_right': (2., 2.)}),
         ('circle', {'center': (-1., -1.), 'radius': 0.25, 'num_points': 18}),
         ('circle', {'center': (1., 1.), 'radius': 0.25, 'num_points': 18})
     ]
    },
    
    # large square track with five obstacles
    {'RADIUS': 0.05, 'X0':-1.7, 'Y0':-1.7, 'GX': 1.7, 'GY': 1.7,
     'obstacles': [
         ('box', {'top_left': (-2., -2.), 'bottom_right': (2., 2.)}),
         ('circle', {'center': (-1., -1.), 'radius': 0.25, 'num_points': 18}),
         ('circle', {'center': (1., 1.), 'radius': 0.25, 'num_points': 18}),
         ('circle', {'center': (0., 0.), 'radius': 0.4, 'num_points': 18}),
         ('circle', {'center': (-1., 1.), 'radius': 0.3, 'num_points': 18}),
         ('circle', {'center': (1., -1.), 'radius': 0.3, 'num_points': 18})
     ]
    }
]
         
inst = RaceCarInstanceGenerator()
inst.save_instances(params)
