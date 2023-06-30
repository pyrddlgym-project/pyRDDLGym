import math
import numpy as np
from typing import Dict

from pyRDDLGym.Examples.InstanceGenerator import InstanceGenerator


class MountainCarInstanceGenerator(InstanceGenerator):
    
    def get_env_path(self) -> str:
        return 'MountainCar'
    
    def get_domain_name(self) -> str:
        return 'mountain_car'
    
    def sample_instance(self, params: Dict[str, object]) -> Dict[str, object]:
        fx, xrange = self.generate_hills(params['terrain_xleft'], 
                                         params['terrain_widths'],
                                         params['terrain_heights'])
        
        points = self.generate_terrain_points(fx, params['num_points'], *xrange)
        segments = [f's{i + 1}' for i in range(len(points))]
        
        nonfluents = {}
        nonfluents['ACTION-PENALTY'] = 0.0
        nonfluents['MIN-POS'] = xrange[0]
        nonfluents['MAX-POS'] = xrange[1]
        nonfluents['GOAL-MIN'] = params['goal-min']
        for (i, (x1, y1, x2, y2)) in enumerate(points):
            nonfluents[f'X-START({segments[i]})'] = x1
            nonfluents[f'Y-START({segments[i]})'] = y1
            nonfluents[f'X-END({segments[i]})'] = x2
            nonfluents[f'Y-END({segments[i]})'] = y2
        
        states = {}
        states['pos'] = params['pos']
        states['vel'] = params['vel']
        
        return {
            'objects': {'segment': segments},
            'non-fluents': nonfluents,
            'init-states': states,
            'horizon': params['horizon'],
            'discount': params['discount'],
            'max-nondef-actions': 'pos-inf'
        }
    
    def generate_hill(self, x1, x2, h, o1, o2):
        slope = (o2 - o1) / (x2 - x1)
        intercept = o1 - slope * x1
        curve = lambda x: h * (np.sin(slope * x + intercept) + 1.0) / 2.
        return curve
    
    def generate_hills(self, x0, widths, heights):
        x1 = x0
        curves, ranges = [], []
        for i, (w, h) in enumerate(zip(widths, heights)):
            x2 = x1 + w
            if i == 0:
                curve = self.generate_hill(x1, x2, h, o1=-3 * math.pi / 2 * 0.7, o2=-math.pi / 2)
            elif i == len(widths) - 1:
                curve = self.generate_hill(x1, x2, h, o1=-math.pi / 2, o2=math.pi / 2 * 1.2)
            else:
                curve = self.generate_hill(x1, x2, h, o1=-math.pi / 2, o2=3 * math.pi / 2)
            curves.append(curve)
            ranges.append((x1, x2))
            x1 = x2
    
        def piecewise(x):
            for i, (a, b) in enumerate(ranges):
                if x >= a and x < b:
                    return curves[i](x)
            return curves[-1](x)
        
        xrange = (ranges[0][0], ranges[-1][1])
        
        return piecewise, xrange
    
    def generate_terrain_points(self, fx, num_points, xmin, xmax):
        xs = np.linspace(xmin, xmax, num_points)
        ys = np.asarray([fx(x) for x in xs])
        xstart = xs[:-1]
        ystart = ys[:-1]
        xend = xs[1:]
        yend = ys[1:]
        return list(zip(xstart, ystart, xend, yend))
        
