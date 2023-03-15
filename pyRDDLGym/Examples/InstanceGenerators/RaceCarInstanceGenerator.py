import random
from typing import Dict

from pyRDDLGym.Examples.InstanceGenerator import InstanceGenerator


class RaceCarInstanceGenerator(InstanceGenerator):
    
    def get_env_path(self) -> str:
        return 'RaceCar'
    
    def get_domain_name(self) -> str:
        return 'racecar'
    
    def sample_instance(self, params: Dict[str, object]) -> Dict[str, object]:
        
        # generate random obstacles as boxes
        # two additional rectangles are generated to bound origin and goal
        boundaries = self._generate_rectangles(params['num_blocks'] + 2)
        boundaries.insert(0, (0., 0., 1., 1.))
        boundaries = [tuple(2 * xi - 1 for xi in pt) for pt in boundaries]
        nonfluents = {}
        objects = []
        i = 0
        for boundary in boundaries[:-2]:
            for (X1, Y1, X2, Y2) in self._line_segments(*boundary):
                obj = f'b{i + 1}'
                objects.append(obj)
                nonfluents[f'X1({obj})'] = X1
                nonfluents[f'Y1({obj})'] = Y1
                nonfluents[f'X2({obj})'] = X2
                nonfluents[f'Y2({obj})'] = Y2
                i += 1
        
        # generate state at random from last two boxes
        states = {}
        x1, y1, x2, y2 = boundaries[-2]
        X0 = random.uniform(x1, x2)
        Y0 = random.uniform(y1, y2)
        x1, y1, x2, y2 = boundaries[-1]
        GX = random.uniform(x1, x2)
        GY = random.uniform(y1, y2)
        states['x'] = nonfluents['X0'] = X0
        states['y'] = nonfluents['Y0'] = Y0
        nonfluents['GX'] = GX
        nonfluents['GY'] = GY
        nonfluents['RADIUS'] = 0.05
        states['vx'] = 0.0
        states['vy'] = 0.0
        
        return {
            'objects': {'b': objects},
            'non-fluents': nonfluents,
            'init-states': states,
            'horizon': 200,
            'discount': 1.0,
            'max-nondef-actions': 'pos-inf'
        }
            
    def _line_segments(self, x1, y1, x2, y2):
        return [(x1, y1, x2, y1), (x2, y1, x2, y2),
                (x2, y2, x1, y2), (x1, y2, x1, y1)]

    def _generate_rectangles(self, n):
        rectangles = []
        for _ in range(n):
            while True:
                width = random.uniform(0., 0.7)
                height = random.uniform(0., 0.7)
                x = random.uniform(0, 1 - width)
                y = random.uniform(0, 1 - height)
                new_rect = (x, y, x + width, y + height)
                overlap = False
                for rect in rectangles:
                    if self._intersect(rect, new_rect):
                        overlap = True
                        break
                if not overlap:
                    rectangles.append(new_rect)
                    break
        return rectangles

    def _intersect(self, rect1, rect2):
        return not (rect1[2] <= rect2[0] or  # left edge of rect1 is to the right of right edge of rect2
                    rect1[0] >= rect2[2] or  # right edge of rect1 is to the left of left edge of rect2
                    rect1[3] <= rect2[1] or  # top edge of rect1 is below bottom edge of rect2
                    rect1[1] >= rect2[3])  # bottom edge of rect1 is above top edge of rect2

params = [
    {'num_blocks': 1},
    {'num_blocks': 2},
    {'num_blocks': 3},
    {'num_blocks': 4},
    {'num_blocks': 5}    
]

inst = RaceCarInstanceGenerator()
for i, param in enumerate(params):
    inst.save_instance(i + 1, param)
