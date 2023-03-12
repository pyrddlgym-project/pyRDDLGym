import numpy as np

from pyRDDLGym.Examples.InstanceGenerator import InstanceGenerator


class RaceCarInstanceGenerator(InstanceGenerator):
    
    def get_env_name(self) -> str:
        return 'RaceCar'
    
    def generate_instance(self, instance: int) -> str:
        
        # square track with no obstacles
        if instance == 1:
            nonfluents = self._draw_box((-1., -1.), (1., 1.))
            X0 = -0.8
            Y0 = -0.8
            GX = 0.8
            GY = 0.8
            RADIUS = 0.05
            horizon = 100
        
        # circular track with no obstacles
        elif instance == 2:
            nonfluents = self._draw_circle(36, (0., 0.), 1.0)
            X0 = -0.7
            Y0 = 0.0
            GX = 0.7
            GY = 0.0
            RADIUS = 0.05
            horizon = 100
        
        # circular track with central circular obstacle
        elif instance == 3:
            nonfluents = self._draw_circle(36, (0., 0.), 1.0) + \
                         self._draw_circle(18, (0., 0.), 0.3)
            X0 = -0.7
            Y0 = 0.0
            GX = 0.7
            GY = 0.0
            RADIUS = 0.05
            horizon = 100
            
        # large square track with two obstacles
        elif instance == 4:
            nonfluents = self._draw_box((-2., -2.), (2., 2.)) + \
                         self._draw_circle(18, (-1.1, -1.1), 0.2) + \
                         self._draw_circle(18, (1.1, 1.1), 0.2)
            X0 = -1.7
            Y0 = -1.7
            GX = 1.7
            GY = 1.7
            RADIUS = 0.05
            horizon = 200
            
        # large square track with three obstacles
        elif instance == 5:
            nonfluents = self._draw_box((-2., -2.), (2., 2.)) + \
                         self._draw_circle(18, (-1.1, -1.1), 0.2) + \
                         self._draw_circle(18, (1.1, 1.1), 0.2) + \
                         self._draw_circle(18, (0., 0.), 0.3)
            X0 = -1.7
            Y0 = -1.7
            GX = 1.7
            GY = 1.7
            RADIUS = 0.05
            horizon = 200
            
        else:
            raise Exception(f'Invalid instance {instance} for RaceCar.')
        
        value = f'non-fluents racecar_{instance}' + ' {'
        value += '\n' + 'domain = racecar;'
        value += '\n' + 'objects {'
        value += '\n\t' + 'b : {' + ','.join(
            f'b{i + 1}' for i in range(len(nonfluents))) + '};'
        value += '\n' + '};'
        value += '\n' + 'non-fluents {' + '\n\t'
        nfs = []
        for (i, (X1, Y1, X2, Y2)) in enumerate(nonfluents):
            nfs.append(f'X1(b{i + 1}) = {X1};')
            nfs.append(f'Y1(b{i + 1}) = {Y1};')
            nfs.append(f'X2(b{i + 1}) = {X2};')
            nfs.append(f'Y2(b{i + 1}) = {Y2};')
        nfs.append(f'X0 = {X0};')
        nfs.append(f'Y0 = {Y0};')
        nfs.append(f'GX = {GX};')
        nfs.append(f'GY = {GY};')
        nfs.append(f'RADIUS = {RADIUS};')
        value += '\n\t'.join(nfs)
        value += '\n\t' + '};'
        value += '\n' + '}'
        
        value += '\n' + f'instance inst_racecar_{instance}' + ' {'
        value += '\n' + 'domain = racecar;'
        value += '\n' + f'non-fluents = racecar_{instance};'
        value += '\n' + 'init-state {'
        value += '\n\t' + f'x = {X0};'
        value += '\n\t' + f'y = {Y0};'
        value += '\n\t' + 'vx = 0.0;'
        value += '\n\t' + 'vy = 0.0;'
        value += '\n' + '};'
        
        value += '\n' + 'max-nondef-actions = pos-inf;'
        value += '\n' + f'horizon = {horizon};'
        value += '\n' + f'discount = 1.0;'
        value += '\n' + '}'
        return value
        
    def _draw_circle(self, num_pt, center, radius):
        cx, cy = center
        angles = np.linspace(0.0, 2.0 * np.pi, num_pt)
        xs = radius * np.cos(angles) + cx
        ys = radius * np.sin(angles) + cy
        X1, Y1 = xs[:-1], ys[:-1]
        X2, Y2 = xs[1:], ys[1:]
        return list(zip(X1, Y1, X2, Y2))
    
    def _draw_box(self, topleft, bottomright):
        x1, y1 = topleft
        x2, y2 = bottomright
        return [(x1, y1, x2, y1), (x2, y1, x2, y2),
                (x2, y2, x1, y2), (x1, y2, x1, y1)]


inst = RaceCarInstanceGenerator()
print(inst.generate_instance(5))