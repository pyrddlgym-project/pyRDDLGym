import math
import numpy as np
from PIL import Image
import pygame
from pygame import gfxdraw

from pyRDDLGym.Core.Compiler.RDDLModel import PlanningModel
from pyRDDLGym.Visualizer.StateViz import StateViz


# code comes from openai gym
class MountainCarVisualizer(StateViz):

    def __init__(self, model: PlanningModel, figure_size=[600, 400], wait_time=100) -> None:
        self._model = model
        self._figure_size = figure_size
        self._wait_time = wait_time
        
        self._nonfluents = model.nonfluents
    
    def init_canvas(self, figure_size):
        screen = pygame.Surface(figure_size)
        surf = pygame.Surface(figure_size)
        return screen, surf
        
    def convert2img(self, screen):
        data = np.transpose(np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2))
        img = Image.fromarray(data)
        return img
    
    def render(self, state):
        screen, surf = self.init_canvas(self._figure_size)
        
        world_width = self._nonfluents['MAX-POS'] - self._nonfluents['MIN-POS']
        scale = self._figure_size[0] / world_width
        carwidth = 40
        carheight = 20
        
        surf.fill((255, 255, 255))
        xs = np.linspace(self._nonfluents['MIN-POS'],
                         self._nonfluents['MAX-POS'], 100)
        ys = np.sin(3 * xs) * self._nonfluents['DEPTH'] + \
            (1. - self._nonfluents['DEPTH'])
        xys = list(zip((xs - self._nonfluents['MIN-POS']) * scale, ys * scale))
        pygame.draw.aalines(surf, points=xys, closed=False, color=(0, 0, 0))
        
        clearance = 10
        l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
        coords = []
        for c in [(l, b), (l, t), (r, t), (r, b)]:
            c = pygame.math.Vector2(c).rotate_rad(math.cos(3 * state['pos']))
            coords.append(
                (
                    c[0] + (state['pos'] - self._nonfluents['MIN-POS']) * scale,
                    c[1] + clearance + (
                        np.sin(3 * state['pos']) * self._nonfluents['DEPTH'] + \
                        (1. - self._nonfluents['DEPTH'])
                    ) * scale,
                )
            )

        gfxdraw.aapolygon(surf, coords, (0, 0, 0))
        gfxdraw.filled_polygon(surf, coords, (0, 0, 0))
        
        for c in [(carwidth / 4, 0), (-carwidth / 4, 0)]:
            c = pygame.math.Vector2(c).rotate_rad(math.cos(3 * state['pos']))
            wheel = (
                int(c[0] + (state['pos'] - self._nonfluents['MIN-POS']) * scale),
                int(c[1] + clearance + (
                    np.sin(3 * state['pos']) * self._nonfluents['DEPTH'] + \
                    (1. - self._nonfluents['DEPTH'])
                ) * scale),
            )
            gfxdraw.aacircle(
                surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128)
            )
            gfxdraw.filled_circle(
                surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128)
            )
            
        flagx = int((self._nonfluents['GOAL-MIN'] - self._nonfluents['MIN-POS']) * scale)
        flagy1 = int((np.sin(3 * self._nonfluents['GOAL-MIN']) * self._nonfluents['DEPTH'] + \
                      (1. - self._nonfluents['DEPTH'])) * scale)
        flagy2 = flagy1 + 50
        gfxdraw.vline(surf, flagx, flagy1, flagy2, (0, 0, 0))
        
        gfxdraw.aapolygon(
            surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )
        gfxdraw.filled_polygon(
            surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )
        
        surf = pygame.transform.flip(surf, False, True)
        screen.blit(surf, (0, 0))

        pygame.time.wait(self._wait_time)
        
        img = self.convert2img(screen)
        
        del screen, surf
        
        return img
    
