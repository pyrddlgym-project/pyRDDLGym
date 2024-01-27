import numpy as np
from PIL import Image
import pygame
from pygame import gfxdraw

from pyRDDLGym.Core.Compiler.RDDLModel import PlanningModel
from pyRDDLGym.Visualizer.StateViz import StateViz


# code comes from openai gym
class PendulumVisualizer(StateViz):

    def __init__(self, model: PlanningModel, screen_dim=500, wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time
        self.screen_dim = screen_dim
    
    def init_canvas(self, figure_size):
        screen = pygame.Surface(figure_size)
        surf = pygame.Surface(figure_size)
        return screen, surf
        
    def convert2img(self, screen):
        data = np.transpose(np.array(pygame.surfarray.pixels3d(screen)), 
                            axes=(1, 0, 2))
        img = Image.fromarray(data)
        return img
    
    def render(self, state):
        screen, surf = self.init_canvas((self.screen_dim, self.screen_dim))
        
        surf.fill((255, 255, 255))
        bound = 2.2
        scale = self.screen_dim / (bound * 2)
        offset = self.screen_dim // 2
        
        rod_length = 1 * scale
        rod_width = 0.2 * scale
        l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
        coords = [(l, b), (l, t), (r, t), (r, b)]
        transformed_coords = []
        for c in coords:
            c = pygame.math.Vector2(c).rotate_rad(state['theta'] + np.pi / 2)
            c = (c[0] + offset, c[1] + offset)
            transformed_coords.append(c)
        gfxdraw.aapolygon(surf, transformed_coords, (204, 77, 77))
        gfxdraw.filled_polygon(surf, transformed_coords, (204, 77, 77))

        gfxdraw.aacircle(surf, offset, offset, int(rod_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(
            surf, offset, offset, int(rod_width / 2), (204, 77, 77)
        )

        rod_end = (rod_length, 0)
        rod_end = pygame.math.Vector2(rod_end).rotate_rad(state['theta'] + np.pi / 2)
        rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
        gfxdraw.aacircle(
            surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )
        gfxdraw.filled_circle(
            surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )
        
        # drawing axle
        gfxdraw.aacircle(surf, offset, offset, int(0.05 * scale), (0, 0, 0))
        gfxdraw.filled_circle(surf, offset, offset, int(0.05 * scale), (0, 0, 0))

        surf = pygame.transform.flip(surf, False, True)
        screen.blit(surf, (0, 0))
        
        pygame.time.wait(self._wait_time)
        
        img = self.convert2img(screen)
        
        del screen, surf        
        
        return img
    
