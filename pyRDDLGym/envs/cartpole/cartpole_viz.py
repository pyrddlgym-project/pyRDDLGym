import numpy as np
from PIL import Image
import pygame
from pygame import gfxdraw

from pyRDDLGym.Core.Compiler.RDDLModel import PlanningModel
from pyRDDLGym.Visualizer.StateViz import StateViz


# code comes from openai gym
class CartPoleVisualizer(StateViz):

    def __init__(self, model: PlanningModel, figure_size=[600, 400], wait_time=100) -> None:
        self._model = model
        self._figure_size = figure_size
        self._wait_time = wait_time
    
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
        screen, surf = self.init_canvas(self._figure_size)
        
        world_width = self._model.nonfluents['POS-LIMIT'] * 2
        scale = self._figure_size[0] / world_width
        polewidth = 10.0
        polelen = scale * (2 * self._model.nonfluents['POLE-LEN'])
        cartwidth = 50.0
        cartheight = 30.0
        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        
        surf.fill((255, 255, 255))
        cartx = state['pos'] * scale + self._figure_size[0] / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(surf, cart_coords, (0, 0, 0))
        
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-state['ang-pos'])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(surf, pole_coords, (202, 152, 101))
         
        gfxdraw.aacircle(
            surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.hline(surf, 0, self._figure_size[0], carty, (0, 0, 0))

        surf = pygame.transform.flip(surf, False, True)
        screen.blit(surf, (0, 0))
        
        pygame.time.wait(self._wait_time)
        
        img = self.convert2img(screen)
        
        del screen, surf        
        
        return img
    
