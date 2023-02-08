import numpy as np
from PIL import Image
import pygame
from pygame import Rect, gfxdraw, freetype

from pyRDDLGym.Core.Compiler.RDDLModel import PlanningModel
from pyRDDLGym.Visualizer.StateViz import StateViz

ROW_SIZE = 50
COL_SIZE = 50
ELEV_WIDTH = int(0.8 * COL_SIZE)
SIGN_RADIUS = 3

# Direction of elevators as texts
freetype.init()
DIR_TEXT = freetype.SysFont('freesansbold', 20)
NUM_IN_TEXT = freetype.SysFont('freesansbold', 20)
NUM_WAIT_TEXT = freetype.SysFont('freesansbold', 15)


# code comes from openai gym
class ElevatorVisualizer(StateViz):

    def __init__(self, model: PlanningModel, figure_size=[600, 400], wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time
        self._n_elev = len(self._model.objects['elevator'])
        self._n_floors = len(self._model.objects['floor'])
        self._n_cols = 7 + 2 * (self._n_elev - 1)
        self._n_rows = self._n_floors
        self._figure_size = (COL_SIZE * self._n_cols, ROW_SIZE * (self._n_rows + 2))
        self._nonfluents = model.nonfluents
        self._left = COL_SIZE 
        self._right = self._figure_size[0] - COL_SIZE
        self._top = ROW_SIZE
        self._bottom = self._figure_size[1] - ROW_SIZE
    
    def init_canvas(self, figure_size):
        screen = pygame.Surface(figure_size)
        surf = pygame.Surface(figure_size)
        return screen, surf
        
    def convert2img(self, screen):
        data = np.transpose(np.array(
            pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2))
        img = Image.fromarray(data)
        return img
    
    def render(self, state):
        
        # Get the state information
        FLUENT_SEP = PlanningModel.FLUENT_SEP
        elev_to_floor = {}
        for e in self._model.objects['elevator']:
            for fl in self._model.objects['floor']:
                state_key = self._model.ground_name('elevator-at-floor', [e, fl])
                if state[state_key]:
                    elev_to_floor[e] = fl
        assert len(elev_to_floor) == self._n_elev
        num_person_waiting_on_floor = 'num-person-waiting' + FLUENT_SEP + '{floor}'
        num_person_in_elevator = 'num-person-in-elevator' + FLUENT_SEP + '{elevator}'
        elev_dir_up = 'elevator-dir-up' + FLUENT_SEP + '{elevator}'
        elev_closed = 'elevator-closed' + FLUENT_SEP + '{elevator}'
        
        # Initialize the canvas
        screen, surf = self.init_canvas(self._figure_size)
        
        surf.fill((255, 255, 255))
        
        # Building exterior
        gfxdraw.box(surf, Rect(COL_SIZE // 2,
                               ROW_SIZE // 2,
                               self._figure_size[0] - COL_SIZE,
                               self._figure_size[1] - ROW_SIZE),
                               (0, 0, 0))
        gfxdraw.box(surf, Rect(self._left,
                               self._top,
                               self._right - self._left,
                               self._bottom - self._top),
                               (255, 255, 255))

        # Draw the vertical lines for elevator passages
        for i in range(self._n_elev):
            col_offset = 2 * i + 2
            gfxdraw.box(
                surf,
                Rect(self._left + col_offset * COL_SIZE,
                     self._top,
                     COL_SIZE,
                     self._bottom - self._top),
                (0, 0, 0, 50),
            )
            gfxdraw.box(
                surf,
                Rect(self._left + col_offset * COL_SIZE + int(0.08 * COL_SIZE),
                     self._top,
                     int(0.84 * COL_SIZE),
                     self._bottom - self._top),
                (255, 255, 255),
            )

        # Draw each floor
        for i in range(1, self._n_floors + 1):
            fl = self._model.objects['floor'][i - 1]
            row = i
            
            floor_y_coord = self._bottom - ROW_SIZE * row
            
            # Draw the floor level
            gfxdraw.hline(
                surf, self._left, self._right, floor_y_coord, (100, 100, 100))

            # Go through elevators
            for j in range(self._n_elev):
                elev = self._model.objects['elevator'][j]
                col_offset = 2 * j + 2
                
                if elev_to_floor[elev] == fl: 
                    n_person_elev = state[num_person_in_elevator.format(elevator=elev)]
                    e_closed = state[elev_closed.format(elevator=elev)]
                    e_up = state[elev_dir_up.format(elevator=elev)]
                    elev_left_top_coord = (
                        self._left + int((col_offset + 0.1) * COL_SIZE),
                        floor_y_coord)
                    
                    # Draw elevator
                    gfxdraw.rectangle(surf,
                                      Rect(elev_left_top_coord[0],
                                          elev_left_top_coord[1],
                                          int(COL_SIZE * 0.8),
                                          ROW_SIZE),
                                      (0, 0, 0))
                    if e_closed:
                        gfxdraw.box(surf,
                                    Rect(elev_left_top_coord[0],
                                        elev_left_top_coord[1],
                                        int(COL_SIZE * 0.8),
                                        ROW_SIZE),
                                    (0, 0, 0, 100))
                        gfxdraw.vline(
                            surf,
                            elev_left_top_coord[0] + ELEV_WIDTH // 2,
                            floor_y_coord,
                            floor_y_coord + ROW_SIZE,
                            (0, 0, 0)
                        )
                    else:
                        gfxdraw.box(surf,
                                    Rect(elev_left_top_coord[0],
                                        elev_left_top_coord[1],
                                        int(COL_SIZE * 0.8),
                                        ROW_SIZE),
                                    (0, 0, 0, 100))
                        gfxdraw.box(surf,
                                    Rect(elev_left_top_coord[0] + int(0.1 * COL_SIZE),
                                        elev_left_top_coord[1] + int(0.1 * ROW_SIZE),
                                        int(COL_SIZE * 0.6),
                                        int(ROW_SIZE * 0.8)),
                                    (255, 255, 255))
                        
                    # Display the direction
                    DIR_TEXT.render_to(surf,
                                       (int(elev_left_top_coord[0] + 0.15 * COL_SIZE),
                                        int(elev_left_top_coord[1] + 0.1 * ROW_SIZE)),
                                       "^" if e_up else "v",
                                       (0, 0, 0))
                    NUM_IN_TEXT.render_to(surf,
                                       (int(elev_left_top_coord[0] + 0.15 * COL_SIZE),
                                        int(elev_left_top_coord[1] + 0.4 * ROW_SIZE)),
                                       str(n_person_elev),
                                       (255, 0, 0) if n_person_elev > 0 else (0, 0, 255)
                                       )
                    gfxdraw.aacircle(
                        surf,
                        (elev_left_top_coord[0] + ELEV_WIDTH // 2),
                        elev_left_top_coord[1] - SIGN_RADIUS,
                        SIGN_RADIUS,
                        (255, 0, 0) if e_closed else (0, 255, 0),
                    )
                    gfxdraw.filled_circle(
                        surf,
                        (elev_left_top_coord[0] + ELEV_WIDTH // 2),
                        elev_left_top_coord[1] - SIGN_RADIUS,
                        SIGN_RADIUS,
                        (255, 0, 0) if e_closed else (0, 255, 0),
                    )

            n_person_waiting = state[num_person_waiting_on_floor.format(floor=fl)]
            col_offset = self._n_cols - 3.1
            NUM_IN_TEXT.render_to(
                surf,
                (self._left + int((col_offset) * COL_SIZE),
                 floor_y_coord + 0.5 * ROW_SIZE),
                str(n_person_waiting),
                (255, 0, 0) if n_person_waiting > 0 else (0, 0, 255)
            )
            
        # surf = pygame.transform.flip(surf, False, True)
        screen.blit(surf, (0, 0))
        
        pygame.time.wait(self._wait_time)
        
        img = self.convert2img(screen)
        
        del screen, surf        
        
        return img
    
