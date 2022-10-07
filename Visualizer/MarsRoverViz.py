from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as plt_img
from PIL import Image

from Visualizer.StateViz import StateViz
from Grounder.RDDLModel import RDDLModel
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)

import Visualizer

class MarsRoverVisualizer(StateViz):
    def __init__(self, model: RDDLModel) -> None:

        self._model= model
        self._states = model.states
        self._nonfluents = model.nonfluents
        self._objects = model.objects
        self._interval = 10
        self._asset_path = "/".join(Visualizer.__file__.split("/")[:-1])      
        self._object_layout = None
        self._fig = None
        self._ax = None
        self._data = None
    
    def build_nonfluents_layout(self):
        picture_point_locaiton = {o:[None,None,None] for o in self._objects['picture-point']}

        # style of fluent_p1
        for k,v in self._nonfluents.items():
            if 'PICT_XPOS_' in k:
                point = k.split('_')[2]
                picture_point_locaiton[point][0] = v
            elif 'PICT_YPOS_' in k:
                point = k.split('_')[2]
                picture_point_locaiton[point][1] = v
            elif 'PICT_VALUE_' in k:
                point = k.split('_')[2]
                picture_point_locaiton[point][2] = v    

        return {'picture_point_location' : picture_point_locaiton}
    
    def build_states_layout(self, state):
        rover_location = {o:[None,None] for o in self._objects['rover']}
        pict_taken ={o:None for o in self._objects['picture-point']}
        

        for k,v in state.items():
            if 'xPos' == k:
                rover_location['r1'][0] = v
            elif 'yPos' == k:
                rover_location['r1'][1] = v

        for k,v in state.items():
            if 'picTaken_' in k:
                point = k.split('_')[1]
                taken = v
                pict_taken[point] = taken

        return {'picture_taken':pict_taken, 'rover_location':rover_location}

    def init_canvas(self, figure_size, dpi):
        fig = plt.figure(figsize = figure_size, dpi = dpi)
        ax = plt.gca()
        plt.xlim([-figure_size[0]//2, figure_size[0]//2])
        plt.ylim([-figure_size[1]//2, figure_size[1]//2])
        return fig, ax

    def build_object_layout(self):
        return {'nonfluents_layout':self.build_nonfluents_layout(), 'states_layout':self.build_states_layout()}

    def render(self, state, figure_size = [50, 50], dpi = 100, fontsize = 8):
        self._state = state
        nonfluent_layout = self.build_nonfluents_layout()
        state_layout = self.build_states_layout(state)
        fig, ax = self.init_canvas(figure_size, dpi)
        plt.axis('scaled')
        # plt.axis('off')


        for k,v in nonfluent_layout['picture_point_location'].items():
            if state_layout['picture_taken'][k] == False:
                p_point = plt.Circle((v[0],v[1]), radius=v[2], ec='forestgreen', fc='g',fill=True, alpha=0.5)
            else:
                p_point = plt.Circle((v[0],v[1]), radius=v[2], ec='forestgreen', fill=False)
            ax.add_patch(p_point)


        rover_img_path = self._asset_path + '/assets/mars-rover.png'
        rover_logo = plt_img.imread(rover_img_path)
        rover_logo_zoom = rover_logo.shape[0]/(dpi*90)

        for k,v in state_layout['rover_location'].items():
            imagebox = OffsetImage(rover_logo, zoom=rover_logo_zoom)
            ab = AnnotationBbox(imagebox, (v[0], v[1]), frameon = False)
            ax.add_artist(ab)

        print(state)
        plt.show()


    
