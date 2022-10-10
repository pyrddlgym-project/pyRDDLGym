from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as plt_img
from PIL import Image
import pprint

from Visualizer.StateViz import StateViz
from Grounder.RDDLModel import RDDLModel
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)

import Visualizer

class TextVisualizer(StateViz):
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
        return self._nonfluents
    
    def build_states_layout(self):
        return self._states
    
    def init_canvas(self, figure_size, dpi):
        self._fig = plt.figure(figsize = figure_size, dpi=dpi)
        self._ax = plt.gca()
        plt.xlim(0, figure_size[0]*self._interval)
        plt.ylim(0, figure_size[1]*self._interval)
        return self._fig, self._ax
        
    def build_object_layout(self):
        return {'nonfluents_layout':self.build_nonfluents_layout(), 'states_layout':self.build_states_layout()}        

    def render(self, state, figure_size = [5, 10], dpi = 100, fontsize = 8):
        
        nonfluent_layout = self.build_nonfluents_layout()
        state_layout = self.build_states_layout()
        text_layout = {'nonfluents': nonfluent_layout, 'state': state_layout}
        fig, ax = self.init_canvas(figure_size, dpi)

        text_str = pprint.pformat(text_layout)[1:-1]
        ax.text(self._interval*0.5, figure_size[1]*self._interval*0.95, text_str, 
                horizontalalignment='left', verticalalignment='top', wrap=True, fontsize = fontsize)
        plt.axis('scaled')
        plt.axis('off')
        ax.set_position((0, 0, 1, 1))
        fig.canvas.draw()

        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        self._data = data

        img = Image.fromarray(data)

        return img


    
    # def display_img(self, duration:float = 0.5) -> None:

    #     plt.imshow(self._data, interpolation='nearest')
    #     # plt.axis('off')
    #     plt.show(block=False)
    #     plt.pause(duration)
    #     plt.close()
    

    # def save_img(self, path:str ='./pict.png') -> None:

    #     im = Image.fromarray(self._data)
    #     im.save(path)
        

    
