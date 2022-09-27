from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as plt_img
from PIL import Image

from Visualizer.StateViz import StateViz
from Grounder.RDDLModel import RDDLModel
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)

import Visualizer

class ReservoirDisplay(StateViz):
    def __init__(self, model: RDDLModel, grid_size: Optional[int] = [50,50], resolution: Optional[int] = [500,500]) -> None:

        self._model= model
        self._states = model.states
        self._nonfluents = model.nonfluents
        self._objects = model.objects
        self._grid_size = grid_size
        self._resolution = resolution

        self._asset_path = "/".join(Visualizer.__file__.split("/")[:-1])      

        self._object_layout = None
        self._render_frame = None
        self._data = None

        self.render()

   
    def build_object_layout(self) -> dict:
        picture_point_locaiton = {o:[None,None,None] for o in self._objects['picture-point']}
        pict_taken ={o:None for o in self._objects['picture-point']}
        rover_location = {o:[None,None] for o in self._objects['rover']}

        # style of fluent(p1)
        for k,v in self._states.items():
            if 'picTaken(' in k:
                point_lst = k[k.find("(")+1:k.find(")")]
                point = point_lst.split(',')[0]
                taken = v
                pict_taken[point] = taken

        # update point location to exclude ones already b

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

        # need to change for later, currently only single rover possible
        for k,v in self._states.items():
            if 'xPos' == k:
                rover_location['r1'][0] = v
            elif 'yPos' == k:
                rover_location['r1'][1] = v


        object_layout = {'picture_point':picture_point_locaiton, 'picture_taken':pict_taken, 'rover_location':rover_location}

        return object_layout

        

    def render(self, display:bool = True) -> np.ndarray:

        self._object_layout = self.build_object_layout()
    
        px = 1/plt.rcParams['figure.dpi']
        fig = plt.figure(figsize=(self._resolution[0]*px,self._resolution[1]*px))
        ax = plt.gca()

        for k,v in self._object_layout['picture_point'].items():
            if self._object_layout['picture_taken'][k] == False:
                p_point = plt.Circle((v[0],v[1]), radius=v[2], ec='forestgreen', fc='g',fill=True, alpha=0.5)
            else:
                p_point = plt.Circle((v[0],v[1]), radius=v[2], ec='forestgreen', fill=False)
            ax.add_patch(p_point)
        
        rover_img_path = self._asset_path + '/assets/mars-rover.png'
        rover_logo = plt_img.imread(rover_img_path)
        rover_logo_zoom = self._resolution[0]*0.05/rover_logo.shape[0]

        for k,v in self._object_layout['rover_location'].items():
            imagebox = OffsetImage(rover_logo, zoom=rover_logo_zoom)
            ab = AnnotationBbox(imagebox, (v[0], v[1]), frameon = False)
            ax.add_artist(ab)
            # r_point = plt.Rectangle((v[0],v[1]), 1, 1, fc='navy')
            # ax.add_patch(r_point)

        plt.axis('scaled')
        plt.axis('off')
        plt.xlim([self._grid_size[0]//2,-self._grid_size[0]//2])
        plt.ylim([self._grid_size[1]//2,-self._grid_size[1]//2])

        fig.canvas.draw()

        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close()

        self._data = data
   
        return data
    
    def display_img(self, duration:float = 0.5) -> None:

        plt.imshow(self._data, interpolation='nearest')
        plt.axis('off')
        plt.show(block=False)
        plt.pause(duration)
        plt.close()
    

    def save_img(self, path:str ='./pict.png') -> None:
        
        im = Image.fromarray(self._data)
        im.save(path)
        

    

