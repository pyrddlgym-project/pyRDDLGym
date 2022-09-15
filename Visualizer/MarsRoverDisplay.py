import matplotlib.pyplot as plt
import numpy as np

from Visualizer.StateViz import StateViz
from Grounder.RDDLModel import RDDLModel

class MarsRoverDisplay(StateViz):
    def __init__(self, model: RDDLModel, resolution: [int,int]) -> None:
        self._model= model
        self._states = model.states
        self._nonfluents = model.nonfluents


        # Temp varaible until grounder is completated
        self._objects = {'rover':['r1'],
                         'picture-point':['p1','p2','p3']}
        

        print(self._states)
        print(self._nonfluents)

        self.build_layout()
    
    def build_layout(self):
        picture_point_locaiton = {o:[None,None,None] for o in self._objects['picture-point']}
        rover_location = {o:[None,None] for o in self._objects['rover']}
        
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


        print(rover_location)
        print(picture_point_locaiton)
        

    def render():
        pass
    

