import matplotlib.pyplot as plt
import numpy as np

from Visualizer.StateViz import StateViz
from Grounder.RDDLModel import RDDLModel

class MarsRoverDisplay(StateViz):
    def __init__(self, model: RDDLModel, resolution: [int,int]) -> None:
        self._model= model
        self._states = model.states
        self._nonfluents = model.nonfluents
        

        print(self._states)
        print(self._nonfluents)
        print(self._model._AST)
    
    def build_layout():
        rover_location = 1

    def render():
        pass
    

