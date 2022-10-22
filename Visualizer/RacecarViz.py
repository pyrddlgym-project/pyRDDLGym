import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os

import numpy as np
from PIL import Image
import scipy

import Visualizer
from Grounder.RDDLModel import RDDLModel
from Visualizer.StateViz import StateViz
import math


class RacecarVisualizer(StateViz):

    def __init__(self, model: RDDLModel, figure_size=(4, 4)) -> None:
        self._model = model
        self._figure_size = figure_size
        
        self._nonfluents = model.nonfluents
        
        this_dir, _ = os.path.split(__file__)
        path = os.path.join(this_dir, 'assets', 'racecar.png')

        # draw boundaries
        obj = model.objects['b']
        X1, X2, Y1, Y2 = [], [], [], []
        for o in obj:
             X1.append(self._nonfluents['X1_' + o])
             X2.append(self._nonfluents['X2_' + o])
             Y1.append(self._nonfluents['Y1_' + o])
             Y2.append(self._nonfluents['Y2_' + o])
        
        self.fig = plt.figure(figsize=self._figure_size)

        self.ax = plt.gca()
        for pt in zip(X1, Y1, X2, Y2):
            self.ax.plot(*pt)
            
        goal = plt.Circle((self._nonfluents['GX'], self._nonfluents['GY']),
                          self._nonfluents['RADIUS'],
                          color='g')
        self.ax.add_patch(goal)
        
        self.car = OffsetImage(plt.imread(path, format='png'), zoom=0.1)

    def convert2img(self, canvas):
        data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(canvas.get_width_height()[::-1] + (3,))
        return Image.fromarray(data)

    def render(self, state):
        vx = state['vx']
        vy = state['vy']
        angle = math.degrees(math.atan2(vy, vx))
        car = scipy.ndimage.rotate(self.car.get_data(), angle)
        car = AnnotationBbox(OffsetImage(car, zoom=0.1), (state['x'], state['y']), frameon=False)
        self.ax.add_artist(car)
        self.fig.canvas.draw()
        img = self.convert2img(self.fig.canvas)
        car.remove()
        return img
    
