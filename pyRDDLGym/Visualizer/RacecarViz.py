import matplotlib.pyplot as plt
# from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Arrow
from matplotlib import collections  as mc

#import os

import numpy as np
from PIL import Image
# import scipy

from pyRDDLGym.Core.Grounder.RDDLModel import RDDLModel
from pyRDDLGym.Visualizer.StateViz import StateViz


# import math


class RacecarVisualizer(StateViz):

    def __init__(self, model: RDDLModel, figure_size=(4, 4), car_radius=0.04, vector_len=0.15, wait_time=100) -> None:
        self._model = model
        self._figure_size = figure_size
        self._car_radius = car_radius
        self._vector_len = vector_len
        self._wait_time = wait_time
        
        self._nonfluents = model.nonfluents
        
        self.fig = plt.figure(figsize=self._figure_size)
        self.ax = plt.gca()
        
        # draw boundaries
        obj = model.objects['b']
        X1, X2, Y1, Y2 = [], [], [], []
        for o in obj:
            X1.append(self._nonfluents['X1_' + o])
            X2.append(self._nonfluents['X2_' + o])
            Y1.append(self._nonfluents['Y1_' + o])
            Y2.append(self._nonfluents['Y2_' + o])
        lines = [[(x1, y1), (x2, y2)] for x1, y1, x2, y2 in zip(X1,Y1, X2, Y2)]
        lc = mc.LineCollection(lines, linewidths=2)
        self.ax.add_collection(lc)
        self.ax.autoscale()
        self.ax.margins(0.1)
        
        # draw goal        
        goal = plt.Circle((self._nonfluents['GX'], self._nonfluents['GY']),
                          self._nonfluents['RADIUS'],
                          color='g')
        self.ax.add_patch(goal)
        
        # velocity vector
        self.arrow = Arrow(0, 0, 0, 0, width=0.2 * self._vector_len, color='black')
        self.move = self.ax.add_patch(self.arrow)

        # this_dir, _ = os.path.split(__file__)
        # path = os.path.join(this_dir, 'assets', 'racecar.png')
        # self.car = OffsetImage(plt.imread(path, format='png'), zoom=0.1)
        

    def convert2img(self, canvas):
        data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(canvas.get_width_height()[::-1] + (3,))
        return Image.fromarray(data)

    def render(self, state):
        vel = np.array([state['vx'], state['vy']])
        if np.max(np.abs(vel)) > 0:
            vel /= np.linalg.norm(vel)
        vel *= self._vector_len
        dx = vel[0]
        dy = vel[1]
        self.move.remove()
        self.arrow = Arrow(state['x'], state['y'], dx, dy, width=0.2 * self._vector_len, color='black')
        self.move = self.ax.add_patch(self.arrow)

        # car = scipy.ndimage.rotate(self.car.get_data(), angle)
        # car = AnnotationBbox(OffsetImage(car, zoom=0.1), (state['x'], state['y']), frameon=False) 
        # self.ax.add_artist(car)
        car = plt.Circle((state['x'], state['y']), self._car_radius)
        self.ax.add_patch(car)
        self.fig.canvas.draw()
        
        plt.pause(self._wait_time / 1000.)
        
        img = self.convert2img(self.fig.canvas)
        
        car.remove()
        del car
        
        return img

