import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import numpy as np

from PIL import Image

from pyRDDLGym.Core.Compiler.RDDLModel import PlanningModel
from pyRDDLGym.Visualizer.StateViz import StateViz


class PongVisualizer(StateViz):

    def __init__(self, model: PlanningModel,
                 figure_size=(4, 4),
                 ball_radius=0.02,
                 wait_time=100) -> None:
        self._model = model
        self._figure_size = figure_size
        self._ball_radius = ball_radius
        self._wait_time = wait_time
        
        self._nonfluents = model.groundnonfluents()
        self._balls = model.objects['ball']
        
        self.fig = plt.figure(figsize=self._figure_size)
        self.ax = plt.gca()
        
        self.ax.get_xaxis().set_ticks([])
        self.ax.get_yaxis().set_ticks([])
        
    def convert2img(self, canvas):
        data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(canvas.get_width_height()[::-1] + (3,))
        return Image.fromarray(data)

    def render(self, state):
        balls = []
        for b in self._balls:
            ball = plt.Circle((state[f'ball-x___{b}'], 
                               state[f'ball-y___{b}']), self._ball_radius, 
                              color='red')
            balls.append(ball)
            self.ax.add_patch(ball)
        
        path = Path([(0.99, state['paddle-y']),
                     (0.99, state['paddle-y'] + self._nonfluents['PADDLE-H'])], 
                    [Path.MOVETO, Path.LINETO])
        paddle = patches.PathPatch(path, color='black', lw=3.0)
        self.ax.add_patch(paddle)
        
        self.fig.canvas.draw()
        
        img = self.convert2img(self.fig.canvas)
        
        for ball in balls:
            ball.remove()
        del balls
        
        paddle.remove()
        del paddle
        
        return img

