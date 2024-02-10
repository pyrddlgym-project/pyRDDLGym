import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pprint

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz


class TextVisualizer(BaseViz):

    def __init__(self, model: RDDLPlanningModel,
                 figure_size=(5, 10),
                 dpi=100,
                 fontsize=10,
                 display=False) -> None:
        self._model = model
        self._figure_size = figure_size
        self._display = display
        self._dpi = dpi
        self._fontsize = fontsize
        self._interval = 10
        self._fig, self._ax = None, None
        self._data = None
        self._img = None
    
    def init_canvas(self, figure_size, dpi):
        fig = plt.figure(figsize=figure_size, dpi=dpi)
        ax = plt.gca()
        plt.xlim(0, figure_size[0] * self._interval)
        plt.ylim(0, figure_size[1] * self._interval)
        plt.axis('scaled')
        plt.axis('off')
        return fig, ax
        
    def convert2img(self, fig, ax):        
        ax.set_position((0, 0, 1, 1))
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = Image.fromarray(data)
        self._data = data
        self._img = img
        return img

    def render(self, state):
        self.states = state

        self._fig, self._ax = self.init_canvas(self._figure_size, self._dpi)
        
        state_layout = state
        text_layout = {'state': state_layout}

        text_str = pprint.pformat(text_layout)[1:-1]
        self._ax.text(self._interval * 0.5,
                      self._figure_size[1] * self._interval * 0.95,
                      text_str,
                      horizontalalignment='left',
                      verticalalignment='top',
                      wrap=True, fontsize=self._fontsize)
        
        img = self.convert2img(self._fig, self._ax)
        self._ax.cla()
        plt.close()

        return img
    
