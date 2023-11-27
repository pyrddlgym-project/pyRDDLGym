import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pprint

from pyRDDLGym.Core.Compiler.RDDLModel import PlanningModel
from pyRDDLGym import Visualizer
from pyRDDLGym.Visualizer.StateViz import StateViz


class SumOfHalfSpacesVisualizer(StateViz):

    def __init__(self, model: PlanningModel,
                 figure_size=[5, 10],
                 dpi=100,
                 fontsize=10,
                 display=False) -> None:
        self._model = model
        self._states = model.groundstates()
        self._nonfluents = model.groundnonfluents()
        self._objects = model.objects
        self._figure_size = figure_size
        self._display = display
        self._dpi = dpi
        self._fontsize = fontsize
        self._interval = 10
        self._asset_path = "/".join(Visualizer.__file__.split("/")[:-1])
        self._nonfluents_layout = None
        self._states_layout = None
        self._fig, self._ax = None, None
        self._data = None
        self._img = None

        self._ndim = len(self._objects['dimension'])
        if self._ndim > 2:
            raise RuntimeError('[SumOfHalfSpacesVisualizer] Can only visualize in 1 or 2 dimensions')
        self._nsummands = len(self._objects['summand'])

        # separators (by default they are '___' and '__', respectively)
        fluent_sep = self._model.FLUENT_SEP
        object_sep = self._model.OBJECT_SEP

        # construct the graph of the objective function
        self._Ws = np.empty(shape=(self._nsummands, self._ndim))
        self._Bs = np.empty(shape=(self._nsummands, self._ndim))

        for i, summand_symbol in enumerate(self._objects['summand']):
            for j, dim_symbol in enumerate(self._objects['dimension']):
                suffix = fluent_sep + summand_symbol + object_sep + dim_symbol
                self._Ws[i,j] = self._nonfluents['W' + suffix]
                self._Bs[i,j] = self._nonfluents['B' + suffix]

        self._xs = np.arange(-10., 10., step=0.1)
        if self._ndim == 2:
            self._xs = np.meshgrid(self._xs, self._xs)

        def eval_objective_fn(x):
            return np.sum( np.sign(self._Ws * (x - self._Bs)), axis=0 )
        self._ys = np.apply_along_axis(eval_objective_fn, axis=0, arr=self._xs)

    def build_nonfluents_layout(self):
        return self._nonfluents

    def init_canvas(self, figure_size, dpi):
        fig = plt.figure(figsize=figure_size, dpi=dpi)
        ax = plt.gca()
        plt.xlim(-10., 10.)
        plt.ylim((min(self._ys), max(self._ys)))
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

        self._ax.plot(self._xs, self._ys)

        img = self.convert2img(self._fig, self._ax)
        self._ax.cla()
        plt.close()
        # plt.show()

        return img

