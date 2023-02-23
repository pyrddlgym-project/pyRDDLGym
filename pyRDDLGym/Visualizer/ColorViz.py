import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image

from pyRDDLGym.Core.Compiler.RDDLModel import PlanningModel
from pyRDDLGym.Visualizer.StateViz import StateViz


class ColorVisualizer(StateViz):

    def __init__(self, model: PlanningModel,
                 steps_history=None,
                 figure_size=[10, 10],
                 dpi=100,
                 fontsize=10,
                 cmap='seismic',
                 loccol='black',
                 scale_full_history: bool=True,
                 ranges=None) -> None:
        self._model = model
        self._figure_size = figure_size
        self._dpi = dpi
        self._fontsize = fontsize
        self._cmap = cmap
        self._loccol = loccol
        self._scale_full_history = scale_full_history
        self._ranges = ranges
        
        self._fig, self._ax = None, None
        self._data = None
        self._img = None
        
        if steps_history is None:
            steps_history = model.horizon
        self._steps = steps_history
        self._state_hist = {}
        self._state_shapes = {}
        self._labels = {}
        self._historic_min_max = {}
        for (state, values) in self._model.states.items():
            values = np.atleast_1d(values)
            self._state_hist[state] = np.full(shape=(len(values), self._steps),
                                              fill_value=np.nan)
            self._state_shapes[state] = self._model.object_counts(
                self._model.param_types[state])
            self._labels[state] = list(map(
                ','.join, self._model.variations(self._model.param_types[state])
            ))
            self._historic_min_max[state] = (np.inf, -np.inf)
        self._step = 0
        
    def convert2img(self, fig, ax): 
        # ax.set_position((0, 0, 1, 1))
        fig.canvas.draw()

        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = Image.fromarray(data)

        self._data = data
        self._img = img
        return img

    def render(self, state):
        
        # update the state info
        if self._step >= self._steps:
            for (_, values) in self._state_hist.items():
                values[:,:-1] = values[:, 1:]
        states = {name: np.full(shape=shape, fill_value=np.nan)
                  for (name, shape) in self._state_shapes.items()}
        for (name, value) in state.items():
            var, objects = self._model.parse(name)
            states[var][self._model.indices(objects)] = value
        index = min(self._step, self._steps - 1)
        for (name, values) in states.items():
            self._state_hist[name][:, index] = np.ravel(values, order='C')
            
        # draw color plots
        self._fig, self._ax = plt.subplots(len(self._state_hist), 1,
                                           squeeze=True,
                                           figsize=self._figure_size,
                                           sharex=True) 
        if len(self._state_hist) == 1:
            self._ax = (self._ax,)
            
        for (y, (state, values)) in enumerate(self._state_hist.items()):
            values = values[::-1, :]
            
            # title, labels and cursor line
            self._ax[y].xaxis.label.set_fontsize(self._fontsize)
            self._ax[y].yaxis.label.set_fontsize(self._fontsize)
            self._ax[y].title.set_text(state)
            self._ax[y].set_xlabel('decision epoch')
            self._ax[y].set_ylabel(state)
            self._ax[y].axvline(x=index + 0.5, 
                                ymin=0.0, 
                                ymax=1.0,
                                color=self._loccol, 
                                linestyle='--', 
                                linewidth=2, 
                                alpha=0.9)
            
            # scaling of y axis
            vmin, vmax = None, None
            if self._scale_full_history:
                hmin, hmax = self._historic_min_max[state]
                valid_values = values[np.isfinite(values)]
                vmin = min(hmin, np.min(valid_values))
                vmax = max(hmax, np.max(valid_values))
                self._historic_min_max[state] = (vmin, vmax)            
            if self._ranges is not None and state in self._ranges:
                vmin, vmax = self._ranges[state]    
            
            # plot fluent as color mesh
            im = self._ax[y].pcolormesh(values, 
                                        edgecolors=self._loccol, 
                                        linewidth=0.5, 
                                        cmap=self._cmap, 
                                        vmin=vmin, 
                                        vmax=vmax)         
            plt.colorbar(im, ax=self._ax[y])
            labels = self._labels[state]
            self._ax[y].yaxis.set_ticks([0, len(labels)])
            self._ax[y].yaxis.set(ticks=np.arange(0.5, len(labels), 1), ticklabels=labels) 
            self._ax[y].set_yticklabels(labels,
                                        fontdict={"fontsize": self._fontsize},
                                        rotation=30)
            
        self._step = self._step + 1
        plt.tight_layout()
        
        img = self.convert2img(self._fig, self._ax)
        
        plt.clf()
        plt.close()

        return img
    
