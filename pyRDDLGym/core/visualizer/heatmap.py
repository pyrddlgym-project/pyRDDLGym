from PIL import Image
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz


class HeatmapVisualizer(BaseViz):

    def __init__(self, model: RDDLPlanningModel,
                 steps_history=None,
                 figure_size=(10, 10),
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
        for (state, values) in self._model.state_fluents.items():
            values = np.atleast_1d(values)
            self._state_hist[state] = np.full(shape=(len(values), self._steps),
                                              fill_value=np.nan)
            self._state_shapes[state] = self._model.object_counts(
                self._model.variable_params[state])
            self._labels[state] = list(map(
                ','.join, 
                self._model.ground_types(self._model.variable_params[state])
            ))
            self._historic_min_max[state] = (np.inf, -np.inf)
        self._step = 0
        self._drawn_artists = False
        
    def _setup(self):
        
        # prepare plot
        self._fig, self._ax = plt.subplots(len(self._state_hist), 1,
                                           squeeze=True,
                                           figsize=self._figure_size,
                                           sharex=True)
        if len(self._state_hist) == 1:
            self._ax = (self._ax,)
            
        # prepare artists
        self._lines = {}
        for (y, (state, values)) in enumerate(self._state_hist.items()):
            values = values[::-1, :]
            
            # title, labels and cursor line
            self._ax[y].xaxis.label.set_fontsize(self._fontsize)
            self._ax[y].yaxis.label.set_fontsize(self._fontsize)
            self._ax[y].title.set_text(state)
            self._ax[y].set_xlabel('decision epoch')
            self._ax[y].set_ylabel(state)
            
            im = self._ax[y].pcolormesh(values, 
                                        edgecolors=self._loccol, 
                                        linewidth=0.5, 
                                        cmap=self._cmap, 
                                        vmin=0.0, 
                                        vmax=0.0)         
            plt.colorbar(im, ax=self._ax[y])
            self._lines[state] = im
            
            labels = self._labels[state]
            self._ax[y].yaxis.set_ticks([0, len(labels)])
            self._ax[y].yaxis.set(ticks=np.arange(0.5, len(labels), 1), 
                                  ticklabels=labels) 
            self._ax[y].set_yticklabels(labels,
                                        fontdict={"fontsize": self._fontsize},
                                        rotation=30)
        plt.tight_layout()
                    
        self._backgrounds = [self._fig.canvas.copy_from_bbox(ax.bbox) for ax in self._ax]
        self._drawn_artists = True
        
    def convert2img(self, fig, ax):
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = Image.fromarray(data)
        self._data = data
        self._img = img
        return img

    def render(self, state):
        if not self._drawn_artists:
            self._setup()
            
        # update the state info
        if self._step >= self._steps:
            for values in self._state_hist.values():
                values[:,:-1] = values[:, 1:]
        states = {name: np.full(shape=shape, fill_value=np.nan)
                  for (name, shape) in self._state_shapes.items()}
        for (name, value) in state.items():
            if name not in self._model.variable_params:  # lifted
                var, objects = RDDLPlanningModel.parse_grounded(name)
                states[var][self._model.object_indices(objects)] = value
            else:  # grounded or scalar
                states[name] = value
        index = min(self._step, self._steps - 1)
        for (name, values) in states.items():
            self._state_hist[name][:, index] = np.ravel(values, order='C')
        
        # draw color plots
        for (y, (state, values)) in enumerate(self._state_hist.items()):
            values = values[::-1, :]
           
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
            im = self._lines[state]
            self._fig.canvas.restore_region(self._backgrounds[y])
            self._lines[state].set_array(np.ravel(values))
            self._ax[y].draw_artist(self._lines[state])
            im.set_clim([vmin, vmax])
            
        self._step = self._step + 1
        
        img = self.convert2img(self._fig, self._ax)
        
        return img
    
