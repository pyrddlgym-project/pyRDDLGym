from pyRDDLGym.Visualizer.StateViz import StateViz
from pyRDDLGym.Core.Compiler.RDDLModel import PlanningModel
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


class ChartVisualizer(StateViz):

    def __init__(self, model: PlanningModel,
                 steps_history=None,
                 figure_size=[10, 10],
                 dpi=100,
                 fontsize=10,
                 markersize=4,
                 boolcol=['red', 'green'],
                 loccol='black',
                 scale_full_history: bool = True,
                 ranges=None) -> None:
        self._model = model
        self._figure_size = figure_size
        self._dpi = dpi
        self._fontsize = fontsize
        self._markersize = markersize
        self._boolcol = boolcol
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
                ','.join, 
                self._model.variations(self._model.param_types[state])
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
        self._fig.canvas.draw()
        
        # prepare artists
        self._lines = {}
        for (y, (state, values)) in enumerate(self._state_hist.items()):
            values = values[::-1, :]
            self._lines[state] = []
            
            # title, labels and cursor line
            self._ax[y].xaxis.label.set_fontsize(self._fontsize)
            self._ax[y].yaxis.label.set_fontsize(self._fontsize)
            if y == len(self._state_hist) - 1:
                self._ax[y].set_xlabel('decision epoch')
            self._ax[y].set_ylabel(state, rotation=15)
            
            # plot boolean fluent
            if self._model.variable_ranges[state] == 'bool':
                line = self._ax[y].pcolormesh(values,
                                              edgecolors=self._loccol,
                                              linewidth=0.5,
                                              cmap=matplotlib.colors.ListedColormap(self._boolcol),
                                              vmin=0, 
                                              vmax=1)
                self._lines[state].append(line)
                
                # set the legend colors
                patches = [
                    matplotlib.patches.Patch(color=self._boolcol[0], label='false'),
                    matplotlib.patches.Patch(color=self._boolcol[1], label='true')
                ]
                self._ax[y].legend(handles=patches, loc='upper right')
                
                # set the state variables to the y-axis labels
                labels = self._labels[state]
                self._ax[y].yaxis.set_ticks([0, len(labels)])
                self._ax[y].yaxis.set(ticks=np.arange(0.5, len(labels), 1),
                                      ticklabels=labels)
                self._ax[y].set_yticklabels(labels,
                                            fontdict={"fontsize": self._fontsize},
                                            rotation=30)
            
            # plot real fluent
            elif self._model.variable_ranges[state] == 'real':
                for (i, var) in enumerate(self._labels[state]):
                    line = self._ax[y].plot(values[i, :],
                                            'o-',
                                            markersize=self._markersize,
                                            label=var)[0]
                    self._lines[state].append(line)
                self._ax[y].set_xlim([0, values[i, :].size])
                self._ax[y].legend(loc='upper right')
            
            # plot int and object fluent
            else:
                for (i, var) in enumerate(self._labels[state]):
                    line = self._ax[y].plot(np.arange(values[i, :].size),
                                            values[i, :],
                                            linestyle=':',
                                            marker='o',
                                            markersize=self._markersize,
                                            markerfacecolor='none',
                                            label=var)[0]
                    self._lines[state].append(line)
                self._ax[y].set_xlim([0, values[i, :].size])
                if var != '':
                    self._ax[y].legend(loc='upper right')
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
                values[:, :-1] = values[:, 1:]
        states = {name: np.full(shape=shape, fill_value=np.nan)
                  for (name, shape) in self._state_shapes.items()}
        for (name, value) in state.items():
            var, objects = self._model.parse(name)
            states[var][self._model.indices(objects)] = value
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

            # plot boolean fluent
            if self._model.variable_ranges[state] == 'bool':
                self._fig.canvas.restore_region(self._backgrounds[y])
                self._lines[state][0].set_array(values)
                self._ax[y].draw_artist(self._lines[state][0])

            # plot real fluent
            elif self._model.variable_ranges[state] == 'real':
                self._fig.canvas.restore_region(self._backgrounds[y])
                for (i, var) in enumerate(self._labels[state]):
                    self._lines[state][i].set_ydata(values[i, :])
                    self._ax[y].draw_artist(self._lines[state][i])
                self._ax[y].set_ylim([vmin, vmax])
                
            # plot int and object fluent
            else:
                
                # offset each curve by a small amount in x so all can be seen
                num_ser = len(self._labels[state])
                if num_ser > 1:
                    offsets = np.linspace(-0.05, 0.05, num=num_ser, endpoint=True)
                else:
                    offsets = np.zeros((num_ser,))
                
                self._fig.canvas.restore_region(self._backgrounds[y])
                for (i, var) in enumerate(self._labels[state]):
                    self._lines[state][i].set_xdata(np.arange(values[i, :].size) + offsets[i])
                    self._lines[state][i].set_ydata(values[i, :])
                    self._ax[y].draw_artist(self._lines[state][i])
                self._ax[y].set_ylim([vmin, vmax])
            self._fig.canvas.blit(self._ax[y].bbox)

        self._step = self._step + 1

        img = self.convert2img(self._fig, self._ax)

        return img
