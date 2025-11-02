from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz


class ChartVisualizer(BaseViz):

    def __init__(self, model: RDDLPlanningModel,
                 steps_history=None,
                 figure_size=(10, 10),
                 dpi=100,
                 fontsize=10, minfontsize=4,
                 markersize=3, linewidth=1.,
                 boolcol=['red', 'green'],
                 scale_full_history: bool=True,
                 ranges=None) -> None:
        self._model = model
        self._figure_size = figure_size
        self._dpi = dpi
        self._fontsize = fontsize
        self._minfontsize = minfontsize
        self._markersize = markersize
        self._linewidth = linewidth
        self._boolcol = boolcol
        self._scale_full_history = scale_full_history
        self._ranges = ranges

        self._fig, self._ax = None, None
        self._data = None
        self._img = None

        if steps_history is None:
            steps_history = self._model.horizon
        self._steps = steps_history
        self._state_hist = {}
        self._state_shapes = {}
        self._labels = {}
        self._historic_min_max = {}
        for (state, values) in self._model.state_fluents.items():
            values = np.atleast_1d(values)
            self._state_hist[state] = np.full(
                shape=(len(values), self._steps), fill_value=np.nan)
            self._state_shapes[state] = self._model.object_counts(
                self._model.variable_params[state])
            self._labels[state] = list(map(
                ','.join, self._model.ground_types(self._model.variable_params[state])))
            self._historic_min_max[state] = (np.inf, -np.inf)
        self._step = 0
        self._drawn_artists = False

    def adjust_tick_label_size(self, ax, labels):
        fig = ax.figure
        dpi = fig.dpi
        _, figure_height = fig.get_size_inches()
        scaling_factor = (figure_height * dpi) / len(labels)
        valid_font = scaling_factor / 5 >= self._minfontsize
        if valid_font:
            font_size = max(self._minfontsize, min(self._fontsize, scaling_factor / 5))
            ax.tick_params(axis='y', labelsize=font_size)
            ax.yaxis.set_ticks([0, len(labels)])
            ax.yaxis.set(ticks=np.arange(0.5, len(labels), 1), ticklabels=labels)
            ax.set_yticklabels(labels)

    def adjust_legend_font_size(self, ax):
        legend = ax.get_legend()
        if legend:
            fig = ax.figure
            dpi = fig.dpi
            _, figure_height = fig.get_size_inches()
            num_labels = len(legend.get_texts())
            if num_labels:
                available_height_pixels = figure_height * dpi
                font_size = max(self._minfontsize, min(
                    self._fontsize, available_height_pixels / (num_labels * 10)))
                for text in legend.get_texts():
                    text.set_fontsize(font_size)
                legend.handletextpad = 0
                legend.borderaxespad = 0
                legend.borderpad = 0
                for handle in legend.legend_handles:
                    handle.set_markersize(font_size / 4)

    def _setup(self):
        
        # prepare plot
        self._fig, self._ax = plt.subplots(
            len(self._state_hist), 1,
            squeeze=True, figsize=self._figure_size, sharex=True)
        if len(self._state_hist) == 1:
            self._ax = (self._ax,)
        self._fig.canvas.draw()
        
        # prepare artists
        self._lines = {}
        for (y, (state, values)) in enumerate(self._state_hist.items()):
            self._lines[state] = []
            
            # title, labels and cursor line
            if y == len(self._state_hist) - 1:
                self._ax[y].set_xlabel('decision epoch')
            self._ax[y].set_ylabel(state, rotation=90)
            self._ax[y].xaxis.label.set_fontsize(self._fontsize)
            self._ax[y].yaxis.label.set_fontsize(self._fontsize)
            self._ax[y].tick_params(axis='x', labelsize=self._fontsize)            
            
            # plot boolean fluent
            if self._model.variable_ranges[state] == 'bool':
                line = self._ax[y].pcolormesh(values,
                                              cmap=matplotlib.colors.ListedColormap(self._boolcol),
                                              vmin=0, vmax=1)
                self._lines[state].append(line)
                
                # set the legend colors
                patches = [
                    matplotlib.patches.Patch(color=self._boolcol[0], label='false'),
                    matplotlib.patches.Patch(color=self._boolcol[1], label='true')
                ]
                self._ax[y].legend(handles=patches, loc='upper right')
                self.adjust_tick_label_size(self._ax[y], self._labels[state])

            # plot real fluent
            elif self._model.variable_ranges[state] == 'real':
                for (i, var) in enumerate(self._labels[state]):
                    line = self._ax[y].plot(values[i,:],
                                            'o-', markersize=self._markersize,
                                            linewidth=self._linewidth,
                                            label=var)[0]
                    self._lines[state].append(line)
                self._ax[y].set_xlim([0, values[i,:].size])
            
            # plot int and object fluent
            else:
                for (i, var) in enumerate(self._labels[state]):
                    line = self._ax[y].plot(np.arange(values[i,:].size), values[i,:],
                                            linestyle=':', linewidth=self._linewidth,
                                            marker='o', markersize=self._markersize, 
                                            markerfacecolor='none',
                                            label=var)[0]
                    self._lines[state].append(line)
                self._ax[y].set_xlim([0, values[i,:].size])
        plt.tight_layout()

        # prepare legends
        for (y, (state, values)) in enumerate(self._state_hist.items()):
            if self._model.variable_ranges[state] != 'bool' and len(self._labels[state]) > 1:
                self._ax[y].legend(loc='upper right', labelspacing=0)
                self.adjust_legend_font_size(self._ax[y])

        self._backgrounds = [self._fig.canvas.copy_from_bbox(ax.bbox) for ax in self._ax]
        self._drawn_artists = True
        
    def convert2img(self, fig, ax):
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        img = Image.fromarray(data)
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
            value = self._model.object_to_index.get(value, value)
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
                    self._lines[state][i].set_ydata(values[i,:])
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
                    self._lines[state][i].set_xdata(np.arange(values[i,:].size) + offsets[i])
                    self._lines[state][i].set_ydata(values[i,:])
                    self._ax[y].draw_artist(self._lines[state][i])
                self._ax[y].set_ylim([vmin, vmax])
            self._fig.canvas.blit(self._ax[y].bbox)

        self._step = self._step + 1

        img = self.convert2img(self._fig, self._ax)

        return img