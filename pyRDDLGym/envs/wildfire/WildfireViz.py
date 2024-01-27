import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from pyRDDLGym.Core.Compiler.RDDLModel import PlanningModel
from pyRDDLGym.Visualizer.StateViz import StateViz
from pyRDDLGym import Visualizer


class WildfireVisualizer(StateViz):

    def __init__(self, model: PlanningModel,
                 dpi=50,
                 fontsize=8,
                 display=False) -> None:
        self._model = model
        self._states = model.groundstates()
        self._nonfluents = model.groundnonfluents()
        self._objects = model.objects
        self._figure_size = None
        self._dpi = dpi
        self._fontsize = fontsize
        self._interval = 5

        self._asset_path = "/".join(Visualizer.__file__.split("/")[:-1])

        self._nonfluent_layout = None
        self._state_layout = None
        self._fig, self._ax = None, None
        self._data = None
        self._img = None
    
    def build_nonfluents_layout(self): 
        targets = []
        
        for k, v in self._nonfluents.items():
            var, objects = self._model.parse(k)
            if var == 'TARGET':
                if v == True:
                    x, y = objects
                    targets.append((x, y))

        return {'targets':targets}

    def build_states_layout(self, state): 
        burning = []
        fuel = []

        for k, v in state.items():
            var, objects = self._model.parse(k)
            if var == 'burning':
                if v == True:
                    x, y = objects
                    burning.append((x, y))
            if var == 'out-of-fuel':
                if v == True:
                    x, y = objects
                    fuel.append((x, y))
        
        return {'burning':burning, 'out-of-fuel':fuel}

    def init_canvas_info(self):
        interval = self._interval
        x_list = self._objects['x-pos']
        y_list = self._objects['y-pos']
        grid_size = (int(len(x_list)), int(len(y_list)))

        grid_init_points = {}
        for x in x_list:
            for y in y_list:
                x_val = int(x[1:])
                y_val = int(y[1:])
                x_init = int((x_val - 1) * interval)
                y_init = int(grid_size[1] * interval - y_val * interval)
                grid_init_points[(x, y)] = (x_init, y_init)

        canvas_info = {'canvas_size': (grid_size[0] * interval,
                                       grid_size[1] * interval),
                       'grid_init_points': grid_init_points}

        return canvas_info
        
    def init_canvas(self, figure_size, dpi):
        fig = plt.figure(figsize=figure_size, dpi=dpi)
        ax = plt.gca()
        plt.xlim([-0.5, figure_size[0] + 0.5])
        plt.ylim([-0.5, figure_size[1] + 0.5])
        plt.axis('scaled')
        # plt.axis('off')
        return fig, ax

    def render_grid(self, pos, nonfluent_layout, state_layout):
        fig = self._fig
        ax = self._ax
        interval = self._interval
        init_x, init_y = self._canvas_info['grid_init_points'][pos]

        plt.text(init_x + 0.2,
                 init_y + 0.2,
                 str(pos),
                 color='black', fontsize=20, zorder=2)

        if pos in nonfluent_layout['targets']:
            if pos in state_layout['out-of-fuel']:
                color = "grey"
            else:
                color = "forestgreen"
            grid_rect = plt.Rectangle((init_x, init_y),
                                      interval,
                                      interval,
                                      fc=color, edgecolor='black',
                                      linewidth=2, zorder=0)
            plt.text(init_x + interval * 0.4,
                     init_y + interval * 0.4,
                     "Target",
                     color='black', fontsize=30, zorder=2)
        else:
            if pos in state_layout['out-of-fuel']:
                color = "lightgrey"
            else:
                color = "lawngreen"
            grid_rect = plt.Rectangle((init_x, init_y),
                                      interval,
                                      interval,
                                      fc=color, edgecolor='black',
                                      linewidth=2, zorder=0)
        
        ax.add_patch(grid_rect)
        
        if pos in state_layout['burning']:
            
            piv_left = (init_x + 0.1, init_y)
            piv_right = (init_x + interval - 0.1, init_y)
            piv_top = (init_x + interval / 2 + 0.1, init_y + interval / 2)
            tri_fire = plt.Polygon([piv_left, piv_right, piv_top],
                                   closed=True, fc='tomato', zorder=1)
            ax.add_patch(tri_fire)

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

    def fig2npa(self, fig):
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data

    def render(self, state):
        self.states = state

        self._nonfluent_layout = self.build_nonfluents_layout()
        self._state_layout = self.build_states_layout(state)
        self._canvas_info = self.init_canvas_info()
        self._figure_size = self._canvas_info['canvas_size']
        self._fig, self._ax = self.init_canvas(self._figure_size, self._dpi)

        for pos in self._canvas_info['grid_init_points'].keys(): 
            self.render_grid(pos, self._nonfluent_layout, self._state_layout)

        img = self.convert2img(self._fig, self._ax)
            
        self._ax.cla()
        plt.cla()
        plt.close()

        return img

