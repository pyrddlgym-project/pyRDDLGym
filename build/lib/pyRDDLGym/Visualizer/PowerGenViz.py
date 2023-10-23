import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from pyRDDLGym.Core.Compiler.RDDLModel import PlanningModel
from pyRDDLGym import Visualizer
from pyRDDLGym.Visualizer.StateViz import StateViz


class PowerGenVisualizer(StateViz):

    def __init__(self, model: PlanningModel, dpi=20, fontsize=8, display=False) -> None:

        self._model = model
        self._states = model.states
        self._nonfluents = model.groundnonfluents()
        self._objects = model.objects
        self._figure_size = None
        self._dpi = dpi
        self._fontsize = fontsize
        self._interval = 15

        self._asset_path = "/".join(Visualizer.__file__.split("/")[:-1])

        self._nonfluent_layout = None
        self._state_layout = None
        self._fig, self._ax = None, None
        self._data = None
        self._img = None
    
    def build_nonfluents_layout(self): 

        prod_units_min = {o: None for o in self._objects['plant']}
        prod_units_max = {o: None for o in self._objects['plant']}
        prod_change_penalty = {o: None for o in self._objects['plant']}
        cost_per_unit = {o: None for o in self._objects['plant']}
        
        # add none-fluents
        for k, v in self._nonfluents.items():
            var, objects = self._model.parse(k)
            if var == 'PROD-UNITS-MIN':
                prod_units_min[objects[0]] = v
            elif var == 'PROD-UNITS-MAX':
                prod_units_max[objects[0]] = v
            elif var == 'PROD-CHANGE-PENALTY':
                prod_change_penalty[objects[0]] = v
            elif var == 'COST-PER-UNIT':
                cost_per_unit[objects[0]] = v

        return {'prod_units_min': prod_units_min,
                'prod_units_max': prod_units_max,
                'prod_change_penalty': prod_change_penalty,
                'cost_per_unit': cost_per_unit}

    def build_states_layout(self, state):
        prevProd = {o: None for o in self._objects['plant']}
        temperature = None

        for k, v in state.items():
            var, objects = self._model.parse(k)
            if var == 'prevProd':
                prevProd[objects[0]] = v
            elif 'temperature' in k:
                temperature = v
        
        return {'prevProd': prevProd, 'temperature': temperature}

    def init_canvas_info(self):
        interval = self._interval
        plant_list = self._objects['plant']
        room_grid_size = (int((len(plant_list)) ** 0.5) + 1,
                          int((len(plant_list)) ** 0.5) + 1)
        room_center = (room_grid_size[0] * self._interval / 2,
                       room_grid_size[1] * self._interval / 2)
        start_points_zone = []
        
        for i in range(room_grid_size[0]):
            for j in range(room_grid_size[1]):
                start_points_zone.append((i * 15, j * 15))

        start_points_zone.sort(key=lambda x: x[1])
        start_points_zone.sort(key=lambda x: x[0], reverse=True)

        init_points = {}
        for i in range(len(plant_list)):
            init_points[plant_list[i]] = (
                start_points_zone[i][0], start_points_zone[i][1])

        canvas_info = {'canvas_size': (room_grid_size[0] * interval,
                                       room_grid_size[1] * interval),
                       'plant_init_points': init_points}

        return canvas_info
        
    def init_canvas(self, figure_size, dpi):
        fig = plt.figure(figsize=figure_size, dpi=dpi)
        ax = plt.gca()
        plt.xlim([-2, figure_size[0]])
        plt.ylim([-2, figure_size[1]])
        plt.axis('scaled')
        plt.axis('off')
        return fig, ax

    def render_plant(self, plant, nonfluent_layout, state_layout):

        fig = self._fig
        ax = self._ax
        interval = self._interval * 2 / 3
        init_x, init_y = self._canvas_info['plant_init_points'][plant]

        prod_min = nonfluent_layout['prod_units_min'][plant]
        prod_max = nonfluent_layout['prod_units_max'][plant]

        change_penalty = nonfluent_layout['prod_change_penalty'][plant]
        unit_cost = nonfluent_layout['cost_per_unit'][plant]

        prev_prod = state_layout['prevProd'][plant]
        temp = state_layout['temperature']

        upL = [init_x, init_y + interval * 1.2]
        upR = [init_x + interval, init_y + interval * 1.2]
        maxL = [init_x, interval + init_y]
        maxR = [init_x + interval, interval + init_y]
        lowL = [init_x, prod_min / prod_max * interval + init_y]
        lowR = [init_x + interval, prod_min / prod_max * interval + init_y]

        set_lw = 5

        line_max = plt.Line2D((maxL[0], maxR[0]),
                              (maxL[1], maxR[1]),
                              ls='--', color='orange', lw=set_lw, zorder=3)
        line_up = plt.Line2D((upL[0], upR[0]),
                             (upL[1], upR[1]),
                             ls='--', color='black', lw=set_lw, zorder=3)
        line_low = plt.Line2D((lowL[0], lowR[0]),
                              (lowL[1], lowR[1]),
                              ls='--', color='orange', lw=set_lw, zorder=3)
        lineL = plt.Line2D((init_x, upL[0]),
                           (init_y, upL[1]),
                           color='black', lw=set_lw, zorder=3)
        lineR = plt.Line2D((init_x + interval, upR[0]),
                           (init_y, upR[1]),
                           color='black', lw=set_lw, zorder=3)
        lineB = plt.Line2D((init_x, init_x + interval),
                           (init_y, init_y),
                           color='black', lw=set_lw, zorder=3)

        ax.add_line(line_max)
        ax.add_line(line_up)
        ax.add_line(line_low)
        ax.add_line(lineL)
        ax.add_line(lineR)
        ax.add_line(lineB)
        
        prod_rect = plt.Rectangle((init_x, init_y),
                                  interval,
                                  prev_prod / prod_max * interval,
                                  fc='limegreen',
                                  zorder=2)
        temp_rect = plt.Rectangle((init_x, init_y),
                                  interval,
                                  interval * 1.2,
                                  fc='firebrick',
                                  alpha=max(min(temp, 100), 0) / 100,
                                  zorder=2)
        cost_rect = plt.Rectangle((init_x, upL[1]),
                                  interval / 2,
                                  interval / 5,
                                  fc='deepskyblue',
                                  alpha=max(min(unit_cost, 10), 0) / 10,
                                  zorder=2)
        pen_rect = plt.Rectangle((init_x + interval / 2, upL[1]),
                                 interval / 2,
                                 interval / 5,
                                 fc='darkcyan',
                                 alpha=max(min(change_penalty, 10), 0) / 10,
                                 zorder=2)

        ax.add_patch(prod_rect)
        ax.add_patch(temp_rect)
        ax.add_patch(cost_rect)
        ax.add_patch(pen_rect)

        plt.text(init_x + interval * 1.1,
                 init_y + interval * 1.3,
                 "%s" % plant,
                 color='black', fontsize=50)

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

        for plant in self._objects['plant']: 
            self.render_plant(plant, self._nonfluent_layout, self._state_layout)

        img = self.convert2img(self._fig, self._ax)
            
        self._ax.cla()
        plt.cla()
        plt.close()

        return img

