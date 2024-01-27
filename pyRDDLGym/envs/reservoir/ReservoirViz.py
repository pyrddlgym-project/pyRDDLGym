from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as plt_img
from PIL import Image

from pyRDDLGym.Visualizer.StateViz import StateViz
from pyRDDLGym.Core.Compiler.RDDLModel import PlanningModel
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
import matplotlib.patches as mpatches

from pyRDDLGym import Visualizer

import sys


class ReservoirVisualizer(StateViz):
    def __init__(self, model: PlanningModel, grid_size: Optional[int] = [50, 50],
                 resolution: Optional[int] = [500, 500]) -> None:

        self._model = model
        self._states = model.groundstates()
        self._nonfluents = model.groundnonfluents()
        self._objects = model.objects
        self._grid_size = grid_size
        self._resolution = resolution
        self._interval = 15

        self._asset_path = "/".join(Visualizer.__file__.split("/")[:-1])

        self._object_layout = None
        self._canvas_info = None
        self._render_frame = None
        self._data = None
        self._fig = None
        self._ax = None

    def build_object_layout(self) -> dict:

        max_res_cap = {o: None for o in self._objects['reservoir']}
        upper_bound = {o: None for o in self._objects['reservoir']}
        lower_bound = {o: None for o in self._objects['reservoir']}
        rain_shape = {o: None for o in self._objects['reservoir']}
        rain_scale = {o: None for o in self._objects['reservoir']}
        downstream = {o: [] for o in self._objects['reservoir']}
        sink_res = {o: None for o in self._objects['reservoir']}

        rlevel = {o: None for o in self._objects['reservoir']}

        # add none-fluents
        for k, v in self._nonfluents.items():
            var, objects = self._model.parse(k)
            if var == 'TOP_RES':
                max_res_cap[objects[0]] = v
            elif var == 'MAX_LEVEL':
                upper_bound[objects[0]] = v
            elif var == 'MIN_LEVEL':
                lower_bound[objects[0]] = v
            elif var == 'RAIN_VAR':
                rain_shape[objects[0]] = v
            elif var == 'RES_CONNECT':
                if v == True:
                    downstream[objects[0]].append(objects[1])
            elif var == 'CONNECTED_TO_SEA':
                sink_res[objects[0]] = v

        # # add states
        for k, v in self._states.items():
            var, objects = self._model.parse(k)
            if var == 'rlevel':
                rlevel[objects[0]] = v


        # adding defaults
        for o in self._objects['reservoir']:
            if max_res_cap[o] == None:
                max_res_cap[o] = self._nonfluents['TOP_RES']
            if upper_bound[o] == None:
                upper_bound[o] = self._nonfluents['MAX_LEVEL']
            if lower_bound[o] == None:
                lower_bound[o] = self._nonfluents['MIN_LEVEL']
            if rain_shape[o] == None:
                rain_shape[o] = self._nonfluents['RAIN_VAR']
            if rlevel[o] == None:
                rlevel[o] = self._states['rlevel']
            if sink_res[o] == None:
                sink_res[o] = False

        object_layout = {'rain_shape': rain_shape, 'max_res_cap': max_res_cap, 'rlevel': rlevel,
                         'upper_bound': upper_bound, 'lower_bound': lower_bound, 'downstream': downstream,
                         'sink_res': sink_res
                         }

        return object_layout

    def init_canvas_info(self):
        interval = self._interval
        objects_set = set(self._objects['reservoir'])
        sink_res_set = set([k for k, v in self._object_layout['sink_res'].items() if v == True])

        all_child_set = []
        for i in self._object_layout['downstream'].values():
            all_child_set += i
        all_child_set = set(all_child_set) - sink_res_set

        all_root_set = objects_set - all_child_set - sink_res_set

        level_list = [list(all_root_set)]
        visited_nodes = objects_set.copy() - all_child_set

        top_set = all_root_set
        bot_set = set()

        while len(visited_nodes) < len(objects_set):
            for i in top_set:
                for j in self._object_layout['downstream'][i]:
                    bot_set.add(j)
                    visited_nodes.add(j)
            level_list.append(list(bot_set))
            top_set = bot_set
            bot_set = set()

        level_list.append(list(sink_res_set))

        row_num = len(level_list)
        col_num = max(len(i) for i in level_list)

        canvas_size = (col_num * interval, row_num * interval)
        init_points = {}
        for i in range(len(level_list)):
            for j in range(len(level_list[i])):
                init_x = 0 + interval * j
                init_y = canvas_size[1] - interval * (i + 1)
                init_points[level_list[i][j]] = (init_x, init_y)
        conn_points = {}

        canvas_info = {'canvas_size': canvas_size, 'init_points': init_points}

        return canvas_info

    def render_conn(self):
        fig = self._fig
        ax = self._ax
        downstream = self._object_layout['downstream']
        init_points = self._canvas_info['init_points']
        interval = self._interval * 2 / 3

        for k, v in downstream.items():
            top_point = (init_points[k][0] + interval / 2, init_points[k][1])
            for p in v:
                bot_point = (init_points[p][0] + interval / 2, init_points[p][1] + interval * 1.25)
                style = mpatches.ArrowStyle('Fancy', head_length=2, head_width=2, tail_width=0.01)
                arrow = mpatches.FancyArrowPatch(top_point, bot_point, arrowstyle=style, color='k')
                ax.add_patch(arrow)

    def render_res(self, res: str) -> tuple:

        fig = self._fig
        ax = self._ax
        interval = self._interval * 2 / 3
        curr_t = res
        init_x, init_y = self._canvas_info['init_points'][curr_t]

        rlevel = self._object_layout['rlevel'][curr_t]
        # rain_scale = self._object_layout['rain_scale'][curr_t]
        rain_shape = self._object_layout['rain_shape'][curr_t]

        max_res_cap = self._object_layout['max_res_cap'][curr_t]
        upper_bound = self._object_layout['upper_bound'][curr_t]
        lower_bound = self._object_layout['lower_bound'][curr_t]

        sink_res = self._object_layout['sink_res'][curr_t]

        maxL = [init_x, max_res_cap / 150 * interval + init_y]
        maxR = [init_x + interval, max_res_cap / 150 * interval + init_y]
        upL = [init_x, upper_bound / 150 * interval + init_y]
        upR = [init_x + interval, upper_bound / 150 * interval + init_y]
        lowL = [init_x, lower_bound / 150 * interval + init_y]
        lowR = [init_x + interval, lower_bound / 150 * interval + init_y]

        line_max = plt.Line2D((maxL[0], maxR[0]), (maxL[1], maxR[1]), ls='-', color='black', lw=0.25)
        line_up = plt.Line2D((upL[0], upR[0]), (upL[1], upR[1]), ls='--', color='orange', lw=0.25)
        line_low = plt.Line2D((lowL[0], lowR[0]), (lowL[1], lowR[1]), ls='--', color='orange', lw=0.25)
        lineL = plt.Line2D((init_x, init_x), (init_y, init_y + interval), color='black', lw=0.5)
        lineR = plt.Line2D((init_x + interval, init_x + interval), (init_y, init_y + interval), color='black', lw=0.5)
        lineB = plt.Line2D((init_x, init_x + interval), (init_y, init_y), color='black', lw=0.5)

        water_rect = plt.Rectangle((init_x, init_y), interval, rlevel / 120 * interval, fc='royalblue')
        res_rect = plt.Rectangle((init_x, init_y), interval, interval, fc='lightgrey', alpha=0.5)
        # scale_rect = plt.Rectangle((init_x, init_y + interval), interval/2, interval/4, fc='deepskyblue', alpha=rain_scale/50)
        shape_rect = plt.Rectangle((init_x, init_y + interval), interval, interval / 4, fc='darkcyan',
                                   alpha=np.clip(rain_shape / 50, 0, 1))

        ax.add_line(line_max)
        ax.add_line(line_up)
        ax.add_line(line_low)
        ax.add_line(lineL)
        ax.add_line(lineR)
        ax.add_line(lineB)

        lineU = plt.Line2D((init_x + interval * 1.25, init_x + interval * 1.25), (init_y, init_y + interval),
                           color='black', lw=1)
        lineD = plt.Line2D((init_x + interval, init_x + interval * 1.25), (init_y, init_y), color='black', lw=1)

        if sink_res:
            land_shape = plt.Rectangle((init_x + interval, init_y), interval / 4, interval, fc='royalblue')
        else:
            land_shape = plt.Rectangle((init_x + interval, init_y), interval / 4, interval, fc='darkgoldenrod')

        plt.text(init_x + interval * 1.1, init_y + interval * 1.1, "%s" % curr_t, color='black', fontsize=5)

        ax.add_patch(water_rect)
        ax.add_patch(res_rect)
        # ax.add_patch(scale_rect)
        ax.add_patch(shape_rect)

        ax.add_line(line_max)
        ax.add_line(line_up)
        ax.add_line(line_low)
        ax.add_line(lineL)
        ax.add_line(lineR)
        ax.add_line(lineB)

        ax.add_line(lineU)
        ax.add_line(lineD)

        ax.add_patch(land_shape)

        return fig, ax

    def fig2npa(self, fig):
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data

    def render(self, state) -> np.ndarray:
        self._states = state
        self._object_layout = self.build_object_layout()
        self._canvas_info = self.init_canvas_info()
        canvas_size = self._canvas_info['canvas_size']
        self._fig = plt.figure(figsize=(canvas_size[0] / 10, canvas_size[1] / 10), dpi=200)
        self._ax = plt.gca()

        plt.xlim([0, canvas_size[0]])
        plt.ylim([0, canvas_size[1]])
        plt.axis('scaled')
        plt.axis('off')

        for res in self._objects['reservoir']:
            curr_t = res
            fig, ax = self.render_res(curr_t)
        self.render_conn()
        img = self.convert2img(self._fig, self._ax)

        self._ax.cla()
        plt.cla()
        plt.close()

        return img

    def convert2img(self, fig, ax):
        fig.canvas.draw()

        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        img = Image.fromarray(data)

        self._data = data
        self._img = img

        return img


