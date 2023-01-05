import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import sys
from typing import Optional

from pyRDDLGym.Core.Compiler.RDDLModel import PlanningModel
from pyRDDLGym import Visualizer
from pyRDDLGym.Visualizer.StateViz import StateViz


class HVACDisplay(StateViz):

    def __init__(self, model: PlanningModel,
                 grid_size: Optional[int]=[50, 50],
                 resolution: Optional[int]=[500, 500]) -> None:
        self._model = model
        self._states = model.states
        self._nonfluents = model.nonfluents
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

        # self.render()
        self.render()
    
    def build_object_layout(self) -> dict: 
        max_res_cap = {o: None for o in self._objects['res']}
        upper_bound = {o: None for o in self._objects['res']}
        lower_bound = {o: None for o in self._objects['res']}
        rain_shape = {o: None for o in self._objects['res']}
        rain_scale = {o: None for o in self._objects['res']}
        downstream = {o: [] for o in self._objects['res']}
        sink_res = {o: None for o in self._objects['res']}

        rlevel = {o: None for o in self._objects['res']}

        # add none-fluents
        for k, v in self._nonfluents.items():
            if 'RAIN_SHAPE_' in k:
                point = k.split('_')[2]
                rain_shape[point] = v
            elif 'MAX_RES_CAP_' in k:
                point = k.split('_')[3]
                max_res_cap[point] = v
            elif 'UPPER_BOUND_' in k:
                point = k.split('_')[2]
                upper_bound[point] = v
            elif 'LOWER_BOUND_' in k:
                point = k.split('_')[2]
                lower_bound[point] = v
            elif 'RAIN_SHAPE_' in k:
                point = k.split('_')[2]
                rain_shape[point] = v
            elif 'RAIN_SCALE_' in k:
                point = k.split('_')[2]
                rain_scale[point] = v
            elif 'DOWNSTREAM_' in k:
                point = k.split('_')[1]
                v = k.split('_')[2]
                downstream[point].append(v)
            elif 'SINK_RES_' in k:
                point = k.split('_')[2]
                sink_res[point] = v

        # add states
        for k, v in self._states.items():
            if 'rlevel' in k:
                rlevel['t1'] = 75

        # adding defaults
        for o in self._objects['res']:
            if rain_shape[o] == None:
                rain_shape[o] = self._nonfluents['RAIN_SHAPE']
            if max_res_cap[o] == None:
                max_res_cap[o] = self._nonfluents['MAX_RES_CAP']
            if upper_bound[o] == None:
                upper_bound[o] = self._nonfluents['UPPER_BOUND']
            if lower_bound[o] == None:
                lower_bound[o] = self._nonfluents['LOWER_BOUND']
            if rain_shape[o] == None:
                rain_shape[o] = self._nonfluents['RAIN_SHAPE']
            if rain_scale[o] == None:
                rain_scale[o] = self._nonfluents['RAIN_SCALE']
            if rlevel[o] == None:
                rlevel[o] = self._states['rlevel']
            if sink_res[o] == None:
                sink_res[o] = False

        object_layout = {'rain_shape': rain_shape, 'rain_scale': rain_scale,
                         'max_res_cap':max_res_cap, 'rlevel':rlevel,
                         'upper_bound':upper_bound, 'lower_bound':lower_bound,
                         'downstream':downstream, 'sink_res':sink_res}
        return object_layout

    def init_canvas_info(self):
        interval = self._interval

        # objects_set = set(self._objects['res'])
        # sink_res_set = set([k for k, v in self._object_layout['sink_res'].items() if v == True])

        zone_list = self._objects['zone']
        heater_list = self._objects['heater']

        room_grid_size = (max(len(zone_list), len(heater_list)),
                          max(len(zone_list), len(heater_list)) + 1)

        room_center = (room_grid_size[0] * self._interval / 2,
                       (room_grid_size[1] - 1) * self._interval / 2 + interval)

        start_points_zone = []

        for i in range(room_grid_size[0] - 1):
            for j in range(room_grid_size[1]):
                start_points_zone.append((i * 15, j * 15))

        start_points_heater = []
        for i in range(room_grid_size[0]):
            start_points_heater.append((i * 15, 0))

        # rank start point by distance from center
        start_points_zone.sort(
            key=lambda x: abs(x[0] + interval / 2 - room_center[0]) + \
                            abs(x[1] + interval / 2 - room_center[1]))
        start_points_heater.sort(
            key=lambda x: abs(x[0] + interval / 2 - room_center[0]) + \
                            abs(x[1] + interval / 2 - room_center[1]))
        
        # rank by number of adj rooms
        zone_list.sort(reverse=True,
                       key=lambda x: len(self._object_layout['adj_zone'][x]))
        heater_list.sort(reverse=True,
                         key=lambda x: len(self._object_layout['adj_heater'][x]))

        zone_init_points = {zone_list[i]:start_points_zone[i] 
                            for i in range(len(zone_list))}
        heater_init_points = {heater_list[i]:start_points_heater[i] 
                              for i in range(len(heater_list))}

        canvas_info = {
            'canvas_size': (room_grid_size[0] * interval,
                            room_grid_size[1] * interval),
            'zone_init_points': zone_init_points,
            'heater_init_points': heater_init_points
        }

        return canvas_info

    def render_conn(self):
        fig = self._fig
        ax = self._ax
        adj_zone = self._object_layout['adj_zone']
        adj_heater = self._object_layout['adj_heater']
        zone_init_points = self._canvas_info['zone_init_points']
        heater_init_points = self._canvas_info['heater_init_points']

        interval = self._interval * 2 / 3

        zone_line_list = []
        for k, v in adj_zone.items():
            from_point = (zone_init_points[k][0] + interval / 2,
                          zone_init_points[k][1] + interval / 2)
            for p in v:
                to_point = (zone_init_points[p][0] + interval / 2,
                            zone_init_points[p][1] + interval / 2)
                line = plt.Line2D((from_point[0], to_point[0]),
                                  (from_point[1], to_point[1]),
                                  ls='-', color='black', lw=2)
                ax.add_line(line)
        for k, v in adj_heater.items():
            from_point = (heater_init_points[k][0] + interval / 2,
                          heater_init_points[k][1] + interval / 2)
            for p in v:
                to_point = (zone_init_points[p][0] + interval / 2,
                            zone_init_points[p][1] + interval / 2)
                line = plt.Line2D((from_point[0], to_point[0]),
                                  (from_point[1], to_point[1]),
                                  ls='-', color='red', lw=2)
                ax.add_line(line)

    def render_zone(self, zone: str) -> tuple:

        fig = self._fig
        ax = self._ax
        
        curr_z = zone

        vol_ratio = self._object_layout['zone_vol'][curr_z] / 500

        print(vol_ratio)

        init_x, init_y = self._canvas_info['zone_init_points'][curr_z]
        center_x = init_x + self._interval / 2
        center_y = init_y + self._interval / 2

        interval = self._interval * 2 / 3 
        length = interval * vol_ratio
        
        print(length)
        print(init_x, init_y)
        print(center_x, center_y)

        init_x = center_x - length
        init_y = center_y - length

        print(init_x, init_y)

        background_rect = plt.Rectangle((init_x, init_y),
                                        length,
                                        length,
                                        fc='blue', alpha=1, zorder=5)
        ax.add_patch(background_rect)

        # plt.show()
        print('here')

        return

        # print(curr_t, init_x, init_y)
        # sys.exit()

        # plt.rcParams['axes.facecolor']='white'
        # fig = plt.figure(figsize = (12,12))
        # ax = plt.gca()
        # plt.text(11, 11, "%s" % curr_t, color='black', fontsize = 35)
        # plt.xlim([-0.1,12])
        # plt.ylim([-0.1,12])
        # plt.axis('scaled')
        # plt.axis('off')

        rlevel = self._object_layout['rlevel'][curr_z]
        rain_scale = self._object_layout['rain_scale'][curr_z]
        rain_shape = self._object_layout['rain_shape'][curr_z]

        max_res_cap = self._object_layout['max_res_cap'][curr_z]
        upper_bound = self._object_layout['upper_bound'][curr_z]
        lower_bound = self._object_layout['lower_bound'][curr_z]

        sink_res = self._object_layout['sink_res'][curr_z]
        # downstream = self._object_layout['downstream'][curr_t]

        maxL = [init_x, max_res_cap / 100 + init_y]
        maxR = [init_x + interval, max_res_cap / 100 + init_y]
        upL = [init_x, upper_bound / 100 + init_y]
        upR = [init_x + interval, upper_bound / 100 + init_y]
        lowL = [init_x, lower_bound / 100 + init_y]
        lowR = [init_x + interval, lower_bound / 100 + init_y]

        line_max = plt.Line2D((maxL[0], maxR[0]),
                              (maxL[1], maxR[1]),
                              ls='--', color='black', lw=1)
        line_up = plt.Line2D((upL[0], upR[0]),
                             (upL[1], upR[1]),
                             ls='--', color='orange', lw=1)
        line_low = plt.Line2D((lowL[0], lowR[0]),
                              (lowL[1], lowR[1]),
                              ls='--', color='orange', lw=1)
        lineL = plt.Line2D((init_x, init_x),
                           (init_y, init_y + interval),
                           color='black', lw=1)
        lineR = plt.Line2D((init_x + interval, init_x + interval),
                           (init_y, init_y + interval),
                           color='black', lw=1)
        lineB = plt.Line2D((init_x, init_x + interval),
                           (init_y, init_y),
                           color='black', lw=1)

        water_rect = plt.Rectangle((init_x, init_y),
                                   interval,
                                   rlevel / 100,
                                   fc='royalblue')
        res_rect = plt.Rectangle((init_x, init_y + rlevel / 100),
                                 interval,
                                 interval - rlevel / 100,
                                 fc='lightgrey', alpha=0.5)
        scale_rect = plt.Rectangle((init_x, init_y + interval),
                                   interval / 2,
                                   interval / 4,
                                   fc='deepskyblue', alpha=rain_scale / 100)
        shape_rect = plt.Rectangle((init_x + interval / 2, init_y + interval),
                                    interval / 2,
                                    interval / 4,
                                    fc='darkcyan', alpha=rain_shape / 100)

        ax.add_line(line_max)
        ax.add_line(line_up)
        ax.add_line(line_low)
        ax.add_line(lineL)
        ax.add_line(lineR)
        ax.add_line(lineB)

        ax.add_patch(water_rect)
        ax.add_patch(res_rect)
        ax.add_patch(scale_rect)
        ax.add_patch(shape_rect)

        lineU = plt.Line2D((init_x + interval * 1.25, init_x + interval * 1.25),
                            (init_y, init_y + interval),
                            color='black', lw=1)
        lineD = plt.Line2D((init_x + interval, init_x + interval * 1.25),
                           (init_y, init_y),
                           color='black', lw=1)
        # conn_shape = plt.Rectangle((init_x + interval, init_y), interval/4, interval, fc='royalblue')
        if sink_res:
            land_shape = plt.Rectangle((init_x + interval, init_y),
                                       interval / 4, interval, fc='royalblue')
        else:
            land_shape = plt.Rectangle((init_x + interval, init_y),
                                       interval / 4, interval, fc='darkgoldenrod')

        plt.text(init_x + interval * 1.1, init_y + interval * 1.1, "%s" % curr_z,
                 color='black', fontsize=5)

        ax.add_patch(water_rect)
        ax.add_patch(res_rect)
        ax.add_patch(scale_rect)
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

        # self.build_res_fill(fig, ax, self._object_layout['rlevel'][curr_t], self._object_layout['rain_scale'][curr_t], self._object_layout['rain_shape'][curr_t])
        # self.build_res_shape(fig, ax, self._object_layout['max_res_cap'][curr_t], self._object_layout['upper_bound'][curr_t], self._object_layout['lower_bound'][curr_t])
        # self.build_conn_shape(fig, ax, self._object_layout['sink_res'][curr_t], self._object_layout['downstream'][curr_t])

        # if render_text:
        #     plt.text(0.1, 11.5, "Rain Scale: %s" % round(rain_scale, 2), color='black', fontsize = 22)        
        #     plt.text(5.1, 0.1, "Water Level: %s" % round(rlevel, 2), color='black', fontsize = 22)
        #     plt.text(5.1, 11.5, "Rain Shape %s" % round(rain_shape, 2), color='black', fontsize = 22)

        #     plt.text(5.1, max_res_cap/100+0.1, "MAX RES CAP: %s" % round(max_res_cap, 2), color='black', fontsize = 22)
        #     plt.text(0.1, upper_bound/100+0.1, "Upper Bound: %s" % round(upper_bound, 2), color='black', fontsize = 22)
        #     plt.text(0.1, lower_bound/100+0.1, "Lower Bound: %s" % round(lower_bound, 2), color='black', fontsize = 22)

        # plt.text(10.1, 0.1, "Down: %s" % downstream, color='black', fontsize = 15)
        # plt.text(10.1, 2.1, "Sink: %s" % sink_res, color='black', fontsize = 15)

        # fig.canvas.draw()
        # return fig, ax
        return fig, ax

    def fig2npa(self, fig):
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data

    def render(self, display:bool=True) -> np.ndarray:

        # self._object_layout = self.build_object_layout()

        self._object_layout = {'temp_zone_max': 25, 'temp_zone_min': 15,
                               'temp_out': 30,
                               'zone_vol': {'z1': 255, 'z2': 100, 'z3': 500},
                                'sigma': {'z1': 0.1, 'z2': 0.5, 'z3': 0.7},
                                'temp_zone': {'z1': 10, 'z2': 20, 'z3': 30},
                                'heater_vol': {'h1': 25, 'h2': 30, 'h3': 40},
                                'adj_zone': {'z1': ['z2'],
                                             'z2': ['z1', 'z3'],
                                             'z3': ['z2']},
                                'adj_heater': {'h1': ['z1'], 'h2': ['z3']}}
        self._objects = {'zone': ['z1', 'z2', 'z3'], 'heater': ['h1', 'h2']}
        self._canvas_info = self.init_canvas_info()
        
        canvas_size = self._canvas_info['canvas_size']

        self._fig = plt.figure(figsize=(canvas_size[0] / 10, canvas_size[1] / 10), 
                               dpi=200)
        self._ax = plt.gca()
        
        plt.xlim([0, canvas_size[0]])
        plt.ylim([0, canvas_size[1]])
        # plt.axis('scaled')
        # plt.axis('off')

        print(self._objects)
        print(self._object_layout)
        print(self._canvas_info)
        
        self.render_conn()
        self.render_zone('z1')
        self.render_zone('z2')
        self.render_zone('z3')

        plt.show()
        sys.exit()

        # layout_data = []
        for res in self._objects['res']:
            curr_t = res
            fig, ax = self.render_res(curr_t)
            # layout_data.append(self.fig2npa(fig))
        self.render_conn()
            
        plt.show()
        sys.exit()
        
        # print(layout_data[0].shape)
        
        plt.imshow(layout_data[0])
        plt.axis('off')
        plt.show(block=False)
        plt.pause(20)
        plt.close()
 
    def display_img(self, duration:float=0.5) -> None:

        plt.imshow(self._data, interpolation='nearest')
        plt.axis('off')
        plt.show(block=False)
        plt.pause(duration)
        plt.close()

    def save_img(self, path:str='./pict.png') -> None:
        
        im = Image.fromarray(self._data)
        im.save(path)

