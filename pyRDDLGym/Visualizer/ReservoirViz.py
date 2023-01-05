import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
import sys
from typing import Optional

from pyRDDLGym.Core.Compiler.RDDLModel import PlanningModel
from pyRDDLGym import Visualizer
from pyRDDLGym.Visualizer.StateViz import StateViz


class ReservoirVisualizer(StateViz):

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
                         'max_res_cap': max_res_cap, 'rlevel': rlevel,
                         'upper_bound': upper_bound, 'lower_bound': lower_bound,
                         'downstream': downstream, 'sink_res': sink_res
                         }
        return object_layout

    def init_canvas_info(self):
        interval = self._interval
        objects_set = set(self._objects['res'])
        sink_res_set = set([k 
                            for k, v in self._object_layout['sink_res'].items() 
                            if v == True])

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
        
        canvas_info = {'canvas_size':canvas_size, 'init_points':init_points}

        return canvas_info

        # print(level_list)
        # print(self._object_layout['downstream'])
        # print(sink_res_set)
        # print(all_child_set)
        # print(all_root_set)

        # node_links = {}
        # for o in self._objects['res']:
        #     for k,v in self._object_layout['downstream'].items():

        # plt.rcParams['axes.facecolor']='white'
        # fig_size = (num_row * 12, num_col*12)
        # fig = plt.figure(fig_size)
        # ax = plt.gca()
        # plt.xlim([-0.1, fig_size[1]])
        # plt.ylim([-0.1, fig_size[1]])
        # plt.axis('scaled')
        # plt.axis('off')
        # return fig, ax

    def render_conn(self):
        fig = self._fig
        ax = self._ax
        downstream = self._object_layout['downstream']
        init_points = self._canvas_info['init_points']
        interval = self._interval * 2 / 3

        print(downstream)
        print(init_points)

        for k, v in downstream.items():
            top_point = (init_points[k][0] + interval / 2, init_points[k][1])
            for p in v:
                bot_point = (init_points[p][0] + interval / 2,
                             init_points[p][1] + interval * 1.25)
                style = mpatches.ArrowStyle('Fancy',
                                            head_length=2, head_width=2,
                                            tail_width=0.01)
                arrow = mpatches.FancyArrowPatch(top_point, bot_point,
                                                 arrowstyle=style, color='k')
                ax.add_patch(arrow)
                
        plt.show()
        sys.exit()

    def render_res(self, res: str) -> tuple:

        fig = self._fig
        ax = self._ax
        interval = self._interval * 2 / 3
        curr_t = res
        init_x, init_y = self._canvas_info['init_points'][curr_t]

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

        rlevel = self._object_layout['rlevel'][curr_t]
        rain_scale = self._object_layout['rain_scale'][curr_t]
        rain_shape = self._object_layout['rain_shape'][curr_t]

        max_res_cap = self._object_layout['max_res_cap'][curr_t]
        upper_bound = self._object_layout['upper_bound'][curr_t]
        lower_bound = self._object_layout['lower_bound'][curr_t]

        sink_res = self._object_layout['sink_res'][curr_t]
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
                                       interval / 4,
                                       interval,
                                       fc='royalblue')
        else:
            land_shape = plt.Rectangle((init_x + interval, init_y),
                                       interval / 4,
                                       interval,
                                       fc='darkgoldenrod')

        plt.text(init_x + interval * 1.1,
                 init_y + interval * 1.1,
                 "%s" % curr_t,
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
        self._object_layout = self.build_object_layout()
        self._canvas_info = self.init_canvas_info()
        canvas_size = self._canvas_info['canvas_size']

        self._fig = plt.figure(figsize=(canvas_size[0] / 10, canvas_size[1] / 10),
                               dpi=500)
        self._ax = plt.gca()

        plt.xlim([0, canvas_size[0]])
        plt.ylim([0, canvas_size[1]])
        plt.axis('scaled')
        plt.axis('off')

        # plt.show()
        # sys.exit()

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

        # fig1, ax1 = self.render_single_fig(curr_t, render_text=True)

        # plt.show()

    # def build_res_shape(self, fig, ax, max_res_cap, upper_bound, lower_bound):

    #     line_max = plt.Line2D((maxL[0], maxR[0]),(maxL[1], maxR[1]), ls='--', color='black',lw=3)
    #     line_up = plt.Line2D((upL[0], upR[0]),(upL[1], upR[1]), ls='--', color='orange',lw=3)
    #     line_low = plt.Line2D((lowL[0], lowR[0]),(lowL[1], lowR[1]), ls='--', color='orange',lw=3)
    #     lineL = plt.Line2D((0, 0), (10, 0), color='black',lw=5)
    #     lineR = plt.Line2D((10, 10), (10, 0), color='black',lw=5)
    #     lineB = plt.Line2D((0, 10), (0, 0), color='black',lw=5)
  
    #     ax.add_line(line_max)
    #     plt.text(5.1, max_res_cap/100+0.1, "MAX RES CAP: %s" % round(max_res_cap, 2), color='black', fontsize = 22)

    #     ax.add_line(line_up)
    #     plt.text(0.1, upper_bound/100+0.1, "Upper Bound: %s" % round(upper_bound, 2), color='black', fontsize = 22)

    #     ax.add_line(line_low)
    #     plt.text(0.1, lower_bound/100+0.1, "Lower Bound: %s" % round(lower_bound, 2), color='black', fontsize = 22)
        
    #     ax.add_line(lineL)
    #     ax.add_line(lineR)
    #     ax.add_line(lineB)
        
    # def build_res_fill(self, fig, ax, rlevel, rain_scale, rain_shape):

    #     water_rect = plt.Rectangle((0,0), 10, rlevel/100, fc='royalblue')
    #     res_rect = plt.Rectangle((0,rlevel/100), 10, 10-rlevel/100, fc='lightgrey', alpha=0.5)
    #     scale_rect = plt.Rectangle((0,10), 5, 2, fc='deepskyblue', alpha=rain_scale/100)
    #     shape_rect = plt.Rectangle((5,10), 5, 2, fc='darkcyan', alpha=rain_shape/100)

    #     ax.add_patch(water_rect)
    #     plt.text(5.1, 0.1, "Water Level: %s" % round(rlevel, 2), color='black', fontsize = 22)

    #     ax.add_patch(res_rect)
    #     ax.add_patch(scale_rect)
    #     plt.text(0.1, 11.5, "Rain Scale: %s" % round(rain_scale, 2), color='black', fontsize = 22)

    #     ax.add_patch(shape_rect)
    #     plt.text(5.1, 11.5, "Rain Shape %s" % round(rain_shape, 2), color='black', fontsize = 22)
    
    # def build_conn_shape(self, fig, ax, sink_res, downstream):
    #     lineU = plt.Line2D((10, 12), (2, 2), color='black',lw=5)
    #     lineD = plt.Line2D((10, 12), (0, 0), color='black',lw=5)
    #     conn_shape = plt.Rectangle((10,0), 2, 2, fc='royalblue')
    #     if sink_res:
    #         land_shape = plt.Rectangle((10,2), 2, 8, fc='royalblue')
    #     else:
    #         land_shape = plt.Rectangle((10,2), 2, 8, fc='darkgoldenrod')
    #     ax.add_line(lineU)
    #     ax.add_line(lineD)
    #     ax.add_patch(conn_shape)
    #     plt.text(10.1, 0.1, "Down: %s" % downstream, color='black', fontsize = 15)

    #     ax.add_patch(land_shape)
    #     plt.text(10.1, 2.1, "Sink: %s" % sink_res, color='black', fontsize = 15)

    def display_img(self, duration:float=0.5) -> None:

        plt.imshow(self._data, interpolation='nearest')
        plt.axis('off')
        plt.show(block=False)
        plt.pause(duration)
        plt.close()

    def save_img(self, path:str='./pict.png') -> None:
        
        im = Image.fromarray(self._data)
        im.save(path)

    # def render(self, display:bool = True) -> np.ndarray:

    #     self._object_layout = self.build_object_layout()
    
    #     px = 1/plt.rcParams['figure.dpi']
    #     fig = plt.figure(figsize=(self._resolution[0]*px,self._resolution[1]*px))
    #     ax = plt.gca()

    #     for k,v in self._object_layout['picture_point'].items():
    #         if self._object_layout['picture_taken'][k] == False:
    #             p_point = plt.Circle((v[0],v[1]), radius=v[2], ec='forestgreen', fc='g',fill=True, alpha=0.5)
    #         else:
    #             p_point = plt.Circle((v[0],v[1]), radius=v[2], ec='forestgreen', fill=False)
    #         ax.add_patch(p_point)
        
    #     rover_img_path = self._asset_path + '/assets/mars-rover.png'
    #     rover_logo = plt_img.imread(rover_img_path)
    #     rover_logo_zoom = self._resolution[0]*0.05/rover_logo.shape[0]

    #     for k,v in self._object_layout['rover_location'].items():
    #         imagebox = OffsetImage(rover_logo, zoom=rover_logo_zoom)
    #         ab = AnnotationBbox(imagebox, (v[0], v[1]), frameon = False)
    #         ax.add_artist(ab)
    #         # r_point = plt.Rectangle((v[0],v[1]), 1, 1, fc='navy')
    #         # ax.add_patch(r_point)

    #     plt.axis('scaled')
    #     plt.axis('off')
    #     plt.xlim([self._grid_size[0]//2,-self._grid_size[0]//2])
    #     plt.ylim([self._grid_size[1]//2,-self._grid_size[1]//2])

    #     fig.canvas.draw()

    #     data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    #     data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    #     plt.close()

    #     self._data = data
   
    #     return data
    
    # def display_img(self, duration:float = 0.5) -> None:

    #     plt.imshow(self._data, interpolation='nearest')
    #     plt.axis('off')
    #     plt.show(block=False)
    #     plt.pause(duration)
    #     plt.close()

    # def save_img(self, path:str ='./pict.png') -> None:
        
    #     im = Image.fromarray(self._data)
    #     im.save(path)

