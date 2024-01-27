import matplotlib
matplotlib.use('agg')
import pygame
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
import pprint
from itertools import product
from collections import defaultdict
from copy import copy
import math

from pyRDDLGym.Core.Compiler.RDDLModel import PlanningModel
from pyRDDLGym import Visualizer
from pyRDDLGym.Visualizer.StateViz import StateViz

ROAD_PAVED_WIDTH = 12                                   # The width of the paved part of the road in m
ROAD_HALF_MEDIAN_WIDTH = 1                              # The width of the median in-between links in m
ROAD_WIDTH = ROAD_PAVED_WIDTH + ROAD_HALF_MEDIAN_WIDTH  # Combined width of a single link (paved + 0.5*median)
ROAD_DELTA = ROAD_PAVED_WIDTH/12                        # Division of a link into parts for drawing turns
TURN_RADIUS_DELTA = 5                                   # Curvature of a turn
LINK_GRID_SAT_DENSITY = 0.2 #per lane                   # Density at which a cell in the flow grid gets a saturated color

RGBA_PAVEMENT_COLOR = (0.6, 0.6, 0.6, 1)
RGBA_INTERSECTION_COLOR = (0.3, 0.3, 0.3, 1)
RGBA_GREEN_TURN_COLOR = (0.2, 1, 0.2, 1)
RGBA_Q_PATCH_COLOR = (1, 0.2, 0.2, 0.4)
RGBA_Q_LINE_COLOR = (1, 0.2, 0.2, 1)
RGBA_LINK_GRID_COLOR = RGBA_PAVEMENT_COLOR
RGBA_LINK_GRID_SAT_COLOR = (0.9, 0.3, 0.9, 1)

def get_fillcolor_by_density(N, capacity):
    p = max(min(1, N/capacity),0)
    return tuple(p*c for c in RGBA_LINK_GRID_SAT_COLOR)

def id(var_id, *objs):
    # Returns <var_id>___<ob1>__<ob2>__<ob3>...
    return var_id + PlanningModel.FLUENT_SEP + PlanningModel.OBJECT_SEP.join(objs)


class TrafficVisualizer(StateViz):

    def __init__(self,
                 model: PlanningModel,
                 figure_size=None, # Pass [int, int] when overriding default
                 dpi=100,
                 fontsize=10,
                 display=False) -> None:
        self._model = model
        self._states = model.groundstates()
        self._nonfluents = model.groundnonfluents()
        self._objects = model.objects

        if figure_size is None:
            pygame.init()
            info = pygame.display.Info()
            if info.current_w == -1 or info.current_h == -1:
                # Display info not available to pygame
                self._figure_size = [10,10]
            else:
                self._figure_size = [0.9*info.current_w/dpi, 0.9*info.current_h/dpi]

        self._display = display
        self._dpi = dpi
        self._fig, self._ax = None, None
        self._fontsize = fontsize
        self._interval = 10
        self._asset_path = "/".join(Visualizer.__file__.split("/")[:-1])
        self._nonfluents_layout = None
        self._states_layout = None
        self._data = None
        self._img = None

        self.parse_nonfluents()
        self.build_nonfluent_patches()

        # if display == True:
        #     self.pygame_thread = threading.Thread(target=self.init_display)
        #     self.pygame_thread.start()

    def parse_nonfluents(self):
        self.veh_len = self._nonfluents['Lv']
        self.sim_step = self._nonfluents['Ts']

        # Register the intersection coordinates and the links incident to them
        self.intersections = {}
        self.linkdata = defaultdict(dict)
        for d in self._objects['intersection']:
            coords = np.array([self._nonfluents[id('X', d)],
                               self._nonfluents[id('Y', d)]])
            inc_links, out_links = [], []

            for L in self._objects['link']:

                if self._nonfluents[id('LINK-TO', L,d)]:
                    self.linkdata[L]['to'] = coords
                    self.linkdata[L]['to_id'] = d
                    inc_links.append(L)

                elif self._nonfluents[id('LINK-FROM', d,L)]:
                    self.linkdata[L]['from'] = coords
                    self.linkdata[L]['from_id'] = d
                    out_links.append(L)

            self.intersections[d] = {
                'coords': coords,
                'inc_links': inc_links,
                'out_links': out_links}


        # Register the links and record the required link data
        ccw_turn_matrix = np.array([[0,-1],[1,0]])
        self.sources, self.sinks = {}, {}

        for L in self.linkdata:

            # Register the sources and sinks, and in addition
            # find the start and ends points of the link
            if self._nonfluents[id('SOURCE', L)]:
                try:
                    vu = np.array([self._nonfluents[id('SOURCE-X', L)],
                                   self._nonfluents[id('SOURCE-Y', L)]])
                except KeyError as e:
                    print('[TrafficViz] Please include the X and Y coordinates of the source '
                          f'{L} in the instance defn')
                    raise e
                self.sources[L] = {'coords': vu}
                self.linkdata[L]['from'] = vu
                self.linkdata[L]['from_id'] = f'source{len(self.sources)}'

            elif self._nonfluents[id('SINK', L)]:
                 try:
                     vd = np.array([self._nonfluents[id('SINK-X', L)],
                                    self._nonfluents[id('SINK-Y', L)]])
                 except KeyError as e:
                     print('[TrafficViz] Please include the X and Y coordinates of the sink '
                           f'{L} in the instance defn')
                     raise e
                 self.sinks[L] = {'coords': vd}
                 self.linkdata[L]['to'] = vd
                 self.linkdata[L]['to_id'] = f'sink{len(self.sinks)}'

            vu, vd = self.linkdata[L]['from'], self.linkdata[L]['to']
            dir = vd-vu
            link_len = np.linalg.norm(dir)

            num_lanes = self._nonfluents[id('Nl', L)]
            v = self._nonfluents[id('Vl', L)]
            num_cells = math.ceil(link_len / (v * self.sim_step))
            cell_capacity = (link_len * num_lanes) / (num_cells * self.veh_len)

            dir = dir/link_len

            self.linkdata[L].update({
                # Unit vector along the link direction
                'dir': dir,
                # Unit normal vector (ccw of unit dir vector)
                'nrm': np.matmul(ccw_turn_matrix, dir),
                # Length of the link
                'len': link_len,
                # Freeflow speed on the link
                'v': v,
                # Number of lanes
                'num_lanes': num_lanes,
                # Number of incoming cells
                'num_cells': num_cells,
                # Cell capacity (for shading)
                'cell_capacity': cell_capacity })


        # Register the turning movements.
        # Turns are indexed by upstream, downstream link pairs
        # Additionally, order turns from each link left-to-right
        def signed_angle(v0, v1):
            if np.cross(v0,v1) == 0:
                return -math.pi if np.dot(v0,v1) < 0 else 0
            return np.sign(np.cross(v0,v1)) * np.arccos(np.dot(v0,v1))

        self.turn_ids = []
        for L in self.linkdata:
            self.linkdata[L]['turns_from'] = []
            for M in self.linkdata:
                if self._nonfluents[id('TURN', L,M)]:
                     self.turn_ids.append((L,M))
                     self.linkdata[L]['turns_from'].append(M)
            self.linkdata[L]['turns_from'].sort(
                key=lambda M: -signed_angle(self.linkdata[L]['dir'], self.linkdata[M]['dir']))

        # Try to find the opposite link by the most negative dot product
        for d, v in self.intersections.items():
            for L in v['inc_links']:
                dL = self.linkdata[L]['dir']
                M = sorted(v['out_links'], key=lambda M: np.dot(dL, self.linkdata[M]['dir']))[0]
                self.linkdata[L]['opposite_link'] = M
                self.linkdata[M]['opposite_link'] = L

        # Register the signal phases
        self.signal_phases = self._objects['signal-phase']

        # Register the green turns for each phase in each intersection
        self.green_turns_by_intersection_phase = {}
        for d, dv in self.intersections.items():
            self.green_turns_by_intersection_phase[d] = {k: [] for k in self.signal_phases}
            for L in dv['inc_links']:
                for M in self.linkdata[L]['turns_from']:
                    for p in self.signal_phases:
                        if self._nonfluents[id('GREEN', L,M,p)]:
                            self.green_turns_by_intersection_phase[d][p].append((L,M))


    def build_nonfluent_patches(self):
        # Find the bbox
        x0, y0, x1, y1 = math.inf, math.inf, -math.inf, -math.inf
        for geopoint in (tuple(self.intersections.values()) +
                         tuple(self.sources.values()) +
                         tuple(self.sinks.values())):
            v = geopoint['coords']
            x0, y0, x1, y1 = min(x0, v[0]), min(y0, v[1]), max(x1, v[0]), max(y1, v[1])

        # Slightly expand bbox
        w, h = x1-x0, y1-y0
        dw, dh = 0.05*w, 0.05*h
        self.bbox = {'x0': x0 - dw, 'x1': x1 + dw,
                     'y0': y0 - dh, 'y1': y1 + dh,
                     'w': round(w+2*dw), 'h': round(h+2*dh)}

        def quad_bezier_middle_ctrl_pt(ctrl0, ctrl2, dir0, dir2):
            # Find the middle control point of a quadratic Bezier curve,
            # given the first and third control points ctrl0 and ctrl2,
            # and the tangent vectors d0 at ctrl0 and d2 at ctrl2
            c = np.cross(dir0,dir2)
            if c == 0:
                # Exceptional case when the tangent vectors are colinear
                ctrl1 = (ctrl0 + ctrl2)/2
            else:
                ctrl1 = ctrl2 + (np.cross(dir0,ctrl0-ctrl2)/c)*dir2
            return ctrl1

        # Construct intersection objects and correct link lengths
        for d in self.intersections:

            # Preliminary computations
            alpha_in, alpha_out = {}, {}
            for L in self.intersections[d]['inc_links']:
                # Right-turn-target from L
                M = self.linkdata[L]['turns_from'][-1]

                # Find the point of intersection of the right boundary of L
                # and the right boundary of M (if both links are extended).
                # The links will be shortened so that they do not intersect,
                # as well as shortened by TURN_RADIUS_DELTA to draw a smoother
                # connecting curve
                vu, vd = self.linkdata[L]['from'], self.linkdata[L]['to']
                dir0, n0 = self.linkdata[L]['dir'], self.linkdata[L]['nrm']
                dir1, n1 = self.linkdata[M]['dir'], self.linkdata[M]['nrm']
                c = np.cross(dir0,dir1)
                if c == 0:
                    # Exceptional case when the right-turn direction happens to be straight ahead
                    alpha_in[L] = 0
                    alpha_out[M] = 0
                else:
                    alpha_in[L]  = ROAD_WIDTH*np.cross(dir1, n1-n0)/np.cross(dir0,dir1)
                    alpha_out[M] = ROAD_WIDTH*np.cross(dir0, n1-n0)/np.cross(dir0,dir1)

            # Construct the intersection contour
            L0 = self.intersections[d]['inc_links'][0]
            L = L0
            contour, codes = [], []
            while True:

                # Right-turn-target from L
                M = self.linkdata[L]['turns_from'][-1]
                Lopp = self.linkdata[L]['opposite_link']
                Mopp = self.linkdata[M]['opposite_link']

                vu, vd = self.linkdata[L]['from'], self.linkdata[L]['to']
                dir0, n0 = self.linkdata[L]['dir'], self.linkdata[L]['nrm']
                dir1, n1 = self.linkdata[M]['dir'], self.linkdata[M]['nrm']

                # Find the control points for the right-turn Bezier curves
                # Shorten the links by the same amount in both travel directions
                beta =  max(-alpha_in[L],    alpha_out[Lopp]) + TURN_RADIUS_DELTA #L shortening
                alpha = max(-alpha_in[Mopp], alpha_out[M]) + TURN_RADIUS_DELTA #M shortening

                # Right-turn Bezier control points
                ctrl0 = vd - ROAD_WIDTH*n0 - beta*dir0
                ctrl2 = vd - ROAD_WIDTH*n1 + alpha*dir1
                ctrl1 = quad_bezier_middle_ctrl_pt(ctrl0, ctrl2, dir0, dir1)

                # Correct the link lengths
                # * Subtract TURN_RADIUS_DELTA
                # * Avoid intersections
                self.linkdata[L]['len'] -= beta
                self.linkdata[M]['from'] = vd + alpha*dir1
                self.linkdata[M]['len'] -= alpha

                # Trace the path
                F = self.linkdata[L].get('from', vu)
                Llen = self.linkdata[L]['len']
                p = F + Llen*dir0 - ROAD_HALF_MEDIAN_WIDTH*n0
                if len(contour) == 0:
                    # Initializer
                    contour.append(p)
                    codes.append(Path.MOVETO)
                contour.append(p)
                codes.append(Path.LINETO)
                contour.extend([ctrl0, ctrl1, ctrl2])
                codes.extend([Path.LINETO, Path.CURVE3, Path.CURVE3])
                p = ctrl2 + ROAD_PAVED_WIDTH*n1
                contour.append(p)
                codes.append(Path.LINETO)
                ctrl0 = p
                ctrl2 = p + 2*ROAD_HALF_MEDIAN_WIDTH*n1
                ctrl1 = p + ROAD_HALF_MEDIAN_WIDTH*n1 - TURN_RADIUS_DELTA*dir1
                contour.extend([ctrl0, ctrl1, ctrl2])
                codes.extend([Path.LINETO, Path.CURVE3, Path.CURVE3])

                L = Mopp
                if L == L0:
                    contour.append(np.array([0,0])) #ignored
                    codes.append(Path.CLOSEPOLY)
                    break

            # Done with the intersection contour!
            path = Path(contour, codes)
            patch = mpatches.PathPatch(
                path,
                color=RGBA_INTERSECTION_COLOR)
            self.intersections[d]['patch'] = patch

        # Construct the link patches,
        # register data for plotting the queues,
        # construct turning curve patches
        self.link_patches = {}
        self.turn_patches = {}
        for L, Lv in self.linkdata.items():
            vu, vd = Lv['from'], Lv['to']
            d0, n0 = Lv['dir'], Lv['nrm']
            road = np.zeros(shape=(4,2))
            Llen = Lv['len']

            # Record the visual shrinking factor for visualizing queues
            self.linkdata[L]['shrink'] = Llen/np.linalg.norm(vd-vu)

            # Create the link patch
            road[0] = vu - ROAD_HALF_MEDIAN_WIDTH*n0
            road[1] = road[0] + Llen*d0
            road[2] = road[1] - ROAD_PAVED_WIDTH*n0
            road[3] = road[2] - Llen*d0
            self.link_patches[L] = plt.Polygon(road, color=RGBA_PAVEMENT_COLOR)

            self.linkdata[L]['cell_len'] = Llen/self.linkdata[L]['num_cells']
            self.linkdata[L]['stopline_left'] = road[1]

            for idx, M in enumerate(Lv['turns_from']):
                # Turns are indexed left-to-right
                d1, n1 = self.linkdata[M]['dir'], self.linkdata[M]['nrm']

                # Turning curve patch
                delta = (2+4*idx)*ROAD_DELTA
                from_ = road[1] - delta*n0
                to_ = self.linkdata[M]['from'] - (ROAD_HALF_MEDIAN_WIDTH+delta)*n1
                ctrl1 = quad_bezier_middle_ctrl_pt(from_, to_, d0, d1)
                path = Path([from_, ctrl1, to_], [Path.MOVETO, Path.CURVE3, Path.CURVE3])
                patch = mpatches.PathPatch(
                    path,
                    edgecolor=RGBA_GREEN_TURN_COLOR,
                    fill=False,
                    linewidth=2)
                self.turn_patches[(L,M)] = patch

    def init_canvas(self, figure_size, dpi):
        fig = plt.figure(figsize=figure_size, dpi=dpi)
        ax = plt.gca()
        plt.xlim(self.bbox['x0'], self.bbox['x1'])
        plt.ylim(self.bbox['y0'], self.bbox['y1'])
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

    def build_nonfluents_layout(self, fig, ax):
        # Seems copy is the only way to re-use patches
        # when using more than one axes object :(
        for link_patch in self.link_patches.values():
            ax.add_patch(copy(link_patch))

        for v in self.intersections.values():
            ax.add_patch(copy(v['patch']))

    def build_states_layout(self, states, fig, ax):

        for d in self.intersections:
            signal = self.signal_phases[states[id('signal', d)]]
            for t in self.green_turns_by_intersection_phase[d][signal]:
                ax.add_patch(copy(self.turn_patches[t]))

        for L, Lv in self.linkdata.items():
            # If the link is not a sink, draw the total queue
            # as well as the queue proportions by turning movement.
            # (for a sink, there is no queue)
            Q = sum(states[id('q', L,M)] for M in Lv['turns_from'])

            if L not in self.sinks and Q > 0:
                Q_dist = (Q*self.veh_len/Lv['num_lanes']) * Lv['shrink']
                Q_frontier_L = Lv['stopline_left'] - Q_dist*Lv['dir']
                Q_frontier_R = Q_frontier_L - ROAD_PAVED_WIDTH*Lv['nrm']

                # Draw the total queue patch
                Q_total_patch = np.zeros(shape=(4,2))
                Q_total_patch[0] = Lv['stopline_left']
                Q_total_patch[1] = Q_total_patch[0] - ROAD_PAVED_WIDTH*Lv['nrm']
                Q_total_patch[2] = Q_frontier_R
                Q_total_patch[3] = Q_frontier_L
                patch = plt.Polygon(Q_total_patch, facecolor=RGBA_Q_PATCH_COLOR, linewidth=1)
                ax.add_patch(patch)

                # Draw the queue-proportions per individual turn
                for idx, M in enumerate(Lv['turns_from']):
                    Q_turn = states[id('q', L,M)]
                    barlen = (Q_turn/Q) * Q_dist
                    Q_turn_bar = np.zeros(shape=(4,2))
                    Q_turn_bar[0] = Lv['stopline_left'] - (1+4*idx)*ROAD_DELTA*Lv['nrm']
                    Q_turn_bar[1] = Q_turn_bar[0] - 2*ROAD_DELTA*Lv['nrm']
                    Q_turn_bar[2] = Q_turn_bar[1] - barlen*Lv['dir']
                    Q_turn_bar[3] = Q_turn_bar[2] + 2*ROAD_DELTA*Lv['nrm']
                    patch = plt.Polygon(Q_turn_bar, color=RGBA_Q_LINE_COLOR, linewidth=0)
                    ax.add_patch(patch)
            else:
                Q_dist, Q_frontier_L, Q_frontier_R = 0, Lv['stopline_left'], Lv['stopline_left'] - ROAD_PAVED_WIDTH*Lv['nrm']

            # Plot the flows through the link
            num_cells_in_use = math.ceil((Lv['len'] - Q_dist) / Lv['cell_len'])
            frontier = np.array([Q_frontier_L, Q_frontier_R])
            saturation_d = LINK_GRID_SAT_DENSITY * Lv['num_lanes']
            for t in range(1, num_cells_in_use):
                flow = states[id('flow-on-link', L,f't{t}')]
                color = get_fillcolor_by_density(flow, saturation_d)
                next_frontier = frontier - Lv['cell_len']*Lv['dir']
                cell = np.array((frontier[0], frontier[1], next_frontier[1], next_frontier[0]))
                patch = plt.Polygon(cell, linewidth=1, facecolor=color,
                                    edgecolor=RGBA_LINK_GRID_COLOR)
                ax.add_patch(patch)
                frontier = next_frontier
            # The last cell may be shorter
            flow = states[id('flow-on-link', L,f't{num_cells_in_use}')]
            color = get_fillcolor_by_density(flow, saturation_d)
            delta = Lv['len'] - (Lv['cell_len']*(num_cells_in_use-1) + Q_dist)
            next_frontier = frontier - delta*Lv['dir']
            cell = np.array((frontier[0], frontier[1], next_frontier[1], next_frontier[0]))
            patch = plt.Polygon(cell, linewidth=1, facecolor=color,
                                edgecolor=RGBA_LINK_GRID_COLOR)
            ax.add_patch(patch)


    def render(self, state):
        self.states = state

        self._fig, self._ax = self.init_canvas(self._figure_size, self._dpi)
        self.build_nonfluents_layout(self._fig, self._ax)
        self.build_states_layout(state, self._fig, self._ax)

        img = self.convert2img(self._fig, self._ax)
        self._ax.cla()
        plt.close()
        return img
