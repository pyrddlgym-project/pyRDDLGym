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
from math import ceil
from math import inf

from pyRDDLGym.Core.Compiler.RDDLModel import PlanningModel
from pyRDDLGym import Visualizer
from pyRDDLGym.Visualizer.StateViz import StateViz

ROAD_PAVED_WIDTH = 12
ROAD_HALF_MEDIAN_WIDTH = 1
ROAD_WIDTH = ROAD_PAVED_WIDTH + ROAD_HALF_MEDIAN_WIDTH
ROAD_DELTA = ROAD_PAVED_WIDTH/12
TURN_RADIUS_DELTA = 5
ARRIVING_RATE_SAT = 0.2 #per lane

RGBA_PAVEMENT_COLOR = (0.6, 0.6, 0.6, 1)
RGBA_INTERSECTION_COLOR = (0.3, 0.3, 0.3, 1)
RGBA_GREEN_TURNING_COLOR = (0.2, 1, 0.2, 1)
RGBA_Q_LINE_COLOR = (1, 0.2, 0.2, 1)
RGBA_Q_BARS_COLOR = (1, 0.2, 0.2, 1)
RGBA_Q_PATCH_COLOR = (1, 0.2, 0.2, 0.4)
RGBA_ARRIVING_CELL_BORDER_COLOR = RGBA_PAVEMENT_COLOR
RGBA_ARRIVING_CELL_SAT_COLOR = (1, 0.2, 1, 0.4)

def get_fillcolor_by_density(N, capacity):
    p = min(1, N/capacity)
    return tuple(p*c for c in RGBA_ARRIVING_CELL_SAT_COLOR)

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

        # Register the intersection coordinates, sinks, and sources
        self.intersections = {}
        self.sources, self.sinks, self.TLs = set(), set(), set()
        for d in self._objects['intersection']:
            self.intersections[d] = {
                'coords': np.array([self._nonfluents[id('X', d)], self._nonfluents[id('Y', d)]]),
                'inc_links': [],
                'out_links': []}
            if self._nonfluents[id('SINK', d)]: self.sinks.add(d)
            if self._nonfluents[id('SOURCE', d)]: self.sources.add(d)
            if self._nonfluents[id('TL', d)]: self.TLs.add(d)

        # Register the links.
        # Links are indexed by the upstream and downstream intersection pairs
        self.link_ids = []
        for u, d in product(self.intersections.keys(), self.intersections.keys()):
            if self._nonfluents[id('LINK', u,d)]:
                self.link_ids.append((u,d))
                self.intersections[u]['out_links'].append((u,d))
                self.intersections[d]['inc_links'].append((u,d))

        # Register the turning movements.
        # Turns are indexed by upstream, downstream, and outgoing intersection triples
        self.turn_ids = []
        for u, d, o in product(self.intersections.keys(), self.intersections.keys(), self.intersections.keys()):
            if u == d or d == o or u == o:
                continue
            if self._nonfluents[id('LINK', u,d)] and self._nonfluents[id('LINK', d,o)]:
                self.turn_ids.append((u,d,o))

        # Record the required link data
        self.linkdata = {}
        ccw_turn_matrix = np.array([[0,-1],[1,0]])
        for (u, d) in self.link_ids:
            vu, vd = self.intersections[u]['coords'], self.intersections[d]['coords']
            dir = vd-vu
            link_len = np.linalg.norm(dir)

            num_lanes = self._nonfluents[id('Nl', u,d)]
            v = self._nonfluents[id('V', u,d)]
            num_cells = ceil(link_len / (v * self.sim_step))
            cell_capacity = (link_len * num_lanes) / (num_cells * self.veh_len)

            dir = dir/link_len

            self.linkdata[(u,d)] = {
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
                'cell_capacity': cell_capacity
            }

        # Order turns from each link left-to-right
        def signed_angle(v0, v1): return np.sign(np.cross(v0,v1)) * np.arccos(np.dot(v0,v1))

        for (u, d) in self.link_ids:
            turns_from_u_d = []
            for o in self.intersections.keys():
                if u == o:
                    continue
                if self._nonfluents[id('LINK', d,o)]:
                    turns_from_u_d.append(o)
            turns_from_u_d.sort(key=lambda o:
                                    -signed_angle(self.linkdata[(u,d)]['dir'], self.linkdata[(d,o)]['dir'])
                               )
            self.linkdata[(u,d)]['turns_from'] = turns_from_u_d

        # Register the green turns for each phase
        self.phase_ids = self._objects['phase']
        self.green_turns_by_phase = defaultdict(list)
        for (u,d,o) in self.turn_ids:
            for p in self.phase_ids:
                if self._nonfluents[id('GREEN', u,d,o,p)]:
                    self.green_turns_by_phase[p].append((u,d,o))

        # Register inverse map from phase index to phase object
        self.phase_by_index_in_intersection = {}
        for d in self.intersections.keys():
             self.phase_by_index_in_intersection[d] = {}
             for p in self.phase_ids:
                 if self._nonfluents[id('PHASE-OF', p,d)]:
                     self.phase_by_index_in_intersection[d][self._nonfluents[id('PHASE-INDEX', p)]] = p


    def build_nonfluent_patches(self):
        # Find the bbox
        x0, y0, x1, y1 = inf, inf, -inf, -inf
        for intersection in self.intersections.values():
            v = intersection['coords']
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
        for d in self.TLs:

            # Preliminary computations
            alpha_ud, alpha_do = {}, {}
            for (u,d) in self.intersections[d]['inc_links']:
                # Right-turn-target from (u,d)
                o = self.linkdata[(u,d)]['turns_from'][-1]

                # Find the point of intersection of the right boundary of (u,d)
                # and the right boundary of (d,o) (if both links are extended)
                # The links will be shortened so that they do not intersect
                vu, vd = self.intersections[u]['coords'], self.intersections[d]['coords']
                dir0, n0 = self.linkdata[(u,d)]['dir'], self.linkdata[(u,d)]['nrm']
                dir1, n1 = self.linkdata[(d,o)]['dir'], self.linkdata[(d,o)]['nrm']
                c = np.cross(dir0,dir1)
                if c == 0:
                    # Exceptional case when the right-turn direction happens to be straight ahead
                    alpha_ud[(u,d)] = 0
                    alpha_do[(d,o)] = 0
                else:
                    alpha_ud[(u,d)] = ROAD_WIDTH*np.cross(dir1, n1-n0)/np.cross(dir0,dir1)
                    alpha_do[(d,o)] = ROAD_WIDTH*np.cross(dir0, n1-n0)/np.cross(dir0,dir1)

            # Construct the intersection contour
            (u0,d0) = self.intersections[d]['inc_links'][0]
            (u,d) = (u0,d0)
            contour, codes = [], []
            while True:
                if u in self.sources:
                    self.linkdata[(u,d)]['from'] = self.intersections[u]['coords']

                # Right-turn-target from (u,d)
                o = self.linkdata[(u,d)]['turns_from'][-1]
                vu, vd = self.intersections[u]['coords'], self.intersections[d]['coords']
                dir0, n0 = self.linkdata[(u,d)]['dir'], self.linkdata[(u,d)]['nrm']
                dir1, n1 = self.linkdata[(d,o)]['dir'], self.linkdata[(d,o)]['nrm']

                # Find the control points for the right-turn Bezier curves
                # Shorten the links by the same amount in both travel directions
                beta =  max(-alpha_ud[(u,d)], alpha_do[(d,u)]) + TURN_RADIUS_DELTA #(u,d) shortening
                alpha = max(-alpha_ud[(o,d)], alpha_do[(d,o)]) + TURN_RADIUS_DELTA #(d,o) shortening

                # Right-turn Bezier control points
                ctrl0 = vd - ROAD_WIDTH*n0 - beta*dir0
                ctrl2 = vd - ROAD_WIDTH*n1 + alpha*dir1
                ctrl1 = quad_bezier_middle_ctrl_pt(ctrl0, ctrl2, dir0, dir1)

                # Correct the link lengths
                # * Subtract TURN_RADIUS_DELTA
                # * Avoid intersections
                self.linkdata[(u,d)]['len'] -= beta
                self.linkdata[(d,o)]['from'] = vd + alpha*dir1
                self.linkdata[(d,o)]['len'] -= alpha

                # Trace the path
                F = self.linkdata[(u,d)].get('from', vu)
                L = self.linkdata[(u,d)]['len']
                p = F + L*dir0 - ROAD_HALF_MEDIAN_WIDTH*n0
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

                (u,d) = (o,d)
                if (u,d) == (u0,d0):
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
        self.incoming_grid_patches = {}
        self.turn_patches = {}
        for (u,d) in self.link_ids:
            vu, vd = self.intersections[u]['coords'], self.intersections[d]['coords']
            d0, n0 = self.linkdata[(u,d)]['dir'], self.linkdata[(u,d)]['nrm']
            road = np.zeros(shape=(4,2))
            L = self.linkdata[(u,d)]['len']

            # Record the visual shrinking factor for visualizing queues
            self.linkdata[(u,d)]['shrink'] = L/np.linalg.norm(vd-vu)

            # Create the link patch
            road[0] = self.linkdata[(u,d)]['from'] - ROAD_HALF_MEDIAN_WIDTH*n0
            road[1] = road[0] + L*d0
            road[2] = road[1] - ROAD_PAVED_WIDTH*n0
            road[3] = road[2] - L*d0
            self.link_patches[(u,d)] = plt.Polygon(road, color=RGBA_PAVEMENT_COLOR)

            self.linkdata[(u,d)]['cell_len'] = L/self.linkdata[(u,d)]['num_cells']
            self.linkdata[(u,d)]['queues_top_left'] = road[1]

            for idx, o in enumerate(self.linkdata[(u,d)]['turns_from']):
                # Turns are indexed left-to-right
                d1, n1 = self.linkdata[(d,o)]['dir'], self.linkdata[(d,o)]['nrm']

                # Turning curve patch
                delta = (2+4*idx)*ROAD_DELTA
                from_ = road[1] - delta*n0
                to_ = self.linkdata[(d,o)]['from'] - (ROAD_HALF_MEDIAN_WIDTH+delta)*n1
                ctrl1 = quad_bezier_middle_ctrl_pt(from_, to_, d0, d1)
                path = Path([from_, ctrl1, to_], [Path.MOVETO, Path.CURVE3, Path.CURVE3])
                patch = mpatches.PathPatch(
                    path,
                    edgecolor=RGBA_GREEN_TURNING_COLOR,
                    fill=False,
                    linewidth=2)
                self.turn_patches[(u,d,o)] = patch

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

        for d in self.TLs:
            intersection_patch = self.intersections[d]['patch']
            ax.add_patch(copy(intersection_patch))

    def build_states_layout(self, states, fig, ax):

        for d in self.TLs:
            if states[id('all-red', d)] > 0:
                continue
            cur_ph_idx = states[id('cur-ph-idx', d)]
            cur_ph = self.phase_by_index_in_intersection[d][cur_ph_idx]
            for t in self.green_turns_by_phase[cur_ph]:
                ax.add_patch(copy(self.turn_patches[t]))


        for d in self.TLs.union(self.sinks):
            for (u,d) in self.intersections[d]['inc_links']:
                ld = self.linkdata[(u,d)]

                # If downstream intersection is not a sink, draw the queue data
                # (for a downstream sink, the queues are zero, so no need)
                if not self._nonfluents[id('SINK', d)]:
                    Q = states[id('qd', u,d)]
                    Q_line = (Q*self.veh_len/ld['num_lanes']) * ld['shrink']
                    Q_line_L = ld['queues_top_left'] - Q_line*ld['dir']
                    Q_line_R = Q_line_L - ROAD_PAVED_WIDTH*ld['nrm']

                    # Draw queue background patch
                    Q_bg_patch = np.zeros(shape=(4,2))
                    Q_bg_patch[0] = ld['queues_top_left']
                    Q_bg_patch[1] = Q_bg_patch[0] - ROAD_PAVED_WIDTH*ld['nrm']
                    Q_bg_patch[2] = Q_line_R
                    Q_bg_patch[3] = Q_line_L
                    patch = plt.Polygon(Q_bg_patch, facecolor=RGBA_Q_PATCH_COLOR, linewidth=0)
                    ax.add_patch(patch)

                    # Draw queue bars
                    for idx, o in enumerate(ld['turns_from']):
                        Q_turn = states[id('q', u,d,o)]
                        Q_turn_height = (Q_turn/(Q+1e-6)) * Q_line
                        Q_turn_bar = np.zeros(shape=(4,2))
                        Q_turn_bar[0] = ld['queues_top_left'] - (1+4*idx)*ROAD_DELTA*ld['nrm']
                        Q_turn_bar[1] = Q_turn_bar[0] - 2*ROAD_DELTA*ld['nrm']
                        Q_turn_bar[2] = Q_turn_bar[1] - Q_turn_height*ld['dir']
                        Q_turn_bar[3] = Q_turn_bar[2] + 2*ROAD_DELTA*ld['nrm']
                        patch = plt.Polygon(Q_turn_bar, color=RGBA_Q_LINE_COLOR, linewidth=0)
                        ax.add_patch(patch)
                else:
                    Q_line, Q_line_L, Q_line_R = 0, ld['queues_top_left'], ld['queues_top_left']-ROAD_PAVED_WIDTH*ld['nrm']

                # Aggregate the incoming flows, depending on the type of u
                num_cells_to_Q_line = max(ceil((ld['len'] - Q_line)/ld['cell_len']), 1)
                if self._nonfluents[id('SOURCE', u)]:
                    flows = tuple( states[id('Mlsrc', u,d,f't{t}')] for t in range(1, num_cells_to_Q_line+1))
                else:
                    flows = np.zeros(num_cells_to_Q_line+1)
                    for t in range(1,num_cells_to_Q_line+1):
                        for (i,u) in self.intersections[u]['inc_links']:
                            if i != d:
                                flows[t] += states[id('Ml', i,u,d,f't{t}')]

                line_L, line_R = Q_line_L, Q_line_R
                for t in range(1,num_cells_to_Q_line):
                    flow = flows[t]
                    next_line_L, next_line_R = line_L-ld['cell_len']*ld['dir'], line_R-ld['cell_len']*ld['dir']
                    cell = np.array([line_L, line_R, next_line_R, next_line_L])
                    color = get_fillcolor_by_density(flow, ARRIVING_RATE_SAT*ld['num_lanes'])
                    patch = plt.Polygon(cell, linewidth=1, facecolor=color, edgecolor=RGBA_ARRIVING_CELL_BORDER_COLOR)
                    ax.add_patch(patch)
                    line_L, line_R = next_line_L, next_line_R
                # Draw the last cell
                flow = flows[-1]
                remainder = ld['len'] - (ld['cell_len']*(num_cells_to_Q_line-1) + Q_line)
                next_line_L, next_line_R = line_L-remainder*ld['dir'], line_R-remainder*ld['dir']
                cell = np.array([line_L, line_R, next_line_R, next_line_L])
                color = get_fillcolor_by_density(flow, ARRIVING_RATE_SAT*ld['num_lanes'])
                patch = plt.Polygon(cell, linewidth=1, facecolor=color, edgecolor=RGBA_ARRIVING_CELL_BORDER_COLOR)
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
