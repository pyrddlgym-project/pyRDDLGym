import matplotlib
matplotlib.use('agg')
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
from time import sleep

from pyRDDLGym.Core.Compiler.RDDLModel import PlanningModel
from pyRDDLGym import Visualizer
from pyRDDLGym.Visualizer.StateViz import StateViz


ROAD_PAVED_WIDTH = 12
ROAD_HALF_MEDIAN_WIDTH = 1
ROAD_WIDTH = ROAD_PAVED_WIDTH + ROAD_HALF_MEDIAN_WIDTH
ROAD_LANE_DELTA = ROAD_PAVED_WIDTH/6
ROAD_QUEUE_DELTA = 2*ROAD_LANE_DELTA
TURN_RADIUS_DELTA = 5

RGBA_PAVEMENT_COLOR = (0.6, 0.6, 0.6, 1)
RGBA_INTERSECTION_COLOR = (0.3, 0.3, 0.3, 1)
RGBA_GREEN_TURNING_COLOR = (0.2, 1, 0.2, 1)

class TrafficVisualizer(StateViz):

    def __init__(self,
                 model: PlanningModel,
                 figure_size=[10, 10],
                 dpi=100,
                 fontsize=10,
                 display=False) -> None:
        self._model = model
        self._states = model.groundstates()
        self._nonfluents = model.groundnonfluents()
        self._objects = model.objects

        self._figure_size = figure_size
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
        self.build_nonfluents_patches()

        # if display == True:
        #     self.pygame_thread = threading.Thread(target=self.init_display)
        #     self.pygame_thread.start()

    def parse_nonfluents(self):
        self.veh_len = self._nonfluents['Lv']
        self.sim_step = self._nonfluents['Ts']

        self.intersection_ids = self._objects['intersection']

        # Register the intersection coordinates, sinks, and sources
        self.intersections = {}
        self.sources = set()
        self.sinks = set()
        for d in self.intersection_ids:
            self.intersections[d] = {
                'coords': np.array([self._nonfluents[f'X_{d}'], self._nonfluents[f'Y_{d}']]),
                'inc_links': [],
                'out_links': []}
            if self._nonfluents[f'SINK_{d}']: self.sinks.add(d)
            if self._nonfluents[f'SOURCE_{d}']: self.sources.add(d)
        self.TLs = set(self.intersection_ids) - self.sources - self.sinks

        # Register the links.
        # Links are indexed by the upstream and downstream intersection pairs
        self.link_ids = []
        for u, d in product(self.intersection_ids, self.intersection_ids):
            if self._nonfluents[f'LINK_{u}_{d}']:
                self.link_ids.append((u,d))
                self.intersections[u]['out_links'].append((u,d))
                self.intersections[d]['inc_links'].append((u,d))

        # Register the turning movements.
        # Turns are indexed by upstream, downstream, and outgoing intersection triples
        self.turn_ids = []
        for u, d, o in product(self.intersection_ids, self.intersection_ids, self.intersection_ids):
            if u == d or d == o or u == o:
                continue
            if self._nonfluents[f'LINK_{u}_{d}'] and self._nonfluents[f'LINK_{d}_{o}']:
                self.turn_ids.append((u,d,o))

        # Record the required link data
        self.linkdata = {}
        ccw_turn_matrix = np.array([[0,-1],[1,0]])
        for (u, d) in self.link_ids:
            vu, vd = self.intersections[u]['coords'], self.intersections[d]['coords']
            dir = vd-vu
            link_len = np.linalg.norm(dir)

            num_lanes = self._nonfluents[f'Nl_{u}_{d}']
            v = self._nonfluents[f'V_{u}_{d}']
            num_cells = ceil(link_len / (v * self.sim_step))

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
            }

        # Order turns from each link left-to-right
        def signed_angle(v0, v1): return np.sign(np.cross(v0,v1)) * np.arccos(np.dot(v0,v1))

        for (u, d) in self.link_ids:
            turns_from_u_d = []
            for o in self.intersection_ids:
                if u == o:
                    continue
                if self._nonfluents[f'LINK_{d}_{o}']:
                    turns_from_u_d.append(o)
            turns_from_u_d.sort(key=lambda o:
                                    -signed_angle(self.linkdata[(u,d)]['dir'], self.linkdata[(d,o)]['dir'])
                               )
            self.linkdata[(u,d)]['turns_from'] = turns_from_u_d


    def build_nonfluents_patches(self):
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
            # and the tangent vectors d0, d2 at ctrl0, ctrl2
            c = np.cross(dir0,dir2)
            if c == 0:
                # Exceptional linear case
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
            v0 = self.intersections[u]['coords']
            d0, n0 = self.linkdata[(u,d)]['dir'], self.linkdata[(u,d)]['nrm']
            road = np.zeros(shape=(4,2))
            L = self.linkdata[(u,d)]['len']
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
                delta = (1+2*idx)*ROAD_LANE_DELTA
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
        # on different axes :(
        for link_patch in self.link_patches.values():
            ax.add_patch(copy(link_patch))

        for d in self.TLs:
            intersection_patch = self.intersections[d]['patch']
            ax.add_patch(copy(intersection_patch))

        for turn_patch in self.turn_patches.values():
            ax.add_patch(copy(turn_patch))

    def build_states_layout(self, states, fig, ax):
        return states

    def render(self, state):
        self.states = state

        self._fig, self._ax = self.init_canvas(self._figure_size, self._dpi)
        self.build_nonfluents_layout(self._fig, self._ax)
        self.build_states_layout(state, self._fig, self._ax)


        img = self.convert2img(self._fig, self._ax)
        self._ax.cla()
        plt.close()
#        img.show()
 #       exit()

        return img
