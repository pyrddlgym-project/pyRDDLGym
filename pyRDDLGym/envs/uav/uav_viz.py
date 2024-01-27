import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from pyRDDLGym.Core.Compiler.RDDLModel import PlanningModel
from pyRDDLGym.Visualizer.StateViz import StateViz
from pyRDDLGym import Visualizer


class UAVsVisualizer(StateViz):

    def __init__(self, model: PlanningModel,
                 figure_size=[200, 200],
                 dpi=5,
                 fontsize=8,
                 display=False) -> None:
        self._model = model
        self._states = model.groundstates()
        self._nonfluents = model.groundnonfluents()
        self._objects = model.objects
        self._figure_size = figure_size
        self._dpi = dpi
        self._fontsize = fontsize
        self._interval = 10
        self._asset_path = "/".join(Visualizer.__file__.split("/")[:-1])      
        self._nonfluent_layout = None
        self._state_layout = None
        self._fig, self._ax = None, None
        self._data = None
        self._img = None
    
    def build_nonfluents_layout(self): 
        goal_location = {o: [None, None, None] for o in self._objects['aircraft']}
        
        for k, v in self._nonfluents.items():
            var, objects = self._model.parse(k)
            if var == 'GOAL-X':
                goal_location[objects[0]][0] = v
            elif var == 'GOAL-Y':
                goal_location[objects[0]][1] = v
            elif var == 'GOAL-Z':
                goal_location[objects[0]][2] = v
        
        return {'goal_location': goal_location}
    
    def build_states_layout(self, state):
        drone_location = {o: [None, None, None] for o in self._objects['aircraft']}
        velocity = {o: None for o in self._objects['aircraft']}        
        
        for k, v in state.items():
            var, objects = self._model.parse(k)
            if var == 'pos-x':
                drone_location[objects[0]][0] = v
            elif var == 'pos-y':
                drone_location[objects[0]][1] = v
            elif var == 'pos-z':
                drone_location[objects[0]][2] = v
            elif var == 'vel':
                velocity[objects[0]] = v

        return {'drone_location': drone_location, 'velocity': velocity}

    def init_canvas(self, figure_size, dpi): 
        # plt.style.use('dark_background')
        fig = plt.figure(figsize=figure_size, dpi=dpi)
        ax = fig.add_subplot(projection='3d')

        ax.axes.set_xlim3d(left=-figure_size[0] // 2, right=figure_size[0] // 2) 
        ax.axes.set_ylim3d(bottom=-figure_size[0] // 2, top=figure_size[0] // 2) 
        ax.axes.set_zlim3d(bottom=-figure_size[0] // 2, top=figure_size[0] // 2)

        fig.subplots_adjust(left=0, right=0.1, bottom=0, top=0.1)

        ax.tick_params(labelsize=100)
        
        # plt.axis('scaled')
        # plt.axis('off')

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

    def render(self, state):
        self.states = state
        self._fig, self._ax = self.init_canvas(self._figure_size, self._dpi)
    
        nonfluent_layout = self.build_nonfluents_layout()
        state_layout = self.build_states_layout(state)

        self._nonfluent_layout = nonfluent_layout
        self._state_layout = state_layout

        for k, v in nonfluent_layout['goal_location'].items():
            self._ax.plot([v[0]], [v[1]], [v[2]],
                          color='seagreen', marker='X', markersize=200)
        
        for k, v in state_layout['drone_location'].items():
            self._ax.plot([v[0]], [v[1]], [v[2]],
                          color='deepskyblue', marker='>', markersize=200)

        img = self.convert2img(self._fig, self._ax)

        self._ax.cla()
        plt.cla()
        plt.close()

        return img
    
    def gen_inter_state(self, beg_state, end_state, steps):
        state_buffer = []
        beg_x = beg_state['xPos']
        beg_y = beg_state['yPos']
        end_x = end_state['xPos']
        end_y = end_state['yPos']

        if (beg_x, beg_y) == (end_x, end_y):
            state_buffer = [beg_state, end_state]
        else:
            for i in range(steps):
                x = beg_x + 1 / steps * (end_x - beg_x) * i
                y = beg_y + 1 / steps * (end_y - beg_y) * i
                state = beg_state.copy()
                state['xPos'] = x
                state['yPos'] = y
                state_buffer.append(state)
            state_buffer.append(end_state)

        return state_buffer
    
