import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from pyRDDLGym.Core.Compiler.RDDLModel import PlanningModel
from pyRDDLGym import Visualizer
from pyRDDLGym.Visualizer.StateViz import StateViz


class MarsRoverVisualizer(StateViz):

    def __init__(self, model: PlanningModel,
                 figure_size=[50, 50],
                 dpi=20,
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
        mineral_locaiton = {o: [None, None, None, None] 
                            for o in self._objects['mineral']}

        # style of fluent_p1
        for k, v in self._nonfluents.items():
            var, objects = self._model.parse(k)
            if var == 'MINERAL-POS-X':
                mineral_locaiton[objects[0]][0] = v
            elif var == 'MINERAL-POS-Y':
                mineral_locaiton[objects[0]][1] = v
            elif var == 'MINERAL-AREA':
                mineral_locaiton[objects[0]][2] = v
            elif var == 'MINERAL-VALUE':
                mineral_locaiton[objects[0]][3] = v

        return {'mineral_location': mineral_locaiton}
    
    def build_states_layout(self, state):
        rover_location = {o: [None, None] for o in self._objects['rover']}
        mineral_harvested = {o: None for o in self._objects['mineral']}
        
        for k, v in state.items():
            var, objects = self._model.parse(k)
            if var == 'pos-x':
                rover_location[objects[0]][0] = v
            elif var == 'pos-y':
                rover_location[objects[0]][1] = v

        for k, v in state.items():
            var, objects = self._model.parse(k)
            if var == 'mineral-harvested':
                mineral_harvested[objects[0]] = v

        return {'mineral_harvested': mineral_harvested,
                'rover_location': rover_location}

    def init_canvas(self, figure_size, dpi):
        fig = plt.figure(figsize=figure_size, dpi=dpi)
        ax = plt.gca()
        plt.xlim([-figure_size[0] // 2, figure_size[0] // 2])
        plt.ylim([-figure_size[1] // 2, figure_size[1] // 2])
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

    def render(self, state):
        self.states = state
        self._fig, self._ax = self.init_canvas(self._figure_size, self._dpi)
    
        nonfluent_layout = self.build_nonfluents_layout()
        state_layout = self.build_states_layout(state)

        self._nonfluent_layout = nonfluent_layout
        self._state_layout = state_layout
        
        max_value = max([v[3] 
                         for k, v in nonfluent_layout['mineral_location'].items()])
        for k, v in nonfluent_layout['mineral_location'].items():
            value = nonfluent_layout['mineral_location'][k][3] / max_value
            if state_layout['mineral_harvested'][k] == False:
                p_point = plt.Circle((v[0], v[1]),
                                     radius=v[2], ec='forestgreen', fc='g',
                                     fill=True, linewidth=10,
                                     alpha=max(min(value, 0.9), 0.1))
            else:
                p_point = plt.Circle((v[0], v[1]),
                                     radius=v[2], ec='forestgreen', linewidth=10,
                                     fill=False, alpha=max(min(value, 0.9), 0.1))
            plt.text(v[0] - 1,
                     v[1],
                     "Value: %s" % self._nonfluent_layout['mineral_location'][k][3],
                     color='black', fontsize=50)
            self._ax.add_patch(p_point)

        for k, v in state_layout['rover_location'].items():
            rover_rec = plt.Rectangle((v[0], v[1]),
                                      0.5,
                                      0.5,
                                      fc='grey', zorder=2)
            self._ax.add_patch(rover_rec)

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
    
    # def animate_buffer(self, states_buffer):

    #     threading.Thread(target=self.animate_buffer).start()

    #     # img_list = [self.render(states_buffer[i]) for i in range(len(states_buffer))]

    #     # img_list[0].save('temp_result.gif', save_all=True,optimize=False, append_images=img_list[1:], loop=0)

    #     def anime(i):
    #         self.render(states_buffer[i])
            
    #     anim = animation.FuncAnimation(self._fig, anime, interval=200)

    #     plt.show()

    #     # plt.show()

    #     # img = self.render(states_buffer[0])
    #     # img.save('./img_folder/0.png')

    #     return
    
