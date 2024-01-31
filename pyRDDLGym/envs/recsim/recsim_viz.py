import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz


class RecSimVisualizer(BaseViz):

    def __init__(self, model: RDDLPlanningModel,
                 figure_size=(5, 10),
                 dpi=100,
                 fontsize=10,
                 display=False) -> None:
        self._model = model
        self._nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._objects = model.type_to_objects
        self._figure_size = figure_size
        self._display = display
        self._dpi = dpi
        self._fontsize = fontsize
        self._interval = 10
        self._object_layout = None
        self._fig, self._ax = None, None
        self._data = None
        self._img = None

        _horizon = 10
        self._interp_steps = 25
        self._creator_disp = 64.0
        self._creator_fan_out = 2
        self._num_creator_clusters = 40
        self._num_topics = 2
        self._creator_boost_cap = 1.2
        self._iter = 0

    def build_nonfluents_layout(self):
        space_dim = len(self._objects['feature'])
        self._feature_index = {
            f: ind for ind, f in enumerate(self._objects['feature'])}
        if space_dim != 2:
            raise RuntimeError('This visualizer can only run in 2d (two features).')
        
        self._num_users = len(self._objects['consumer'])
        creators = self._objects['provider'].copy()
        creators.remove('pn')
        self._user_index = {
            c: ind for ind, c in enumerate(self._objects['consumer'])}
        self._num_creators = len(creators)
        self._creator_index = {p: ind for ind, p in enumerate(creators)}
        self._creator_means = np.zeros((self._num_creators, space_dim))
        self._user_interests = np.zeros((self._num_users, space_dim))
        
        for key, value in self._nonfluents.items():
            var, objects = RDDLPlanningModel.parse_grounded(key)
            if var == 'CONSUMER-AFFINITY':
                self._user_interests[
                    self._user_index[objects[0]]][
                        self._feature_index[objects[1]]] = value
            elif var == 'PROVIDER-COMPETENCE':
                if objects[0] == 'pn': continue
                self._creator_means[
                    self._creator_index[objects[0]]][
                        self._feature_index[objects[1]]] = value
        self._user_plot = None
        self._creator_plot = None
        self._creator_util_scatter = None
        return self._nonfluents
    
    def build_states_layout(self, states):
        self._user_satisfaction = np.zeros(self._num_users)
        self._creator_satisfaction = np.zeros(self._num_creators)
        for key, value in states.items():
            var, objects = RDDLPlanningModel.parse_grounded(key)
            if var == 'consumer-satisfaction':
                self._user_satisfaction[self._user_index[objects[0]]] = value
            elif var == 'provider-satisfaction':
                if objects[0] == 'pn': continue
                self._creator_satisfaction[self._creator_index[objects[0]]] = value
        return states
    
    def init_canvas(self, figure_size, dpi):
        width_box = 0.15
        rect_scatter = [0, 0, 1, 0.95]
        rect_creator_util = [0.62, 0.95, width_box, 0.05]
        rect_reset = [0.51, 0.95, width_box * 0.6, 0.05]
        rect_regenerate = [0.41, 0.95, width_box * 0.6, 0.05]
        rect_text = [0.01, 0.95, width_box, 0.05]
        rect_fair = [0.8, 0.95, width_box, 0.05]

        self._fig = plt.figure()
        self._ax_scatter = plt.axes(rect_scatter)
        self._ax_creator_util = plt.axes(rect_creator_util)
        self._ax_text = plt.axes(rect_text)
        self._axes = [self._ax_scatter, self._ax_creator_util, self._ax_text]

        return None, None

    def build_object_layout(self):
        return {'nonfluents_layout': self.build_nonfluents_layout(),
                'states_layout': self.build_states_layout()}

    def convert2img(self, fig, ax):        
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = Image.fromarray(data)
        self._data = data
        self._img = img
        return img

    def render(self, state):
        self.states = state

        self.init_canvas(self._figure_size, self._dpi)
        nonfluent_layout = self.build_nonfluents_layout()
        states_layout = self.build_states_layout(state)

        reward = self._user_satisfaction + 0.1
        rel_sizes = self._creator_satisfaction + 0.1
        rel_sizes = rel_sizes / np.sum(rel_sizes)
        if self._user_plot is None:
            self._user_plot = self._ax_scatter.scatter(
                self._user_interests[:, 0],
                self._user_interests[:, 1],
                s=np.maximum(reward, 0.),
                c=20 * np.sqrt(np.maximum(reward, 0.)),
                cmap='plasma',
                edgecolors='black')
        self._creator_plot = self._ax_scatter.scatter(
            self._creator_means[:, 0],
            self._creator_means[:, 1],
            s=200 * rel_sizes,
            c='red')
        self._ax_scatter.axis([
            np.min(self._user_interests[:, 0]) - 2,
            np.max(self._user_interests[:, 0]) + 2,
            np.min(self._user_interests[:, 1]) - 2,
            np.max(self._user_interests[:, 1]) + 2
        ])

        self._creator_util_scatter, = self._ax_creator_util.plot(
            range(self._num_creators), 20 * np.sort(rel_sizes), color='red')

        # plot design
        for ax in self._axes:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        # state_layout = self.build_states_layout(state)
        # text_layout = {'state': state_layout}
        # text_str = pprint.pformat(text_layout)[1:-1]
        # self._ax.text(self._interval*0.5, self._figure_size[1]*self._interval*0.95, text_str, 
        #         horizontalalignment='left', verticalalignment='top', wrap=True, fontsize = self._fontsize)
        
        img = self.convert2img(self._fig, None)
        self._iter += 1

        plt.close()

        return img
    
