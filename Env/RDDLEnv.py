import copy
import sys

import gym
from gym.spaces import Discrete, Dict, Box
import numpy as np
# from PIL import ImageDraw, ImageTk
# import matplotlib.pyplot as plt
import tkinter
import pygame

import Simulator.RDDLSimulator
from Parser import parser as parser
from Parser import RDDLReader as RDDLReader
import Grounder.RDDLGrounder as RDDLGrounder
from Simulator.RDDLSimulator import RDDLSimulatorWConstraints

from Visualizer.TextViz import TextVisualizer

class RDDLEnv(gym.Env):
    def __init__(self, domain, instance=None, is_grounded=False):
        super(RDDLEnv, self).__init__()

        # max allowed action value
        self.BigM = 100

        # read and parser domain and instance
        if instance is None:
            MyReader = RDDLReader.RDDLReader(domain)
        else:
            MyReader = RDDLReader.RDDLReader(domain, instance)
        domain = MyReader.rddltxt

        # build parser - built in lexer, non verbose
        MyRDDLParser = parser.RDDLParser(None, False)
        MyRDDLParser.build()

        # parse RDDL file
        rddl_ast = MyRDDLParser.parse(domain)

        # ground domain
        if is_grounded == True:
            grounder = RDDLGrounder.RDDLGroundedGrounder(rddl_ast)
        else:
            grounder = RDDLGrounder.RDDLGrounder(rddl_ast)
        self.model = grounder.Ground()

        # define the model sampler
        self.sampler = RDDLSimulatorWConstraints(self.model, max_bound=self.BigM)

        # set the horizon
        self.horizon = self.model.horizon
        self.currentH = 0

        # set the discount factor
        self.discount = self.model.discount

        # set the number of concurrent actions allowed
        self._NumConcurrentActions = self.model.max_allowed_actions

        # set default actions dic
        self.defaultAction = copy.deepcopy(self.model.actions)

        # define the actions bounds
        action_space = Dict()
        # action_space_range = {}
        for act in self.model.actions:
            range = self.model.actionsranges[act]
            if range == 'real':
                if act in self.sampler.bounds:
                    action_space[act] = Box(low=self.sampler.bounds[act][0], high=self.sampler.bounds[act][1],
                                            dtype=np.float32)
                else:
                    action_space[act] = Box(low=-self.BigM, high=self.BigM, dtype=np.float32)
            elif range == 'bool':
                action_space[act] = Discrete(2)
            elif range == 'int':
                action_space[act] = Discrete(2*self.BigM + 1 ,start = -self.BigM)
            else:
                raise Exception("unknown action range in gym environment init")
            # action_space_range[act] = range

        self.action_space = action_space
        # self.action_space_range = action_space_range

        # define the states bounds
        state_space = Dict()
        for state in self.model.states:
            range = self.model.statesranges[state]
            if range == 'real':
                if state in self.sampler.bounds:
                    state_space[state] = Box(low=self.sampler.bounds[state][0], high=self.sampler.bounds[state][1],
                                             dtype=np.float32)
                else:
                    state_space[state] = Box(low=-self.BigM, high=self.BigM, dtype=np.float32)
            elif range == 'bool':
                state_space[state] = Discrete(2)
            elif range == 'int':
                state_space[state] = Discrete(2 * self.BigM + 1, start=-self.BigM)
            else:
                raise Exception("unknown state range in gym environment init")
        self.observation_space = state_space

        # TODO
        # set the visualizer, the next line should be changed for the default behaviour - TextVix
        self._visualizer = TextVisualizer(self.model)
        self.state = None
        self.image = None
        # self.window = tkinter.Tk()
        self.window = None
        self.to_render = False
        self.image_size = None
        # pygame.init()

    def set_visualizer(self, viz):
        # set the vizualizer with self.model
        self._visualizer = viz(self.model)

    def step(self, at):

        # make sure the action length is of currect size
        action_length = len(at)
        if (action_length > self._NumConcurrentActions):
            raise Exception(
                "Invalid action, expected maximum of {} entries, {} were were given".format(self._NumConcurrentActions,
                                                                                            action_length))

        # set full action vector, values are clipped to be inside the feasible action space
        action = copy.deepcopy(self.defaultAction)
        for act in at:
            action[act] = self.clip(at[act], self.sampler.bounds[act][0], self.sampler.bounds[act][1])

        # sample next state and reward
        state = self.sampler.sample_next_state(action)
        reward = self.sampler.sample_reward()

        # TODO: The following chunk of code should be removed and replaced only by self.sampler.check_state_invariants()

        # check if the state is within the invariant constraints
        for st in state:
            if self.model.statesranges[st] == 'real':
                state[st] = self.clip(state[st], self.sampler.bounds[st][0], self.sampler.bounds[st][1])

        self.sampler.update_state(state)

        # check for non-linear constraint violation
        try:
            self.sampler.check_state_invariants()
        except Exception:
            if isinstance(sys.exc_info()[1], Simulator.RDDLSimulator.RDDLRuntimeError):
                print("WARNING:",sys.exc_info()[1])
            else:
                raise Exception("Error in state constraint validation").with_traceback(sys.exc_info()[2])

        # update step horizon
        self.currentH += 1
        if self.currentH == self.horizon:
            done = True
        else:
            done = False

        # for visualization purposes
        self.state = state

        return state, reward, done, {}

    def reset(self):
        self.total_reward = 0
        self.currentH = 0
        self.state = self.sampler.reset_state()

        # if self.to_render:
        image = self._visualizer.render(self.state)
        self.image_size = image.size
            # self.window = pygame.display.set_mode((image.size[0], image.size[1]))
        # self.image = None
        return self.state

    def pilImageToSurface(self, pilImage):
        return pygame.image.fromstring(
            pilImage.tobytes(), pilImage.size, pilImage.mode).convert()

    def render(self):
        if self._visualizer is not None:
            if not self.to_render:
                self.to_render = True
                pygame.init()
                self.window = pygame.display.set_mode((self.image_size[0], self.image_size[1]))
            image = self._visualizer.render(self.state)
            self.window.fill(0)
            pygameSurface = self.pilImageToSurface(image)
            self.window.blit(pygameSurface, (0, 0))
            pygame.display.flip()
        return image
            # return image
            # this_images = pygame.image.fromstring(image.data, image.size, image.mode)

            # self.window.geometry('%dx%d' % (image.size[0], image.size[1]))
            # tkpi = ImageTk.PhotoImage(image)
            # label_image = tkinter.Label(self.window, image=tkpi)
            # label_image.pack()
            # label_image.place(x=0, y=0, width=image.size[0], height=image.size[1])
            # self.window.title("title")
            # self.window.mainloop()
            # if self.image is not None:
            #     self.image.destroy()
            # self.image = label_image



            # if self.image is not None:
            #     plt.close()
                # self.image.close()
            # self.image = self._visualizer.render(self.state)
            # plt.close()
            # plt.imshow(self.image)
            # plt.show()
            # self.image.show()
            # self.image.close()


    @property
    def NumConcurrentActions(self):
        return self._NumConcurrentActions

    def clip(self, val, low, high):
        return max(min(val, high), low)