import copy

import gym
from gym.spaces import Discrete, Dict, Box
import numpy as np

from Parser import parser as parser
from Parser import RDDLReader as RDDLReader
import Grounder.RDDLGrounder as RDDLGrounder
from Simulator.RDDLSimulator import RDDLSimulator

class RDDLEnv(gym.Env):
    def __init__(self, domain, instance=None):
        super(RDDLEnv, self).__init__()

        # max allowed action value
        self.BigM = 10

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
        grounder = RDDLGrounder.RDDLGroundedGrounder(rddl_ast)
        self.model = grounder.Ground()

        # define the model sampler
        self.sampler = RDDLSimulator(self.model)

        # set the horizon
        self.horizon = self.model.horizon
        self.currentH = 0

        # set the discount factor
        self.discount = self.model.discount

        # set the number of concurrent actions allowed
        self._NumConcurrentActions = self.model.max_allowed_actions

        # set default actions dic
        self.defaultAction = copy.deepcopy(self.model.actions)

        #TODO
        # define the action correct bounds as specified in the RDDL file
        action_space = Dict()
        for act in self.model.actions:
            range = self.model.actionsranges[act]
            if range == 'real':
                action_space[act] = Box(low=-self.BigM, high=self.BigM, dtype=np.float32)
            elif range == 'bool':
                action_space[act] = Discrete(2)
            elif range == 'int':
                action_space[act] = Discrete(2*self.BigM + 1 ,start = -self.BigM)
            else:
                raise Exception("unknown action range in gym environment init")
        self.action_space = action_space

        #TODO
        # define the observation space correct bounds as specified in the RDDL file
        state_space = Dict()
        for state in self.model.states:
            range = self.model.statesranges[state]
            if range == 'real':
                state_space[state] = Box(low=-self.BigM, high=self.BigM, dtype=np.float32)
            elif range == 'bool':
                state_space[state] = Discrete(2)
            elif range == 'int':
                state_space[state] = Discrete(2 * self.BigM + 1, start=-self.BigM)
            else:
                raise Exception("unknown state range in gym environment init")
        self.observation_space = state_space

        # TODO
        # set the visualizer


    def step(self, at):
        action_length = len(at)
        if (action_length > self._NumConcurrentActions):
            raise Exception(
                "Invalid action, expected maximum of {} entries, {} were were given".format(self._NumConcurrentActions,
                                                                                            action_length))

        action = copy.deepcopy(self.defaultAction)
        for act in at:
            action[act] = at[act]
        state = self.sampler.sample_next_state(action)
        reward = self.sampler.sample_reward()
        self.sampler.update_state()
        self.currentH += 1
        if self.currentH == self.horizon:
            done = True
        else:
            done = False
        return state, reward, done, {}

    def reset(self):
        self.total_reward = 0
        self.currentH = 0
        self.state = self.sampler.reset_state()
        return self.state

    def render(self):
        pass

    @property
    def NumConcurrentActions(self):
        return self._NumConcurrentActions
