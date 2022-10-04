import copy
import sys
import warnings

import gym
from gym.spaces import Discrete, Dict, Box
import numpy as np

import Simulator.RDDLSimulator
from Parser import parser as parser
from Parser import RDDLReader as RDDLReader
import Grounder.RDDLGrounder as RDDLGrounder
from Simulator.RDDLSimulator import RDDLSimulatorWConstraints

class RDDLEnv(gym.Env):
    def __init__(self, domain, instance=None):
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
        grounder = RDDLGrounder.RDDLGroundedGrounder(rddl_ast)
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
        self.action_space = action_space

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
        self._visualizer = None
        self.state = None

    def set_visualizer(self, viz):
        # set the vizualizer with self.model
        self._visualizer = viz

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
        return self.sampler.reset_state()

    def render(self):
        if self._visualizer is not None:
            pass

    @property
    def NumConcurrentActions(self):
        return self._NumConcurrentActions

    def clip(self, val, low, high):
        return max(min(val, high), low)