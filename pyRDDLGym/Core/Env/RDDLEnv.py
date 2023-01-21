import copy
import gym
from gym.spaces import Discrete, Dict, Box
import numpy as np
import pygame

from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidNumberOfArgumentsError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLTypeError

from pyRDDLGym.Core.Compiler.RDDLLiftedModel import RDDLLiftedModel
from pyRDDLGym.Core.Debug.Logger import Logger
from pyRDDLGym.Core.Env.RDDLConstraints import RDDLConstraints
from pyRDDLGym.Core.Parser.parser import RDDLParser
from pyRDDLGym.Core.Parser.RDDLReader import RDDLReader
from pyRDDLGym.Core.Simulator.RDDLSimulator import RDDLSimulator
from pyRDDLGym.Visualizer.TextViz import TextVisualizer


class RDDLEnv(gym.Env):
    
    def __init__(self, domain: str, 
                 instance: str=None, 
                 enforce_action_constraints: bool=False,
                 debug: bool=False, 
                 sim_class: RDDLSimulator=RDDLSimulator):
        '''Creates a new gym environment from the given RDDL domain + instance.
        
        :param domain: the RDDL domain
        :param instance: the RDDL instance
        :param enforce_action_constraints: whether to raise an exception if the
        action constraints are violated
        :param debug: whether to log compilation information to a log file
        :param sim_class: the subclass of RDDLSimulator to use as backend for
        simulation (currently supports numpy and Jax)
        '''
        super(RDDLEnv, self).__init__()
        self.enforce_action_constraints = enforce_action_constraints
        
        # read and parse domain and instance
        reader = RDDLReader(domain, instance)
        domain = reader.rddltxt

        # parse RDDL file
        parser = RDDLParser(lexer=None, verbose=False)
        parser.build()
        rddl = parser.parse(domain)
        self.model = RDDLLiftedModel(rddl)
        
        # for logging
        ast = self.model._AST
        logger = Logger(f'{ast.domain.name}_{ast.instance.name}.log') if debug else None
        
        # define the model sampler and bounds    
        self.sampler = sim_class(self.model, logger=logger)
        bounds = RDDLConstraints(self.sampler).bounds

        # set roll-out parameters
        self.horizon = self.model.horizon
        self.discount = self.model.discount
        self.max_allowed_actions = self.model.max_allowed_actions
            
        self.currentH = 0
        self.done = False

        # set default actions
        self.defaultAction = self.model.groundactions()

        # define the actions bounds
        self.actionsranges = self.model.groundactionsranges()
        action_space = Dict()
        for act in self.defaultAction:
            act_range = self.actionsranges[act]
            if act_range in self.model.enum_types:
                action_space[act] = Discrete(len(self.model.objects[act_range]))            
            elif act_range == 'real':
                action_space[act] = Box(low=bounds[act][0],
                                        high=bounds[act][1],
                                        dtype=np.float32)
            elif act_range == 'bool':
                action_space[act] = Discrete(2)
            elif act_range == 'int':
                high = bounds[act][1]
                if high == np.inf:
                    high = np.iinfo(np.int32).max
                low = bounds[act][0]
                if low == -np.inf:
                    low = np.iinfo(np.int32).min
                action_space[act] = Discrete(int(high - low + 1), start=int(low))
            else:
                raise RDDLTypeError(
                    f'Unknown action value type <{act_range}> in environment.')
        self.action_space = action_space

        # define the states bounds
        if self.sampler.isPOMDP:
            search_dict = self.model.groundobserv()
            ranges = self.model.groundobservranges()
        else:
            search_dict = self.model.groundstates()
            ranges = self.model.groundstatesranges()
            
        state_space = Dict()
        for state in search_dict:
            state_range = ranges[state]
            if state_range in self.model.enum_types:
                state_space[state] = Discrete(len(self.model.objects[state_range]))          
            elif state_range == 'real':
                state_space[state] = Box(low=bounds[state][0],
                                         high=bounds[state][1],
                                         dtype=np.float32)
            elif state_range == 'bool':
                state_space[state] = Discrete(2)
            elif state_range == 'int':
                high = bounds[state][1]
                if high == np.inf:
                    high = np.iinfo(np.int32).max
                low = bounds[state][0]
                if low == -np.inf:
                    low = np.iinfo(np.int32).min
                state_space[state] = Discrete(int(high - low + 1), start=int(low))
            else:
                raise RDDLTypeError(
                    f'Unknown state value type <{state_range}> in environment.')
        self.observation_space = state_space

        # set the visualizer
        # the next line should be changed for the default behaviour - TextVix
        self._visualizer = TextVisualizer(self.model)
        self._movie_generator = None
        self.state = None
        self.image = None
        self.window = None
        self.to_render = False
        self.image_size = None

    def set_visualizer(self, viz, movie_gen=None, movie_per_episode=False):
        self._visualizer = viz(self.model)
        self._movie_generator = movie_gen
        self._movie_per_episode = movie_per_episode
        self._movies = 0
        self.to_render = False

    def step(self, actions):
        if self.done:
            return self.state, 0.0, self.done, {}

        # make sure the action length is of currect size
        action_length = len(actions)
        if (action_length > self.max_allowed_actions):
            raise RDDLInvalidNumberOfArgumentsError(
                f'Invalid action, expected at most '
                f'{self.max_allowed_actions} entries, '
                f'but got {action_length}.')
        
        # set full action vector
        # values are clipped to be inside the feasible action space
        clipped_actions = copy.deepcopy(self.defaultAction)
        for act in actions:
            if str(self.action_space[act]) == 'Discrete(2)':
                if self.actionsranges[act] == 'bool':
                    clipped_actions[act] = bool(actions[act])
            else:
                clipped_actions[act] = actions[act]
                
        # check action constraints
        if self.enforce_action_constraints:
            self.sampler.check_action_preconditions(clipped_actions)
        
        # sample next state and reward
        obs, reward, self.done = self.sampler.step(clipped_actions)
        state = self.sampler.states

        # check if the state invariants are satisfied
        if not self.done:
            self.sampler.check_state_invariants()               

        # update step horizon
        self.currentH += 1
        if self.currentH == self.horizon:
            self.done = True

        # for visualization purposes
        self.state = state

        return obs, reward, self.done, {}

    def reset(self):
        self.total_reward = 0
        self.currentH = 0
        obs, self.done = self.sampler.reset()
        self.state = self.sampler.states

        image = self._visualizer.render(self.state)
        if self._movie_generator is not None:
            if self._movie_per_episode:
                self._movie_generator.save_gif(
                    self._movie_generator.env_name + '_' + str(self._movies))
                self._movies += 1
            self._movie_generator.save_frame(image)            
        self.image_size = image.size
        return obs

    def pilImageToSurface(self, pilImage):
        return pygame.image.fromstring(
            pilImage.tobytes(), pilImage.size, pilImage.mode).convert()

    def render(self, to_display=True):
        if self._visualizer is not None:
            image = self._visualizer.render(self.state)
            if to_display:
                if not self.to_render:
                    self.to_render = True
                    pygame.init()
                    self.window = pygame.display.set_mode(
                        (self.image_size[0], self.image_size[1]))
                self.window.fill(0)
                pygameSurface = self.pilImageToSurface(image)
                self.window.blit(pygameSurface, (0, 0))
                pygame.display.flip()
    
            if self._movie_generator is not None:
                self._movie_generator.save_frame(image)
    
        return image
    
    def close(self):
        if self.to_render:
            pygame.display.quit()
            pygame.quit()
    
            if self._movie_generator is not None:
                self._movie_generator.save_gif(
                    self._movie_generator.env_name + '_' + str(self._movies))
                self._movies += 1

    @property
    def numConcurrentActions(self):
        return self.max_allowed_actions
    
    @property
    def non_fluents(self):
        return self.model.groundnonfluents()
