import copy
import gym
from gym.spaces import Discrete, Dict, Box
import numpy as np
import pygame
import os

from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidNumberOfArgumentsError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLTypeError, RDDLLogFolderError

from pyRDDLGym.Core.Compiler.RDDLLiftedModel import RDDLLiftedModel
from pyRDDLGym.Core.Debug.Logger import Logger, SimLogger
from pyRDDLGym.Core.Env.RDDLConstraints import RDDLConstraints
from pyRDDLGym.Core.Env.RDDLEnvSeeder import RDDLEnvSeederFibonacci as RDDLSeeder
from pyRDDLGym.Core.Parser.parser import RDDLParser
from pyRDDLGym.Core.Parser.RDDLReader import RDDLReader
from pyRDDLGym.Core.Simulator.RDDLSimulator import RDDLSimulator
from pyRDDLGym.Visualizer.ChartViz import ChartVisualizer


def _make_dir(simlogname, domain_name, instance_name):
    curpath = os.path.abspath(__file__)
    for _ in range(3):
        curpath = os.path.split(curpath)[0]
    path = os.path.join(curpath, 'Logs', simlogname, domain_name)
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except Exception as e:
            if not isinstance(e, FileExistsError):
                raise RDDLLogFolderError(
                    f'Could not create log folder for domain {domain_name} '
                    f'of method {simlogname} at path {path}')
    path = os.path.join(path, instance_name)
    return path


class RDDLEnv(gym.Env):
    '''A gym environment class for RDDL domains.'''
    
    def __init__(self, domain: str,
                 instance: str=None,
                 enforce_action_constraints: bool=False,
                 enforce_action_count_non_bool: bool=True,
                 debug: bool=False,
                 log: bool=False,
                 simlogname: str=None,
                 backend: RDDLSimulator=RDDLSimulator,
                 backend_kwargs: Dict={},
                 seeds: list=None):
        '''Creates a new gym environment from the given RDDL domain + instance.
        
        :param domain: the RDDL domain
        :param instance: the RDDL instance
        :param enforce_action_constraints: whether to raise an exception if the
        action constraints are violated
        :param enforce_action_count_non_bool: whether to include non-bool actions
        in check that number of nondef actions don't exceed max-nondef-actions
        :param debug: whether to log compilation information to a log file
        :param log: whether to log simulation data to file
        :param simlogname: name of the file to save simulation data log
        :param backend: the subclass of RDDLSimulator to use as backend for
        simulation (currently supports numpy and Jax)
        :param backend_kwargs: dictionary of additional named arguments to
        pass to backend (must not include logger)
        :param seeds: list of seeds for the cyclic iterator. Will be seeded in the reset function.
        '''
        super(RDDLEnv, self).__init__()
        self.domain_text = domain
        self.instance_text = instance
        self.enforce_action_constraints = enforce_action_constraints
        self.enforce_action_count_non_bool = enforce_action_count_non_bool

        # time budget for applications limiting time on episodes.
        # hardcoded so cannot be changed externally for the purpose of the competition.
        # TODO: add it to the API after the competition
        self.budget = 240
        self.seeds = RDDLSeeder(a1=1)

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
        self.trial = 0
        log_fname = f'{ast.domain.name}_{ast.instance.name}'
        logger = Logger(f'{log_fname}_debug.log') if debug else None
        self.simlogger = None
        if log:
            simlog_fname = _make_dir(simlogname, ast.domain.name, ast.instance.name)
            self.simlogger = SimLogger(f'{simlog_fname}_log.csv')
        if self.simlogger:
            self.simlogger.clear(overwrite=False)
        
        # define the model sampler and bounds    
        self.sampler = backend(self.model, logger=logger, **backend_kwargs)
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
            if act_range in self.model.enums:
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
            if state_range in self.model.enums:
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
        self._visualizer = ChartVisualizer(self.model)
        self._movie_generator = None
        self.state = None
        self.image = None
        self.window = None
        self.to_render = False
        self.image_size = None
    
    def seed(self, seed=None):
        # super(RDDLEnv, self).seed(seed)
        self.sampler.seed(seed)
        return [seed]
    
    def set_visualizer(self, viz, movie_gen=None, movie_per_episode=False, **viz_kwargs):
        if viz is not None:
            self._visualizer = viz(self.model, **viz_kwargs)
        self._movie_generator = movie_gen
        self._movie_per_episode = movie_per_episode
        self._movies = 0
        self.to_render = False

    def step(self, actions):
        if self.done:
            return self.state, 0.0, self.done, {}

        # make sure the action length is of currect size
        if self.enforce_action_count_non_bool:
            action_length = len(actions)
        else:
            action_length = len([
                action 
                for action in actions 
                if self.model.groundactionsranges()[action] == 'bool'
            ])
        if action_length > self.max_allowed_actions:
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

        # log to file
        if self.simlogger is not None:
            self.simlogger.log(
                obs, clipped_actions, reward, self.done, self.currentH)
        
        # update step horizon
        self.currentH += 1
        if self.currentH == self.horizon:
            self.done = True

        # for visualization purposes
        self.state = state

        return obs, reward, self.done, {}

    def reset(self, seed=None):
        self.total_reward = 0
        self.currentH = 0
        obs, self.done = self.sampler.reset()
        self.state = self.sampler.states

        image = self._visualizer.render(self.state)
        if self._movie_generator is not None:
            if self._movie_per_episode:
                self._movie_generator.save_animation(
                    self._movie_generator.env_name + '_' + str(self._movies))
                self._movies += 1
            self._movie_generator.save_frame(image)            
        self.image_size = image.size

        if seed is not None:
            self.seed(seed)
        else:
            seed = self.seeds.Next()
            if seed is not None:
                self.seed(seed)

        # Logging
        if self.simlogger:
            self.trial += 1
            text = '######################################################\n'
            if seed is not None:
                text += f'New Trial, seed={seed}\n'
            else:
                text += f'New Trial\n'
            text += '######################################################'
            self.simlogger.log_free(text)

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
        if self.simlogger:
            self.simlogger.close()
                        
        if self.to_render:
            pygame.display.quit()
            pygame.quit()
    
            if self._movie_generator is not None:
                self._movie_generator.save_animation(
                    self._movie_generator.env_name + '_' + str(self._movies))
                self._movies += 1

    @property
    def numConcurrentActions(self):
        return self.max_allowed_actions
    
    @property
    def non_fluents(self):
        return self.model.groundnonfluents()

    @property
    def Budget(self):
        return self.budget
