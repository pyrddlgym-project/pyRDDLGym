import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete
import numpy as np
import os
import pygame
import typing

from pyRDDLGym.core.compiler.model import RDDLLiftedModel
from pyRDDLGym.core.constraints import RDDLConstraints
from pyRDDLGym.core.debug.exception import (
    RDDLEpisodeAlreadyEndedError,
    RDDLLogFolderError,
    RDDLTypeError
)
from pyRDDLGym.core.debug.logger import Logger, SimLogger
from pyRDDLGym.core.parser.parser import RDDLParser
from pyRDDLGym.core.parser.reader import RDDLReader
from pyRDDLGym.core.seeding import RDDLEnvSeeder, RDDLEnvSeederFibonacci
from pyRDDLGym.core.simulator import RDDLSimulator
from pyRDDLGym.core.visualizer.chart import ChartVisualizer
from pyRDDLGym.core.visualizer.heatmap import HeatmapVisualizer
from pyRDDLGym.core.visualizer.text import TextVisualizer


def _make_dir(log_path):
    root_path = os.path.dirname(log_path)
    if not os.path.exists(root_path):
        try:
            os.makedirs(root_path)
        except Exception as e:
            if not isinstance(e, FileExistsError):
                raise RDDLLogFolderError(
                    f'Could not create folder at path {root_path}.')
    return log_path

    
class RDDLEnv(gym.Env):
    '''A gym environment class for RDDL domains.'''
    
    def __init__(self, domain: str,
                 instance: str,
                 enforce_action_constraints: bool=False,
                 enforce_action_count_non_bool: bool=True,
                 vectorized: bool=False,
                 debug: bool=False,
                 log_path: str=None,
                 backend: RDDLSimulator=RDDLSimulator,
                 backend_kwargs: typing.Dict={},
                 seeds: RDDLEnvSeeder=RDDLEnvSeederFibonacci()):
        '''Creates a new gym environment from the given RDDL domain + instance.
        
        :param domain: the RDDL domain
        :param instance: the RDDL instance
        :param enforce_action_constraints: whether to raise an exception if the
        action constraints are violated
        :param enforce_action_count_non_bool: whether to include non-bool actions
        in check that number of nondef actions don't exceed max-nondef-actions
        :param vectorized: whether actions and states are represented as
        dictionaries of numpy arrays (if True), or as dictionaries of scalars
        :param debug: whether to log compilation information to a log file
        :param log_path: absolute path to file where simulation log is saved,
        excluding the file extension, None means no logging
        :param backend: the subclass of RDDLSimulator to use as backend for
        simulation (currently supports numpy and Jax)
        :param backend_kwargs: dictionary of additional named arguments to
        pass to backend (must not include logger)
        :param seeds: an instance of RDDLEnvSeeder for generating RNG seeds
        '''
        super(RDDLEnv, self).__init__()
        
        self.domain_text = domain
        self.instance_text = instance
        self.enforce_action_constraints = enforce_action_constraints
        self.enforce_count_non_bool = enforce_action_count_non_bool
        self.vectorized = vectorized
        
        # read and parse domain and instance
        reader = RDDLReader(domain, instance)
        domain = reader.rddltxt
        parser = RDDLParser(lexer=None, verbose=False)
        parser.build()
        rddl = parser.parse(domain)
        
        # define the RDDL model
        self.model = RDDLLiftedModel(rddl)
        self.horizon = self.model.horizon
        self.discount = self.model.discount
        self.max_allowed_actions = self.model.max_allowed_actions 
                
        # for logging compilation data
        log_fname = f'{self.model.domain_name}_{self.model.instance_name}'
        logger = Logger(f'{log_fname}_debug.log') if debug else None
        self.logger = logger
        
        # for logging simulation data
        self.simlogger = None
        if log_path is not None and log_path:
            new_log_path = _make_dir(log_path)
            self.simlogger = SimLogger(f'{new_log_path}_log.csv')
            self.simlogger.clear(overwrite=False)
        
        # define the simulation backend  
        self.sampler = backend(self.model,
                               logger=logger,
                               keep_tensors=self.vectorized,
                               **backend_kwargs)
        
        # compute the bounds on fluents from the constraints
        constraints = RDDLConstraints(self.sampler, vectorized=self.vectorized)
        self._bounds = constraints.bounds
        self._shapes = {var: np.shape(values[0]) 
                        for (var, values) in self._bounds.items()}
        
        # construct the gym observation space
        if self.sampler.is_pomdp:
            state_ranges = self.model.observ_ranges
        else:
            state_ranges = self.model.state_ranges
        if not self.vectorized:
            state_ranges = self.model.ground_vars_with_value(state_ranges)   
             
        self.observation_space = self._rddl_to_gym_bounds(state_ranges)
        
        # construct the gym action space      
        if self.vectorized:
            self._action_ranges = self.model.action_ranges
            self._noop_actions = self.sampler.noop_actions
        else:
            self._action_ranges = self.sampler.grounded_action_ranges
            self._noop_actions = self.sampler.grounded_noop_actions
            
        self.action_space = self._rddl_to_gym_bounds(self._action_ranges)
        
        # set the visualizer
        self._visualizer = ChartVisualizer(self.model)
        self._movie_generator = None
        self.state = None
        self.image = None
        self.window = None
        self.to_render = False
        self.image_size = None
        
        # set roll-out parameters           
        self.trial = 0
        self.timestep = 0
        self.done = False
        self.seeds = iter(seeds)
            
    def _rddl_to_gym_bounds(self, ranges):
        result = Dict()
        for (var, prange) in ranges.items():
            shape = self._shapes[var]
            
            # enumerated values converted to Discrete space
            if prange in self.model.enum_types:
                num_objects = len(self.model.type_to_objects[prange])
                if self.vectorized:
                    result[var] = Box(0, num_objects - 1, shape=shape, dtype=np.int32)
                else:
                    result[var] = Discrete(num_objects) 
            
            # real values define a box
            elif prange == 'real':
                low, high = self._bounds[var]
                result[var] = Box(low, high, dtype=np.float32)
            
            # boolean values converted to Discrete space
            elif prange == 'bool':
                if self.vectorized:
                    result[var] = Box(0, 1, shape=shape, dtype=np.int32)
                else:
                    result[var] = Discrete(2)
            
            # integer values converted to Discrete space
            elif prange == 'int':
                low, high = self._bounds[var]
                low = np.maximum(low, np.iinfo(np.int32).min)
                high = np.minimum(high, np.iinfo(np.int32).max)
                if self.vectorized:
                    result[var] = Box(low, high, shape=shape, dtype=np.int32)
                else:
                    result[var] = Discrete(int(high - low + 1), start=int(low))
            
            # unknown type
            else:
                raise RDDLTypeError(
                    f'Type <{prange}> of fluent <{var}> is not valid, '
                    f'must be an enumerated or primitive type (real, int, bool).')
                
        return result
    
    def seed(self, seed=None):
        self.sampler.seed(seed)
        return [seed]
    
    VISUALIZER_CLASSES = {
        'chart': ChartVisualizer,
        'heatmap': HeatmapVisualizer,
        'text': TextVisualizer
    }
    
    def set_visualizer(self, viz, movie_gen=None, movie_per_episode=False, **viz_kwargs):
        if viz is None:
            self._visualizer = ChartVisualizer(self.model)
        else:
            if isinstance(viz, str):
                viz = self.VISUALIZER_CLASSES.get(viz.lower(), None)
                if viz is None:
                    raise ValueError(
                        f'Visualizer type {viz} is invalid, '
                        f'must be one of {list(self.VISUALIZER_CLASSES.keys())}')
            self._visualizer = viz(self.model, **viz_kwargs)
        self._movie_generator = movie_gen
        self._movie_per_episode = movie_per_episode
        self._movies = 0
        self.to_render = False
    
    def _fix_boolean_actions(self, actions):
        fixed_actions = self._noop_actions.copy()
        for (var, values) in actions.items():
            if self._action_ranges.get(var, '') == 'bool':
                if np.shape(values):
                    fixed_actions[var] = np.asarray(values, dtype=bool)
                else:
                    fixed_actions[var] = bool(values)
            else:
                fixed_actions[var] = values
        return fixed_actions

    def step(self, actions):
        sampler = self.sampler
        
        if self.done:
            raise RDDLEpisodeAlreadyEndedError(
                'The step() function has been called even though the '
                'current episode has terminated or truncated: please call reset().')
            
        # fix actions and check constraints
        actions = self._fix_boolean_actions(actions)
        sampler.check_default_action_count(actions, self.enforce_count_non_bool)
        if self.enforce_action_constraints:
            sampler.check_action_preconditions(actions, silent=False)
        
        # sample next state and reward
        obs, reward, terminated = sampler.step(actions)
        self.state = sampler.states
            
        # produce array outputs for vectorized option
        if self.vectorized:
            obs = {var: np.atleast_1d(value) for (var, value) in obs.items()}
            
        # check if the state invariants are satisfied
        truncated = not sampler.check_state_invariants(silent=True)
        self.done = terminated or truncated
            
        # log to file
        if self.simlogger is not None:
            if self.vectorized:
                log_obs = self.model.ground_vars_with_values(obs)
                log_action = self.model.ground_vars_with_values(actions)
            else:
                log_obs = obs
                log_action = actions
            self.simlogger.log(log_obs, log_action, reward, self.done, self.timestep)
        
        # update step horizon
        self.timestep += 1
        if self.timestep == self.horizon:
            truncated = True
            self.done = True
        
        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        sampler = self.sampler
        
        # reset counters and internal state
        obs, terminated = sampler.reset()
        self.done = terminated
        self.state = sampler.states
        self.trial += 1
        self.timestep = 0
        
        # produce array outputs for vectorized option
        if self.vectorized:
            obs = {var: np.atleast_1d(value) for (var, value) in obs.items()}
            
        # update movie generator
        if self._movie_generator is not None and self._visualizer is not None:
            if self.vectorized:
                state = self.model.ground_vars_with_values(self.state)
            else:
                state = self.state
                
            image = self._visualizer.render(state)
            self.image_size = image.size
            if self._movie_per_episode:
                self._movie_generator.save_animation(
                    self._movie_generator.env_name + '_' + str(self._movies))
                self._movies += 1
            self._movie_generator.save_frame(image)            
        
        # update random generator seed
        if seed is not None:
            self.seed(seed)
        else:
            seed = next(self.seeds)
            if seed is not None:
                self.seed(seed)

        # logging
        if self.simlogger:
            text = (f'######################################################\n'
                    f'New Trial, seed={seed}\n'
                    f'######################################################')
            self.simlogger.log_free(text)
            
        return obs, {}

    def pilImageToSurface(self, pilImage):
        return pygame.image.fromstring(
            pilImage.tobytes(), pilImage.size, pilImage.mode).convert()

    def render(self, to_display=True):
        image = None
        if self._visualizer is not None:
            if self.vectorized:
                state = self.model.ground_vars_with_values(self.state)
            else:
                state = self.state
            
            # update the screen
            image = self._visualizer.render(state)
            self.image_size = image.size
            if to_display:
                if not self.to_render:
                    self.to_render = True
                    pygame.init()
                    self.window = pygame.display.set_mode(
                        (self.image_size[0], self.image_size[1]))
                self.window.fill(0)
                pygameSurface = self.pilImageToSurface(image)
                self.window.blit(pygameSurface, (0, 0))
                pygame.display.set_caption(self.model.instance_name)
                pygame.display.flip()
                
                # prevents the window from freezing up midway
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pass
            
            # capture frame to disk
            if self._movie_generator is not None:
                self._movie_generator.save_frame(image)
    
        return image
    
    def close(self):
        if self.simlogger:
            self.simlogger.close()
        
        # close rendering and save animation  
        if self.to_render:
            pygame.display.quit()
            pygame.quit()    
            
            # prepare the animation from captured frames
            if self._movie_generator is not None:
                self._movie_generator.save_animation(
                    self._movie_generator.env_name + '_' + str(self._movies))
                self._movies += 1
