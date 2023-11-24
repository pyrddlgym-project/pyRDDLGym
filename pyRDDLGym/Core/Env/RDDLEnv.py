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
from pyRDDLGym.Core.Env.RDDLEnvSeeder import RDDLEnvSeeder, RDDLEnvSeederFibonacci
from pyRDDLGym.Core.Parser.parser import RDDLParser
from pyRDDLGym.Core.Parser.RDDLReader import RDDLReader
from pyRDDLGym.Core.Simulator.RDDLSimulator import RDDLSimulator
from pyRDDLGym.Visualizer.ChartViz import ChartVisualizer


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
    
    @staticmethod
    def build(env_info, env: str, **env_kwargs):
        env = RDDLEnv(domain=env_info.get_domain(),
                      instance=env_info.get_instance(env),
                      **env_kwargs)
        env.set_visualizer(env_info.get_visualizer())
        return env

    def __init__(self, domain: str,
                 instance: str=None,
                 enforce_action_constraints: bool=False,
                 enforce_action_count_non_bool: bool=True,
                 debug: bool=False,
                 log_path: str=None,
                 backend: RDDLSimulator=RDDLSimulator,
                 backend_kwargs: Dict={},
                 seeds: RDDLEnvSeeder=RDDLEnvSeederFibonacci()):
        '''Creates a new gym environment from the given RDDL domain + instance.
        
        :param domain: the RDDL domain
        :param instance: the RDDL instance
        :param enforce_action_constraints: whether to raise an exception if the
        action constraints are violated
        :param enforce_action_count_non_bool: whether to include non-bool actions
        in check that number of nondef actions don't exceed max-nondef-actions
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
        self.enforce_action_count_non_bool = enforce_action_count_non_bool

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
        ast = self.model._AST
        log_fname = f'{ast.domain.name}_{ast.instance.name}'
        logger = Logger(f'{log_fname}_debug.log') if debug else None
        self.logger = logger
        
        # for logging simulation data
        self.simlogger = None
        if log_path is not None and log_path:
            new_log_path = _make_dir(log_path)
            self.simlogger = SimLogger(f'{new_log_path}_log.csv')
            self.simlogger.clear(overwrite=False)
        
        # define the simulation backend  
        self.sampler = backend(self.model, logger=logger, **backend_kwargs)
        self.vectorized = self.sampler.keep_tensors
        
        # impute the bounds on fluents from the constraints
        self.bounds = RDDLConstraints(self.sampler).bounds
        
        if self.sampler.isPOMDP:
            self.statesranges = self.model.groundobservranges()
        else:
            self.statesranges = self.model.groundstatesranges()
        self.actionsranges = self.model.groundactionsranges()
        self.default_actions = self.model.groundactions()
            
        self.action_space = self._rddl_to_gym_bounds(self.actionsranges)        
        self.observation_space = self._rddl_to_gym_bounds(self.statesranges)

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
        self.currentH = 0
        self.done = False
        self.seeds = iter(seeds)
    
    def _rddl_to_gym_bounds(self, ranges):
        result = Dict()
        for (var, prange) in ranges.items():
            
            # enumerated values converted to Discrete space
            if prange in self.model.enums:
                result[var] = Discrete(len(self.model.objects[prange])) 
            
            # real values define a box
            elif prange == 'real':
                low, high = self.bounds[var]
                result[var] = Box(low=low, high=high, dtype=np.float32)
            
            # boolean values converted to Discrete space
            elif prange == 'bool':
                result[var] = Discrete(2)
            
            # integer values converted to Discrete space
            elif prange == 'int':
                low, high = self.bounds[var]
                if high == np.inf:
                    high = np.iinfo(np.int32).max
                if low == -np.inf:
                    low = np.iinfo(np.int32).min
                result[var] = Discrete(int(high - low + 1), start=int(low))
            
            # unknown type
            else:
                raise RDDLTypeError(
                    f'Type <{prange}> of fluent <{var}> is not valid.')
        return result
        
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

        # make sure the action length is of correct size
        if self.enforce_action_count_non_bool:
            action_length = len(actions)
        else:
            action_length = len([
                action 
                for action in actions 
                if self.actionsranges[action] == 'bool'
            ])
        if action_length > self.max_allowed_actions:
            raise RDDLInvalidNumberOfArgumentsError(
                f'Invalid action provided: expected {self.max_allowed_actions} '
                f'entries, but got {action_length}.')
        
        # values are clipped to be inside the feasible action space
        clipped_actions = copy.deepcopy(self.default_actions)
        for act in actions:
            if self.actionsranges[act] == 'bool':
                clipped_actions[act] = bool(actions[act])
            else:
                clipped_actions[act] = actions[act]
                
        # check action constraints
        sampler = self.sampler
        if self.enforce_action_constraints:
            sampler.check_action_preconditions(clipped_actions)
        
        # sample next state and reward
        obs, reward, self.done = sampler.step(clipped_actions)
        self.state = sampler.states
            
        # check if the state invariants are satisfied
        if not self.done:
            sampler.check_state_invariants()               

        # log to file
        if self.simlogger is not None:
            self.simlogger.log(obs, clipped_actions, reward, self.done, self.currentH)
        
        # update step horizon
        self.currentH += 1
        if self.currentH == self.horizon:
            self.done = True

        return obs, reward, self.done, {}

    def reset(self, seed=None):
        
        # reset counters and internal state
        sampler = self.sampler
        self.currentH = 0
        obs, self.done = sampler.reset()
        self.state = sampler.states
        
        # update movie generator
        if self._movie_generator is not None and self._visualizer is not None:
            image = self._visualizer.render(self.state)
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
                pygame.display.set_caption(self.model.instanceName())
                pygame.display.flip()
                
                # prevents the window from freezing up midway
                # https://www.reddit.com/r/pygame/comments/eq970n/pygame_window_freezes_seconds_into_animation/?rdt=63412
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pass
        
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
