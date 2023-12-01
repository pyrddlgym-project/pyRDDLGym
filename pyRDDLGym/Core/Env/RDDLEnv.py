import gym
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete
import numpy as np
import os
import pygame
import typing

from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLEpisodeAlreadyEndedError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidActionError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLLogFolderError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLNotImplementedError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLTypeError

from pyRDDLGym.Core.Compiler.RDDLLiftedModel import RDDLLiftedModel
from pyRDDLGym.Core.Compiler.RDDLValueInitializer import RDDLValueInitializer
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
        self.sampler = backend(self.model, 
                               logger=logger, 
                               keep_tensors=self.vectorized, 
                               **backend_kwargs)
        
        # compute the bounds on fluents from the constraints
        self._bounds = RDDLConstraints(
            self.sampler, vectorized=self.vectorized).bounds
        self._shapes = {var: np.shape(values[0]) 
                        for (var, values) in self._bounds.items()}
        
        # construct the gym observation space
        if self.sampler.isPOMDP:
            statesranges = self.model.observranges
        else:
            statesranges = self.model.statesranges
        if not self.vectorized:
            statesranges = self.model.ground_ranges_from_dict(statesranges)    
        self.observation_space = self._rddl_to_gym_bounds_obs(statesranges)
        
        # construct the gym action space      
        if self.vectorized:
            self._actionsranges = self.model.actionsranges
            self._noop_actions = self.sampler.noop_actions
        else:
            self._actionsranges = self.sampler.grounded_actionsranges
            self._noop_actions = self.sampler.grounded_noop_actions
        self.action_space = self._rddl_to_gym_bounds_act(self._actionsranges)
        
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
    
    def _rddl_to_gym_bounds_obs(self, ranges):
        result = Dict()
        for (var, prange) in ranges.items():
            
            # enumerated values converted to Discrete space
            if prange in self.model.enums:
                num_objects = len(self.model.objects[prange])
                if self.vectorized:
                    result[var] = Box(0, num_objects - 1,
                                      shape=self._shapes[var], 
                                      dtype=np.int32)
                else:
                    result[var] = Discrete(num_objects) 
            
            # real values define a box
            elif prange == 'real':
                low, high = self._bounds[var]
                result[var] = Box(low, high, dtype=np.float32)
            
            # boolean values converted to Discrete space
            elif prange == 'bool':
                if self.vectorized:
                    result[var] = Box(0, 1, 
                                      shape=self._shapes[var],
                                      dtype=np.int32)
                else:
                    result[var] = Discrete(2)
            
            # integer values converted to Discrete space
            elif prange == 'int':
                low, high = self._bounds[var]
                low = np.maximum(low, np.iinfo(np.int32).min)
                high = np.minimum(high, np.iinfo(np.int32).max)
                if self.vectorized:
                    result[var] = Box(low, high,
                                      shape=self._shapes[var], 
                                      dtype=np.int32)
                else:
                    result[var] = Discrete(int(high - low + 1), start=int(low))
            
            # unknown type
            else:
                raise RDDLTypeError(
                    f'Type <{prange}> of fluent <{var}> is not valid, '
                    f'must be an enumerated or primitive type (real, int, bool).')
        return result
    
    def _rddl_to_gym_bounds_act(self, ranges):
        return (self._rddl_to_gym_bounds_obs(ranges), None)
    
    def _gym_to_rddl_actions(self, gym_actions):
        return gym_actions
    
    def seed(self, seed=None):
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
            raise RDDLEpisodeAlreadyEndedError(
                'The step() function has been called even though the '
                'current episode has terminated: please call reset().')            
            
        actions = self._gym_to_rddl_actions(actions)
        
        # cast non-boolean actions to boolean
        fixed_actions = self._noop_actions.copy()
        for (var, values) in actions.items():
            if self._actionsranges.get(var, '') == 'bool':
                if np.shape(values):
                    fixed_actions[var] = np.asarray(values, dtype=bool)
                else:
                    fixed_actions[var] = bool(values)
            else:
                fixed_actions[var] = values
        actions = fixed_actions

        # check action constraints
        sampler = self.sampler
        sampler.check_default_action_count(actions, self.enforce_count_non_bool)
        if self.enforce_action_constraints:
            sampler.check_action_preconditions(actions)
        
        # sample next state and reward
        obs, reward, self.done = sampler.step(actions)
        self.state = sampler.states
            
        # check if the state invariants are satisfied
        if not self.done:
            sampler.check_state_invariants()               

        # log to file
        if self.simlogger is not None:
            if self.vectorized:
                log_obs = self.model.ground_values_from_dict(obs)
                log_action = self.model.ground_values_from_dict(actions)
            else:
                log_obs = obs
                log_action = actions
            self.simlogger.log(log_obs, log_action, reward, self.done, self.currentH)
        
        # update step horizon
        self.currentH += 1
        if self.currentH == self.horizon:
            self.done = True

        return obs, reward, self.done, {}

    def reset(self, seed=None):
        
        # reset counters and internal state
        sampler = self.sampler
        obs, self.done = sampler.reset()
        self.state = sampler.states
        self.trial += 1
        self.currentH = 0
        
        # update movie generator
        if self._movie_generator is not None and self._visualizer is not None:
            if self.vectorized:
                state = self.model.ground_values_from_dict(self.state)
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

        return obs

    def pilImageToSurface(self, pilImage):
        return pygame.image.fromstring(
            pilImage.tobytes(), pilImage.size, pilImage.mode).convert()

    def render(self, to_display=True):
        image = None
        if self._visualizer is not None:
            if self.vectorized:
                state = self.model.ground_values_from_dict(self.state)
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
                pygame.display.set_caption(self.model.instanceName())
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


class RDDLEnvCompact(RDDLEnv):
    '''A gym environment class for RDDL domains with a compact action space.
    Most suitable for RL implementations.'''
    
    @staticmethod
    def build(env_info, env: str, **env_kwargs):
        env = RDDLEnvCompact(domain=env_info.get_domain(),
                             instance=env_info.get_instance(env),
                             **env_kwargs)
        env.set_visualizer(env_info.get_visualizer())
        return env
    
    def __init__(self, *args, **kwargs):
        super(RDDLEnvCompact, self).__init__(*args, vectorized=True, **kwargs)
        
        self.action_space, self._action_info = self._rddl_to_gym_bounds_act(
            self._actionsranges)
    
    def _rddl_to_gym_bounds_act(self, ranges):
        
        # collect information about action ranges
        enum_shape = []
        real_bounds = ([], [])
        int_bounds = ([], [])
        count_enum, count_real, count_int, count_bool = 0, 0, 0, 0
        locational = {}
        for (var, prange) in ranges.items():
            shape = self._shapes[var]
            num_elements = np.prod(shape, dtype=np.int64)
            if prange in self.model.enums:
                num_objects = len(self.model.objects[prange])
                enum_shape.extend([num_objects] * num_elements)
                locational[var] = ('finite', (count_enum, num_elements, shape))
                count_enum += num_elements
            elif prange == 'real':
                low, high = self._bounds[var]
                low = np.ravel(low, order='C').tolist()
                high = np.ravel(high, order='C').tolist()
                real_bounds[0].extend(low)
                real_bounds[1].extend(high)
                locational[var] = ('real', (count_real, num_elements, shape))
                count_real += num_elements
            elif prange == 'int':
                low, high = self._bounds[var]
                low = np.ravel(low, order='C').tolist()
                high = np.ravel(high, order='C').tolist()
                int_bounds[0].extend(low)
                int_bounds[1].extend(high)
                locational[var] = ('int', (count_int, num_elements, shape))
                count_int += num_elements
            elif prange == 'bool':
                locational[var] = ('finite', (count_bool, num_elements, shape))
                count_bool += num_elements
            else:
                raise RDDLTypeError(
                    f'Type <{prange}> of fluent <{var}> is not valid, '
                    f'must be an enumerated or primitive type (real, int, bool).')
        
        # simplify real space
        if real_bounds[0]:
            low, high = real_bounds
            real_space = Box(np.asarray(low), np.asarray(high), dtype=np.float32)
        else:
            real_space = None
        
        # simplify int space
        if int_bounds[0]:
            low, high = int_bounds
            int_space = Box(np.asarray(low), np.asarray(high), dtype=np.int32)
        else:
            int_space = None
            
        # simplify finite space
        finite_dims = enum_shape.copy()
        if count_bool == 1:
            finite_dims.append(2)
        elif count_bool > 1:
            if self.max_allowed_actions == 1:
                finite_dims.append(count_bool + 1)
            elif self.max_allowed_actions >= count_bool:
                finite_dims.extend([2] * count_bool)
            else:
                raise RDDLNotImplementedError(
                    'Simplification of bool action space with max-nondef-actions '
                    'other than 1 or pos-inf is not currently supported.')  
        if len(finite_dims) == 1:
            finite_space = Discrete(finite_dims[0])
        elif len(finite_dims) > 1:
            if all(dim == 2 for dim in finite_dims):
                finite_space = MultiBinary(len(finite_dims))
            else:
                finite_space = MultiDiscrete(finite_dims)
        else:
            finite_space = None
        
        # simplify space
        combined_space = {}
        if real_space is not None:
            combined_space['real'] = real_space
        if int_space is not None:
            combined_space['int'] = int_space
        if finite_space is not None:
            combined_space['finite'] = finite_space
        if not combined_space:
            raise RDDLInvalidActionError(
                'RDDL action specification resulted in an empty action space.')
        keys = list(combined_space.keys())
        if len(combined_space) == 1:
            combined_space = next(iter(combined_space.values()))
            
        return combined_space, (locational, keys, count_bool)

    def _gym_to_rddl_actions(self, gym_actions):
        locational, keys, count_bool = self._action_info
        if len(keys) == 1:
            gym_actions = {keys[0]: gym_actions}        
        bool_constraint = count_bool > 1 and self.max_allowed_actions == 1
        
        # process all actions except if active max-nondef-actions constraint
        actions = {}
        for (var, prange) in self._actionsranges.items():
            key, (start, count, shape) = locational[var]
            action = gym_actions[key]
            if not (bool_constraint and prange == 'bool'):
                action = action[start:start + count]
                dtype = RDDLValueInitializer.NUMPY_TYPES.get(
                    prange, RDDLValueInitializer.INT)
                actions[var] = np.reshape(action, shape, order='C').astype(dtype)
        
        # process the active max-nondef-actions constraint
        if bool_constraint:
            index = np.atleast_1d(gym_actions['finite'])[-1]
            for (var, prange) in self._actionsranges.items():
                if prange == 'bool':
                    _, (start, count, shape) = locational[var]
                    offset = index - start
                    if 0 <= offset < count:
                        default_value = self.model.default_values[var]
                        action = np.full(shape=count, fill_value=default_value, dtype=bool)
                        action[offset] ^= True
                        actions[var] = np.reshape(action, shape, order='C')
                        break
        
        return actions
              
