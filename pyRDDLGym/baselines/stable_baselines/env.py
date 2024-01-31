import gym
from gym.spaces import Box, Dict, Discrete, MultiDiscrete
import numpy as np
import typing

from pyRDDLGym.core.compiler.initializer import RDDLValueInitializer
from pyRDDLGym.core.compiler.model import RDDLLiftedModel
from pyRDDLGym.core.constraints import RDDLConstraints
from pyRDDLGym.core.debug.exception import (
    RDDLInvalidActionError,
    RDDLNotImplementedError,
    RDDLTypeError
)
from pyRDDLGym.core.debug.logger import Logger, SimLogger
from pyRDDLGym.core.env import RDDLEnv
from pyRDDLGym.core.parser.parser import RDDLParser
from pyRDDLGym.core.parser.reader import RDDLReader
from pyRDDLGym.core.seeding import RDDLEnvSeeder, RDDLEnvSeederFibonacci
from pyRDDLGym.core.simulator import RDDLSimulator
from pyRDDLGym.core.visualizer.chart import ChartVisualizer

    
class StableBaselinesRDDLEnv(RDDLEnv):
    '''A gym environment class for RDDL domains, modified to compact the
    action space for stable-baselines.'''
    
    def __init__(self, domain: str,
                 instance: str,
                 enforce_action_constraints: bool=False,
                 enforce_action_count_non_bool: bool=True,
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
        :param debug: whether to log compilation information to a log file
        :param log_path: absolute path to file where simulation log is saved,
        excluding the file extension, None means no logging
        :param backend: the subclass of RDDLSimulator to use as backend for
        simulation (currently supports numpy and Jax)
        :param backend_kwargs: dictionary of additional named arguments to
        pass to backend (must not include logger)
        :param seeds: an instance of RDDLEnvSeeder for generating RNG seeds
        '''
        self.domain_text = domain
        self.instance_text = instance
        self.enforce_action_constraints = enforce_action_constraints
        self.enforce_count_non_bool = enforce_action_count_non_bool
        
        # needed for parent class
        self.new_gym_api = True
        self.vectorized = True
        self.compact_action_space = True
        
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
        self.sampler = backend(self.model, logger=logger, keep_tensors=True,
                               **backend_kwargs)
        
        # compute the bounds on fluents from the constraints
        self._bounds = RDDLConstraints(self.sampler, vectorized=True).bounds
        self._shapes = {var: np.shape(values[0]) 
                        for (var, values) in self._bounds.items()}
        
        # construct the gym observation space
        if self.sampler.is_pomdp:
            state_ranges = self.model.observ_ranges
        else:
            state_ranges = self.model.state_ranges
        self.observation_space = self._rddl_to_gym_bounds(state_ranges)
        
        # construct the gym action space      
        self._action_ranges = self.model.action_ranges
        self._noop_actions = self.sampler.noop_actions
        self.action_space, self._action_info = self._rddl_to_gym_bounds_act(self._action_ranges)
        
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
    
    def _rddl_to_gym_bounds_act(self, ranges):
        
        # compute whether or not a constraint must be placed on boolean actions
        count_bool = 0
        for (var, prange) in ranges.items():
            if prange == 'bool':
                count_bool += np.prod(self._shapes[var], dtype=np.int64)   
        if self.max_allowed_actions < count_bool:
            if self.max_allowed_actions != 1:
                raise RDDLNotImplementedError(
                    'Simplification of bool action space with max-nondef-actions '
                    'other than 1 or pos-inf is not currently supported.')  
            bool_constraint = True
        else:
            bool_constraint = False
                
        # collect information about action ranges
        locational = {}
        count_disc, count_cont, count_bool = 0, 0, 0
        disc_start, disc_nelem = [], []
        cont_low, cont_high = [], []
        for (var, prange) in ranges.items():
            shape = self._shapes[var]
            num_elements = np.prod(shape, dtype=np.int64)  
            
            # boolean actions without constraint are stored as Discrete
            if prange == 'bool':
                if bool_constraint:
                    locational[var] = ('discrete', (count_bool, num_elements, shape))
                    count_bool += num_elements
                else:
                    disc_start.extend([0] * num_elements)
                    disc_nelem.extend([2] * num_elements)  
                    locational[var] = ('discrete', (count_disc, num_elements, shape))
                    count_disc += num_elements
            
            # integer actions are stored as Discrete
            elif prange == 'int':
                low, high = self._bounds[var]
                low = np.ravel(low, order='C')
                high = np.ravel(high, order='C')
                low = np.maximum(low, np.iinfo(np.int32).min).astype(np.int32)
                high = np.minimum(high, np.iinfo(np.int32).max).astype(np.int32)
                disc_start.extend(low.tolist())
                disc_nelem.extend((high - low + 1).tolist())                
                locational[var] = ('discrete', (count_disc, num_elements, shape))
                count_disc += num_elements
            
            # enum-valued actions are stored as Discrete
            elif prange in self.model.enum_types:
                num_objects = len(self.model.type_to_objects[prange])
                disc_start.extend([0] * num_elements)
                disc_nelem.extend([num_objects] * num_elements)
                locational[var] = ('discrete', (count_disc, num_elements, shape))
                count_disc += num_elements
            
            # real actions are stored as Box
            elif prange == 'real':
                low, high = self._bounds[var]
                low = np.ravel(low, order='C').tolist()
                high = np.ravel(high, order='C').tolist()
                cont_low.extend(low)
                cont_high.extend(high)
                locational[var] = ('continuous', (count_cont, num_elements, shape))
                count_cont += num_elements
            
            # not a valid action type
            else:
                raise RDDLTypeError(
                    f'Type <{prange}> of fluent <{var}> is not valid, '
                    f'must be an enumerated or primitive type (real, int, bool).')
        
        # boolean actions with constraint are stored in the last place instead
        if bool_constraint:
            disc_start.append(0)
            disc_nelem.append(count_bool + 1)
            count_disc += 1
        
        # discrete space
        if len(disc_nelem) == 1:
            disc_space = Discrete(disc_nelem[0])
        elif len(disc_nelem) > 1:
            disc_space = MultiDiscrete(disc_nelem)
        else:
            disc_space = None
        
        # real space
        if count_cont:
            cont_space = Box(np.asarray(cont_low), np.asarray(cont_high), dtype=np.float32)
        else:
            cont_space = None
        
        # simplify space
        combined_space = Dict()
        if disc_space is not None:
            combined_space['discrete'] = disc_space
        if cont_space is not None:
            combined_space['continuous'] = cont_space
        if not combined_space:
            raise RDDLInvalidActionError(
                'RDDL action specification resulted in an empty action space.')
        keys = list(combined_space.keys())
        if len(keys) == 1:
            combined_space = combined_space[keys[0]]
        
        # log information
        if self.logger is not None:
            act_info = '\n\t'.join(
                f'{act}: action_tensor={key}, '
                f'start={start}, count={count}, shape={shape}'
                for (act, (key, (start, count, shape))) in locational.items())
            bound_info = (f'\tdiscrete_start={disc_start}, '
                          f'discrete_n={disc_nelem}\n'
                          f'\tcontinuous_low={cont_low}, '
                          f'continuous_high={cont_high}')
            self.logger.log(f'[info] computed gym action space:\n' 
                            f'{bound_info}\n'
                            f'\t{act_info}\n'
                            f'[info] final space: {combined_space}\n')
            
        return combined_space, \
            (locational, keys, bool_constraint, np.asarray(disc_start))
    
    def _gym_to_rddl_actions(self, gym_actions):
        locational, keys, bool_constraint, disc_start = self._action_info
        if len(keys) == 1:
            gym_actions = {keys[0]: gym_actions}  
        
        # process all actions except if active max-nondef-actions constraint
        actions = {}
        for (var, prange) in self._action_ranges.items():
            if not (bool_constraint and prange == 'bool'):
                key, (start, count, shape) = locational[var]
                action_key = gym_actions.get(key, None)
                if action_key is not None:
                    action = np.atleast_1d(action_key)[start:start + count]
                    if key == 'discrete':
                        action = action + disc_start[start:start + count]
                    dtype = RDDLValueInitializer.NUMPY_TYPES.get(
                        prange, RDDLValueInitializer.INT)
                    actions[var] = np.reshape(
                        action, newshape=shape, order='C').astype(dtype)
        
        # process the active max-nondef-actions constraint
        action_key = gym_actions.get('discrete', None)
        if bool_constraint and action_key is not None:
            index = np.atleast_1d(action_key)[-1]
            for (var, prange) in self._action_ranges.items():
                if prange == 'bool':
                    _, (start, count, shape) = locational[var]                    
                    index_in_var = index - start
                    if 0 <= index_in_var < count:
                        default_value = self.model.variable_defaults[var]
                        action = np.full(shape=count, fill_value=default_value, dtype=bool)
                        action[index_in_var] ^= True
                        actions[var] = np.reshape(action, newshape=shape, order='C')
                        break
            
        return actions
    
    def step(self, actions):
        actions = self._gym_to_rddl_actions(actions)
        return super(StableBaselinesRDDLEnv, self).step(actions)
