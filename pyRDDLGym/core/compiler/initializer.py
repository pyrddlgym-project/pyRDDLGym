import numpy as np
from typing import Dict, Optional, Union

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.debug.exception import (
    RDDLInvalidObjectError,
    RDDLTypeError
)
from pyRDDLGym.core.debug.logger import Logger


class RDDLValueInitializer:
    '''Compiles all initial values in pvariables scope and init-fluents scope
    in a RDDL domain + instance to scalars or numpy arrays.
    '''
    
    INT = np.int64
    REAL = np.float64
        
    NUMPY_TYPES = {
        'int': INT,
        'real': REAL,
        'bool': bool
    }
    
    DEFAULT_VALUES = {
        'int': 0,
        'real': 0.0,
        'bool': False
    }
        
    def __init__(self, rddl: RDDLPlanningModel, 
                 logger: Optional[Logger]=None) -> None:
        '''Creates a new object to compile initial values from a RDDL file. 
        Initial values of parameterized variables are stored in numpy arrays.
        For a variable var(?x1, ?x2, ... ?xn), the numpy array has n dimensions, 
        where the i-th dimension has number of elements equal to the number of 
        elements in the type of ?xi.
        
        :param rddl: the RDDL file whose initial values to extract
        :param logger: to log information about initial values to file
        '''
        self.rddl = rddl
        self.logger = logger
    
    def initialize(self) -> Dict[str, Union[np.ndarray, np.integer, np.floating, bool]]:
        '''Compiles all initial values of all variables for the current RDDL file.
        A dictionary is returned with variable names as keys (as they appear in
        the RDDL) and value arrays as values.'''
        rddl = self.rddl
                
        # initial values consists of non-fluents, state and action fluents
        init_values = {}
        init_values.update(rddl.non_fluents)
        init_values.update(rddl.state_fluents)
        init_values.update(rddl.action_fluents)

        # domain objects are converted to integers
        for (var, values) in init_values.items():
            prange = rddl.variable_ranges[var]
            if prange in rddl.type_to_objects:
                init_values[var] = self._objects_to_ints(values, prange, var)
        
        # create a tensor for each pvar with the init_values
        # if the init_values are missing use the default value of range
        np_init_values = {}
        for (var, prange) in rddl.variable_ranges.items():
            
            # domain objects are treated as int
            if prange in rddl.type_to_objects:

                # do not allow default statements for non-enum types
                if prange not in rddl.enum_types \
                and rddl.variable_defaults.get(var, None) is not None:
                    raise RDDLTypeError(
                        f'Setting a default value for pvariable <{var}> of range <{prange}> '
                        f'would require a concrete object '
                        f'that can not be specified in the domain, '
                        f'please specify initial value(s) in the instance.')
                prange = 'int'
            
            # convert a parameterized variable to dimensioned numpy array
            ptypes = rddl.variable_params[var]
            shape = rddl.object_counts(ptypes) if ptypes else None
            np_init_values[var] = self._create_np_tensor(var, init_values, shape, prange)    
        
        # log shapes of initial values
        if self.logger is not None:
            tensor_info = '\n\t'.join((
                f'{k}{rddl.variable_params[k]}, '
                f'shape={v.shape if type(v) is np.ndarray else ()}, '
                f'dtype={v.dtype if type(v) is np.ndarray else type(v).__name__}'
            ) for (k, v) in np_init_values.items())
            message = (
                f'[info] initializing pvariable tensors:' 
                f'\n\t{tensor_info}\n'
            )
            self.logger.log(message)
        
        return np_init_values
    
    def initialize_policy(self) -> Dict[str, Union[np.ndarray, np.integer, np.floating, bool]]:
        rddl = self.rddl
        policy = rddl.policy
        if policy is None:
            return {}
        
        # initial values are just policy non-fluents, only allow numerical types
        init_values = {}
        for (var, values) in policy.non_fluents.items():
            prange = policy.variable_ranges[var]
            if prange in rddl.type_to_objects:
                raise RDDLTypeError(
                    f'Range <{prange}> of policy non-fluent is not valid, '
                    f'must be a numeric type.')
            if values is None:
                raise RDDLTypeError(
                    f'Policy non-fluent <{var}> does not have a default value.')
            init_values[var] = values

        # create a tensor for each pvar with the init_values
        # if the init_values are missing use the default value of range
        np_init_values = {}
        for (var, prange) in policy.variable_ranges.items():
            
            # domain objects are treated as int
            if prange in rddl.type_to_objects:

                # do not allow default statements for non-enum types
                if prange not in rddl.enum_types \
                and policy.variable_defaults.get(var, None) is not None:
                    raise RDDLTypeError(
                        f'Setting a default value for policy pvariable <{var}> of '
                        f'range <{prange}> would require a concrete object '
                        f'that can not be specified in the domain.')
                prange = 'int'
            
            # convert a parameterized variable to dimensioned numpy array
            ptypes = policy.variable_params[var]
            shape = rddl.object_counts(ptypes) if ptypes else None
            np_init_values[var] = self._create_np_tensor(var, init_values, shape, prange)        
        
        return np_init_values
    
    def _create_np_tensor(self, var, init_values, shape, prange):   

        # get the numpy dtype and default value
        default = RDDLValueInitializer.DEFAULT_VALUES.get(prange, None)
        dtype = RDDLValueInitializer.NUMPY_TYPES.get(prange, None)
        if default is None or dtype is None:
            raise RDDLTypeError(
                f'Range <{prange}> of pvariable <{var}> is not valid, '
                f'must be an object type in {set(self.rddl.type_to_objects.keys())} '
                f'or a numeric type in {set(RDDLValueInitializer.DEFAULT_VALUES.keys())}.'
            )

        # has a valid shape and objects 
        if shape is not None:
            values = init_values.get(var, None)    
            if values is None:
                values = np.full(shape=shape, fill_value=default, dtype=dtype)
            else:
                values = np.reshape(
                    [(default if v is None else v) for v in values], shape, order='C')
                
                # cast to the required type
                if not np.can_cast(values, dtype):
                    raise RDDLTypeError(
                        f'Initial values {values} of pvariable <{var}> '
                        f'can not all be cast to required type <{prange}>.')
                values = np.asarray(values, dtype=dtype)
        
        # scalar pvariable
        else:
            values = init_values.get(var, default)
            if isinstance(values, str) or not np.can_cast(np.atleast_1d(values), dtype):
                raise RDDLTypeError(
                    f'Initial values {values} of pvariable <{var}> '
                    f'can not all be cast to required type <{prange}>.')
            values = dtype(values)
        return values
            
    def _objects_to_ints(self, literals, prange, var):
        is_scalar = isinstance(literals, str)
        if is_scalar:
            literals = [literals]
            
        indices = [0] * len(literals)
        for (i, obj) in enumerate(literals):
            if obj is not None:
                if self.rddl.object_to_type.get(obj, None) != prange:
                    raise RDDLInvalidObjectError(
                        f'Value <{obj}> assigned to pvariable <{var}> in instance '
                        f'is not an object of type <{prange}>.')
                indices[i] = self.rddl.object_to_index[obj]
                
        if is_scalar:
            indices, = indices
        return indices
    
