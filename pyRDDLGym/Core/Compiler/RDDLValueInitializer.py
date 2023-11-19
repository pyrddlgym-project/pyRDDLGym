import numpy as np
from typing import Dict, Union

from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidObjectError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLTypeError

from pyRDDLGym.Core.Compiler.RDDLModel import PlanningModel
from pyRDDLGym.Core.Debug.Logger import Logger


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
        
    def __init__(self, rddl: PlanningModel, logger: Logger=None) -> None:
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
    
    def initialize(self) -> Dict[str, Union[np.ndarray, INT, REAL, bool]]:
        '''Compiles all initial values of all variables for the current RDDL file.
        A dictionary is returned with variable names as keys (as they appear in
        the RDDL) and value arrays as values.'''
        rddl = self.rddl
                
        # initial values consists of non-fluents, state and action fluents
        init_values = {}
        init_values.update(rddl.nonfluents)
        init_values.update(rddl.init_state)
        init_values.update(rddl.actions)

        # domain objects are converted to integers
        for (var, values) in init_values.items():
            prange = rddl.variable_ranges[var]
            if prange in rddl.enums:
                init_values[var] = self._objects_to_ints(values, prange, var)
        
        # create a tensor for each pvar with the init_values
        # if the init_values are missing use the default value of range
        np_init_values = {}
        for (var, prange) in rddl.variable_ranges.items():
            
            # domain objects are treated as int
            if prange in rddl.enums:
                prange = 'int'
            
            # get default value and dtype
            default = RDDLValueInitializer.DEFAULT_VALUES.get(prange, None)
            dtype = RDDLValueInitializer.NUMPY_TYPES.get(prange, None)
            if default is None or dtype is None:
                raise RDDLTypeError(
                    f'Type <{prange}> of variable <{var}> is not valid, '
                    f'must be either an enumerated type in '
                    f'{rddl.enums} or an object type in '
                    f'{set(RDDLValueInitializer.DEFAULT_VALUES.keys())}.')
            
            # scalar value is just cast to the desired type
            # list values are converted to numpy arrays and reshaped such that 
            # number of axes matches number of pvariable arguments
            ptypes = rddl.param_types[var]
            if ptypes:
                shape = rddl.object_counts(ptypes)
                if var in init_values:
                    values = [(default if v is None else v) for v in init_values[var]]
                    values = np.reshape(values, newshape=shape, order='C')
                    if not np.can_cast(values, dtype):
                        raise RDDLTypeError(
                            f'Initial values {values} for variable <{var}> '
                            f'cannot all be cast to required type <{prange}>.')
                    values = np.asarray(values, dtype=dtype)
                else:
                    values = np.full(shape=shape, fill_value=default, dtype=dtype)
            else:
                values = init_values.get(var, default)
                if isinstance(values, str) or not np.can_cast(values, dtype):
                    raise RDDLTypeError(
                        f'Initial value {values} for variable <{var}> '
                        f'cannot be cast to required type <{prange}>.')
                values = dtype(values)
            np_init_values[var] = values
        
        # log shapes of initial values
        if self.logger is not None:
            tensor_info = '\n\t'.join((
                f'{k}{rddl.param_types[k]}, '
                f'shape={v.shape if type(v) is np.ndarray else ()}, '
                f'dtype={v.dtype if type(v) is np.ndarray else type(v).__name__}'
            ) for (k, v) in np_init_values.items())
            message = (f'[info] initializing pvariable tensors:' 
                       f'\n\t{tensor_info}\n')
            self.logger.log(message)
        
        return np_init_values
    
    def _objects_to_ints(self, literals, prange, var):
        is_scalar = isinstance(literals, str)
        if is_scalar:
            literals = [literals]
            
        indices = [0] * len(literals)
        for (i, obj) in enumerate(literals):
            if obj is not None:
                if self.rddl.objects_rev.get(obj, None) != prange:
                    raise RDDLInvalidObjectError(
                        f'<{obj}> assigned to pvariable <{var}> in instance '
                        f'is not an object of type <{prange}>.')
                indices[i] = self.rddl.index_of_object[obj]
                
        if is_scalar:
            indices, = indices
        return indices
    
