import datetime
import numpy as np
from typing import Callable, Iterable, List, Tuple

from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidNumberOfArgumentsError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidObjectError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLNotImplementedError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLTypeError

from pyRDDLGym.Core.Compiler.RDDLModel import RDDLModel
from pyRDDLGym.Core.Parser.expr import Value


class RDDLTensors:
    
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
    
    VALID_SYMBOLS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    def __init__(self, rddl: RDDLModel, debug: bool=False) -> None:
        self.rddl = rddl
        self.debug = debug
        
        self.filename = f'debug_{rddl._AST.domain.name}_{rddl._AST.instance.name}.txt'
        if self.debug:
            fp = open(self.filename, 'w')
            fp.write('')
            fp.close()
            
        self.index_of_object, self.grounded = self._compile_objects()
        self.init_values = self._compile_init_values()
        
        self._cached_transforms = {}

    def _compile_objects(self):
        grounded = {}
        for var, types in self.rddl.param_types.items():
            grounded[var] = list(self.rddl.grounded_names(var, types))
        
        index_of_object = {}
        for objects in self.rddl.objects.values():
            for i, obj in enumerate(objects):
                index_of_object[obj] = i      
                  
        return index_of_object, grounded
    
    def _compile_init_values(self):
        init_values = {}
        init_values.update(self.rddl.nonfluents)
        init_values.update(self.rddl.init_state)
        init_values.update(self.rddl.actions)
        init_values = {name: value 
                       for name, value in init_values.items() 
                       if value is not None}
        
        init_arrays = {}
        for var in self.rddl.variable_ranges.keys():
            prange = self.rddl.variable_ranges[var]
            valid_ranges = RDDLTensors.DEFAULT_VALUES
            if prange not in valid_ranges:
                raise RDDLTypeError(
                    f'Type <{prange}> of variable <{var}> is not valid, '
                    f'must be one of {set(valid_ranges.keys())}.')                
            default = valid_ranges[prange]
            
            types = self.rddl.param_types[var]
            dtype = RDDLTensors.NUMPY_TYPES[prange]
            if types:
                if self.rddl.is_grounded:
                    for name in self.rddl.grounded_names(var, types):
                        init_arrays[name] = dtype(init_values.get(name, default)) 
                else:
                    grounded_values = [
                        init_values.get(name, default) 
                        for name in self.rddl.grounded_names(var, types)
                    ]
                    array = np.asarray(grounded_values, dtype=dtype)
                    array = np.reshape(array, newshape=self.shape(types), order='C')
                    init_arrays[var] = array                
            else:
                init_arrays[var] = dtype(init_values.get(var, default))
        
        if self.debug:
            tensor_info = '\n\t'.join(
                (f'{k}{[] if self.rddl.is_grounded else self.rddl.param_types[k]}, '
                 f'shape={v.shape if type(v) is np.ndarray else ()}, '
                 f'dtype={v.dtype if type(v) is np.ndarray else type(v).__name__}')
                for k, v in init_arrays.items())
            self.write_debug_message(
                f'initializing pvariable tensors:' 
                    f'\n\t{tensor_info}\n'
            )
            
        return init_arrays
        
    def coordinates(self, objects: Iterable[str], msg: str='') -> Tuple[int, ...]:
        '''Converts a list of objects into their coordinate representation.
        
        :param objects: object instances corresponding to valid types defined
        in the RDDL domain
        :param msg: an error message to print in case the conversion fails.
        '''
        index_of_obj = self.index_of_object
        try:
            return tuple(index_of_obj[obj] for obj in objects)
        except:
            for obj in objects:
                if obj not in index_of_obj:
                    raise RDDLInvalidObjectError(
                        f'Object <{obj}> is not valid, '
                        f'must be one of {set(index_of_obj.keys())}.\n'
                        f'{msg}')
    
    def shape(self, types: Iterable[str], msg: str='') -> Tuple[int, ...]:
        '''Given a list of RDDL types, returns the shape of a tensor
        that would hold all values of a pvariable with those type arguments
        
        :param types: a list of RDDL types
        :param msg: an error message to print in case the calculation fails
        '''
        objects = self.rddl.objects
        try:
            return tuple(len(objects[ptype]) for ptype in types)
        except:
            for ptype in types:
                if ptype not in objects:
                    raise RDDLInvalidObjectError(
                        f'Type <{ptype}> is not valid, '
                        f'must be one of {set(objects.keys())}.\n'
                        f'{msg}')
            
    def write_debug_message(self, msg: str) -> None:
        if self.debug:
            fp = open(self.filename, 'a')
            timestamp = str(datetime.datetime.now())
            fp.write(timestamp + ': ' + msg + '\n')
            fp.close()
        
    def map(self, var: str,
            obj_in: List[str],
            sign_out: List[Tuple[str, str]],
            gnp=np,
            msg: str='') -> Callable[[np.ndarray], np.ndarray]:
        '''Returns a function that transforms a pvariable value tensor to one
        whose shape matches a desired output signature. This operation is
        achieved by adding new dimensions to the value as needed, and performing
        a combination of transposition/reshape/einsum operations to coerce the 
        value to the desired shape.
        
        :param var: a string pvariable defined in the domain
        :param obj_in: a list of desired object quantifications, e.g. ?x, ?y at
        which var will be evaluated
        :param sign_out: a list of tuples (objecti, typei) representing the
            desired signature of the output pvariable tensor
        :param gnp: the library in which to perform tensor arithmetic 
        (either numpy or jax.numpy)
        :param msg: a stack trace to print for error handling
        '''
                
        # check that the input objects match fluent type definition
        types_in = self.rddl.param_types.get(var, [])
        if obj_in is None:
            obj_in = []
        n_in = len(obj_in)
        n_req = len(types_in)
        if n_in != n_req:
            raise RDDLInvalidNumberOfArgumentsError(
                f'Variable <{var}> requires {n_req} parameters, got {n_in}.'
                f'\n{msg}')
            
        # reached limit on number of valid dimensions
        valid_symbols = RDDLTensors.VALID_SYMBOLS
        n_max = len(valid_symbols)
        n_out = len(sign_out)
        if n_out > n_max:
            raise RDDLNotImplementedError(
                f'At most {n_max} object arguments are supported, '
                f'but variable <{var}> has {n_out} arguments.'
                f'\n{msg}')
        
        # find a map permutation(a,b,c...) -> (a,b,c...) for correct transpose
        sign_in = tuple(zip(obj_in, types_in))
        permutation = [None] * len(obj_in)
        new_dims = []
        for i_out, (o_out, t_out) in enumerate(sign_out):
            new_dim = True
            for i_in, (o_in, t_in) in enumerate(sign_in):
                if o_in == o_out:
                    permutation[i_in] = i_out
                    new_dim = False
                    if t_out != t_in: 
                        raise RDDLInvalidObjectError(
                            f'Argument {i_in + 1} of variable <{var}> '
                            f'expects object of type <{t_in}>, '
                            f'got <{o_out}> of type <{t_out}>.'
                            f'\n{msg}')
            
            # need to expand the shape of the value array
            if new_dim:
                permutation.append(i_out)
                new_dims.append(len(self.rddl.objects[t_out]))
                
        # safeguard against any free types
        free = {sign_in[i][0] for i, p in enumerate(permutation) if p is None}
        if free:
            raise RDDLInvalidNumberOfArgumentsError(
                f'Variable <{var}> has unresolved parameter(s) {free}.'
                f'\n{msg}')
        
        # compute the mapping function as follows:
        # 1. append new axes to value tensor equal to # of missing variables
        # 2. broadcast new axes to the desired shape (# of objects of each type)
        # 3. rearrange the axes as needed to match the desired variables in order
        #     3a. in most cases, it suffices to use np.transform (cheaper)
        #     3b. in cases where we have a more complex contraction like 
        #         fluent(?x) = matrix(?x, ?x), we will use np.einsum
        in_shape = self.shape(types_in)
        out_shape = in_shape + tuple(new_dims)
        new_axis = tuple(range(len(in_shape), len(out_shape)))
         
        lhs = ''.join(valid_symbols[p] for p in permutation)        
        rhs = valid_symbols[:n_out]
        use_einsum = len(set(lhs)) != len(lhs)
        use_tr = lhs != rhs
        if use_einsum:
            subscripts = lhs + '->' + rhs
        elif use_tr:
            subscripts = tuple(np.argsort(permutation))  # inverse permutation
        else:
            subscripts = None
        
        # check if the tensor transform with the given signature already exists
        # if so, just retrieve it from the cache
        # if not, create it and store it in the cache
        transform_id = f'{new_axis}_{out_shape}_{use_einsum}_{use_tr}_{subscripts}'
        _transform = self._cached_transforms.get(transform_id, None)        
        if _transform is None:
            
            def _transform(arg):
                sample = arg
                if new_axis:
                    sample = gnp.expand_dims(sample, axis=new_axis)
                    sample = gnp.broadcast_to(sample, shape=out_shape)
                if use_einsum:
                    return gnp.einsum(subscripts, sample)
                elif use_tr:
                    return gnp.transpose(sample, axes=subscripts)
                else:
                    return sample
            
            self._cached_transforms[transform_id] = _transform
            
        operation = gnp.einsum if use_einsum else (
                        gnp.transpose if use_tr else 'None')
        self.write_debug_message(
            f'computing info for pvariable transform:' 
                f'\n\tvar        ={var}'
                f'\n\tinputs     ={sign_in}'
                f'\n\ttargets    ={sign_out}'
                f'\n\tnew axes   ={new_axis}'
                f'\n\toperation  ={operation}, subscripts={subscripts}'
                f'\n\tunique id  ={id(_transform)}\n'
        )
            
        return _transform
    
    def expand(self, var: str, values: np.ndarray) -> Iterable[Tuple[str, Value]]:
        '''Produces a grounded representation of the pvariable var from its 
        tensor representation. The output is a dict whose keys are grounded
        representations of the var, and values are read from the tensor.
        
        :param var: the pvariable
        :param values: the tensor whose values correspond to those of var(?...)        
        '''
        keys = self.grounded[var]
        values = np.ravel(values)
        if len(keys) != values.size:
            raise RDDLInvalidNumberOfArgumentsError(
                f'Size of value array is not compatible with variable <{var}>.')
        return zip(keys, values)
