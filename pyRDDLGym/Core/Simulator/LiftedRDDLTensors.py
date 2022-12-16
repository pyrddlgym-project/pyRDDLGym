import numpy as np
from typing import Dict, Iterable, List, Tuple
import warnings

from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidNumberOfArgumentsError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidObjectError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLNotImplementedError
from pyRDDLGym.Core.Parser.expr import Value
from pyRDDLGym.Core.Simulator.LiftedRDDLModel import LiftedRDDLModel


class LiftedRDDLTensors:
    
    VALID_SYMBOLS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    def __init__(self, rddl: LiftedRDDLModel, debug: bool=False):
        self.rddl = rddl
        self.debug = debug
        
        self.index_of_object, self.grounded = self._compile()

    def _compile(self):
        grounded = {}
        for name, (objects, _) in self.rddl.cpfs.items():
            types = self.rddl.param_types[name]
            key = name.replace('\'', '')
            grounded[key] = list(self.rddl.grounded_names(name, types))
        
        index_of_object = {}
        for objects in self.rddl.objects.values():
            for i, obj in enumerate(objects):
                index_of_object[obj] = i      
                  
        return index_of_object, grounded
                        
    def coordinates(self, objects: Iterable[str], msg: str='') -> Tuple[int, ...]:
        '''Converts a list of objects into their coordinate representation.
        
        :param objects: object instances corresponding to valid types defined
        in the RDDL domain
        :param msg: an error message to print in case the conversion fails.
        '''
        try:
            return tuple(self.index_of_object[obj] for obj in objects)
        except:
            for obj in objects:
                if obj not in self.index_of_object:
                    raise RDDLInvalidObjectError(
                        f'Object <{obj}> is not valid, '
                        f'must be one of {set(self.index_of_object.keys())}.'
                        f'\n{msg}')
    
    def shape(self, types: Iterable[str], msg: str='') -> Tuple[int, ...]:
        '''Given a list of RDDL types, returns the shape of a tensor
        that would hold all values of a pvariable with those type arguments
        
        :param types: a list of RDDL types
        :param msg: an error message to print in case the calculation fails
        '''
        try:
            return tuple(len(self.rddl.objects[ptype]) for ptype in types)
        except:
            for ptype in types:
                if ptype not in self.rddl.objects:
                    raise RDDLInvalidObjectError(
                        f'Type <{ptype}> is not valid, '
                        f'must be one of {set(self.rddl.objects.keys())}.'
                        f'\n{msg}')
            
    def map(self, var: str,
            obj_in: List[str],
            sign_out: List[Tuple[str, str]],
            expr: str, 
            msg: str='') -> Tuple[str, bool, Tuple[int, ...]]:
        '''Given:       
        1. a pvariable var
        2. a list of objects [o1, ...] at which var should be evaluated, and 
        3. a desired signature [(object1, type1), (object2, type2)...],
          
        a call to map produces a tuple of three things if it is possible (and 
        raises an exception if not):
        
        1. a mapping 'permutation(a,b,c...) -> (a,b,c...)' that represents the 
        np.einsum transform on the tensor var with [o1, ...] to produce the
        tensor representation of var with the desired signature
        2. whether the permutation above is the identity, and
        3. a tuple of new dimensions that need to be added to var in order to 
        broadcast to the desired signature shape.
        
        :param var: a string pvariable defined in the domain
        :param obj_in: a list of desired objects [o1, ...] as above
        :param sign_out: a list of tuples (objecti, typei) representing the
            desired signature of the output pvariable tensor
        :param expr: a string representation of the expression for logging
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
        valid_symbols = LiftedRDDLTensors.VALID_SYMBOLS
        n_max = len(valid_symbols)
        n_out = len(sign_out)
        if n_out > n_max:
            raise RDDLNotImplementedError(
                f'Up to {n_max}-D are supported, '
                f'but variable <{var}> is {n_out}-D.'
                f'\n{msg}')
        
        # find a map permutation(a,b,c...) -> (a,b,c...) for the correct einsum
        sign_in = tuple(zip(obj_in, types_in))
        lhs = [None] * len(obj_in)
        new_dims = []
        for i_out, (o_out, t_out) in enumerate(sign_out):
            new_dim = True
            for i_in, (o_in, t_in) in enumerate(sign_in):
                if o_in == o_out:
                    lhs[i_in] = valid_symbols[i_out]
                    new_dim = False
                    if t_out != t_in: 
                        raise RDDLInvalidObjectError(
                            f'Argument {i_in + 1} of variable <{var}> '
                            f'expects object of type <{t_in}>, '
                            f'got <{o_out}> of type <{t_out}>.'
                            f'\n{msg}')
            
            # need to expand the shape of the value array
            if new_dim:
                lhs.append(valid_symbols[i_out])
                new_dims.append(len(self.rddl.objects[t_out]))
                
        # safeguard against any free types
        free = {sign_in[i][0] for i, p in enumerate(lhs) if p is None}
        if free:
            raise RDDLInvalidNumberOfArgumentsError(
                f'Variable <{var}> has unresolved parameter(s) {free}.'
                f'\n{msg}')
        
        # this is the necessary information for np.einsum
        lhs = ''.join(lhs)
        rhs = valid_symbols[:n_out]
        permute = lhs + ' -> ' + rhs
        identity = lhs == rhs
        new_dims = tuple(new_dims)
        
        if self.debug:
            warnings.warn(
                f'computing info for pvariable transform:' 
                f'\n\texpr     ={expr}'
                f'\n\tinputs   ={sign_in}'
                f'\n\ttargets  ={sign_out}'
                f'\n\tnew axes ={new_dims}'
                f'\n\teinsum   ={permute}\n'
            )
        
        return (permute, identity, new_dims)
    
    def expand(self, var: str, values: np.ndarray) -> Dict[str, Value]:
        '''Produces a grounded representation of the pvariable var from its 
        tensor representation. The output is a dict whose keys are grounded
        representations of the var, and values are read from the tensor.
        
        :param var: the pvariable
        :param values: the tensor whose values correspond to those of var(?...)        
        '''
        keys = self.grounded[var]
        values = np.ravel(values)
        return dict(zip(keys, values, strict=True))
