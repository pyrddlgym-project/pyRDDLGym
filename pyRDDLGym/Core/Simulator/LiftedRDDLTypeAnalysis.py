import itertools
import numpy as np
import re
from typing import Dict, Iterable, List, Set, Tuple, Union
import warnings

from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidNumberOfArgumentsError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidObjectError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLNotImplementedError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLParseError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLTypeError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLUndefinedVariableError
from pyRDDLGym.Core.Parser.expr import Value
from pyRDDLGym.Core.Parser.rddl import RDDL


class LiftedRDDLTypeAnalysis:
    '''Utility class that takes a RDDL domain, compiles its type and object info,
    and provides a set of utility functions to query information about them. It
    can also assist in producing tensor representations of variables that have
    type arguments, e.g. x(?o) maps to a 1-D tensor, y(?p, ?q) to a 2-D, etc...
    '''
    
    VALID_SYMBOLS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    def __init__(self, rddl: RDDL, debug: bool=False):
        '''Creates a new Type analysis object for the given rddl.
        
        :param rddl: a RDDL domain to compile
        :param debug: whether to print information about the compilation
        '''
        self.rddl = rddl
        self.debug = debug
        
        self.domain = rddl.domain
        self.instance = rddl.instance
        self.non_fluents = rddl.non_fluents
        
        self._compile_types()
        self._compile_objects()
        
        self.action_pattern = re.compile(r'(\S*?)\((\S.*?)\)', re.VERBOSE)
        
    # ===========================================================================
    # compilation
    # ===========================================================================
    
    def _compile_types(self):
        self.pvar_types = {}
        for pvar in self.domain.pvariables:
            primed_name = name = pvar.name
            if pvar.is_state_fluent():
                primed_name = name + '\''
            ptypes = pvar.param_types
            if ptypes is None:
                ptypes = []
            self.pvar_types[name] = ptypes
            self.pvar_types[primed_name] = ptypes
            
        self.cpf_types = {}
        for cpf in self.domain.cpfs[1]:
            _, (name, objects) = cpf.pvar
            if objects is None:
                objects = [] 
            types = self.pvar_types[name]
            self.cpf_types[name] = [(o, types[i]) for i, o in enumerate(objects)]
        
        if self.debug:
            pvar = ''.join(f'\n\t\t{k}: {v}' for k, v in self.pvar_types.items())
            cpf = ''.join(f'\n\t\t{k}: {v}' for k, v in self.cpf_types.items())
            warnings.warn(
                f'compiling type info:'
                f'\n\tpvar types ={pvar}'
                f'\n\tcpf types  ={cpf}\n'
            )
    
    def _compile_objects(self): 
        self.objects = {}
        self.objects_to_index = {}
        self.objects_to_type = {}
        for name, ptype in self.non_fluents.objects:
            self.objects[name] = {obj: i for i, obj in enumerate(ptype)}
            for obj in ptype:
                if obj in self.objects_to_type:
                    other_name = self.objects_to_type[obj]
                    raise RDDLInvalidObjectError(
                        f'Types <{other_name}> and <{name}> '
                        f'can not share the same object <{obj}>.')
            self.objects_to_index.update(self.objects[name])
            self.objects_to_type.update({obj: name for obj in ptype})
        
        self.grounded = {}
        for var in self.cpf_types.keys():
            objects = [self.objects[ptype] for ptype in self.pvar_types[var]]
            variations = itertools.product(*objects)
            args = map(','.join, variations)
            if var.endswith('\''):
                var = var[:-1]
            self.grounded[var] = [f'{var}({arg})' if arg != '' else f'{var}' 
                                  for arg in args]
            
        if self.debug:
            obj = ''.join(f'\n\t\t{k}: {v}' for k, v in self.objects.items())
            warnings.warn(
                f'compiling object info:'
                f'\n\tobjects ={obj}\n'
            )
            
    # ===========================================================================
    # utility functions
    # ===========================================================================
    
    def count_type_args(self, var: str) -> int:
        '''Return the number of type arguments of pvariable var.'''
        return len(self.pvar_types.get(var, []))
    
    def coordinates(self, objects: Iterable[str], stack: str) -> Tuple[int, ...]:
        '''Converts a list of objects into their coordinate representation.
        
        :param objects: object instances corresponding to valid types defined
        in the RDDL domain
        :param stack: an error message to print in case the conversion fails.
        
        Examples:
        
        For a fluent x(?o) where ?o has objects {o1, o2, ... on},
        calling coordinates on [oi] will return (i - 1,).
        
        For a fluent x(?p, ?q) where ?p has objects {p1, ... pm}
        and ?q has objects {q1, ... qn}, calling coordinates on [pi, qj] will
        return (i - 1, j - 1).
        '''
        try:
            return tuple(self.objects_to_index[obj] for obj in objects)
        except:
            for obj in objects:
                if obj not in self.objects_to_index:
                    raise RDDLInvalidObjectError(
                        f'Object <{obj}> is not valid, '
                        f'must be one of {set(self.objects_to_index.keys())}.'
                        f'\n{stack}')
    
    def shape(self, types: Iterable[str], stack: str) -> Tuple[int, ...]:
        '''Given a list of RDDL types, returns the shape of a tensor
        that would hold all values of a pvariable with those type arguments
        
        :param types: a list of RDDL types
        :param stack: an error message to print in case the calculation fails
        
        Examples:
        
        Calling shape on type ?o with objects {o1, o2, ... on} will return (n,).
        
        Calling shape on types [?p, ?q] where ?p has objects {p1, ... pm}
        and ?q has objects {q1, ... qn} will return (m, n).
        
        '''
        try:
            return tuple(len(self.objects[ptype]) for ptype in types)
        except:
            for ptype in types:
                if ptype not in self.objects:
                    raise RDDLInvalidObjectError(
                        f'Type <{ptype}> is not valid, '
                        f'must be one of {set(self.objects.keys())}.'
                        f'\n{stack}')
    
    def is_compatible(self, var: str, objects: List[str]) -> bool:
        '''Determines whether or not the given pvariable var can be evaluated
        for the given list of objects.
        '''
        types = self.pvar_types[var]
        if objects is None:
            objects = []
        if len(types) != len(objects):
            return False
        for ptype, obj in zip(types, objects):
            if obj not in self.objects_to_type or ptype != self.objects_to_type[obj]:
                return False
        return True
    
    def validate_types(self, types: Iterable[Tuple[str, str]],
                       stack: str) -> List[str]:
        '''Given a list of tuples of the form (_, type), raises an exception
        if the types are not defined in the domain.
        
        :param types: a list of tuples (_, type)
        :param stack: a stack trace to print in case the validation fails
        '''
        fails = [ptype for _, ptype in types if ptype not in self.objects]
        if fails:
            raise RDDLUndefinedVariableError(
                f'Type(s) {fails} are not defined, '
                f'must be one of {set(self.objects.keys())}.'
                f'\n{stack}')
        
    def map(self, var: str,
            obj_in: List[str],
            sign_out: List[Tuple[str, str]],
            expr: str, stack: str) -> Tuple[str, bool, Tuple[int, ...]]:
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
        :param stack: a stack trace to print for error handling
        '''
                
        # check that the input objects match fluent type definition
        types_in = self.pvar_types.get(var, [])
        if obj_in is None:
            obj_in = []
        n_in = len(obj_in)
        n_req = len(types_in)
        if n_in != n_req:
            raise RDDLInvalidNumberOfArgumentsError(
                f'Variable <{var}> requires {n_req} parameters, '
                f'got {n_in}.\n{stack}')
            
        # reached limit on number of valid dimensions
        valid_symbols = LiftedRDDLTypeAnalysis.VALID_SYMBOLS
        n_max = len(valid_symbols)
        n_out = len(sign_out)
        if n_out > n_max:
            raise RDDLNotImplementedError(
                f'Up to {n_max}-D are supported, '
                f'but variable <{var}> is {n_out}-D.\n{stack}')
        
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
                            f'Argument <{i_in + 1}> of variable <{var}> '
                            f'expects object of type <{t_in}>, '
                            f'got <{o_out}> of type <{t_out}>.\n{stack}')
            
            # need to expand the shape of the value array
            if new_dim:
                lhs.append(valid_symbols[i_out])
                new_dims.append(len(self.objects[t_out]))
                
        # safeguard against any free types
        free = [types_in[i] for i, p in enumerate(lhs) if p is None]
        if free:
            raise RDDLInvalidNumberOfArgumentsError(
                f'Variable <{var}> has free parameter(s) {free}.\n{stack}')
        
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
    
    def tensor(self, ptypes: Iterable[str], 
               value: object, 
               dtype: object) -> Union[Value, np.ndarray]:
        '''Creates a scalar or tensor representation for the pvariable
        with the given types as arguments, and initial value.
        
        :param ptypes: the types of the pvariable
        :param value: the initial (e.g. default) value of the pvariable
        :param dtype: the return type of the tensor
        '''
        # check that the value is compatible with output tensor
        value_type = type(value)
        if not np.can_cast(value_type, dtype, casting='same_kind'):
            raise RDDLTypeError(
                f'Value <{value}> of type {value_type} '
                f'cannot be safely cast to type {dtype}.')
            
        if ptypes is None:
            return dtype(value)              
        else: 
            shape = self.shape(ptypes, '')
            return np.full(shape=shape, fill_value=value, dtype=dtype)
                
    def put(self, var: str, 
            objects: List[str], 
            value: Value, 
            out: np.ndarray) -> None:
        '''Places the value into the output array representation of pvariable
        var at the coordinates specified by the list of objects.
        
        :param var: the pvariable for which this substitution is done
        :param objects: the list of objects
        :param value: the value to set
        :param out: the array to write the value to at the coordinates specified
            by objects
        '''
        
        # check that the value is compatible with output tensor
        prange_val = type(value)
        prange_req = out.dtype
        if not np.can_cast(prange_val, prange_req, casting='same_kind'):
            raise RDDLTypeError(
                f'Value for pvariable <{var}> of type {prange_val} '
                f'cannot be safely cast to type {prange_req}.')
        
        # check that the arguments are correct  
        if not self.is_compatible(var, objects):
            ptypes = self.pvar_types[var]
            raise RDDLInvalidNumberOfArgumentsError(
                f'Type arguments {objects} for pvariable <{var}> '
                f'do not match definition {ptypes}.')
        
        # update the output tensor
        coords = self.coordinates(objects, '')
        out[coords] = value
        
    def expand(self, var: str, values: np.ndarray) -> Dict[str, Value]:
        '''Produces a grounded representation of the pvariable var from its 
        tensor representation. The output is a dict whose keys are grounded
        representations of the var, and values are read from the tensor.
        
        :param var: the pvariable
        :param values: the tensor whose values correspond to those of var(?...)
        
        Examples:
        - if var(?o) is a pvariable where ?o is one of {o1, ... on}, and values 
        is an array X of compatible shape, then the output is 
        {var(o1): X(0), var(o2): X(1), ... }.
        
        '''
        keys = self.grounded[var]
        values = np.ravel(values)
        return dict(zip(keys, values, strict=True))
    
    def parse(self, expr: str, valid_vars: Set[str]):
        '''Parses an expression of the form <name> or <name>(<type1>,<type2>...)
        into a tuple of <name>, [<type1>, <type2>, ...].
        
        :param expr: the string expression to be parsed
        :param valid_vars: checks to make sure name is in this set
        '''
        name, objects = expr, ''
        if '(' in expr:
            parsed = self.action_pattern.match(expr)
            if not parsed:
                raise RDDLParseError(
                    f'Pvariable expression <{expr}> is not valid, '
                    f'must be of the form <name> or <name>(<type1>,<type2>...).')
            name, objects = parsed.groups()
            
        if name not in valid_vars:
            raise RDDLUndefinedVariableError(
                f'Variable <{name}> is not valid, must be one of {valid_vars}.')
        
        objects = [obj.strip() for obj in objects.split(',')]
        objects = [obj for obj in objects if obj != '']
        return name, objects