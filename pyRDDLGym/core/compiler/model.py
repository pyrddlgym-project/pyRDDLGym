from abc import ABCMeta
import itertools
import numpy as np
from typing import Dict, Iterable, List, Tuple

from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidNumberOfArgumentsError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidObjectError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLTypeError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLUndefinedVariableError

from pyRDDLGym.Core.Parser.expr import Expression, Value


class PlanningModel(metaclass=ABCMeta):
    '''The base class representing all RDDL domain + instance.
    '''
    
    FLUENT_SEP = '___'
    OBJECT_SEP = '__'
    NEXT_STATE_SYM = '\''

    def __init__(self):
        self._AST = None
        self._nonfluents = None
        self._states = None
        self._statesranges = None
        self._nextstates = None
        self._prevstates = None
        self._initstate = None
        self._actions = None
        self._objects = None
        self._objects_rev = None
        self._enums = None
        self._actionsranges = None
        self._derived = None
        self._interm = None
        self._observ = None
        self._observranges = None
        self._cpfs = None
        self._cpforder = None
        self._gvar_to_cpforder = None
        self._reward = None
        self._terminals = None
        self._preconditions = None
        self._invariants = None
        self._gvar_to_pvar = None
        self._pvar_to_type = None
        self._gvar_to_type = None

        self._max_allowed_actions = None
        self._horizon = None
        self._discount = None
        
        self._param_types = None
        self._variable_types = None
        self._variable_ranges = None
        
        self._grounded_names = None
        self._index_of_object = None
        self._default_values = None
        
    # ===========================================================================
    # properties
    # ===========================================================================
    
    def SetAST(self, AST):
        self._AST = AST
    
    def domainName(self):
        if self._AST is None:
            return None
        else:
            return self._AST.domain.name
    
    def instanceName(self):
        if self._AST is None:
            return None
        else:
            return self._AST.instance.name
        
    @property
    def objects(self):
        return self._objects

    @objects.setter
    def objects(self, value):
        self._objects = value
    
    @property
    def objects_rev(self):
        return self._objects_rev

    @objects_rev.setter
    def objects_rev(self, value):
        self._objects_rev = value
    
    @property
    def enums(self):
        return self._enums

    @enums.setter
    def enums(self, value):
        self._enums = value
    
    @property
    def nonfluents(self):
        return self._nonfluents

    @nonfluents.setter
    def nonfluents(self, val):
        self._nonfluents = val

    @property
    def states(self):
        return self._states

    @states.setter
    def states(self, val):
        self._states = val

    @property
    def statesranges(self):
        return self._statesranges

    @statesranges.setter
    def statesranges(self, value):
        self._statesranges = value

    @property
    def next_state(self):
        return self._nextstates

    @next_state.setter
    def next_state(self, val):
        self._nextstates = val

    @property
    def prev_state(self):
        return self._prevstates

    @prev_state.setter
    def prev_state(self, val):
        self._prevstates = val

    @property
    def init_state(self):
        return self._initstate

    @init_state.setter
    def init_state(self, val):
        self._initstate = val

    @property
    def actions(self):
        return self._actions

    @actions.setter
    def actions(self, val):
        self._actions = val

    @property
    def actionsranges(self):
        return self._actionsranges

    @actionsranges.setter
    def actionsranges(self, value):
        self._actionsranges = value

    @property
    def derived(self):
        return self._derived

    @derived.setter
    def derived(self, val):
        self._derived = val

    @property
    def interm(self):
        return self._interm

    @interm.setter
    def interm(self, val):
        self._interm = val

    @property
    def observ(self):
        return self._observ

    @observ.setter
    def observ(self, value):
        self._observ = value

    @property
    def observranges(self):
        return self._observranges

    @observranges.setter
    def observranges(self, value):
        self._observranges = value

    @property
    def cpfs(self):
        return self._cpfs

    @cpfs.setter
    def cpfs(self, val):
        self._cpfs = val

    @property
    def cpforder(self):
        return self._cpforder

    @cpforder.setter
    def cpforder(self, val):
        self._cpforder = val

    @property
    def gvar_to_cpforder(self):
        return self._gvar_to_cpforder

    @gvar_to_cpforder.setter
    def gvar_to_cpforder(self, val):
        self._gvar_to_cpforder = val
    
    @property
    def reward(self):
        return self._reward

    @reward.setter
    def reward(self, val):
        self._reward = val

    @property
    def terminals(self):
        return self._terminals

    @terminals.setter
    def terminals(self, value):
        self._terminals = value

    @property
    def preconditions(self):
        return self._preconditions

    @preconditions.setter
    def preconditions(self, val):
        self._preconditions = val

    @property
    def invariants(self):
        return self._invariants

    @invariants.setter
    def invariants(self, val):
        self._invariants = val
    
    @property
    def gvar_to_type(self):
        return self._gvar_to_type

    @gvar_to_type.setter
    def gvar_to_type(self, val):
        self._gvar_to_type = val
    
    @property
    def pvar_to_type(self):
        return self._pvar_to_type

    @pvar_to_type.setter
    def pvar_to_type(self, val):
        self._pvar_to_type = val

    @property
    def gvar_to_pvar(self):
        return self._gvar_to_pvar

    @gvar_to_pvar.setter
    def gvar_to_pvar(self, val):
        self._gvar_to_pvar = val

    @property
    def discount(self):
        return self._discount

    @discount.setter
    def discount(self, val):
        self._discount = val

    @property
    def horizon(self):
        return self._horizon

    @horizon.setter
    def horizon(self, val):
        self._horizon = val

    @property
    def max_allowed_actions(self):
        return self._max_allowed_actions

    @max_allowed_actions.setter
    def max_allowed_actions(self, val):
        self._max_allowed_actions = val
    
    @property
    def param_types(self):
        return self._param_types

    @param_types.setter
    def param_types(self, val):
        self._param_types = val
        
    @property
    def variable_types(self):
        return self._variable_types

    @variable_types.setter
    def variable_types(self, val):
        self._variable_types = val
    
    @property
    def variable_ranges(self):
        return self._variable_ranges

    @variable_ranges.setter
    def variable_ranges(self, val):
        self._variable_ranges = val
        
    @property
    def grounded_names(self):
        return self._grounded_names

    @grounded_names.setter
    def grounded_names(self, val):
        self._grounded_names = val
        
    @property
    def index_of_object(self):
        return self._index_of_object

    @index_of_object.setter
    def index_of_object(self, val):
        self._index_of_object = val
    
    @property
    def default_values(self):
        return self._default_values
    
    @default_values.setter
    def default_values(self, val):
        self._default_values = val
        
    # ===========================================================================
    # utility methods
    # ===========================================================================
    
    def ground_name(self, name: str, objects: Iterable[str]) -> str:
        '''Given a variable name and list of objects as arguments, produces a 
        grounded representation <variable>___<obj1>__<obj2>__...'''
        PRIME = PlanningModel.NEXT_STATE_SYM
        is_primed = name.endswith(PRIME)
        var = name
        if is_primed:
            var = var[:-len(PRIME)]
        if objects is not None and objects:
            objects = PlanningModel.OBJECT_SEP.join(objects)
            var += PlanningModel.FLUENT_SEP + objects
        if is_primed:
            var += PRIME
        return var
    
    def parse(self, expr: str) -> Tuple[str, List[str]]:
        '''Parses an expression of the form <name> or <name>___<type1>__<type2>...)
        into a tuple of <name>, [<type1>, <type2>, ...].'''
        PRIME = PlanningModel.NEXT_STATE_SYM
        is_primed = expr.endswith(PRIME)
        if is_primed:
            expr = expr[:-len(PRIME)]
        var, *objects = expr.split(PlanningModel.FLUENT_SEP)
        if objects:
            if len(objects) != 1:
                raise RDDLInvalidObjectError(f'Invalid pvariable expression {expr}.')
            objects = objects[0].split(PlanningModel.OBJECT_SEP)
        if is_primed:
            var += PRIME
        return var, objects
    
    def variations(self, ptypes: Iterable[str]) -> Iterable[Tuple[str, ...]]:
        '''Given a list of types, computes the Cartesian product of all object
        enumerations that corresponds to those types.'''
        if ptypes is None or not ptypes:
            return [()]
        objects_by_type = []
        for ptype in ptypes:
            objects = self.objects.get(ptype, None)
            if objects is None:
                raise RDDLTypeError(
                    f'Type <{ptype}> is not valid, '
                    f'must be one of {set(self.objects.keys())}.')
            objects_by_type.append(objects)
        return itertools.product(*objects_by_type)
    
    def ground_names(self, name: str, ptypes: Iterable[str]) -> Iterable[str]:
        '''Given a variable name and list of types, produces a new iterator
        whose elements are the grounded representations in the cartesian product 
        of all object enumerations that corresponds to those types.'''
        for objects in self.variations(ptypes):
            yield self.ground_name(name, objects)
    
    def is_free_variable(self, name: str) -> bool:
        '''Determines whether the quantity is a free variable (e.g., ?x).'''
        return name[0] == '?'
        
    def object_name(self, name: str) -> str:
        '''Extracts the internal object name representation. 
        Used to read objects with string quantification, e.g. @obj.'''
        if name[0] == '@':
            name = name[1:]
        return name
        
    def is_object(self, name: str, msg='') -> bool:
        '''Determines whether the given string points to an object.'''
        
        # this must be an object
        if name[0] == '@':
            name = name[1:]
            if name not in self.objects_rev:
                raise RDDLInvalidObjectError(
                    f'Object <@{name}> is not defined. {msg}')
            return True
        
        # this could either be an object or variable
        # object if a variable does not exist with the same name
        # object if a variable has the same name but free parameters
        # ambiguous if a variable has the same name but no free parameters
        if name in self.objects_rev:
            params = self.param_types.get(name, None)
            if params is not None and not params:
                raise RDDLInvalidObjectError(
                    f'Ambiguous reference to object or '
                    f'parameter-free variable with identical name <{name}>. {msg}')
            return True
        
        # this must be a variable or free parameter
        return False
    
    def is_compatible(self, var: str, objects: List[str]) -> bool:
        '''Determines whether or not the given variable can be evaluated
        for the given list of objects.'''
        ptypes = self.param_types.get(var, None)
        if ptypes is None:
            return False
        if objects is None:
            objects = []
        if len(ptypes) != len(objects):
            return False
        for (ptype, obj) in zip(ptypes, objects):
            ptype_of_obj = self.objects_rev.get(obj, None)
            if ptype_of_obj is None or ptype != ptype_of_obj:
                return False
        return True
    
    def is_non_fluent_expression(self, expr: Expression) -> bool:
        '''Determines whether or not expression is a non-fluent.'''
        if isinstance(expr, (tuple, list, set)):
            for arg in expr:
                if not self.is_non_fluent_expression(arg):
                    return False
            return True
        
        elif not isinstance(expr, Expression):
            return True
        
        etype, _ = expr.etype
        if etype == 'constant':
            return True
        
        elif etype == 'randomvar' or etype == 'randomvector':
            return False
        
        elif etype == 'pvar':
            name, pvars = expr.args
            
            # a free variable (e.g., ?x) is non-fluent
            if self.is_free_variable(name):
                return True
                
            # an object is non-fluent
            elif not pvars and self.is_object(name):
                return True
                        
            # variable must be well-defined and non-fluent
            else:
                
                # check well-defined
                var_type = self.variable_types.get(name, None)     
                if var_type is None:
                    raise RDDLUndefinedVariableError(
                        f'Variable <{name}> is not defined.')
                
                # check nested fluents
                if pvars is None:
                    pvars = []
                return var_type == 'non-fluent' \
                    and self.is_non_fluent_expression(pvars)
        
        else:
            return self.is_non_fluent_expression(expr.args)
    
    def indices(self, objects: Iterable[str], msg: str='') -> Tuple[int, ...]:
        '''Returns the canonical indices of a sequence of objects according to the
        order they are listed in the instance
        
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
                        f'must be one of {set(index_of_obj.keys())}. {msg}')
    
    def object_counts(self, types: Iterable[str], msg: str='') -> Tuple[int, ...]:
        '''Returns a tuple containing the number of objects of each type.
        
        :param types: a list of RDDL types
        :param msg: an error message to print in case the calculation fails
        '''
        objects = self.objects
        try:
            return tuple(len(objects[ptype]) for ptype in types)
        except:
            for ptype in types:
                if ptype not in objects:
                    raise RDDLTypeError(
                        f'Type <{ptype}> is not valid, '
                        f'must be one of {set(objects.keys())}. {msg}')
    
    def ground_values(self, var: str, values: Iterable[Value]) -> Iterable[Tuple[str, Value]]:
        '''Produces a sequence of pairs where the first element is the 
        grounded variables of var and the second are the corresponding values
        from values array.
        
        :param var: the pvariable as it appears in RDDL
        :param values: the values of var(?...) in C-based order      
        '''
        keys = self.grounded_names[var]
        values = np.ravel(values, order='C')
        if len(keys) != values.size:
            raise RDDLInvalidNumberOfArgumentsError(
                f'Variable <{var}> requires {len(keys)} argument(s), '
                f'got {values.size}.')
        return zip(keys, values)
    
    def ground_values_from_dict(self, dict_values: Dict[str, object]) -> Dict[str, Value]:
        '''Converts a dictionary of values such as nonfluents, states, observ
        which has entries <var>: <list of values> to a grounded <var>: <value>
        form, where each variable <var> is grounded out.
        
        :param dict_values: the value dictionary, where values are scalars
        or lists in C-based order
        '''
        grounded = {}
        for (var, values) in dict_values.items():
            grounded.update(self.ground_values(var, values))
        return grounded
    
    def ground_ranges_from_dict(self, dict_ranges: Dict[str, str]) -> Dict[str, str]:
        '''Converts a dictionary of ranges such as statesranges
        which has entries <var>: <range> to a grounded <var>: <range>
        form, where each variable <var> is grounded out.
        
        :param dict_ranges: the ranges dictionary mapping variable to range
        '''
        return {name: prange 
                for (action, prange) in dict_ranges.items()
                    for name in self.grounded_names[action]}
    
    def groundnonfluents(self):
        return self.ground_values_from_dict(self.nonfluents)
    
    def groundstates(self):
        return self.ground_values_from_dict(self.states)
    
    def groundstatesranges(self):
        return self.ground_ranges_from_dict(self.statesranges)
    
    def groundactions(self):
        return self.ground_values_from_dict(self.actions)
    
    def groundactionsranges(self):
        return self.ground_ranges_from_dict(self.actionsranges)
    
    def groundobserv(self):
        return self.ground_values_from_dict(self.observ)
    
    def groundobservranges(self):
        return self.ground_ranges_from_dict(self.observranges)
    
    def print_expr(self):
        '''Returns a dictionary containing string representations of all 
        expressions in the current RDDL.
        '''
        printed = {}
        printed['cpfs'] = {name: str(expr) 
                           for (name, (_, expr)) in self.cpfs.items()}
        printed['reward'] = str(self.reward)
        printed['invariants'] = [str(expr) for expr in self.invariants]
        printed['preconditions'] = [str(expr) for expr in self.preconditions]
        printed['terminations'] = [str(expr) for expr in self.terminals]
        return printed
    
    def dump_to_stdout(self):
        '''Dumps a pretty printed representation of the current model to stdout.
        '''
        from pprint import pprint
        pprint(vars(self))

    
class RDDLGroundedModel(PlanningModel):
    '''A class representing a RDDL domain + instance in grounded form.
    '''

    def __init__(self):
        super().__init__()

