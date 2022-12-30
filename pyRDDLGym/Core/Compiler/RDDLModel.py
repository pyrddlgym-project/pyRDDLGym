from abc import ABCMeta
import itertools
from typing import Iterable, List, Tuple

from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidObjectError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLUndefinedVariableError

from pyRDDLGym.Core.Parser.expr import Expression


class PlanningModel(metaclass=ABCMeta):

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
        self._enum_types = None
        self._enum_literals = None
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
        
    def SetAST(self, AST):
        self._AST = AST

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
    def enum_types(self):
        return self._enum_types

    @enum_types.setter
    def enum_types(self, value):
        self._enum_types = value
    
    @property
    def enum_literals(self):
        return self._enum_literals

    @enum_literals.setter
    def enum_literals(self, value):
        self._enum_literals = value
    
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
    def is_grounded(self):
        return True
    
    def ground_name(self, name: str, objects: Iterable[str]) -> str:
        '''Given a variable name and list of objects as arguments, produces a 
        grounded representation <variable>_<obj1>_<obj2>_...
        '''
        is_primed = name.endswith('\'')
        var = name
        if is_primed:
            var = var[:-1]
        if objects is not None and objects:
            var += '_' + '_'.join(objects)
        if is_primed:
            var += '\''
        return var
    
    def parse(self, expr: str) -> Tuple[str, List[str]]:
        '''Parses an expression of the form <name> or <name>_<type1>_<type2>...)
        into a tuple of <name>, [<type1>, <type2>, ...].
        '''
        is_primed = expr.endswith('\'')
        if is_primed:
            expr = expr[:-1]
        tokens = expr.split('_')
        var = tokens[0]
        if is_primed:
            var += '\''
        objects = tokens[1:]
        return var, objects
    
    def variations(self, ptypes: Iterable[str]) -> Iterable[Tuple[str, ...]]:
        '''Given a list of types, computes the cartesian product of all object
        enumerations that corresponds to those types.
        '''
        if ptypes is None or not ptypes:
            return [()]
        objects_by_type = []
        for ptype in ptypes:
            objects = self.objects.get(ptype, None)
            if objects is None:
                raise RDDLInvalidObjectError(
                    f'Type <{ptype}> is not valid, '
                    f'must be one of {set(self.objects.keys())}.')
            objects_by_type.append(objects)
        return itertools.product(*objects_by_type)
    
    def grounded_names(self, name: str, ptypes: Iterable[str]) -> Iterable[str]:
        '''Given a variable name and list of types, produces a new iterator
        whose elements are the grounded representations in the cartesian product 
        of all object enumerations that corresponds to those types.
        '''
        for objects in self.variations(ptypes):
            yield self.ground_name(name, objects)
        
    def is_compatible(self, var: str, objects: List[str]) -> bool:
        '''Determines whether or not the given variable can be evaluated
        for the given list of objects.
        '''
        ptypes = self.param_types.get(var, None)
        if ptypes is None:
            return False
        if objects is None:
            objects = []
        if len(ptypes) != len(objects):
            return False
        for ptype, obj in zip(ptypes, objects):
            if obj not in self.objects_rev or ptype != self.objects_rev[obj]:
                return False
        return True
    
    def is_non_fluent_expression(self, expr: Expression) -> bool:
        '''Determines whether or not expression is a non-fluent.
        '''
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
        
        elif etype == 'randomvar':
            return False
        
        elif etype == 'pvar':
            name = expr.args[0]
            if name in self.enum_literals:
                return True  # enum literal
            
            else:
                var = self.parse(name)[0]  # pvariable
                var_type = self.variable_types.get(var, None)      
                if var_type is None:
                    raise RDDLUndefinedVariableError(
                        f'Variable or literal <{var}> is not defined, '
                        f'must be an enum literal in {self.enum_literals} '
                        f'or one of {set(self.variable_types.keys())}.')
                return var_type == 'non-fluent'
        
        else:
            for arg in expr.args:
                if not self.is_non_fluent_expression(arg):
                    return False
            return True

    
class RDDLModel(PlanningModel):

    def __init__(self):
        super().__init__()

