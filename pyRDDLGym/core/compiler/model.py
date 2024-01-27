from abc import ABCMeta
import itertools
import numpy as np
from pprint import pprint
from typing import Dict, Iterable, List, Tuple

from pyRDDLGym.core.debug.exception import (
    RDDLInvalidNumberOfArgumentsError,
    RDDLInvalidObjectError,
    RDDLMissingCPFDefinitionError,
    RDDLRepeatedVariableError,
    RDDLTypeError,
    RDDLUndefinedCPFError,
    RDDLUndefinedVariableError,
    RDDLValueOutOfRangeError
)
from pyRDDLGym.core.parser.expr import Expression, Value


class RDDLPlanningModel(metaclass=ABCMeta):
    '''The base class representing all RDDL domain + instance.
    '''
    
    FLUENT_SEP = '___'
    OBJECT_SEP = '__'
    NEXT_STATE_SYM = '\''

    def __init__(self):
        
        # base
        self._AST = None
        
        # objects
        self._type_to_objects = None
        self._object_to_type = None
        self._object_to_index = None
        self._enum_types = None
        
        # variable info
        self._variable_types = None
        self._variable_ranges = None
        self._variable_params = None
        self._variable_defaults = None
        self._variable_groundings = None
        self._grounded_name_to_pvar_name = None        
        
        self._non_fluents = None        
        self._state_fluents = None
        self._state_ranges = None
        self._prev_state = None
        self._next_state = None    
        self._action_fluents = None
        self._action_ranges = None        
        self._derived_fluents = None
        self._interm_fluents = None        
        self._observ_fluents = None
        self._observ_ranges = None
        
        # cpf info
        self._cpfs = None
        self._level_to_cpfs = None
        self._cpf_to_level = None
        self._reward = None
        
        # constraint info
        self._terminations = None
        self._preconditions = None
        self._invariants = None
        
        # instance info
        self._discount = None
        self._horizon = None
        self._max_allowed_actions = None        
        
    # ===========================================================================
    # base properties
    # ===========================================================================
    
    @property
    def ast(self):
        return self._AST

    @ast.setter
    def ast(self, value):
        self._AST = value
    
    @property
    def domain_name(self):
        if self.ast is None:
            return None
        else:
            return self.ast.domain.name
    
    @property
    def instance_name(self):
        if self.ast is None:
            return None
        else:
            return self.ast.instance.name
    
    # ===========================================================================
    # objects
    # ===========================================================================
    
    @property
    def type_to_objects(self):
        return self._type_to_objects

    @type_to_objects.setter
    def type_to_objects(self, value):
        self._type_to_objects = value
    
    @property
    def object_to_type(self):
        return self._object_to_type

    @object_to_type.setter
    def object_to_type(self, value):
        self._object_to_type = value
    
    @property
    def object_to_index(self):
        return self._object_to_index

    @object_to_index.setter
    def object_to_index(self, val):
        self._object_to_index = val
    
    @property
    def enum_types(self):
        return self._enum_types

    @enum_types.setter
    def enum_types(self, value):
        self._enum_types = value
    
    # ===========================================================================
    # variables
    # ===========================================================================
    
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
    def variable_params(self):
        return self._variable_params

    @variable_params.setter
    def variable_params(self, val):
        self._variable_params = val
    
    @property
    def variable_defaults(self):
        return self._variable_defaults
    
    @variable_defaults.setter
    def variable_defaults(self, val):
        self._variable_defaults = val
        
    @property
    def variable_groundings(self):
        return self._variable_groundings

    @variable_groundings.setter
    def variable_groundings(self, val):
        self._variable_groundings = val
    
    @property
    def grounded_name_to_pvar_name(self):
        return self._grounded_name_to_pvar_name

    @grounded_name_to_pvar_name.setter
    def grounded_name_to_pvar_name(self, val):
        self._grounded_name_to_pvar_name = val
        
    @property
    def non_fluents(self):
        return self._non_fluents

    @non_fluents.setter
    def non_fluents(self, val):
        self._non_fluents = val

    @property
    def state_fluents(self):
        return self._state_fluents

    @state_fluents.setter
    def state_fluents(self, val):
        self._state_fluents = val
    
    @property
    def state_ranges(self):
        return self._state_ranges

    @state_ranges.setter
    def state_ranges(self, value):
        self._state_ranges = value

    @property
    def next_state(self):
        return self._next_state

    @next_state.setter
    def next_state(self, val):
        self._next_state = val

    @property
    def prev_state(self):
        return self._prev_state

    @prev_state.setter
    def prev_state(self, val):
        self._prev_state = val

    @property
    def action_fluents(self):
        return self._action_fluents

    @action_fluents.setter
    def action_fluents(self, val):
        self._action_fluents = val

    @property
    def action_ranges(self):
        return self._action_ranges

    @action_ranges.setter
    def action_ranges(self, value):
        self._action_ranges = value
        
    @property
    def derived_fluents(self):
        return self._derived_fluents

    @derived_fluents.setter
    def derived_fluents(self, val):
        self._derived_fluents = val

    @property
    def interm_fluents(self):
        return self._interm_fluents

    @interm_fluents.setter
    def interm_fluents(self, val):
        self._interm_fluents = val
    
    @property
    def observ_fluents(self):
        return self._observ_fluents

    @observ_fluents.setter
    def observ_fluents(self, value):
        self._observ_fluents = value

    @property
    def observ_ranges(self):
        return self._observ_ranges

    @observ_ranges.setter
    def observ_ranges(self, value):
        self._observ_ranges = value
    
    # ===========================================================================
    # CPFs
    # ===========================================================================
    
    @property
    def cpfs(self):
        return self._cpfs

    @cpfs.setter
    def cpfs(self, val):
        self._cpfs = val

    @property
    def level_to_cpfs(self):
        return self._level_to_cpfs

    @level_to_cpfs.setter
    def level_to_cpfs(self, val):
        self._level_to_cpfs = val

    @property
    def cpf_to_level(self):
        return self._cpf_to_level

    @cpf_to_level.setter
    def cpf_to_level(self, val):
        self._cpf_to_level = val
    
    @property
    def reward(self):
        return self._reward

    @reward.setter
    def reward(self, val):
        self._reward = val
    
    # ===========================================================================
    # constraints
    # ===========================================================================
    
    @property
    def terminations(self):
        return self._terminations

    @terminations.setter
    def terminations(self, value):
        self._terminations = value

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
    
    # ===========================================================================
    # init-block
    # ===========================================================================
    
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
    
    # ===========================================================================
    # utility methods
    # ===========================================================================
    
    def ground_name(self, name: str, objects: Iterable[str]) -> str:
        '''Given a variable name and list of objects as arguments, produces a 
        grounded representation <variable>___<obj1>__<obj2>__...'''
        PRIME = RDDLPlanningModel.NEXT_STATE_SYM
        is_primed = name.endswith(PRIME)
        var = name
        if is_primed:
            var = var[:-len(PRIME)]
        if objects is not None and objects:
            objects = RDDLPlanningModel.OBJECT_SEP.join(objects)
            var += RDDLPlanningModel.FLUENT_SEP + objects
        if is_primed:
            var += PRIME
        return var
    
    def parse(self, expr: str) -> Tuple[str, List[str]]:
        '''Parses an expression of the form <name> or <name>___<type1>__<type2>...)
        into a tuple of <name>, [<type1>, <type2>, ...].'''
        PRIME = RDDLPlanningModel.NEXT_STATE_SYM
        is_primed = expr.endswith(PRIME)
        if is_primed:
            expr = expr[:-len(PRIME)]
        var, *objects = expr.split(RDDLPlanningModel.FLUENT_SEP)
        if objects:
            if len(objects) != 1:
                raise RDDLInvalidObjectError(f'Invalid pvariable expression {expr}.')
            objects = objects[0].split(RDDLPlanningModel.OBJECT_SEP)
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
            objects = self.type_to_objects.get(ptype, None)
            if objects is None:
                raise RDDLTypeError(
                    f'Type <{ptype}> is not valid, '
                    f'must be one of {set(self.type_to_objects.keys())}.')
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
        
    def is_valid_type(self, ptype: str) -> bool:
        '''Returns whether the given string is a valid type.'''
        return ptype in self.enum_types or ptype in self.type_to_objects
        
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
            if name not in self.object_to_type:
                raise RDDLInvalidObjectError(
                    f'Object <@{name}> is not defined, '
                    f'must be one of {set(self.object_to_type.keys())}. {msg}')
            return True
        
        # this could either be an object or variable
        # object if a variable does not exist with the same name
        # object if a variable has the same name but free parameters
        # ambiguous if a variable has the same name but no free parameters
        if name in self.object_to_type:
            params = self.variable_params.get(name, None)
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
        ptypes = self.variable_params.get(var, None)
        if ptypes is None:
            return False
        if objects is None:
            objects = []
        if len(ptypes) != len(objects):
            return False
        for (ptype, obj) in zip(ptypes, objects):
            ptype_of_obj = self.object_to_type.get(obj, None)
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
        index_of_obj = self.object_to_index
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
        objects = self.type_to_objects
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
        groundings = self.variable_groundings[var]
        values = np.ravel(values, order='C')
        if len(groundings) != values.size:
            raise RDDLInvalidNumberOfArgumentsError(
                f'Variable <{var}> requires {len(groundings)} argument(s), '
                f'got {values.size}.')
        return zip(groundings, values)
    
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
        return {grounded_name: prange 
                for (action, prange) in dict_ranges.items()
                    for grounded_name in self.variable_groundings[action]}
    
    def grounded_non_fluents(self):
        return self.ground_values_from_dict(self.non_fluents)
    
    def grounded_state_fluents(self):
        return self.ground_values_from_dict(self.state_fluents)
    
    def grounded_state_ranges(self):
        return self.ground_ranges_from_dict(self.state_ranges)
    
    def grounded_action_fluents(self):
        return self.ground_values_from_dict(self.action_fluents)
    
    def grounded_action_ranges(self):
        return self.ground_ranges_from_dict(self.action_ranges)
    
    def grounded_observ_fluents(self):
        return self.ground_values_from_dict(self.observ_fluents)
    
    def grounded_observ_ranges(self):
        return self.ground_ranges_from_dict(self.observ_ranges)
    
    def print_expr(self):
        '''Returns a dictionary containing string representations of all 
        expressions in the current RDDL.
        '''
        printed = {
            'cpfs': {name: str(expr) for (name, (_, expr)) in self.cpfs.items()},
            'reward': str(self.reward),
            'invariants': [str(expr) for expr in self.invariants],
            'preconditions': [str(expr) for expr in self.preconditions],
            'terminations': [str(expr) for expr in self.terminations]
        }
        return printed
    
    def dump_to_stdout(self):
        '''Dumps a pretty printed representation of the current model to stdout.
        '''
        pprint(vars(self))

    
class RDDLGroundedModel(RDDLPlanningModel):
    '''A class representing a RDDL domain + instance in grounded form.
    '''

    def __init__(self):
        super().__init__()


class RDDLLiftedModel(RDDLPlanningModel):
    '''Represents a RDDL domain + instance in lifted form.'''
    
    PRIMITIVE_TYPES = {
        'int': int,
        'real': float,
        'bool': bool
    }
    
    def __init__(self, rddl):
        super(RDDLLiftedModel, self).__init__()
        
        self.ast = rddl
        
        self._extract_objects()            
        self._extract_variable_information()  
                
        self._extract_states()
        self._extract_actions()
        self._extract_derived_and_interm()
        self._extract_observ()
        self._extract_non_fluents()
        
        self.reward = self.ast.domain.reward
        self._extract_cpfs()
        self._extract_constraints()
        
        self._extract_horizon()
        self._extract_discount()
        self._extract_max_actions()
        
    def _extract_objects(self):
        
        # objects of each type as defined in the non-fluents {..} block
        ast_objects = self.ast.non_fluents.objects
        if not ast_objects or ast_objects[0] is None:
            ast_objects = []
        ast_objects = dict(ast_objects)
        
        # record the set of objects of each type defined in the domain
        objects, objects_rev, enums = {}, {}, set() 
        for (name, pvalues) in self.ast.domain.types:
            
            # check duplicated type
            if name in objects:
                raise RDDLInvalidObjectError(
                    f'Type <{name}> is repeated in types block.')
            
            # instance object
            if pvalues == 'object': 
                objects[name] = ast_objects.get(name, None)
                if objects[name] is None:
                    raise RDDLInvalidObjectError(
                        f'Type <{name}> has no objects defined in the instance.')
                objects[name] = list(map(self.object_name, objects[name]))
                
            # domain object
            else: 
                objects[name] = list(map(self.object_name, pvalues))
                enums.add(name)
        
            # make sure types do not share an object - record type of each object
            for obj in objects[name]:
                if obj in objects_rev:
                    if objects_rev[obj] == name:
                        raise RDDLInvalidObjectError(
                            f'Type <{name}> contains duplicated object <{obj}>.')
                    else:
                        raise RDDLInvalidObjectError(
                            f'Types <{name}> and <{objects_rev[obj]}> '
                            f'can not share the same object <{obj}>.')
                objects_rev[obj] = name
        
        # check that all types in instance are declared in domain
        for ptype in ast_objects:
            if ptype not in objects:
                raise RDDLInvalidObjectError(
                    f'Type <{ptype}> defined in the instance is not declared in '
                    f'the domain.')
        
        # maps each object to its canonical order as it appears in definition
        self.type_to_objects = objects
        self.object_to_type = objects_rev
        self.object_to_index = {obj: i 
                                for objs in objects.values() 
                                    for (i, obj) in enumerate(objs)}
        self.enum_types = enums
    
    def _extract_variable_information(self):
        
        # extract some basic information about variables in the domain:
        # 1. object parameters needed to evaluate
        # 2. type (e.g. state-fluent, action-fluent)
        # 3. range (e.g. int, real, bool, type)
        # 4. save default values
        var_params, var_types, var_ranges, var_ground, var_default = {}, {}, {}, {}, {}
        for pvar in self.ast.domain.pvariables: 
            
            # make sure variable is not defined more than once        
            primed_name = name = pvar.name
            if name in var_params:
                raise RDDLRepeatedVariableError(
                    f'{pvar.fluent_type} <{name}> has the same name as '
                    f'another {var_types[name]} variable.')
                
            # make sure name does not contain separators
            SEPARATORS = [RDDLPlanningModel.FLUENT_SEP, RDDLPlanningModel.OBJECT_SEP] 
            for separator in SEPARATORS:
                if separator in name:
                    raise RDDLInvalidObjectError(
                        f'Variable name <{name}> contains an '
                        f'illegal separator {separator}.')
            
            # record its type, parameters and range
            if pvar.is_state_fluent():
                primed_name = name + RDDLPlanningModel.NEXT_STATE_SYM
            ptypes = pvar.param_types
            if ptypes is None:
                ptypes = []
            var_params[name] = var_params[primed_name] = ptypes        
            var_types[name] = pvar.fluent_type
            if pvar.is_state_fluent():
                var_types[primed_name] = 'next-state-fluent'                
            var_ranges[name] = var_ranges[primed_name] = pvar.range 
            var_ground[name] = list(self.ground_names(name, ptypes))
            if pvar.is_state_fluent():
                var_ground[primed_name] = list(self.ground_names(primed_name, ptypes))
            
            # save default value
            var_default[name] = self._extract_default_value(pvar)
            
        # maps each variable (as appears in RDDL) to list of grounded variations
        self.variable_types = var_types
        self.variable_ranges = var_ranges
        self.variable_params = var_params
        self.variable_defaults = var_default
        self.variable_groundings = var_ground
        
        self.grounded_name_to_pvar_name = {
            grounded_name: var
            for (var, grounded_names) in var_ground.items()
                for grounded_name in grounded_names
        }
    
    def _grounded_dict_to_dict_of_list(self, grounded_dict):
        new_dict = {}
        for (var, values_dict) in grounded_dict.items():
            grounded_names = list(values_dict.values())
            if self.variable_params[var]:
                new_dict[var] = grounded_names
            else:
                assert len(grounded_names) == 1
                new_dict[var] = grounded_names[0]
        return new_dict
    
    def _extract_default_value(self, pvar):
        prange, default = pvar.range, pvar.default
        if default is not None:
            
            # is an object
            if isinstance(default, str):
                default = self.object_name(default)
                if default not in self.type_to_objects.get(prange, set()):
                    raise RDDLTypeError(
                        f'Default value {default} of variable <{pvar.name}> '
                        f'is not an object of type <{prange}>.')                     
            
            # is a primitive
            else:
                dtype = RDDLLiftedModel.PRIMITIVE_TYPES.get(prange, None)
                if dtype is None:
                    raise RDDLTypeError(
                        f'Type <{prange}> of variable <{pvar.name}> is an object '
                        f'or enumerated type, but assigned a default value '
                        f'{default}.')
                
                # type cast
                if not np.can_cast(default, dtype):
                    raise RDDLTypeError(
                        f'Default value {default} of variable <{pvar.name}> '
                        f'cannot be cast to required type <{prange}>.')
                default = dtype(default)
            
        return default
    
    def _extract_states(self):
        
        # get the information for each state from the domain
        PRIME = RDDLPlanningModel.NEXT_STATE_SYM
        states, statesranges, nextstates, prevstates = {}, {}, {}, {}
        for pvar in self.ast.domain.pvariables:
            if pvar.is_state_fluent():
                name, ptypes = pvar.name, pvar.param_types
                statesranges[name] = pvar.range
                nextstates[name] = name + PRIME
                prevstates[name + PRIME] = name
                default = self.variable_defaults[name]           
                states[name] = {gname: default 
                                for gname in self.ground_names(name, ptypes)} 
                
        # update the state values with the values in the instance
        init_state_info = getattr(self.ast.instance, 'init_state', [])
        for ((name, params), value) in init_state_info:
                
            # check whether name is a valid state-fluent
            if name not in states:
                raise RDDLUndefinedVariableError(
                    f'Variable <{name}> referenced in init-state block '
                    f'is not a valid state-fluent.')
                    
            # extract the grounded name and check that parameters are valid
            if params is not None:
                params = list(map(self.object_name, params))
            gname = self.ground_name(name, params)
            if gname not in states[name]:
                raise RDDLInvalidObjectError(
                    f'Parameter(s) {params} of state-fluent <{name}> '
                    f'declared in the init-state block are not valid.')
                
            # make sure value is correct type
            if isinstance(value, str):
                value = self.object_name(value)
                value_type = self.object_to_type.get(value, None)
                required_type = statesranges[name]
                if value_type != required_type:
                    if value_type is None:
                        raise RDDLInvalidObjectError(
                            f'State-fluent <{name}> of type <{required_type}> '
                            f'is initialized in init-state block with undefined '
                            f'object <{value}>.')
                    else:
                        raise RDDLInvalidObjectError(
                            f'State-fluent <{name}> of type <{required_type}> '
                            f'is initialized in init-state block with object '
                            f'<{value}> of type {value_type}.')
                        
            states[name][gname] = value
                
        # state dictionary associates the variable lifted name with a list of
        # values for all variations of parameter arguments in C-based order
        # if the fluent does not have parameters, then the value is a scalar
        self.state_fluents = self._grounded_dict_to_dict_of_list(states)  
        self.state_ranges = statesranges   
        self.next_state = nextstates
        self.prev_state = prevstates
    
    def _value_list_or_scalar_from_default(self, pvar):
        default = self.variable_defaults[pvar.name]
        ptypes = pvar.param_types   
        if ptypes is None:
            return default
        else:
            num_variations = 1
            for ptype in ptypes:
                num_variations *= len(self.type_to_objects[ptype])
            return [default] * num_variations
    
    def _extract_actions(self):
        
        # actions are stored similar to states described above
        actions, actionsranges = {}, {}
        for pvar in self.ast.domain.pvariables:
            if pvar.is_action_fluent():
                actionsranges[pvar.name] = pvar.range
                actions[pvar.name] = self._value_list_or_scalar_from_default(pvar)
        self.action_fluents = actions
        self.action_ranges = actionsranges
    
    def _extract_derived_and_interm(self):
        
        # derived and interm are stored similar to states described above
        derived, interm = {}, {}
        for pvar in self.ast.domain.pvariables:
            if pvar.is_derived_fluent():
                derived[pvar.name] = self._value_list_or_scalar_from_default(pvar)
            elif pvar.is_intermediate_fluent():
                interm[pvar.name] = self._value_list_or_scalar_from_default(pvar)
        self.derived_fluents = derived
        self.interm_fluents = interm
    
    def _extract_observ(self):
        
        # observed are stored similar to states described above
        observ, observranges = {}, {}
        for pvar in self.ast.domain.pvariables:
            if pvar.is_observ_fluent():
                observranges[pvar.name] = pvar.range
                observ[pvar.name] = self._value_list_or_scalar_from_default(pvar)
        self.observ_fluents = observ
        self.observ_ranges = observranges
        
    def _extract_non_fluents(self):
        
        # extract non-fluents values from the domain defaults
        non_fluents = {}
        for pvar in self.ast.domain.pvariables:
            if pvar.is_non_fluent():
                name, ptypes = pvar.name, pvar.param_types
                default = self.variable_defaults[name] 
                non_fluents[name] = {gname: default
                                     for gname in self.ground_names(name, ptypes)}
        
        # update non-fluent values with the values in the instance
        non_fluent_info = getattr(self.ast.non_fluents, 'init_non_fluent', [])
        for ((name, params), value) in non_fluent_info:
                
            # check whether name is a valid non-fluent
            grounded_names = non_fluents.get(name, None)
            if grounded_names is None:
                raise RDDLUndefinedVariableError(
                    f'Variable <{name}> referenced in non-fluents block '
                    f'is not a valid non-fluent.')
                
            # extract the grounded name and check that parameters are valid
            if params is not None:
                params = list(map(self.object_name, params))
            gname = self.ground_name(name, params)                           
            if gname not in grounded_names:
                raise RDDLInvalidObjectError(
                    f'Parameter(s) {params} of non-fluent <{name}> '
                    f'as declared in the non-fluents block are not valid.')
                    
            # make sure value is correct type
            if isinstance(value, str):
                value = self.object_name(value)
                value_type = self.object_to_type.get(value, None)
                required_type = self.variable_ranges[name]
                if value_type != required_type:
                    if value_type is None:
                        raise RDDLInvalidObjectError(
                            f'Non-fluent <{name}> of type <{required_type}> '
                            f'is initialized in non-fluents block with '
                            f'undefined object <{value}>.')
                    else:
                        raise RDDLInvalidObjectError(
                            f'Non-fluent <{name}> of type <{required_type}> '
                            f'is initialized in non-fluents block with object '
                            f'<{value}> of type <{value_type}>.')
                        
            grounded_names[gname] = value
                                        
        # non-fluents are stored similar to states described above
        self.non_fluents = self._grounded_dict_to_dict_of_list(non_fluents)
    
    def _extract_cpfs(self):
        cpfs = {}
        for cpf in self.ast.domain.cpfs[1]:
            name, objects = cpf.pvar[1]
            
            # make sure the CPF is defined in pvariables {...} scope
            types = self.variable_params.get(name, None)
            if types is None:
                raise RDDLUndefinedCPFError(
                    f'CPF <{name}> is not defined in pvariable block.')
            
            # make sure the number of parameters matches that in cpfs {...}
            if objects is None:
                objects = []
            if len(types) != len(objects):
                raise RDDLInvalidNumberOfArgumentsError(
                    f'l.h.s. of expression for CPF <{name}> requires '
                    f'{len(types)} parameter(s), got {objects}.')
            
            # CPFs are stored as dictionary that associates cpf name with a pair 
            # the first element is the type argument list
            # the second element is the AST expression
            objects = list(zip(objects, types))
            cpfs[name] = (objects, cpf.expr)
        
        # make sure all CPFs have a valid expression in cpfs {...}
        for (var, fluent_type) in self.variable_types.items():
            if fluent_type in {'derived-fluent', 'interm-fluent',
                               'next-state-fluent', 'observ-fluent'} \
            and var not in cpfs:
                raise RDDLMissingCPFDefinitionError(
                    f'{fluent_type} CPF <{var}> is not defined in cpfs block.')
                    
        self.cpfs = cpfs
    
    def _extract_constraints(self):
        self.terminations = getattr(self.ast.domain, 'terminals', [])
        self.preconditions = getattr(self.ast.domain, 'preconds', [])
        self.invariants = getattr(self.ast.domain, 'invariants', [])

    def _extract_horizon(self):
        horizon = self.ast.instance.horizon
        if not (horizon >= 0):
            raise RDDLValueOutOfRangeError(
                f'Horizon {horizon} in the instance is not >= 0.')
        self.horizon = horizon

    def _extract_max_actions(self):
        numactions = getattr(self.ast.instance, 'max_nondef_actions', 'pos-inf')
        if numactions == 'pos-inf':
            self.max_allowed_actions = sum(map(np.size, self.action_fluents.values()))
        else:
            self.max_allowed_actions = int(numactions)

    def _extract_discount(self):
        discount = self.ast.instance.discount
        if not (0. <= discount):
            raise RDDLValueOutOfRangeError(
                f'Discount factor {discount} in the instance is not >= 0')
        self.discount = discount
    
