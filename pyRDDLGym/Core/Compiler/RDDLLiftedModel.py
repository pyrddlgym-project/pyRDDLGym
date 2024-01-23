import copy
import numpy as np

from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidNumberOfArgumentsError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidObjectError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLMissingCPFDefinitionError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLRepeatedVariableError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLTypeError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLValueOutOfRangeError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLUndefinedCPFError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLUndefinedVariableError

from pyRDDLGym.Core.Compiler.RDDLModel import PlanningModel


class RDDLLiftedModel(PlanningModel):
    '''Represents a RDDL domain + instance in lifted form.'''
    
    PRIMITIVE_TYPES = {
        'int': int,
        'real': float,
        'bool': bool
    }
    
    def __init__(self, rddl):
        super(RDDLLiftedModel, self).__init__()
        
        self.SetAST(rddl)
        
        self._extract_objects()            
        self._extract_variable_information()  
                
        self._extract_states()
        self._extract_actions()
        self._extract_derived_and_interm()
        self._extract_observ()
        self._extract_non_fluents()
        
        self.reward = self._AST.domain.reward
        self._extract_cpfs()
        self._extract_constraints()
        
        self._extract_horizon()
        self._extract_discount()
        self._extract_max_actions()
        
    def _extract_objects(self):
        
        # objects of each type as defined in the non-fluents {..} block
        ast_objects = self._AST.non_fluents.objects
        if not ast_objects or ast_objects[0] is None:
            ast_objects = []
        ast_objects = dict(ast_objects)
        
        # record the set of objects of each type defined in the domain
        objects, objects_rev, enums = {}, {}, set() 
        for (name, pvalues) in self._AST.domain.types:
            
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
        self.index_of_object = {obj: i 
                                for objs in objects.values() 
                                    for (i, obj) in enumerate(objs)}
        
        self.objects, self.objects_rev, self.enums = objects, objects_rev, enums
    
    def _extract_variable_information(self):
        
        # extract some basic information about variables in the domain:
        # 1. object parameters needed to evaluate
        # 2. type (e.g. state-fluent, action-fluent)
        # 3. range (e.g. int, real, bool, type)
        # 4. save default values
        var_params, var_types, var_ranges, default_values = {}, {}, {}, {}
        for pvar in self._AST.domain.pvariables: 
            
            # make sure variable is not defined more than once        
            primed_name = name = pvar.name
            if name in var_params:
                raise RDDLRepeatedVariableError(
                    f'{pvar.fluent_type} <{name}> has the same name as '
                    f'another {var_types[name]} variable.')
                
            # make sure name does not contain separators
            SEPARATORS = [PlanningModel.FLUENT_SEP, PlanningModel.OBJECT_SEP] 
            for separator in SEPARATORS:
                if separator in name:
                    raise RDDLInvalidObjectError(
                        f'Variable name <{name}> contains an '
                        f'illegal separator {separator}.')
            
            # record its type, parameters and range
            if pvar.is_state_fluent():
                primed_name = name + PlanningModel.NEXT_STATE_SYM
            ptypes = pvar.param_types
            if ptypes is None:
                ptypes = []
            var_params[name] = var_params[primed_name] = ptypes        
            var_types[name] = pvar.fluent_type
            if pvar.is_state_fluent():
                var_types[primed_name] = 'next-state-fluent'                
            var_ranges[name] = var_ranges[primed_name] = pvar.range    
            
            # save default value
            default_values[name] = self._extract_default_value(pvar)
            
        # maps each variable (as appears in RDDL) to list of grounded variations
        self.grounded_names = {var: list(self.ground_names(var, types))
                               for (var, types) in var_params.items()}        
        
        self.param_types, self.variable_types, self.variable_ranges = \
            var_params, var_types, var_ranges
        self.default_values = default_values
    
    def _grounded_dict_to_dict_of_list(self, grounded_dict):
        new_dict = {}
        for (var, values_dict) in grounded_dict.items():
            grounded_names = list(values_dict.values())
            if self.param_types[var]:
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
                if default not in self.objects.get(prange, set()):
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
        PRIME = PlanningModel.NEXT_STATE_SYM
        states, statesranges, nextstates, prevstates = {}, {}, {}, {}
        for pvar in self._AST.domain.pvariables:
            if pvar.is_state_fluent():
                name, ptypes = pvar.name, pvar.param_types
                statesranges[name] = pvar.range
                nextstates[name] = name + PRIME
                prevstates[name + PRIME] = name
                default = self.default_values[name]           
                states[name] = {gname: default 
                                for gname in self.ground_names(name, ptypes)} 
                
        # update the state values with the values in the instance
        initstates = copy.deepcopy(states)
        init_state_info = getattr(self._AST.instance, 'init_state', [])
        for ((name, params), value) in init_state_info:
                
            # check whether name is a valid state-fluent
            if name not in initstates:
                raise RDDLUndefinedVariableError(
                    f'Variable <{name}> referenced in init-state block '
                    f'is not a valid state-fluent.')
                    
            # extract the grounded name and check that parameters are valid
            if params is not None:
                params = list(map(self.object_name, params))
            gname = self.ground_name(name, params)
            if gname not in initstates[name]:
                raise RDDLInvalidObjectError(
                    f'Parameter(s) {params} of state-fluent <{name}> '
                    f'declared in the init-state block are not valid.')
                
            # make sure value is correct type
            if isinstance(value, str):
                value = self.object_name(value)
                value_type = self.objects_rev.get(value, None)
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
                        
            initstates[name][gname] = value
                
        # state dictionary associates the variable lifted name with a list of
        # values for all variations of parameter arguments in C-based order
        # if the fluent does not have parameters, then the value is a scalar
        states = self._grounded_dict_to_dict_of_list(states)
        initstates = self._grounded_dict_to_dict_of_list(initstates)     
        
        self.states, self.statesranges = states, statesranges
        self.next_state, self.prev_state = nextstates, prevstates
        self.init_state = initstates
    
    def _value_list_or_scalar_from_default(self, pvar):
        default = self.default_values[pvar.name]
        ptypes = pvar.param_types   
        if ptypes is None:
            return default
        else:
            num_variations = 1
            for ptype in ptypes:
                num_variations *= len(self.objects[ptype])
            return [default] * num_variations
    
    def _extract_actions(self):
        
        # actions are stored similar to states described above
        actions, actionsranges = {}, {}
        for pvar in self._AST.domain.pvariables:
            if pvar.is_action_fluent():
                actionsranges[pvar.name] = pvar.range
                actions[pvar.name] = self._value_list_or_scalar_from_default(pvar)
        self.actions, self.actionsranges = actions, actionsranges
    
    def _extract_derived_and_interm(self):
        
        # derived and interm are stored similar to states described above
        derived, interm = {}, {}
        for pvar in self._AST.domain.pvariables:
            if pvar.is_derived_fluent():
                derived[pvar.name] = self._value_list_or_scalar_from_default(pvar)
            elif pvar.is_intermediate_fluent():
                interm[pvar.name] = self._value_list_or_scalar_from_default(pvar)
        self.derived, self.interm = derived, interm
    
    def _extract_observ(self):
        
        # observed are stored similar to states described above
        observ, observranges = {}, {}
        for pvar in self._AST.domain.pvariables:
            if pvar.is_observ_fluent():
                observranges[pvar.name] = pvar.range
                observ[pvar.name] = self._value_list_or_scalar_from_default(pvar)
        self.observ, self.observranges = observ, observranges
        
    def _extract_non_fluents(self):
        
        # extract non-fluents values from the domain defaults
        non_fluents = {}
        for pvar in self._AST.domain.pvariables:
            if pvar.is_non_fluent():
                name, ptypes = pvar.name, pvar.param_types
                default = self.default_values[name] 
                non_fluents[name] = {gname: default
                                     for gname in self.ground_names(name, ptypes)}
        
        # update non-fluent values with the values in the instance
        non_fluent_info = getattr(self._AST.non_fluents, 'init_non_fluent', [])
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
                value_type = self.objects_rev.get(value, None)
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
        self.nonfluents = self._grounded_dict_to_dict_of_list(non_fluents)
    
    def _extract_cpfs(self):
        cpfs = {}
        for cpf in self._AST.domain.cpfs[1]:
            name, objects = cpf.pvar[1]
            
            # make sure the CPF is defined in pvariables {...} scope
            types = self.param_types.get(name, None)
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
        self.terminals = getattr(self._AST.domain, 'terminals', [])
        self.preconditions = getattr(self._AST.domain, 'preconds', [])
        self.invariants = getattr(self._AST.domain, 'invariants', [])

    def _extract_horizon(self):
        horizon = self._AST.instance.horizon
        if not (horizon >= 0):
            raise RDDLValueOutOfRangeError(
                f'Horizon {horizon} in the instance is not >= 0.')
        self.horizon = horizon

    def _extract_max_actions(self):
        numactions = getattr(self._AST.instance, 'max_nondef_actions', 'pos-inf')
        if numactions == 'pos-inf':
            self.max_allowed_actions = sum(map(np.size, self.actions.values()))
        else:
            self.max_allowed_actions = int(numactions)

    def _extract_discount(self):
        discount = self._AST.instance.discount
        if not (0. <= discount):
            raise RDDLValueOutOfRangeError(
                f'Discount factor {discount} in the instance is not >= 0')
        self.discount = discount
    
