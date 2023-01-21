import copy

from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidNumberOfArgumentsError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidObjectError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLMissingCPFDefinitionError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLRepeatedVariableError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLValueOutOfRangeError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLUndefinedCPFError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLUndefinedVariableError

from pyRDDLGym.Core.Compiler.RDDLModel import PlanningModel


class RDDLLiftedModel(PlanningModel):
    '''Represents a RDDL domain + instance in lifted form.
    '''
    
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
        if (not ast_objects) or ast_objects[0] is None:
            ast_objects = []
        ast_objects = dict(ast_objects)
        
        # record the set of objects of each type defined in the domain
        objects, objects_rev = {}, {}
        enum_types, enum_literals = set(), set()    
        for (name, pvalues) in self._AST.domain.types:
            
            # object
            if pvalues == 'object': 
                objects[name] = ast_objects.get(name, None)
                if objects[name] is None:
                    raise RDDLInvalidObjectError(
                        f'Type <{name}> has no objects defined in the instance.')
            
            # enumerated type
            else: 
                objects[name] = pvalues
                enum_types.add(name)
                enum_literals.update(pvalues)        
        
            # make sure types do not share an object
            # record the type that each object corresponds to
            for obj in objects[name]:
                if obj in objects_rev:
                    raise RDDLInvalidObjectError(
                        f'Types <{name}> and <{objects_rev[obj]}> '
                        f'can not share the same object <{obj}>.')
                objects_rev[obj] = name
        
        # check that all types in instance are declared in domain
        for (ptype, _) in self._AST.non_fluents.objects:
            if ptype not in objects:
                raise RDDLInvalidObjectError(
                    f'Type <{ptype}> declared in instance is not declared in '
                    f'types {{ ... }} domain block.')
        
        # maps each object to its canonical order as appears in RDDL definition
        objects_index = {obj: i 
                         for objs in objects.values() 
                            for (i, obj) in enumerate(objs)}
        
        self.objects, self.objects_rev = objects, objects_rev
        self.index_of_object = objects_index
        self.enum_types, self.enum_literals = enum_types, enum_literals
    
    def _extract_variable_information(self):
        
        # extract some basic information about variables in the domain:
        # 1. their object parameters needed to evaluate
        # 2. their types (e.g. state-fluent, action-fluent)
        # 3. their ranges (e.g. int, real, enum-type)
        var_params, var_types, var_ranges = {}, {}, {}
        for pvar in self._AST.domain.pvariables: 
            
            # make sure variable is not defined more than once        
            primed_name = name = pvar.name
            if name in var_params:
                raise RDDLRepeatedVariableError(
                    f'{pvar.fluent_type} <{name}> has the same name as '
                    f'another {var_types[name]}.')
        
            # variable is new: record its type, parameters and range
            if pvar.is_state_fluent():
                primed_name = name + '\''
            ptypes = pvar.param_types
            if ptypes is None:
                ptypes = []
            var_params[name] = var_params[primed_name] = ptypes        
            var_types[name] = pvar.fluent_type
            if pvar.is_state_fluent():
                var_types[primed_name] = 'next-state-fluent'                
            var_ranges[name] = var_ranges[primed_name] = pvar.range    
        
        # maps each variable (as appears in RDDL) to list of grounded variations
        var_grounded = {var: list(self.ground_names(var, types))
                        for (var, types) in var_params.items()}        
        
        self.param_types, self.variable_types, self.variable_ranges = \
            var_params, var_types, var_ranges
        self.grounded_names = var_grounded
    
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
    
    def _extract_states(self):
        
        # get the default value for each grounded state variable
        states, statesranges, nextstates, prevstates = {}, {}, {}, {}
        for pvar in self._AST.domain.pvariables:
            if pvar.is_state_fluent():
                name, ptypes = pvar.name, pvar.param_types
                statesranges[name] = pvar.range
                nextstates[name] = name + '\''
                prevstates[name + '\''] = name
                states[name] = {gname: pvar.default 
                                for gname in self.ground_names(name, ptypes)}                
        
        # update the state values with the values in the instance
        initstates = copy.deepcopy(states)
        if hasattr(self._AST.instance, 'init_state'):
            for ((name, params), value) in self._AST.instance.init_state:
                
                # check whether name is a valid state-fluent
                if name not in initstates:
                    raise RDDLUndefinedVariableError(
                        f'Variable <{name}> referenced in init-state block '
                        f'is not a valid state-fluent.')
                    
                # extract the grounded name, and check that parameters are valid
                gname = self.ground_name(name, params)
                if gname not in initstates[name]:
                    raise RDDLInvalidObjectError(
                        f'Parameter(s) {params} of state-fluent {name} '
                        f'as declared in the init-state block are not valid.')
                    
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
        ptypes = pvar.param_types
        default = pvar.default
        if ptypes is None:
            return default
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
        
        # derived/interm are stored similar to states described above
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
                non_fluents[name] = {gname: pvar.default
                                     for gname in self.ground_names(name, ptypes)}
        
        # update non-fluent values with the values in the instance
        if hasattr(self._AST.non_fluents, 'init_non_fluent'):
            for ((name, params), value) in self._AST.non_fluents.init_non_fluent:
                
                # check whether name is a valid non-fluent
                grounded_names = non_fluents.get(name, None)
                if grounded_names is None:
                    raise RDDLUndefinedVariableError(
                        f'Variable <{name}> referenced in non-fluents block '
                        f'is not a valid non-fluent.')
                
                # extract the grounded name, and check that parameters are valid
                gname = self.ground_name(name, params)                           
                if gname not in grounded_names:
                    raise RDDLInvalidObjectError(
                        f'Parameter(s) {params} of non-fluent {name} '
                        f'as declared in the non-fluents block are not valid.')
                
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
                    f'CPF <{name}> expects {len(types)} parameter(s), '
                    f'got {len(objects)}.')
            
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
        terminals, preconds, invariants = [], [], []
        if hasattr(self._AST.domain, 'terminals'):
            terminals = self._AST.domain.terminals
        if hasattr(self._AST.domain, 'preconds'):
            preconds = self._AST.domain.preconds
        if hasattr(self._AST.domain, 'invariants'):
            invariants = self._AST.domain.invariants
        self.terminals, self.preconditions, self.invariants = \
            terminals, preconds, invariants

    def _extract_horizon(self):
        horizon = self._AST.instance.horizon
        if not (horizon >= 0):
            raise RDDLValueOutOfRangeError(
                f'Horizon {horizon} in the instance is not >= 0.')
        self.horizon = horizon

    def _extract_max_actions(self):
        numactions = self._AST.instance.max_nondef_actions
        if numactions == 'pos-inf':
            self.max_allowed_actions = len(self.actions)
        else:
            self.max_allowed_actions = int(numactions)

    def _extract_discount(self):
        discount = self._AST.instance.discount
        if not (0. <= discount <= 1.):
            raise RDDLValueOutOfRangeError(
                f'Discount factor {discount} in the instance is not in [0, 1].')
        self.discount = discount
    
