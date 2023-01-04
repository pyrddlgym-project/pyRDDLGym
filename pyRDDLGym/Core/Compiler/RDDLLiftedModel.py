import copy

from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidObjectError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLMissingCPFDefinitionError
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
        self.objects, self.objects_rev, self.enum_types, self.enum_literals = \
            self._extract_objects()
            
        self.param_types, self.variable_types, self.variable_ranges = \
            self._extract_variable_information()  
                
        self.states, self.statesranges, self.next_state, self.prev_state, \
            self.init_state = self._extract_states()
        self.actions, self.actionsranges = self._extract_actions()
        self.derived, self.interm = self._extract_derived_and_interm()
        self.observ, self.observranges = self._extract_observ()
        self.nonfluents = self._extract_non_fluents()
        
        self.reward = self._AST.domain.reward
        self.cpfs = self._extract_cpfs()
        self.terminals, self.preconditions, self.invariants = \
            self._extract_constraints()
        
        self.horizon = self._extract_horizon()
        self.discount = self._extract_discount()
        self.max_allowed_actions = self._extract_max_actions()
                
        self.grounded_names = {var: list(self.ground_names(var, types))
                               for (var, types) in self.param_types.items()}                
        self.index_of_object = {obj: i 
                                for objects in self.objects.values() 
                                    for (i, obj) in enumerate(objects)}
        
    def _extract_objects(self):
        ast_objects = self._AST.non_fluents.objects
        if (not ast_objects) or ast_objects[0] is None:
            ast_objects = []
        ast_objects = dict(ast_objects)
        
        # record the set of objects of each type
        objects = {}
        enum_types, enum_literals = set(), set()    
        for (name, pvalues) in self._AST.domain.types:
            
            # objects
            if pvalues == 'object':  
                objects[name] = ast_objects.get(name, None)
                if objects[name] is None:
                    raise RDDLInvalidObjectError(
                        f'Type <{name}> has no objects defined in the instance.')
            
            # enumerated types
            else:  
                objects[name] = pvalues
                enum_types.add(name)
                enum_literals.update(pvalues)        
        
        # make sure different types do not share the same object
        # then record the type that each object corresponds to
        objects_rev = {}
        for (name, values) in objects.items():
            for obj in values:
                if obj in objects_rev:
                    raise RDDLInvalidObjectError(
                        f'Types <{name}> and <{objects_rev[obj]}> '
                        f'can not share the same object <{obj}>.')
                objects_rev[obj] = name
            
        return objects, objects_rev, enum_types, enum_literals
    
    def _extract_variable_information(self):
        var_params, var_types, var_ranges = {}, {}, {}
        for pvar in self._AST.domain.pvariables:
            primed_name = name = pvar.name
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
        return var_params, var_types, var_ranges
    
    def _flatten_grounded_dict(self, grounded_dict):
        new_dict = {}
        for (var, values_dict) in grounded_dict.items():
            if self.param_types[var]:
                new_dict[var] = list(values_dict.values())
            else:
                assert len(values_dict) == 1
                new_dict[var] = next(iter(values_dict.values()))
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
                        f'Parameters {params} of state-fluent {name} '
                        f'as declared in the init-state block are not valid.')
                    
                initstates[name][gname] = value
                
        # state dictionary associates the variable lifted name with a list of
        # values for all variations of parameter arguments in C-based order
        # if the fluent does not have parameters, then the value is a scalar
        states = self._flatten_grounded_dict(states)
        initstates = self._flatten_grounded_dict(initstates)     
        return states, statesranges, nextstates, prevstates, initstates
    
    def _num_variations(self, ptypes):
        if ptypes is None:
            ptypes = []
        prod = 1
        for ptype in ptypes:
            prod *= len(self.objects[ptype])
        return prod
    
    def _extract_actions(self):
        
        # actions are stored similar to states described above
        actions, actionsranges = {}, {}
        for pvar in self._AST.domain.pvariables:
            if pvar.is_action_fluent():
                name, ptypes = pvar.name, pvar.param_types
                actionsranges[name] = pvar.range
                actions[name] = [pvar.default] * self._num_variations(ptypes)
        return actions, actionsranges
    
    def _extract_derived_and_interm(self):
        
        # derived/interm are stored similar to states described above
        derived, interm = {}, {}
        for pvar in self._AST.domain.pvariables:
            name, ptypes = pvar.name, pvar.param_types
            if pvar.is_derived_fluent():
                derived[name] = [pvar.default] * self._num_variations(ptypes)
            elif pvar.is_intermediate_fluent():
                interm[name] = [pvar.default] * self._num_variations(ptypes)
        return derived, interm
    
    def _extract_observ(self):
        
        # observed are stored similar to states described above
        observ, observranges = {}, {}
        for pvar in self._AST.domain.pvariables:
            if pvar.is_observ_fluent():
                name, ptypes = pvar.name, pvar.param_types
                observranges[name] = pvar.range
                observ[name] = [pvar.default] * self._num_variations(ptypes)
        return observ, observranges
        
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
                if name not in non_fluents:
                    raise RDDLUndefinedVariableError(
                        f'Variable <{name}> referenced in non-fluents block '
                        f'is not a valid non-fluent.')
                
                # extract the grounded name, and check that parameters are valid
                gname = self.ground_name(name, params)                
                if gname not in non_fluents[name]:
                    raise RDDLInvalidObjectError(
                        f'Parameters {params} of non-fluent {name} '
                        f'as declared in the non-fluents block are not valid.')
                
                non_fluents[name][gname] = value
                                        
        # non-fluents are stored similar to states described above
        return self._flatten_grounded_dict(non_fluents)
    
    def _extract_cpfs(self):
        cpfs = {}
        for cpf in self._AST.domain.cpfs[1]:
            name, objects = cpf.pvar[1]
            if objects is None:
                objects = []
            
            # make sure the CPF is defined in pvariables {...} scope
            types = self.param_types.get(name, None)
            if types is None:
                raise RDDLUndefinedCPFError(
                    f'CPF <{name}> is not defined in pvariable scope.')
            
            # make sure the number of parameters matches that in cpfs {...}
            if len(types) != len(objects):
                raise RDDLInvalidObjectError(
                    f'CPF <{name}> expects {len(types)} parameters, '
                    f'got {len(objects)}.')
            
            # CPFs are stored as dictionary that associates cpf name
            # with a pair, where the first element is the type argument info,
            # and the second element is the AST expression
            objects = [(o, types[i]) for (i, o) in enumerate(objects)]
            cpfs[name] = (objects, cpf.expr)
        
        # make sure all CPFs have a valid expression in cpfs {...}
        for (var, fluent_type) in self.variable_types.items():
            if fluent_type in {'derived-fluent', 'interm-fluent', 
                               'next-state-fluent', 'observ-fluent'} \
            and var not in cpfs:
                raise RDDLMissingCPFDefinitionError(
                    f'{fluent_type} CPF <{var}> is missing a valid definition.')
                    
        return cpfs
    
    def _extract_constraints(self):
        terminals, preconds, invariants = [], [], []
        if hasattr(self._AST.domain, 'terminals'):
            terminals = self._AST.domain.terminals
        if hasattr(self._AST.domain, 'preconds'):
            preconds = self._AST.domain.preconds
        if hasattr(self._AST.domain, 'invariants'):
            invariants = self._AST.domain.invariants
        return terminals, preconds, invariants

    def _extract_horizon(self):
        horizon = self._AST.instance.horizon
        if not (horizon >= 0):
            raise RDDLValueOutOfRangeError(
                f'Horizon {horizon} in the instance is not >= 0.')
        return horizon

    def _extract_max_actions(self):
        numactions = self._AST.instance.max_nondef_actions
        if numactions == 'pos-inf':
            return len(self.actions)
        else:
            return int(numactions)

    def _extract_discount(self):
        discount = self._AST.instance.discount
        if not (0. <= discount <= 1.):
            raise RDDLValueOutOfRangeError(
                f'Discount factor {discount} in the instance is not in [0, 1].')
        return discount
    