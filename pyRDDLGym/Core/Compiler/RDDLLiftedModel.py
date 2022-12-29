from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidObjectError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLMissingCPFDefinitionError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLValueOutOfRangeError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLUndefinedCPFError

from pyRDDLGym.Core.Compiler.RDDLModel import RDDLModel


class RDDLLiftedModel(RDDLModel):
    
    def __init__(self, rddl):
        super(RDDLLiftedModel, self).__init__()
        
        self.SetAST(rddl)
        self.objects, self.objects_rev, self.enums, self.enums_rev = \
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
                
    def _extract_objects(self):
        ast_objects = self._AST.non_fluents.objects
        if (not ast_objects) or ast_objects[0] is None:
            ast_objects = []
        ast_objects = dict(ast_objects)
         
        objects, enums = {}, {}     
        for name, pvalues in self._AST.domain.types:
            if pvalues == 'object':
                objects[name] = ast_objects.get(name, None)
                if objects[name] is None:
                    raise RDDLInvalidObjectError(
                        f'Type <{name}> has no objects defined in the instance.')
            else:
                enums[name] = pvalues        
        
        objects_rev = {}
        for name, values in objects.items():
            for obj in values:
                if obj in objects_rev:
                    raise RDDLInvalidObjectError(
                        f'Types <{name}> and <{objects_rev[obj]}> '
                        f'can not share the same object <{obj}>.')
                objects_rev[obj] = name
        
        enums_rev = {}
        for name, values in enums.items():
            for obj in values:
                if obj in enums_rev:
                    raise RDDLInvalidObjectError(
                        f'Enums <{name}> and <{enums_rev[obj]}> '
                        f'can not share the same literal <{obj}>.')
                enums_rev[obj] = name
        
        return objects, objects_rev, enums, enums_rev
    
    def _extract_variable_information(self):
        var_params, var_types, var_ranges = {}, {}, {}
        for pvar in self._AST.domain.pvariables:
            primed_name = name = pvar.name
            if pvar.is_state_fluent():
                primed_name = name + '\''
            ptypes = pvar.param_types
            if ptypes is None:
                ptypes = []
            var_params[name] = ptypes
            var_params[primed_name] = ptypes        
            var_types[name] = pvar.fluent_type
            if pvar.is_state_fluent():
                var_types[primed_name] = 'next-state-fluent'                
            var_ranges[name] = pvar.range
            var_ranges[primed_name] = pvar.range            
        return var_params, var_types, var_ranges

    def _extract_states(self):
        states, statesranges = {}, {}
        nextstates, prevstates = {}, {}
        for pvar in self._AST.domain.pvariables:
            if pvar.is_state_fluent():
                for name in self.grounded_names(pvar.name, pvar.param_types):
                    states[name] = pvar.default
                    statesranges[name] = pvar.range
                    nextstates[name] = name + '\''
                    prevstates[name + '\''] = name
                
        initstates = states.copy()
        if hasattr(self._AST.instance, 'init_state'):
            for (var, params), value in self._AST.instance.init_state:
                name = self.ground_name(var, params)
                if name in initstates:
                    initstates[name] = value
        return states, statesranges, nextstates, prevstates, initstates
    
    def _extract_actions(self):
        actions, actionsranges = {}, {}
        for pvar in self._AST.domain.pvariables:
            if pvar.is_action_fluent():
                for name in self.grounded_names(pvar.name, pvar.param_types):
                    actionsranges[name] = pvar.range
                    actions[name] = pvar.default
        return actions, actionsranges
    
    def _extract_derived_and_interm(self):
        derived, interm = {}, {}
        for pvar in self._AST.domain.pvariables:
            name = pvar.name
            if pvar.is_derived_fluent():
                for name in self.grounded_names(pvar.name, pvar.param_types):
                    derived[name] = pvar.default
            elif pvar.is_intermediate_fluent():
                for name in self.grounded_names(pvar.name, pvar.param_types):
                    interm[name] = pvar.default
        return derived, interm
    
    def _extract_observ(self):
        observ, observranges = {}, {}
        for pvar in self._AST.domain.pvariables:
            if pvar.is_observ_fluent():
                for name in self.grounded_names(pvar.name, pvar.param_types):
                    observranges[name] = pvar.range
                    observ[name] = pvar.default
        return observ, observranges
        
    def _extract_non_fluents(self):
        non_fluents = {}
        for pvar in self._AST.domain.pvariables:
            if pvar.is_non_fluent():
                for name in self.grounded_names(pvar.name, pvar.param_types):
                    non_fluents[name] = pvar.default
                        
        if hasattr(self._AST.non_fluents, 'init_non_fluent'):
            for (var, params), value in self._AST.non_fluents.init_non_fluent:
                name = self.ground_name(var, params)
                if name in non_fluents:
                    non_fluents[name] = value             
        return non_fluents
    
    def _extract_cpfs(self):
        cpfs = {}
        for cpf in self._AST.domain.cpfs[1]:
            name, objects = cpf.pvar[1]
            if objects is None:
                objects = []
                
            types = self.param_types.get(name, None)
            if types is None:
                raise RDDLUndefinedCPFError(
                    f'CPF <{name}> is not defined in pvariable scope.')
                
            if len(types) != len(objects):
                raise RDDLInvalidObjectError(
                    f'CPF <{name}> expects {len(types)} parameters, '
                    f'got {len(objects)}.')
                
            objects = [(o, types[i]) for i, o in enumerate(objects)]
            cpfs[name] = (objects, cpf.expr)
        
        for var, fluent_type in self.variable_types.items():
            if fluent_type in {'derived-fluent', 'interm-fluent', 
                               'next-state-fluent', 'observ-fluent'}:
                if var not in cpfs:
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
    
    @property
    def is_grounded(self):
        return False
    
