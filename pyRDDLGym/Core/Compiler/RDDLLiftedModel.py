import itertools
from typing import Iterable, List, Tuple

from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidObjectError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLUndefinedVariableError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLValueOutOfRangeError

from pyRDDLGym.Core.Compiler.RDDLModel import RDDLModel
from pyRDDLGym.Core.Parser.expr import Expression


class RDDLLiftedModel(RDDLModel):
    
    def __init__(self, rddl):
        super(RDDLLiftedModel, self).__init__()
        
        self.SetAST(rddl)
        self.objects, self.objects_rev = self._extract_objects()
        self.param_types = self._extract_param_types()  
                
        self.states, self.statesranges, self.next_state, self.prev_state, \
            self.init_state = self._extract_states()
        self.actions, self.actionsranges = self._extract_actions()
        self.derived, self.interm = self._extract_derived_and_interm()
        self.observ, self.observranges = self._extract_observ()
        self.nonfluents = self._extract_non_fluents()
        self.variable_types, self.variable_ranges = self._extract_variables()
        
        self.reward = self._AST.domain.reward
        self.cpfs = self._extract_cpfs()
        self.terminals, self.preconditions, self.invariants = self._extract_constraints()
        
        self.horizon = self._extract_horizon()
        self.discount = self._extract_discount()
        self.max_allowed_actions = self._extract_max_actions()
                
    def _extract_objects(self):
        ast_objects = self._AST.non_fluents.objects
        if (not ast_objects) or (ast_objects[0] is None):
            ast_objects = []
        objects = dict(ast_objects)
        
        objects_rev = {}
        for ptype, objs in objects.items():
            for obj in objs:
                if obj in objects_rev:
                    raise RDDLInvalidObjectError(
                        f'Types <{ptype}> and <{objects_rev[obj]}> '
                        f'cannot share the same object <{obj}>.')
                objects_rev[obj] = ptype
        return objects, objects_rev
    
    def _extract_param_types(self):
        param_types = {}
        for pvar in self._AST.domain.pvariables:
            primed_name = name = pvar.name
            if pvar.is_state_fluent():
                primed_name = name + '\''
            ptypes = pvar.param_types
            if ptypes is None:
                ptypes = []
            param_types[name] = ptypes
            param_types[primed_name] = ptypes
        return param_types

    def variations(self, ptypes: Iterable[str]) -> Iterable[Tuple[str, ...]]:
        '''Given a list of types, computes the cartesian product of all object
        enumerations that corresponds to those types.
        '''
        if ptypes is None or not ptypes:
            return [()]
        objects_by_type = []
        for ptype in ptypes:
            if ptype not in self.objects:
                raise RDDLInvalidObjectError(
                    f'Type <{ptype}> is not valid, '
                    f'must be one of {set(self.objects.keys())}.')
            objects_by_type.append(self.objects[ptype])
        return itertools.product(*objects_by_type)
    
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
    
    def grounded_names(self, name: str, ptypes: Iterable[str]) -> Iterable[str]:
        '''Given a variable name and list of types, produces a new iterator
        whose elements are the grounded representations in the cartesian product 
        of all object enumerations that corresponds to those types.
        '''
        for objects in self.variations(ptypes):
            yield self.ground_name(name, objects)
        
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
    
    def _extract_variables(self):
        variable_types, variable_ranges = {}, {}
        for pvar in self._AST.domain.pvariables:
            variable_types[pvar.name] = pvar.fluent_type
            variable_ranges[pvar.name] = pvar.range
            if pvar.is_state_fluent():
                variable_types[pvar.name + '\''] = 'next-state-fluent'
                variable_ranges[pvar.name + '\''] = pvar.range
        return variable_types, variable_ranges
    
    def _extract_cpfs(self):
        cpfs = {}
        for cpf in self._AST.domain.cpfs[1]:
            name, objects = cpf.pvar[1]
            if objects is None:
                objects = []
            types = self.param_types[name]
            if len(types) != len(objects):
                raise RDDLInvalidObjectError(
                    f'CPF <{name}> expects {len(types)} parameters, '
                    f'got {len(objects)}.')
            objects = [(o, types[i]) for i, o in enumerate(objects)]
            cpfs[name] = (objects, cpf.expr)
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
    
    def is_compatible(self, var: str, objects: List[str]) -> bool:
        '''Determines whether or not the given variable can be evaluated
        for the given list of objects.
        '''
        if var not in self.param_types:
            return False
        ptypes = self.param_types[var]
        if objects is None:
            objects = []
        if len(ptypes) != len(objects):
            return False
        for ptype, obj in zip(ptypes, objects):
            if obj not in self.objects_rev or ptype != self.objects_rev[obj]:
                return False
        return True
    
    def parse(self, expr: str) -> Tuple[str, List[str]]:
        '''Parses an expression of the form <name> or <name>_<type1>_<type2>...)
        into a tuple of <name>, [<type1>, <type2>, ...].
        '''
        tokens = expr.split('_')
        var = tokens[0]
        objects = tokens[1:]
        return var, objects
    
    def is_non_fluent_expression(self, expr: Expression) -> bool:
        '''Determines whether or not expression is a non-fluent.
        '''
        if isinstance(expr, tuple):
            return True
        etype, _ = expr.etype
        if etype == 'constant':
            return True
        elif etype == 'randomvar':
            return False
        elif etype == 'pvar':
            name = expr.args[0]
            if name not in self.variable_types:
                raise RDDLUndefinedVariableError(
                    f'Variable <{name}> is not defined in the domain, '
                    f'must be one of {set(self.variable_types.keys())}.')
            return self.variable_types[name] == 'non-fluent'
        else:
            for arg in expr.args:
                if not self.is_non_fluent_expression(arg):
                    return False
            return True
