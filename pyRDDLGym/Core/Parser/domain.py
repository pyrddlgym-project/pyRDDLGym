# This file is based on thiago pbueno's pyrddl.
# https://github.com/thiagopbueno/pyrddl
# it was adapted and extended for pyRDDLGym


from pyRDDLGym.Core.Parser import utils
from pyRDDLGym.Core.Parser.pvariable import PVariable
from pyRDDLGym.Core.Parser.cpf import CPF
from pyRDDLGym.Core.Parser.expr import Expression

from typing import Dict, List, Sequence, Optional, Tuple

Type = Tuple[str, str]


class Domain(object):
    '''Domain class for accessing RDDL domain sections.
    Note:
        This class is intended to be solely used by the parser and compiler.
        Do not attempt to directly use this class to build a Domain object.
    Args:
        name: Name of RDDL domain.
        requirements: List of RDDL requirements.
        sections: Mapping from string to domain section.
    Attributes:
        name (str): Domain identifier.
        requirements (List[str]): List of requirements.
        types (List[:obj:`Type`]): List of types.
        pvariables (List[:obj:`PVariable`]): List of parameterized variables.
        cpfs (List[:obj:`CPF`]): List of Conditional Probability Functions.
        reward (:obj:`Expression`): Reward function.
        preconds (List[:obj:`Expression`]): List of action preconditions.
        terminals (List[:obj:`Expression`]): List of termination conditions.
        constraints (List[:obj:`Expression`]): List of state-action constraints.
        invariants (List[:obj:`Expression`]): List of state invariants.
    '''

    def __init__(self, name: str, requirements: List[str], sections: Dict[str, Sequence]) -> None:
        self.name = name
        self.requirements = requirements

        self.pvariables = sections['pvariables']
        self.cpfs = sections['cpfs']
        self.reward = sections['reward']

        self.types = sections.get('types', [])
        self.preconds = sections.get('preconds', [])
        self.terminals = sections.get('terminals', [])
        self.invariants = sections.get('invariants', [])
        self.constraints = sections.get('constraints', [])

    # @property
    # def Requirements(self):
    #     return self.requirements

    def build(self):
        self._build_preconditions_table()
        self._build_action_bound_constraints_table()

    def _build_preconditions_table(self):
        '''Builds the local action precondition expressions.'''
        self.local_action_preconditions = dict()
        self.global_action_preconditions = []
        action_fluents = self.action_fluents
        for precond in self.preconds:
            scope = precond.scope
            action_scope = [action for action in scope if action in action_fluents]
            if len(action_scope) == 1:
                name = action_scope[0]
                self.local_action_preconditions[name] = self.local_action_preconditions.get(name, [])
                self.local_action_preconditions[name].append(precond)
            else:
                self.global_action_preconditions.append(precond)

    def _build_action_bound_constraints_table(self):
        '''Builds the lower and upper action bound constraint expressions.'''
        self.action_lower_bound_constraints = {}
        self.action_upper_bound_constraints = {}

        for name, preconds in self.local_action_preconditions.items():

            for precond in preconds:
                expr_type = precond.etype
                expr_args = precond.args

                bounds_expr = None

                if expr_type == ('aggregation', 'forall'):
                    inner_expr = expr_args[1]
                    if inner_expr.etype[0] == 'relational':
                        bounds_expr = inner_expr
                elif expr_type[0] == 'relational':
                    bounds_expr = precond

                if bounds_expr:
                    # lower bound
                    bound = self._extract_lower_bound(name, bounds_expr)
                    if bound is not None:
                        self.action_lower_bound_constraints[name] = bound
                    else: # upper bound
                        bound = self._extract_upper_bound(name, bounds_expr)
                        if bound is not None:
                            self.action_upper_bound_constraints[name] = bound


    def _extract_lower_bound(self, name: str, expr: Expression) -> Optional[Expression]:
        '''Returns the lower bound expression of the action with given `name`.'''
        etype = expr.etype
        args = expr.args
        if etype[1] in ['<=', '<']:
            if args[1].is_pvariable_expression() and args[1].name == name:
                return args[0]
        elif etype[1] in ['>=', '>']:
            if args[0].is_pvariable_expression() and args[0].name == name:
                return args[1]
        return None

    def _extract_upper_bound(self, name: str, expr: Expression) -> Optional[Expression]:
        '''Returns the upper bound expression of the action with given `name`.'''
        etype = expr.etype
        args = expr.args
        if etype[1] in ['<=', '<']:
            if args[0].is_pvariable_expression() and args[0].name == name:
                return args[1]
        elif etype[1] in ['>=', '>']:
            if args[1].is_pvariable_expression() and args[1].name == name:
                return args[0]
        return None

    @property
    def non_fluents(self) -> Dict[str, PVariable]:
        '''Returns non-fluent pvariables.'''
        return { str(pvar): pvar for pvar in self.pvariables if pvar.is_non_fluent() }

    @property
    def state_fluents(self) -> Dict[str, PVariable]:
        '''Returns state-fluent pvariables.'''
        return { str(pvar): pvar for pvar in self.pvariables if pvar.is_state_fluent() }

    @property
    def action_fluents(self) -> Dict[str, PVariable]:
        '''Returns action-fluent pvariables.'''
        return { str(pvar): pvar for pvar in self.pvariables if pvar.is_action_fluent() }

    @property
    def intermediate_fluents(self) -> Dict[str, PVariable]:
        '''Returns interm-fluent pvariables.'''
        return { str(pvar): pvar for pvar in self.pvariables if pvar.is_intermediate_fluent() }

    @property
    def derived_fluents(self) -> Dict[str, PVariable]:
        '''Returns derived-fluent pvariables.'''
        return {str(pvar): pvar for pvar in self.pvariables if pvar.is_derived_fluent()}

    @property
    def observation_fluents(self) -> Dict[str, PVariable]:
        '''Returns observ-fluents pvaraibles'''
        return {str(pvar): pvar for pvar in self.pvariables if pvar.is_observ_fluent()}
        return

    @property
    def intermediate_cpfs(self) -> List[CPF]:
        '''Returns list of intermediate-fluent CPFs in level order.'''
        _, cpfs = self.cpfs
        interm_cpfs = [cpf for cpf in cpfs if cpf.name in self.intermediate_fluents]
        interm_cpfs = sorted(interm_cpfs, key=lambda cpf: (self.intermediate_fluents[cpf.name].level, cpf.name))
        return interm_cpfs

    @property
    def derived_cpfs(self) -> List[CPF]:
        '''Returns list of derived-fluent CPFs in level order.'''
        _, cpfs = self.cpfs
        der_cpfs = [cpf for cpf in cpfs if cpf.name in self.derived_fluents]
        der_cpfs = sorted(der_cpfs, key=lambda cpf: (self.derived_fluents[cpf.name].level, cpf.name))
        return der_cpfs

    @property
    def observation_cpfs(self) -> List[CPF]:
        '''Returns list of observation_fluents CPFs in level order.'''
        _, cpfs = self.cpfs
        der_cpfs = [cpf for cpf in cpfs if cpf.name in self.observation_fluents]
        der_cpfs = sorted(der_cpfs, key=lambda cpf: (self.observation_fluents[cpf.name].level, cpf.name))
        return der_cpfs

    def get_intermediate_cpf(self, name):
        for cpf in self.intermediate_cpfs:
            if cpf.name == name:
                return cpf

    def get_derived_cpf(self, name):
        for cpf in self.derived_cpfs:
            if cpf.name == name:
                return cpf

    @property
    def state_cpfs(self) -> List[CPF]:
        '''Returns list of state-fluent CPFs.'''
        _, cpfs = self.cpfs
        state_cpfs = []
        for cpf in cpfs:
            name = utils.rename_next_state_fluent(cpf.name)
            if name in self.state_fluents:
                state_cpfs.append(cpf)
        state_cpfs = sorted(state_cpfs, key=lambda cpf: cpf.name)
        return state_cpfs

    @property
    def non_fluent_ordering(self) -> List[str]:
        '''The list of non-fluent names in canonical order.
        Returns:
            List[str]: A list of fluent names.
        '''
        return sorted(self.non_fluents)

    @property
    def state_fluent_ordering(self) -> List[str]:
        '''The list of state-fluent names in canonical order.
        Returns:
            List[str]: A list of fluent names.
        '''
        return sorted(self.state_fluents)

    @property
    def action_fluent_ordering(self) -> List[str]:
        '''The list of action-fluent names in canonical order.
        Returns:
            List[str]: A list of fluent names.
        '''
        return sorted(self.action_fluents)

    @property
    def interm_fluent_ordering(self) -> List[str]:
        '''The list of intermediate-fluent names in canonical order.
        Returns:
            List[str]: A list of fluent names.
        '''
        interm_fluents = self.intermediate_fluents.values()
        key = lambda pvar: (pvar.level, pvar.name)
        return [str(pvar) for pvar in sorted(interm_fluents, key=key)]

    @property
    def derived_fluent_ordering(self) -> List[str]:
        '''The list of derived-fluent names in canonical order.
        Returns:
            List[str]: A list of fluent names.
        '''
        derived_fluents = self.derived_fluents.values()
        key = lambda pvar: (pvar.level, pvar.name)
        return [str(pvar) for pvar in sorted(derived_fluents, key=key)]

    @property
    def next_state_fluent_ordering(self) -> List[str]:
        '''The list of next state-fluent names in canonical order.
        Returns:
            List[str]: A list of fluent names.
        '''
        key = lambda x: x.name
        return [cpf.name for cpf in sorted(self.state_cpfs, key=key)]
