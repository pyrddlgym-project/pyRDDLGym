from typing import Dict, Set
import warnings

from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidDependencyInCPFError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLMissingCPFDefinitionError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLNotImplementedError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLUndefinedVariableError
from pyRDDLGym.Core.Parser.rddl import RDDL

VALID_DEPENDENCIES = {
    'derived-fluent': {'state-fluent', 'derived-fluent'},
    'interm-fluent': {'action-fluent', 'state-fluent', 
                      'derived-fluent', 'interm-fluent'},
    'next-state-fluent': {'action-fluent', 'state-fluent', 
                          'derived-fluent', 'interm-fluent', 'next-state-fluent'},
    'observ-fluent': {'action-fluent', 
                      'derived-fluent', 'interm-fluent', 'next-state-fluent'},
    'reward': {'action-fluent', 'state-fluent', 
               'derived-fluent', 'interm-fluent', 'next-state-fluent'},
    'invariant': {'state-fluent'},
    'precondition': {'state-fluent', 'action-fluent'},
    'termination': {'state-fluent'}
}

    
class LiftedRDDLLevelAnalysis:
    
    def __init__(self, rddl: RDDL, allow_synchronous_state: bool=True) -> None:
        self.rddl = rddl
        self.allow_synchronous_state = allow_synchronous_state
        
        self.domain = rddl.domain
        self.cpfs = {cpf.pvar[1][0]: cpf.expr for cpf in self.domain.cpfs[1]}
        self.pvars = {pvar.name: pvar for pvar in self.domain.pvariables}
        self.next_states = {pvar.name + '\'': pvar.name
                            for pvar in self.domain.pvariables
                            if pvar.is_state_fluent()}
        
    # ===========================================================================
    # call graph construction
    # ===========================================================================
    
    def build_call_graph(self) -> Dict[str, Set[str]]:
        
        # compute call graph of CPs and check validity
        cpf_graph = {}
        for name, expr in self.cpfs.items():
            self._update_call_graph(cpf_graph, name, expr)
            if name not in cpf_graph:
                cpf_graph[name] = set()
        self._validate_dependencies(cpf_graph)
        self._validate_cpf_definitions(cpf_graph)
        
        # check validity of reward, constraints, termination
        for name, exprs in [
            ('reward', [self.domain.reward]),
            ('precondition', self.domain.preconds),
            ('invariant', self.domain.invariants),
            ('termination', self.domain.terminals)
        ]:
            call_graph = {}
            for expr in exprs:
                self._update_call_graph(call_graph, name, expr)
            self._validate_dependencies(call_graph)
        return cpf_graph
    
    def _update_call_graph(self, graph, cpf, expr):
        if isinstance(expr, tuple):
            pass
        elif expr.is_pvariable_expression():
            name, *_ = expr.args
            if name in self.next_states:
                pvar = self.pvars[self.next_states[name]]
            elif name in self.pvars:
                pvar = self.pvars[name]
            else:
                raise RDDLUndefinedVariableError(
                    f'Variable <{name}> in CPF <{cpf}> is not defined.')
            if pvar.is_fluent():
                graph.setdefault(cpf, set()).add(name)
        elif not expr.is_constant_expression():
            for arg in expr.args:
                self._update_call_graph(graph, cpf, arg)
    
    # ===========================================================================
    # call graph validation
    # ===========================================================================
    
    def _fluent_type(self, fluent):
        if fluent in self.next_states:
            return 'next-state-fluent'
        elif fluent in self.pvars:
            return self.pvars[fluent].fluent_type       
        else:
            return fluent 

    def _validate_dependencies(self, graph):
        for cpf, deps in graph.items():
            cpf_type = self._fluent_type(cpf)
            
            # warn use of derived fluent
            if cpf_type == 'derived-fluent':
                warnings.warn(
                    f'The use of derived-fluent is discouraged in this version: '
                    f'please change <{cpf}> to interm-fluent.',
                    FutureWarning, stacklevel=2)
            
            # not a recognized type
            if cpf_type not in VALID_DEPENDENCIES:
                if cpf_type == 'state-fluent':
                    raise RDDLInvalidDependencyInCPFError(
                        f'CPF definition for state-fluent <{cpf}> is not valid: '
                        f'did you mean <{cpf}\'>?')
                else:
                    raise RDDLNotImplementedError(
                        f'Type {cpf_type} of CPF <{cpf}> is not valid.')
            
            # check that all dependencies are valid
            for dep in deps: 
                dep_type = self._fluent_type(dep)
                if dep_type not in VALID_DEPENDENCIES[cpf_type] or (
                    not self.allow_synchronous_state
                    and cpf_type == dep_type == 'next-state-fluent'):
                    raise RDDLInvalidDependencyInCPFError(
                        f'{cpf_type} <{cpf}> cannot depend on {dep_type} <{dep}>.')                
    
    def _validate_cpf_definitions(self, graph): 
        for name in self.pvars.keys():
            fluent_type = self._fluent_type(name)
            if fluent_type == 'state-fluent':
                fluent_type = 'next-state-fluent'
                name = name + '\''
            if fluent_type in VALID_DEPENDENCIES and name not in graph:
                raise RDDLMissingCPFDefinitionError(
                    f'{fluent_type} CPF <{name}> is missing a valid definition.')
                    
    # ===========================================================================
    # topological sort
    # ===========================================================================
    
    def compute_levels(self) -> Dict[int, Set[str]]:
        graph = self.build_call_graph()
        order = LiftedRDDLLevelAnalysis._topological_sort(graph)
        
        levels, result = {}, {}
        for var in order:
            if var in self.cpfs:
                level = 0
                for child in graph[var]:
                    if child in self.cpfs:
                        level = max(level, levels[child] + 1)
                result.setdefault(level, set()).add(var)
                levels[var] = level
        return result
    
    @staticmethod
    def _topological_sort(graph):
        order = []
        unmarked = set(graph.keys()).union(*graph.values())
        temp = set()
        while unmarked:
            var = next(iter(unmarked))
            LiftedRDDLLevelAnalysis._sort_variables(
                order, graph, var, unmarked, temp)
        return order
    
    @staticmethod
    def _sort_variables(order, graph, var, unmarked, temp):
        if var not in unmarked:
            return
        elif var in temp:
            cycle = ','.join(temp)
            raise RDDLInvalidDependencyInCPFError(
                f'Cyclic dependency detected, suspected CPFs {{{cycle}}}.')
        else:
            temp.add(var)
            if var in graph:
                for dep in graph[var]:
                    LiftedRDDLLevelAnalysis._sort_variables(
                        order, graph, dep, unmarked, temp)
            temp.remove(var)
            unmarked.remove(var)
            order.append(var)
    
