from typing import Dict, Set
import warnings

from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidDependencyInCPFError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLMissingCPFDefinitionError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLNotImplementedError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLUndefinedVariableError

from pyRDDLGym.Core.Compiler.RDDLModel import RDDLModel
from pyRDDLGym.Core.Parser.expr import Expression

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

    
class RDDLLevelAnalysis:
    '''Performs graphical analysis of a RDDL domain, including dependency
    structure of CPFs, performs topological sort to figure out order of evaluation,
    and checks for cyclic dependencies and ensures dependencies are valid 
    according to the RDDL language specification.     
    '''
        
    def __init__(self, rddl: RDDLModel,
                 allow_synchronous_state: bool=True) -> None:
        '''Creates a new level analysis for the given RDDL domain.
        
        :param rddl: the RDDL domain to analyze
        :param allow_synchronous_state: whether state variables can depend on
        one another (cyclic dependencies will still not be permitted)
        '''
        self.rddl = rddl
        self.allow_synchronous_state = allow_synchronous_state
                
    # ===========================================================================
    # call graph construction
    # ===========================================================================
    
    def build_call_graph(self) -> Dict[str, Set[str]]:
        
        # compute call graph of CPs and check validity
        cpf_graph = {}
        for name, (_, expr) in self.rddl.cpfs.items():
            self._update_call_graph(cpf_graph, name, expr)
            if name not in cpf_graph:
                cpf_graph[name] = set()
        self._validate_dependencies(cpf_graph)
        self._validate_cpf_definitions(cpf_graph)
        
        # check validity of reward, constraints, termination
        for name, exprs in [
            ('reward', [self.rddl.reward]),
            ('precondition', self.rddl.preconditions),
            ('invariant', self.rddl.invariants),
            ('termination', self.rddl.terminals)
        ]:
            call_graph = {}
            for expr in exprs:
                self._update_call_graph(call_graph, name, expr)
            self._validate_dependencies(call_graph)
        return cpf_graph
    
    def _update_call_graph(self, graph, cpf, expr):
        if isinstance(expr, (tuple, list, set)):
            for arg in expr:
                self._update_call_graph(graph, cpf, arg)
        
        elif not isinstance(expr, Expression):
            pass
        
        elif expr.is_pvariable_expression():
            name, *_ = expr.args
            if name in self.rddl.enum_literals:
                pass  # enum literal
            else:
                var = self.rddl.parse(name)[0]  # pvariable
                var_type = self.rddl.variable_types.get(var, None)
                if var_type is None:
                    raise RDDLUndefinedVariableError(
                        f'Variable or literal <{name}> in CPF <{cpf}> '
                        f'is not defined.')
                elif var_type != 'non-fluent':
                    graph.setdefault(cpf, set()).add(name)
                
        elif not expr.is_constant_expression():
            for arg in expr.args:
                self._update_call_graph(graph, cpf, arg)
    
    # ===========================================================================
    # call graph validation
    # ===========================================================================
    
    def _validate_dependencies(self, graph):
        for cpf, deps in graph.items():
            var = self.rddl.parse(cpf)[0]
            cpf_type = self.rddl.variable_types.get(var, var)
            
            # warn use of derived fluent
            if cpf_type == 'derived-fluent':
                warnings.warn(
                    f'The use of derived-fluent is discouraged in this version: '
                    f'please change <{cpf}> to interm-fluent.', stacklevel=2)
            
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
                var = self.rddl.parse(dep)[0]
                dep_type = self.rddl.variable_types.get(var, var)
                if dep_type not in VALID_DEPENDENCIES[cpf_type] or (
                    not self.allow_synchronous_state
                    and cpf_type == dep_type == 'next-state-fluent'
                ):
                    raise RDDLInvalidDependencyInCPFError(
                        f'{cpf_type} <{cpf}> cannot depend on {dep_type} <{dep}>.')                
    
    def _validate_cpf_definitions(self, graph): 
        for cpf in self.rddl.cpfs.keys():
            var = self.rddl.parse(cpf)[0]
            fluent_type = self.rddl.variable_types.get(var, var)
            if fluent_type == 'state-fluent':
                fluent_type = 'next-' + fluent_type
                var = var + '\''
            if fluent_type in VALID_DEPENDENCIES and cpf not in graph:
                raise RDDLMissingCPFDefinitionError(
                    f'{fluent_type} CPF <{cpf}> is missing a definition.')
                    
    # ===========================================================================
    # topological sort
    # ===========================================================================
    
    def compute_levels(self) -> Dict[int, Set[str]]:
        graph = self.build_call_graph()
        order = RDDLLevelAnalysis._topological_sort(graph)
        
        levels, result = {}, {}
        for var in order:
            if var in self.rddl.cpfs:
                level = 0
                for child in graph[var]:
                    if child in self.rddl.cpfs:
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
            RDDLLevelAnalysis._sort_variables(order, graph, var, unmarked, temp)
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
                    RDDLLevelAnalysis._sort_variables(
                        order, graph, dep, unmarked, temp)
            temp.remove(var)
            unmarked.remove(var)
            order.append(var)
    
