from typing import Dict, Optional, List

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.debug.exception import (
    raise_warning,
    RDDLInvalidDependencyInCPFError,
    RDDLMissingCPFDefinitionError,
    RDDLNotImplementedError,
    RDDLUndefinedVariableError
)
from pyRDDLGym.core.debug.logger import Logger
from pyRDDLGym.core.parser.expr import Expression


class RDDLLevelAnalysis:
    '''Performs graphical analysis of a RDDL domain, including dependency
    structure of CPFs, performs topological sort to figure out order of evaluation,
    and checks for cyclic dependencies and ensures dependencies are valid 
    according to the RDDL language specification.     
    '''
        
    # this specifies the valid dependencies that can occur between variable types
    # in the RDDL language
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
    
    def __init__(self, rddl: RDDLPlanningModel,
                 allow_synchronous_state: bool=True,
                 logger: Optional[Logger]=None) -> None:
        '''Creates a new level analysis for the given RDDL domain.
        
        :param rddl: the RDDL domain to analyze
        :param allow_synchronous_state: whether state variables can depend on
        one another (cyclic dependencies will still not be permitted)
        :param logger: to log information about dependency analysis to file
        '''
        self.rddl = rddl
        self.allow_synchronous_state = allow_synchronous_state
        self.logger = logger
                
    # ===========================================================================
    # call graph construction
    # ===========================================================================
    
    def build_call_graph(self) -> Dict[str, List[str]]:
        '''Builds a call graph for the current RDDL, where keys represent parent
        expressions (e.g., CPFs, reward) and values are sets of variables on 
        which each parent depends. Also handles validation of the call graph to
        make sure it follows RDDL rules.
        '''
        rddl = self.rddl
        
        # compute call graph of CPs and check validity
        cpf_graph = {}
        for (name, (_, expr)) in rddl.cpfs.items():
            self._update_call_graph(cpf_graph, name, expr)
            if name not in cpf_graph:
                cpf_graph[name] = set()
        self._validate_dependencies(cpf_graph)
        self._validate_cpf_definitions(cpf_graph)
        
        # check validity of reward, constraints, termination
        for (name, exprs) in (
            ('reward', [rddl.reward]),
            ('precondition', rddl.preconditions),
            ('invariant', rddl.invariants),
            ('termination', rddl.terminations)
        ):
            call_graph_expr = {}
            for expr in exprs:
                self._update_call_graph(call_graph_expr, name, expr)
            self._validate_dependencies(call_graph_expr)
        
        # produce reproducible order of graph dependencies
        cpf_graph = {name: sorted(deps) 
                     for (name, deps) in cpf_graph.items()}
        return cpf_graph
    
    def _update_call_graph(self, graph, cpf, expr):
        if isinstance(expr, (tuple, list, set)):
            for arg in expr:
                self._update_call_graph(graph, cpf, arg)
        
        elif not isinstance(expr, Expression):
            pass
        
        elif expr.is_pvariable_expression():
            name, pvars = expr.args
            rddl = self.rddl
            
            # free objects (e.g., ?x) are ignored
            if RDDLPlanningModel.is_free_object(name):
                pass
            
            # objects are ignored
            elif not pvars and rddl.is_object(
                name, f'Please check expression for CPF {cpf}.'): 
                pass
            
            # variable defined in pvariables {..} scope
            else:
                
                # check that name is valid variable
                var_type = rddl.variable_types.get(name, None)
                if var_type is None:
                    raise RDDLUndefinedVariableError(
                        f'Variable <{name}> is not defined in '
                        f'expression for CPF <{cpf}>.')
                
                # if var is a fluent assign it as dependent of cpf
                elif var_type != 'non-fluent':
                    graph.setdefault(cpf, set()).add(name)
                
                # if a nested fluent
                if pvars is not None:
                    self._update_call_graph(graph, cpf, pvars)
        
        # scan compound expression
        elif not expr.is_constant_expression():
            self._update_call_graph(graph, cpf, expr.args)
    
    # ===========================================================================
    # call graph validation
    # ===========================================================================
    
    def _validate_dependencies(self, graph):
        for (cpf, deps) in graph.items():
            cpf_type = self.rddl.variable_types.get(cpf, cpf)
            
            # warn use of derived fluent
            if cpf_type == 'derived-fluent':
                raise_warning(
                    f'The use of derived-fluent is no longer supported, '
                    f'please change <{cpf}> to interm-fluent.', 'red')
            
            # not a recognized type
            if cpf_type not in RDDLLevelAnalysis.VALID_DEPENDENCIES:
                if cpf_type == 'state-fluent':
                    PRIME = RDDLPlanningModel.NEXT_STATE_SYM
                    raise RDDLInvalidDependencyInCPFError(
                        f'CPF definition for state-fluent <{cpf}> is not valid, '
                        f'did you mean <{cpf}{PRIME}>?')
                else:
                    raise RDDLNotImplementedError(
                        f'Type <{cpf_type}> of CPF <{cpf}> is not valid.')
            
            # check that all dependencies are valid
            for dep in deps: 
                dep_type = self.rddl.variable_types.get(dep, dep)
                
                # completely illegal dependency
                if dep_type not in RDDLLevelAnalysis.VALID_DEPENDENCIES[cpf_type]:
                    raise RDDLInvalidDependencyInCPFError(
                        f'{cpf_type} <{cpf}> cannot depend on {dep_type} <{dep}>.') 
                
                # s' cannot depend on s' if allow_synchronous_state = False
                elif not self.allow_synchronous_state \
                and cpf_type == dep_type == 'next-state-fluent':
                    raise RDDLInvalidDependencyInCPFError(
                        f'{cpf_type} <{cpf}> cannot depend on {dep_type} <{dep}>, '
                        f'set allow_synchronous_state=True to allow this.')                
    
    def _validate_cpf_definitions(self, graph):
        for cpf in self.rddl.cpfs:
            fluent_type = self.rddl.variable_types.get(cpf, cpf)
            if fluent_type == 'state-fluent':
                fluent_type = 'next-' + fluent_type
                cpf = cpf + RDDLPlanningModel.NEXT_STATE_SYM
            if fluent_type in RDDLLevelAnalysis.VALID_DEPENDENCIES \
            and cpf not in graph:
                raise RDDLMissingCPFDefinitionError(
                    f'{fluent_type} CPF <{cpf}> is not defined '
                    f'in cpfs {{...}} block.')
                    
    # ===========================================================================
    # topological sort
    # ===========================================================================
    
    def compute_levels(self) -> Dict[int, List[str]]:
        '''Constructs a call graph for the current RDDL, and then runs a 
        topological sort to determine the optimal order in which the CPFs in the 
        RDDL should be be evaluated.
        '''
        rddl = self.rddl
        graph = self.build_call_graph()
        order = _topological_sort(graph)
        
        # use the graph structure to group CPFs into levels 0, 1, 2, ...
        # two CPFs in the same level cannot depend on each other
        # a CPF can only depend on another CPF of a lower level than it
        levels, result = {}, {}
        for var in order:
            if var in rddl.cpfs:
                level = 0
                for child in graph[var]:
                    if child in rddl.cpfs:
                        level = max(level, levels[child] + 1)
                result.setdefault(level, set()).add(var)
                levels[var] = level
        
        # produce reproducible order of graph dependencies
        result = {level: sorted(cpfs) 
                  for (level, cpfs) in result.items()}
        
        # log dependency graph information to file
        if self.logger is not None: 
            graph_info = '\n\t'.join(f"{rddl.variable_types[k]} {k}: "
                                     f"{{{', '.join(v)}}}"
                                     for (k, v) in graph.items())
            self.logger.log(f'[info] computed fluent dependencies in CPFs:\n' 
                            f'\t{graph_info}\n')
            
            levels_info = '\n\t'.join(f"{k}: {{{', '.join(v)}}}"
                                      for (k, v) in result.items())
            self.logger.log(f'[info] computed order of CPF evaluation:\n' 
                            f'\t{levels_info}\n')
        
        return result
    
# ===========================================================================
# helper functions for performing topological sort
# ===========================================================================

    
def _topological_sort(graph):
    order = []
    unmarked = set(graph.keys()).union(*graph.values())
    temp = set()
    while unmarked:
        var = next(iter(unmarked))
        _sort_variables(order, graph, var, unmarked, temp)
    return order

    
def _sort_variables(order, graph, var, unmarked, temp):
        
    # var has already been visited
    if var not in unmarked: 
        return
        
    # a cycle is detected
    elif var in temp:
        cycle = ', '.join(temp)
        raise RDDLInvalidDependencyInCPFError(
            f'Cyclic dependency detected among CPFs {{{cycle}}}.')
        
    # recursively sort all variables on which var depends
    else: 
        temp.add(var)
        deps = graph.get(var, None)
        if deps is not None:
            for dep in deps:
                _sort_variables(order, graph, dep, unmarked, temp)
        temp.remove(var)
        unmarked.remove(var)
        order.append(var)
    
