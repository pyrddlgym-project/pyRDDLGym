from collections import deque
import itertools
import numpy as np

from typing import Dict, Optional, List, Set

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
    according to the RDDL language specification.'''
        
    # this specifies the valid dependencies that can occur between variable types
    # in the RDDL language
    VALID_DEPENDENCIES = {
        'derived-fluent': {'state-fluent', 'derived-fluent', 'non-fluent'},
        'interm-fluent': {'action-fluent', 'state-fluent',
                          'derived-fluent', 'interm-fluent', 'non-fluent'},
        'next-state-fluent': {'action-fluent', 'state-fluent',
                              'derived-fluent', 'interm-fluent', 'next-state-fluent', 
                              'non-fluent'},
        'observ-fluent': {'action-fluent',
                          'derived-fluent', 'interm-fluent', 'next-state-fluent', 
                          'non-fluent'},
        'reward': {'action-fluent', 'state-fluent',
                   'derived-fluent', 'interm-fluent', 'next-state-fluent', 'non-fluent'},
        'invariant': {'state-fluent', 'non-fluent'},
        'precondition': {'state-fluent', 'action-fluent', 'non-fluent'},
        'termination': {'state-fluent', 'non-fluent'}
    }

    VALID_DEPENDENCIES_POLICY = {
        'derived-fluent': {'state-fluent', 'derived-fluent', 'non-fluent', 'param-fluent'},
        'action-fluent': {'state-fluent', 'action-fluent', 'derived-fluent', 'non-fluent',
                          'param-fluent'},
        'next-state-fluent': {'action-fluent', 'state-fluent', 'derived-fluent',
                              'next-state-fluent', 'non-fluent', 'param-fluent'}
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
    
    def build_call_graph(self, include_non_fluents: bool=False,
                         include_reward: bool=False) -> Dict[str, List[str]]:
        '''Builds a call graph for the current RDDL, where keys represent parent
        expressions (CPFs, and the reward if include_reward) and values are the 
        sorted lists of variables on which each parent depends. Also handles 
        validation of the call graph to make sure it follows RDDL rules.

        :param include_non_fluents: whether to include non-fluents in call graph
        :param include_reward: whether to include the reward in call graph'
        '''
        rddl = self.rddl
        
        # compute call graph of CPs and check validity
        cpf_graph = {}
        for (name, (_, expr)) in rddl.cpfs.items():
            self._update_call_graph(cpf_graph, name, expr, include_non_fluents)
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
                self._update_call_graph(call_graph_expr, name, expr, include_non_fluents)
            self._validate_dependencies(call_graph_expr)
            if name == 'reward':
                reward_deps = call_graph_expr.get(name, set())
        
        # produce reproducible order of graph dependencies
        cpf_graph = {name: sorted(deps) 
                     for (name, deps) in cpf_graph.items()}
        if include_reward:
            cpf_graph['reward'] = sorted(reward_deps)
        return cpf_graph
    
    def _update_call_graph(self, graph, cpf, expr, nfs):
        if isinstance(expr, (tuple, list, set)):
            for arg in expr:
                self._update_call_graph(graph, cpf, arg, nfs)
        
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
                name, f'Please check expression for CPF <{cpf}>.'): 
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
                elif nfs or var_type != 'non-fluent':
                    graph.setdefault(cpf, set()).add(name)
                
                # if a nested fluent
                if pvars is not None:
                    self._update_call_graph(graph, cpf, pvars, nfs)
        
        # scan compound expression
        elif not expr.is_constant_expression():
            self._update_call_graph(graph, cpf, expr.args, nfs)
    
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
                        f'CPF <{cpf}> of type {cpf_type} '
                        f'can not depend on variable <{dep}> of type {dep_type}.') 
                
                # s' cannot depend on s' if allow_synchronous_state = False
                elif not self.allow_synchronous_state \
                and cpf_type == dep_type == 'next-state-fluent':
                    raise RDDLInvalidDependencyInCPFError(
                        f'CPF <{cpf}> of type {cpf_type} '
                        f'can not depend on variable <{dep}> of type {dep_type}, '
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
                    f'CPF <{cpf}> of type {fluent_type} is not defined in cpfs block.')
                    
    # ===========================================================================
    # call graph construction for policy
    # ===========================================================================
    
    def build_call_graph_policy(self, include_non_fluents: bool=True) -> Dict[str, List[str]]:
        '''Builds a call graph for the policy.'''
        if self.rddl.policy is None:
            return {}

        # build policy call graph and check validity
        cpf_graph = {}
        for (name, (_, expr)) in self.rddl.policy.cpfs.items():
            self._update_call_graph_policy(cpf_graph, name, expr, include_non_fluents)
            if name not in cpf_graph:
                cpf_graph[name] = set()
        self._validate_dependencies_policy(cpf_graph)

        # produce reproducible order of graph dependencies
        cpf_graph = {name: sorted(deps) 
                     for (name, deps) in cpf_graph.items()}
        return cpf_graph

    def _update_call_graph_policy(self, graph, cpf, expr, nfs):
        if isinstance(expr, (tuple, list, set)):
            for arg in expr:
                self._update_call_graph_policy(graph, cpf, arg, nfs)
        
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
                name, f'Please check expression for CPF <{cpf}>.'): 
                pass
            
            # variable defined in pvariables {..} scope
            else:
                
                # check that name is valid variable
                var_type = rddl.variable_types.get(
                    name, rddl.policy.variable_types.get(name, None))
                if var_type is None:
                    raise RDDLUndefinedVariableError(
                        f'Variable <{name}> is not defined in '
                        f'expression for policy CPF <{cpf}>.')
                
                # if var is a fluent assign it as dependent of cpf
                elif nfs or var_type != 'non-fluent':
                    graph.setdefault(cpf, set()).add(name)
                
                # if a nested fluent
                if pvars is not None:
                    self._update_call_graph_policy(graph, cpf, pvars, nfs)
        
        # scan compound expression
        elif not expr.is_constant_expression():
            self._update_call_graph_policy(graph, cpf, expr.args, nfs)
    
    def _validate_dependencies_policy(self, graph):
        rddl = self.rddl
        policy = rddl.policy
        
        for (cpf, deps) in graph.items():

            # check if cpf is action fluent or policy derived fluent
            if cpf in rddl.action_fluents:
                cpf_type = rddl.variable_types[cpf]
            else:
                cpf_type = policy.variable_types.get(cpf, cpf)

            # not a recognized type
            if cpf_type not in RDDLLevelAnalysis.VALID_DEPENDENCIES_POLICY:
                if cpf_type == 'state-fluent':
                    PRIME = RDDLPlanningModel.NEXT_STATE_SYM
                    raise RDDLInvalidDependencyInCPFError(
                        f'CPF definition for policy state-fluent <{cpf}> is not valid, '
                        f'did you mean <{cpf}{PRIME}>?')
                else:
                    raise RDDLNotImplementedError(
                        f'Type <{cpf_type}> of policy CPF <{cpf}> is not valid.')
            
            # check that all dependencies are valid
            for dep in deps: 
                if dep in rddl.variable_types:
                    dep_type = rddl.variable_types[dep]
                else:
                    dep_type = policy.variable_types.get(dep, dep)
                    
                # completely illegal dependency
                if dep_type not in RDDLLevelAnalysis.VALID_DEPENDENCIES_POLICY[cpf_type]:
                    raise RDDLInvalidDependencyInCPFError(
                        f'Policy CPF <{cpf}> of type {cpf_type} '
                        f'can not depend on p-variable <{dep}> of type {dep_type}.') 
                
                # derived-fluent dependency must be policy defined
                if (dep_type == 'derived-fluent' or dep_type == 'next-state-fluent') \
                and dep not in policy.variable_types:
                    raise RDDLInvalidDependencyInCPFError(
                        f'Policy CPF <{cpf}> can not depend on {dep_type} <{dep}> '
                        f'defined in the domain.') 

    # ===========================================================================
    # level and reachability analysis
    # ===========================================================================
    
    def compute_levels(self) -> Dict[int, List[str]]:
        '''Constructs a call graph for the current RDDL, and then runs a 
        topological sort to determine the optimal order in which the CPFs in the 
        RDDL should be be evaluated.'''
        rddl = self.rddl
        graph = self.build_call_graph()
        order = _topological_sort(graph)
        result = _group_by_level(graph, order, rddl.cpfs)
        
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
    
    def compute_dependencies(self, include_non_fluents: bool=False) -> Dict[str, Set[str]]:
        '''Computes all direct/indirect dependencies between fluents for the current RDDL.

        :param include_non_fluents: whether to include non-fluents in call graph
        '''
        graph = self.build_call_graph(include_non_fluents=include_non_fluents)
        return _bfs_search(graph)
    
    # ===========================================================================
    # level and reachability analysis for the policy
    # ===========================================================================
    
    def variable_type_from_rddl_or_policy(self, var, default=None):
        rddl = self.rddl
        return rddl.variable_types.get(
            var, 
            (default if rddl.policy is None 
             else rddl.policy.variable_types.get(var, default))
        )
    
    def compute_levels_policy(self) -> Dict[int, List[str]]:
        '''Constructs a call graph for the current RDDL policy, and then runs a 
        topological sort to determine the optimal order in which the CPFs in the 
        policy should be be evaluated.
        '''
        if self.rddl.policy is None:
            return {}
        graph = self.build_call_graph_policy()
        order = _topological_sort(graph)
        result = _group_by_level(graph, order, self.rddl.policy.cpfs)
        
        # log dependency graph information to file
        if self.logger is not None: 
            graph_info = '\n\t'.join(f"{self.variable_type_from_rddl_or_policy(k)} {k}: "
                                     f"{{{', '.join(v)}}}"
                                     for (k, v) in graph.items())
            self.logger.log(f'[info] computed fluent dependencies in policy CPFs:\n' 
                            f'\t{graph_info}\n')
            
            levels_info = '\n\t'.join(f"{k}: {{{', '.join(v)}}}"
                                      for (k, v) in result.items())
            self.logger.log(f'[info] computed order of policy CPF evaluation:\n' 
                            f'\t{levels_info}\n')
            
        return result
    
    def compute_dependencies_policy(self, include_non_fluents: bool=True) -> Dict[str, Set[str]]:
        '''Computes all direct/indirect dependencies between fluents for the current RDDL policy.

        :param include_non_fluents: whether to include non-fluents in call graph
        '''
        graph = self.build_call_graph_policy(include_non_fluents=include_non_fluents)
        return _bfs_search(graph)


class RDDLInitStateAnalysis:
    '''Computes the evaluation order of init-state expressions.'''

    MATRIX_OPS = ('inverse', 'pinverse', 'cholesky')

    def __init__(self, rddl: RDDLPlanningModel) -> None:
        self.rddl = rddl
        self._non_fluents = None

    def compute_schedule(self) -> list:
        '''Returns the schedule of the init-state: its assignments in evaluation order, as
        tuples (name, expr, target, copy), where target is a tuple of object indices or a
        boolean mask of the groundings to write, and copy is the grounding (objects) when
        expr is the lifted entry's expression evaluated for that grounding on its own.'''
        rddl = self.rddl
        exprs = rddl.init_state_exprs
        defined = {(name, grounding) for (name, entry) in exprs.items()
                   for grounding in itertools.chain(entry['covered'], entry['ground'])}

        # dependencies of each grounding
        graph, discarded_reads = {}, {}
        for (name, entry) in exprs.items():
            for (grounding, expr) in entry['ground'].items():
                graph[(name, grounding)] = self._reads(expr, {}, (name, grounding), defined)
            if entry['lifted'] is None:
                continue
            objects, expr = entry['lifted']
            variables = [var for (var, _) in objects]
            covered = set(entry['covered'])
            discarded_reads[name] = set()
            scope = {var: tuple(rddl.type_to_objects[ptype]) for (var, ptype) in objects}
            if not self._reads(expr, scope, (name, tuple(variables)), defined):
                graph.update(((name, grounding), set()) for grounding in covered)
                continue
            for grounding in rddl.ground_types(rddl.variable_params[name]):
                scope = {var: (obj,) for (var, obj) in zip(variables, grounding)}
                reads = self._reads(expr, scope, (name, grounding), defined)
                if grounding in covered:
                    graph[(name, grounding)] = reads
                else:
                    discarded_reads[name].update(reads)
        levels, users = self._levels_of_groundings(graph)

        # level of each lifted evaluation
        lifted_level = {}
        for (name, reads) in discarded_reads.items():
            covered = exprs[name]['covered']
            if covered:
                lifted_level[name] = max(itertools.chain(
                    (levels[(name, grounding)] for grounding in covered),
                    (levels[node] + 1 for node in reads)))
        read_by_lifted = set().union(*discarded_reads.values())

        schedule = []
        masks = {name: np.zeros(rddl.object_counts(rddl.variable_params[name]), dtype=bool)
                 for name in lifted_level}
        for (node, level) in levels.items():
            name, grounding = node
            entry = exprs[name]
            indices = rddl.object_indices(grounding)
            if grounding in entry['ground']:
                schedule.append((level, name, entry['ground'][grounding], indices, None))
                continue
            at = lifted_level[name]
            if node in read_by_lifted or any(
                    levels[user] <= at for user in users.get(node, ())):
                objects, expr = entry['lifted']
                literals = {var: '@' + obj for ((var, _), obj) in zip(objects, grounding)}
                op = self._matrix_op_over(expr, set(literals))
                if op is not None:
                    lifted = self.grounding_str((name, tuple(literals)))
                    raise RDDLNotImplementedError(
                        f'Init-state grounding <{self.grounding_str(node)}> is read before '
                        f'<{lifted}> can be evaluated, so it must be evaluated on its '
                        f'own, but {op} over the parameters of <{lifted}> requires '
                        f'the whole matrix.')
                schedule.append(
                    (level, name, self._substitute(expr, literals), indices, grounding))
            else:
                masks[name][indices] = True

        for (name, mask) in masks.items():
            if mask.any():
                schedule.append(
                    (lifted_level[name], name, exprs[name]['lifted'][1], mask, None))
        schedule.sort(key=lambda assignment: assignment[0])
        return [assignment[1:] for assignment in schedule]

    @staticmethod
    def _substitute(expr, literals):
        '''Returns a copy of expr with free variables replaced by object literals.'''
        if isinstance(expr, Expression):
            return Expression((expr[0], RDDLInitStateAnalysis._substitute(expr[1], literals)))
        elif isinstance(expr, (tuple, list)):
            return type(expr)(RDDLInitStateAnalysis._substitute(arg, literals) for arg in expr)
        else:
            return literals.get(expr, expr) if isinstance(expr, str) else expr

    @staticmethod
    def _matrix_op_over(expr, variables):
        '''Returns the first matrix operation in expr with a row or column parameter
        in variables, or None.'''
        if isinstance(expr, Expression):
            etype, op = expr.etype
            if etype == 'matrix' and op in RDDLInitStateAnalysis.MATRIX_OPS \
            and variables.intersection(expr.args[0]):
                return op
            expr = expr[1]
        if isinstance(expr, (tuple, list)):
            for arg in expr:
                op = RDDLInitStateAnalysis._matrix_op_over(arg, variables)
                if op is not None:
                    return op
        return None

    def _reads(self, expr, scope, node, defined):
        references = set()
        self._references(expr, scope, references, node)
        return references.intersection(defined)

    def _references(self, expr, scope, references, node):
        rddl = self.rddl
        if isinstance(expr, (tuple, list)):
            for arg in expr:
                self._references(arg, scope, references, node)
        elif not isinstance(expr, Expression) or expr.is_constant_expression():
            pass
        elif not expr.is_pvariable_expression():
            for arg in expr.args:
                if isinstance(arg, tuple) and arg[:1] == ('typed_var',):
                    var, ptype = arg[1]
                    scope = {**scope, var: tuple(rddl.type_to_objects[ptype])}

            # row and column parameters come from the outer scope, but range over
            # the whole matrix inside the operation
            etype, op = expr.etype
            if etype == 'matrix' and op in RDDLInitStateAnalysis.MATRIX_OPS:
                scope = {**scope, **{
                    var: tuple(rddl.type_to_objects[rddl.object_to_type[scope[var][0]]])
                    for var in expr.args[0]}}
            self._references(expr.args, scope, references, node)
        else:
            name, args = expr.args
            var_type = rddl.variable_types.get(name, None)
            if var_type is None or (not args and rddl.is_object(name)):
                return
            elif var_type not in ('state-fluent', 'non-fluent'):
                raise RDDLInvalidDependencyInCPFError(
                    f'Init-state expression for <{self.grounding_str(node)}> can not '
                    f'depend on <{name}> of type {var_type}.')
            args = args or []
            self._references(args, scope, references, node)
            if var_type == 'state-fluent':
                choices = [self._objects(arg, ptype, scope) 
                           for (arg, ptype) in zip(args, rddl.variable_params[name])]
                references.update((name, objs) for objs in itertools.product(*choices))

    @staticmethod
    def grounding_str(node):
        name, objects = node
        return f'{name}({", ".join(objects)})' if objects else name

    @staticmethod
    def _levels_of_groundings(graph):
        '''Returns the level of each node and the nodes that depend on each node.'''
        users = {}
        for (node, deps) in graph.items():
            for dep in deps:
                users.setdefault(dep, []).append(node)
        waiting = {node: len(deps) for (node, deps) in graph.items()}
        levels, level = {}, 0
        current = [node for (node, count) in waiting.items() if count == 0]
        while current:
            following = []
            for node in current:
                levels[node] = level
                for user in users.get(node, ()):
                    waiting[user] -= 1
                    if waiting[user] == 0:
                        following.append(user)
            current, level = following, level + 1
        if len(levels) < len(graph):
            nodes = sorted(set(graph).difference(levels))
            nodes = ', '.join(map(RDDLInitStateAnalysis.grounding_str, nodes))
            raise RDDLInvalidDependencyInCPFError(
                f'Cyclic dependency in init-state: groundings {{{nodes}}} are in '
                f'or depend on a cycle.')
        return levels, users

    def _objects(self, arg, ptype, scope):
        '''The objects an argument can evaluate to, resolved at compile time for objects, 
        bound variables and non-fluents.'''
        rddl = self.rddl
        if isinstance(arg, str):
            if arg == '_':
                return tuple(rddl.type_to_objects[ptype])
            elif RDDLPlanningModel.is_free_object(arg):
                return scope[arg]
            else:
                return (RDDLPlanningModel.strip_literal(arg),)
        name, args = arg.args if arg.is_pvariable_expression() else (None, None)
        if name is not None and not args and rddl.is_object(name):
            return (RDDLPlanningModel.strip_literal(name),)
        elif rddl.variable_types.get(name, None) == 'non-fluent':
            if self._non_fluents is None:
                self._non_fluents = rddl.ground_vars_with_values(rddl.non_fluents)
            choices = [self._objects(a, t, scope) 
                       for (a, t) in zip(args or [], rddl.variable_params[name])]
            return tuple({self._non_fluents[rddl.ground_var(name, objs)] 
                          for objs in itertools.product(*choices)})
        else:
            return tuple(rddl.type_to_objects[ptype])
    

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
            f'Cyclic dependency detected within CPFs {{{cycle}}}.')
        
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


def _group_by_level(graph, order, cpfs):
    levels, result = {}, {}
    for var in order:
        if var in cpfs:
            level = 0
            for child in graph[var]:
                if child in cpfs:
                    level = max(level, levels[child] + 1)
            result.setdefault(level, set()).add(var)
            levels[var] = level
    result = {level: sorted(cpfs) 
              for (level, cpfs) in result.items()}
    return result


def _bfs_search(graph):
    def bfs(start):
        queue = deque([start])
        visited = set([start])
        while queue:
            node = queue.popleft()
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        visited.remove(start)
        return visited 
    reachability = {node: bfs(node) for node in graph}
    return reachability
