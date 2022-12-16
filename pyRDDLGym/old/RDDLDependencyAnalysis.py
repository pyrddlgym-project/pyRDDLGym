# from typing import Dict, Set
# import warnings
#
# from pyRDDLGym.Core.Grounder.RDDLModel import PlanningModel
# from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidDependencyInCPFError
# from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLUndefinedVariableError
#
#
#
# # we generally have state -> derived -> interm -> next state -> obs/reward
# VALID_DEPENDENCIES = {
#     'derived': {'state', 'derived'},
#     'interm': {'action', 'state', 'derived', 'interm'},
#     'next state': {'action', 'state', 'derived', 'interm', 'next state'},
#     'observ': {'action', 'derived', 'interm', 'next state'},
#     'reward': {'action', 'state', 'derived', 'interm', 'next state'},
#     'state invariant': {'state'},
#     'action precondition': {'state', 'action'},
#     'termination': {'state'}
# }
#
#
# class RDDLDependencyAnalysis:
#
#     def __init__(self,
#                  model: PlanningModel,
#                  disallow_state_synchrony: bool=False) -> None:
#         self._model = model
#         self._disallow_state_synchrony = disallow_state_synchrony
#
#     # dependency analysis
#     def build_call_graph(self) -> Dict[str, Set[str]]:
#
#         # compute call graph of CPs and check validity
#         cpf_graph = {}
#         for cpf, expr in self._model.cpfs.items():
#             self._update_call_graph(cpf_graph, cpf, expr)
#             if cpf not in cpf_graph:
#                 cpf_graph[cpf] = set()
#         self._check_deps_by_fluent_type(cpf_graph)
#
#         # check validity of reward, constraints, termination
#         for name, exprs in [
#             ('reward', [self._model.reward]),
#             ('action precondition', self._model.preconditions),
#             ('state invariant', self._model.invariants),
#             ('termination', self._model.terminals)
#         ]:
#             call_graph = {}
#             for expr in exprs:
#                 self._update_call_graph(call_graph, name, expr)
#             self._check_deps_by_fluent_type(call_graph)
#
#         return cpf_graph
#
#     def _update_call_graph(self, graph, cpf, expr):
#         etype, _ = expr.etype
#         if etype == 'pvar':
#             var = expr.args[0]
#             if var not in self._model.nonfluents:
#                 self._check_is_fluent(cpf, var)
#                 graph.setdefault(cpf, set()).add(var)
#         elif etype != 'constant':
#             for arg in expr.args:
#                 self._update_call_graph(graph, cpf, arg)
#
#     # verification of fluent definitions    
#     def _check_is_fluent(self, cpf, var):
#         if var not in self._model.derived \
#         and var not in self._model.interm \
#         and var not in self._model.states \
#         and var not in self._model.prev_state \
#         and var not in self._model.actions \
#         and var not in self._model.observ:
#             raise RDDLUndefinedVariableError(
#                 'Variable <{}> found in CPF <{}> is not defined.'.format(var, cpf))
#
#     def fluent_type(self, fluent: str) -> str:
#         if fluent in self._model.actions: return 'action'
#         elif fluent in self._model.states: return 'state'
#         elif fluent in self._model.interm: return 'interm'
#         elif fluent in self._model.derived: return 'derived'
#         elif fluent in self._model.prev_state: return 'next state'
#         elif fluent in self._model.observ: return 'observ'
#         else: return fluent
#
#     def _check_deps_by_fluent_type(self, graph):
#         for cpf, deps in graph.items():
#             cpf_type = self.fluent_type(cpf)
#             if cpf_type == 'derived':
#                 warnings.warn('The use of derived fluents is discouraged in this RDDL version: ' + 
#                               'please change the type of fluent <{}> to interm.'.format(cpf),
#                               FutureWarning, stacklevel=2)
#             if cpf_type in VALID_DEPENDENCIES:
#                 for dep in deps: 
#                     dep_type = self.fluent_type(dep)
#                     if dep_type not in VALID_DEPENDENCIES[cpf_type] or (
#                         self._disallow_state_synchrony and cpf_type == dep_type == 'next state'):
#                         raise RDDLInvalidDependencyInCPFError(
#                             '{} fluent <{}> cannot depend on {} fluent <{}>.'.format(
#                                 cpf_type, cpf, dep_type, dep))
#
#     # topological sort
#     def compute_levels(self) -> Dict[int, Set[str]]:
#         graph = self.build_call_graph()
#
#         # topological sort of variables
#         order = []
#         temp = set()
#         unmarked = set(graph.keys()).union(*graph.values())
#         while unmarked:
#             var = next(iter(unmarked))
#             self._sort_variables(order, graph, var, unmarked, temp)
#
#         # stratify levels
#         levels = {}
#         result = {}
#         for var in order:
#             if var in self._model.cpfs:
#                 level = 0
#                 for child in graph[var]:
#                     if child in self._model.cpfs:
#                         level = max(level, levels[child] + 1)
#                 unprimed = self._model.prev_state.get(var, var)
#                 result.setdefault(level, set()).add(unprimed)
#                 levels[var] = level
#         return result
#
#     def _sort_variables(self, order, graph, var, unmarked, temp):
#         if var not in unmarked:
#             return
#         elif var in temp:
#             raise RDDLInvalidDependencyInCPFError(
#                 'Cyclic dependency detected, suspected CPFs <{}>.'.format(
#                     ','.join(temp)))
#         else:
#             temp.add(var)
#             if var in graph:
#                 for dep in graph[var]:
#                     self._sort_variables(order, graph, dep, unmarked, temp)
#             temp.remove(var)
#             unmarked.remove(var)
#             order.append(var)
#

