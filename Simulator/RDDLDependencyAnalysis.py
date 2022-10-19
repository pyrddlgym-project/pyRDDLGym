import itertools
from typing import Dict, List, Set

from Grounder.RDDLModel import PlanningModel
from Parser.expr import Expression


class RDDLCyclicDependencyInCPFError(SyntaxError):
    pass


class RDDLUndefinedVariableError(SyntaxError):
    pass


class RDDLDependencyAnalysis:
    
    def __init__(self, model: PlanningModel) -> None:
        self._model = model
    
    @staticmethod
    def _print_stack_trace(expr):
        return '...\n' + str(expr) + '\n...'
    
    # dependency analysis
    def build_call_graph(self) -> Dict[str, Set[str]]:
        graph = {}
        for var, expr in self._model.cpfs.items():
            self._update_call_graph(graph, var, expr, expr)
        
        # check validity of reward, constraints etc
        expr = self._model.reward
        self._update_call_graph({}, '', expr, expr)
        for expr in self._model.preconditions:
            self._update_call_graph({}, '', expr, expr)
        for expr in self._model.invariants:
            self._update_call_graph({}, '', expr, expr)
            
        return graph
    
    def _update_call_graph(self, graph, root_var, expr, parent_expr):
        etype, _ = expr.etype
        args = expr.args
        if etype == 'pvar':
            var = expr.args[0]
            if var not in self._model.nonfluents:
                self._assert_valid_variable(parent_expr, var)
                if root_var not in graph:
                    graph[root_var] = set()
                graph[root_var].add(var)
        elif etype != 'constant':
            for arg in args:
                self._update_call_graph(graph, root_var, arg, parent_expr)
    
    def _assert_valid_variable(self, expr, var):
        if var not in self._model.derived \
            and var not in self._model.interm \
            and var not in self._model.states \
            and var not in self._model.prev_state \
            and var not in self._model.actions:
                raise RDDLUndefinedVariableError(
                    'Variable {} is not defined in the instance.'.format(var) + 
                    '\n' + RDDLDependencyAnalysis._print_stack_trace(expr))
                
    # topological sort
    def compute_levels(self) -> Dict[int, Set[str]]:
        graph = self.build_call_graph()
        
        # topological sort of variables
        order = []
        temp = set()
        unmarked = set(graph.keys()).union(*graph.values())
        while unmarked:
            var = next(iter(unmarked))
            self._sort_variables(order, graph, var, unmarked, temp)
        
        # stratify levels
        levels = {var: 0 for var in order}
        result = {}
        for var in order:
            if var in self._model.cpfs:
                for child in graph[var]:
                    levels[var] = max(levels[var], levels[child] + 1)
                unprimed = self._model.prev_state.get(var, var)
                result.setdefault(levels[var], set()).add(unprimed)
        return result
    
    def _has_dependency(self, graph, level, var):
        return not level.isdisjoint(graph[var])
    
    def _sort_variables(self, order, graph, var, unmarked, temp):
        if var not in unmarked:
            return
        elif var in temp:
            raise RDDLCyclicDependencyInCPFError(
                'CPF {} has a cyclic dependency!'.format(var) + 
                '\n' + RDDLDependencyAnalysis._print_stack_trace(self._model.cpfs[var]))
        else:
            temp.add(var)
            if var in graph:
                for dep in graph[var]:
                    self._sort_variables(order, graph, dep, unmarked, temp)
            temp.remove(var)
            unmarked.remove(var)
            order.append(var)
        
