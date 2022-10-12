from typing import Dict, List, Set

from Grounder.RDDLModel import PlanningModel
from Parser.expr import Expression

AdjacencyList = Dict[str, Set[str]]


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
    def build_call_graph(self) -> AdjacencyList:
        graph = {}
        for var, expr in self._model.cpfs.items():
            self._update_call_graph(graph, var, expr)
        return graph
    
    def _assert_valid_variable(self, var):
        if var not in self._model.derived \
            and var not in self._model.interm \
            and var not in self._model.states \
            and var not in self._model.prev_state \
            and var not in self._model.actions:
                raise RDDLUndefinedVariableError(
                    'Variable {} is not defined in the instance.'.format(var))
                
    def _update_call_graph(self, graph, root_var, expr):
        etype, _ = expr.etype
        args = expr.args
        if etype == 'pvar':
            var = expr.args[0]
            if var not in self._model.nonfluents:
                self._assert_valid_variable(var)
                if root_var not in graph:
                    graph[root_var] = set()
                graph[root_var].add(var)
        elif etype != 'constant':
            for arg in args:
                self._update_call_graph(graph, root_var, arg)
    
    # topological sort
    def compute_levels(self) -> Dict[int, str]:
        graph = self.build_call_graph()
        order = []
        temp = set()
        unmarked = set(graph.keys()).union(*graph.values())
        while unmarked:
            var = next(iter(unmarked))
            self._sort_variables(order, graph, var, unmarked, temp)
        order = [var for var in order if var in self._model.cpfs]
        order = [(self._model.prev_state[var] if var in self._model.prev_state else var) 
                 for var in order]
        order = [[var] for var in order]
        levels = dict(enumerate(order))
        return levels
        
    def _sort_variables(self, order, graph, var, unmarked, temp):
        if var not in unmarked:
            return
        if var in temp:
            raise RDDLCyclicDependencyInCPFError(
                'CPF {} has a cyclic dependency!'.format(var) + 
                '\n' + RDDLDependencyAnalysis._print_stack_trace(self._model.cpfs[var]))
        temp.add(var)
        if var in graph:
            for dep in graph[var]:
                self._sort_variables(order, graph, dep, unmarked, temp)
        temp.remove(var)
        unmarked.remove(var)
        order.append(var)
        
