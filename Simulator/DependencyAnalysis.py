from typing import Dict, List, Set

from Grounder.RDDLModel import PlanningModel
from Parser.expr import Expression

AdjacencyList = Dict[str, Set[str]]


class DependencyAnalysis:
    
    def __init__(self, model: PlanningModel) -> None:
        self._model = model
    
    def build_call_graph(self) -> AdjacencyList:
        graph = {}
        for root_var, cpf in self._model.cpfs.items():
            self._update_call_graph(graph, root_var, cpf.expr)
        return graph
    
    def _update_call_graph(self, graph, root_var, expr):
        etype, _ = expr.etype
        args = expr.args
        if etype == 'pvar':
            var = expr.args[0]
            if var not in self._model.nonfluents:
                if root_var not in graph:
                    graph[root_var] = set()
                graph[root_var].add(var)
        elif etype != 'constant':
            for arg in args:
                self._update_call_graph(graph, root_var, arg)
    
    def compute_levels(self) -> List[str]:
        graph = self.build_call_graph()
        sort = []
        unmarked = set(graph.keys()).union(*graph.values())
        temp = set()
        while unmarked:
            var = next(iter(unmarked))
            self._sort_variables(sort, graph, var, unmarked, temp)
        return sort
        
    def _sort_variables(self, sort, graph, var, unmarked, temp):
        if var not in unmarked:
            return
        if var in temp:
            raise Exception('RDDL cpfs have a cyclic dependency!')
        temp.add(var)
        if var in graph:
            for dep in graph[var]:
                self._sort_variables(sort, graph, dep, unmarked, temp)
        temp.remove(var)
        unmarked.remove(var)
        sort.append(var)
        
