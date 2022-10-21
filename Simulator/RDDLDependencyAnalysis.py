import itertools
from typing import Dict, List, Set

from Grounder.RDDLModel import PlanningModel
from Parser.expr import Expression


class RDDLCyclicDependencyInCPFError(SyntaxError):
    pass


class RDDLUndefinedVariableError(SyntaxError):
    pass


class RDDLInvalidDependencyInCPFError(SyntaxError):
    pass


class RDDLDependencyAnalysis:
    
    def __init__(self, model: PlanningModel) -> None:
        self._model = model
    
    # dependency analysis
    def build_call_graph(self) -> Dict[str, Set[str]]:
        graph = {}
        for cpf, expr in self._model.cpfs.items():
            self._update_call_graph(graph, cpf, expr)
            if cpf not in graph:  # CPF without dependencies
                graph[cpf] = set()

        # check validity of reward, constraints etc
        expr = self._model.reward
        self._update_call_graph({}, '', expr)
        for expr in self._model.preconditions:
            self._update_call_graph({}, '', expr)
        for expr in self._model.invariants:
            self._update_call_graph({}, '', expr)
        
        # check validity of dependencies according to CPF type
        self._verify_fluents(graph)
        
        return graph
    
    def _update_call_graph(self, graph, cpf, expr):
        etype, _ = expr.etype
        if etype == 'pvar':
            var = expr.args[0]
            if var not in self._model.nonfluents:
                self._assert_valid_variable(cpf, var)
                graph.setdefault(cpf, set()).add(var)
        elif etype != 'constant':
            for arg in expr.args:
                self._update_call_graph(graph, cpf, arg)
    
    # verification of fluent definitions
    # TODO: add any additional constraints here
    def _verify_fluents(self, graph):
        for cpf, deps in graph.items():
            
             # check that next state DND on obs
            if cpf in self._model.prev_state:
                for var in deps:
                    if var in self._model.observ:
                        raise RDDLInvalidDependencyInCPFError(
                            'State-fluent {} depends on observ-fluent {}.'.format(cpf, var))
            
            # check that interm DND on obs, next_state
            elif cpf in self._model.interm or cpf in self._model.derived:
                for var in deps:
                    if var in self._model.observ:
                        raise RDDLInvalidDependencyInCPFError(
                            'Interm-fluent {} depends on observ-fluent {}.'.format(cpf, var))
                    elif var in self._model.prev_state:
                        raise RDDLInvalidDependencyInCPFError(
                            'Interm-fluent {} depends on state-fluent {}.'.format(cpf, var))
                
        
    def _assert_valid_variable(self, cpf, var):
        if var not in self._model.derived \
            and var not in self._model.interm \
            and var not in self._model.states \
            and var not in self._model.prev_state \
            and var not in self._model.actions \
            and var not in self._model.observ:
                raise RDDLUndefinedVariableError(
                    'Variable {} found in CPF {} is not defined.'.format(var, cpf))
    
    def _assert_valid_dependencies(self, cpf, deps):
        pass
        
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
        levels = {}
        result = {}
        for var in order:
            if var in self._model.cpfs:
                level = 0
                for child in graph[var]:
                    if child in self._model.cpfs:
                        level = max(level, levels[child] + 1)
                unprimed = self._model.prev_state.get(var, var)
                result.setdefault(level, set()).add(unprimed)
                levels[var] = level
        return result
    
    def _sort_variables(self, order, graph, var, unmarked, temp):
        if var not in unmarked:
            return
        elif var in temp:
            raise RDDLCyclicDependencyInCPFError(
                'Cyclic dependency detected, suspected CPFs {}.'.format(temp))
        else:
            temp.add(var)
            if var in graph:
                for dep in graph[var]:
                    self._sort_variables(order, graph, dep, unmarked, temp)
            temp.remove(var)
            unmarked.remove(var)
            order.append(var)
    