from typing import Union, cast, Dict, Set
from xaddpy.xadd import XADD

from pyRDDLGym.Core.Compiler.RDDLLevelAnalysis import RDDLLevelAnalysis
from pyRDDLGym.Core.Compiler.RDDLModel import PlanningModel
from pyRDDLGym.XADD.RDDLModelXADD import RDDLModelWXADD
from pyRDDLGym.Core.Parser.expr import Expression


class RDDLLevelAnalysisWXADD(RDDLLevelAnalysis):
    
    def __init__(self, model: PlanningModel, 
                 disallow_state_synchrony: bool = False) -> None:
        super().__init__(model, disallow_state_synchrony)
        self._model = cast(RDDLModelWXADD, self._model)
        self._context: XADD = self._model._context
    
    def _update_call_graph(self, graph: Dict[str, Set[str]], 
                           cpf: str, 
                           expr: Union[int, Expression]):
        if isinstance(expr, Expression):
            super()._update_call_graph(graph, cpf, expr)
        else:
            var_set = self._context.collect_vars(expr)
            for v in var_set:
                v_name = self._model._sympy_var_name_to_var_name.get(str(v), str(v))
                if v_name not in self._model.nonfluents and v not in self._model.rvs:
                    self._check_is_fluent(cpf, v_name)
                    graph.setdefault(cpf, set()).add(v_name)
    
    # dependency analysis
    def build_call_graph(self) -> Dict[str, Set[str]]:
        
        # compute call graph of CPs and check validity
        cpf_graph = {}
        for cpf, expr in self._model.cpfs.items():
            self._update_call_graph(cpf_graph, cpf, expr)
            if cpf not in cpf_graph:
                cpf_graph[cpf] = set()
        self._check_deps_by_fluent_type(cpf_graph)
        
        # check validity of reward, constraints, termination
        for name, exprs in [
            ('reward', [self._model.reward]),
            ('action precondition', self._model.preconditions),
            ('state invariant', self._model.invariants),
            ('termination', self._model.terminals)
        ]:
            call_graph = {}
            for expr in exprs:
                if isinstance(expr, int):
                    self._update_call_graph(call_graph, name, expr)
                else:
                    super()._update_call_graph(call_graph, name, expr)
            self._check_deps_by_fluent_type(call_graph)
            
        return cpf_graph
