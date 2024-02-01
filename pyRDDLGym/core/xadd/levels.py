from typing import cast, Dict, List, Union
from xaddpy.xadd import XADD

from pyRDDLGym.core.compiler.levels import RDDLLevelAnalysis
from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.parser.expr import Expression

from pyRDDLGym.xadd.model import RDDLModelXADD


class RDDLLevelAnalysisXADD(RDDLLevelAnalysis):
    
    def __init__(self, model: RDDLPlanningModel, 
                 allow_synchronous_state: bool = True) -> None:
        super().__init__(model, allow_synchronous_state)
        
        self._model = cast(RDDLModelXADD, model)
        self._context: XADD = self._model._context
    
    def _update_call_graph(self, graph: Dict[str, List[str]], 
                           cpf: str, 
                           expr: Union[int, Expression]):
        if isinstance(expr, Expression):
            super()._update_call_graph(graph, cpf, expr)
        else:
            model = self._model
            var_set = self._context.collect_vars(expr)
            for v in var_set:
                v_name = model._sympy_var_name_to_var_name.get(str(v), str(v))
                if v_name not in model.non_fluents and v not in model.rvs:
                    graph.setdefault(cpf, set()).add(v_name)
    
    def build_call_graph(self) -> Dict[str, List[str]]:
        model = self._model
        
        # compute call graph of CPs and check validity
        cpf_graph = {}
        for (name, expr) in model.cpfs.items():
            self._update_call_graph(cpf_graph, name, expr)
            if name not in cpf_graph:
                cpf_graph[name] = set()
        self._validate_dependencies(cpf_graph)
        self._validate_cpf_definitions(cpf_graph)
        
        # check validity of reward, constraints, termination
        for (name, exprs) in (
            ('reward', [model.reward]),
            ('precondition', model.preconditions),
            ('invariant', model.invariants),
            ('termination', model.terminations)
        ):
            call_graph_expr = {}
            for expr in exprs:
                if isinstance(expr, int):
                    self._update_call_graph(call_graph_expr, name, expr)
                else:
                    super()._update_call_graph(call_graph_expr, name, expr)
            self._validate_dependencies(call_graph_expr)
            
        return cpf_graph
