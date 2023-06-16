import gurobipy
from gurobipy import GRB
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from pyRDDLGym.Core.Gurobi.GurobiRDDLCompiler import GurobiRDDLCompiler


class GurobiRDDLPlan:
    
    def action_vars(self, compiled: 'GurobiRDDLCompiler',
                    step: int,
                    model: gurobipy.Model,
                    subs: Dict[str, object]) -> Dict[str, object]:
        raise NotImplementedError
    

class GurobiRDDLStraightLinePlan(GurobiRDDLPlan):
    
    def action_vars(self, compiled: 'GurobiRDDLCompiler',
                    step: int,
                    model: gurobipy.Model,
                    subs: Dict[str, object]) -> Dict[str, object]:
        rddl = compiled.rddl
        variables = {}
        for (action, prange) in rddl.actionsranges.items():
            name = f'{action}___{step}'
            lb, ub = (0, 1) if prange == 'bool' else (-GRB.INFINITY, GRB.INFINITY)
            vtype = compiled.GUROBI_TYPES[prange]
            var = compiled._add_var(name, model, vtype, lb, ub, name=name)
            variables[action] = (var, vtype, lb, ub, True)
        return variables
