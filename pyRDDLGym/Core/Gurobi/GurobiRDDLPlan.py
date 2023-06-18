import gurobipy
from gurobipy import GRB
from typing import Dict, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from pyRDDLGym.Core.Gurobi.GurobiRDDLCompiler import GurobiRDDLCompiler


class GurobiRDDLPlan:
    
    def action_vars(self, compiled: 'GurobiRDDLCompiler',
                    step: int,
                    model: gurobipy.Model,
                    subs: Dict[str, object],
                    init_values: Dict[str, object]) -> Tuple[Dict[str, object], List[object]]:
        '''Returns a tuple consisting of the action variables at the specified
        time step, along with a list of auxiliary variables of the plan/policy
        that are to be optimized.
        
        :param compiled: A gurobi compiler where the current plan is initialized
        :param step: the decision epoch
        :param model: the gurobi model instance
        :param subs: the set of fluent and non-fluent variables available at the
        current step
        :param init_values: the set of initial fluent and non-fluent values
        '''
        raise NotImplementedError
    

class GurobiRDDLStraightLinePlan(GurobiRDDLPlan):
    
    def action_vars(self, compiled: 'GurobiRDDLCompiler',
                    step: int,
                    model: gurobipy.Model,
                    subs: Dict[str, object],
                    init_values: Dict[str, object]) -> Tuple[Dict[str, object], List[object]]:
        rddl = compiled.rddl
        variables = {}
        for (action, prange) in rddl.actionsranges.items():
            name = f'{action}___{step}'
            lb, ub = (0, 1) if prange == 'bool' else (-GRB.INFINITY, GRB.INFINITY)
            vtype = compiled.GUROBI_TYPES[prange]
            var = compiled._add_var(name, model, vtype, lb, ub, name=name)
            variables[action] = (var, vtype, lb, ub, True)
        aux = []
        return variables, aux
