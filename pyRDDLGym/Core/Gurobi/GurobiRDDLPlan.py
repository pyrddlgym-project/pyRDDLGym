import gurobipy
from gurobipy import GRB
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from pyRDDLGym.Core.Gurobi.GurobiRDDLCompiler import GurobiRDDLCompiler


class GurobiRDDLPlan:
    
    def initialize(self, compiled: 'GurobiRDDLCompiler',
                   model: gurobipy.Model) -> Dict[str, object]:
        '''Returns the parameters of this plan/policy to be optimized.
        
        :param compiled: A gurobi compiler where the current plan is initialized
        :param model: the gurobi model instance
        '''
        raise NotImplementedError
        
    def predict(self, compiled: 'GurobiRDDLCompiler',
                model: gurobipy.Model,
                params: Dict[str, object],
                step: int,
                subs: Dict[str, object]) -> Dict[str, object]:
        '''Returns a dictionary of action variables predicted by the plan.
        
        :param compiled: A gurobi compiler where the current plan is initialized
        :param model: the gurobi model instance
        :param params: parameter variables of the plan/policy
        :param step: the decision epoch
        :param subs: the set of fluent and non-fluent variables available at the
        current step
        '''
        raise NotImplementedError
    

class GurobiRDDLStraightLinePlan(GurobiRDDLPlan):
    
    def initialize(self, compiled: 'GurobiRDDLCompiler',
                   model: gurobipy.Model) -> Dict[str, object]:
        rddl = compiled.rddl
        params = {}
        for (action, prange) in rddl.actionsranges.items():
            if prange == 'bool':
                lb, ub = 0, 1
            else:
                lb, ub = -GRB.INFINITY, GRB.INFINITY
            vtype = compiled.GUROBI_TYPES[prange]
            for step in range(compiled.horizon):
                name = f'{action}___{step}'
                params[name] = compiled._add_var(model, vtype, lb, ub, name)
        return params
        
    def predict(self, compiled: 'GurobiRDDLCompiler',
                model: gurobipy.Model,
                params: Dict[str, object],
                step: int,
                subs: Dict[str, object]) -> Dict[str, object]:
        action_vars = {action: params[f'{action}___{step}'] 
                       for action in compiled.rddl.actions}
        return action_vars
    
