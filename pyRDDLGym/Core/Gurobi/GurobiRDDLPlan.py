import numpy as np
import gurobipy
from gurobipy import GRB
from typing import Callable, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from pyRDDLGym.Core.Gurobi.GurobiRDDLCompiler import GurobiRDDLCompiler


class GurobiRDDLPlan:
    
    def parameterize(self, compiled: 'GurobiRDDLCompiler',
                     model: gurobipy.Model,
                     values: Dict[str, object]=None) -> Dict[str, object]:
        '''Returns the parameters of this plan/policy to be optimized.
        
        :param compiled: A gurobi compiler where the current plan is initialized
        :param model: the gurobi model instance
        :param values: if None, freeze policy parameters to these values
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
    
    def init_params(self, compiled: 'GurobiRDDLCompiler',
                    model: gurobipy.Model) -> Dict[str, object]:
        '''Return initial parameter values for the current policy class.
        '''
        raise NotImplementedError


class GurobiRDDLStraightLinePlan(GurobiRDDLPlan):
    
    def parameterize(self, compiled: 'GurobiRDDLCompiler',
                     model: gurobipy.Model,
                     values: Dict[str, object]=None) -> Dict[str, object]:
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
                if values is None:
                    var = compiled._add_var(model, vtype, lb, ub)
                    params[name] = (var, vtype, lb, ub, True)
                else:
                    value = values[name]
                    params[name] = (value, vtype, value, value, False)
        return params
        
    def predict(self, compiled: 'GurobiRDDLCompiler',
                model: gurobipy.Model,
                params: Dict[str, object],
                step: int,
                subs: Dict[str, object]) -> Dict[str, object]:
        action_vars = {action: params[f'{action}___{step}'] 
                       for action in compiled.rddl.actions}
        return action_vars
    
    def init_params(self, compiled: 'GurobiRDDLCompiler',
                    model: gurobipy.Model) -> Dict[str, object]:
        params = {}
        for action in compiled.rddl.actions:
            for step in range(compiled.horizon):
                params[f'{action}___{step}'] = compiled.init_values[action]
        return params


class GurobiLinearPolicy(GurobiRDDLPlan):
    
    def __init__(self, state_feature: Callable=None, noise: float=0.01) -> None:
        if state_feature is None:
            state_feature = lambda action, states: states.values()
        self.state_feature = state_feature
        self.noise = noise
        
    def parameterize(self, compiled: 'GurobiRDDLCompiler',
                     model: gurobipy.Model,
                     values: Dict[str, object]=None) -> Dict[str, object]:
        rddl = compiled.rddl
        params = {}
        for action in rddl.actions:
            
            # bias
            name = f'bias_{action}'
            if values is None: 
                var = compiled._add_real_var(model)
                params[name] = (var, GRB.CONTINUOUS, -GRB.INFINITY, GRB.INFINITY, True)
            else:
                value = values[name]
                params[name] = (value, GRB.CONTINUOUS, value, value, False)
            
            # feature weights
            features = self.state_feature(action, rddl.states)
            for i in range(len(features)):
                name = f'weight_{action}_{i}'
                if values is None:
                    var = compiled._add_real_var(model)
                    params[name] = (var, GRB.CONTINUOUS, -GRB.INFINITY, GRB.INFINITY, True)
                else:
                    value = values[name]
                    params[name] = (value, GRB.CONTINUOUS, value, value, False)
        return params
    
    def predict(self, compiled: 'GurobiRDDLCompiler',
                model: gurobipy.Model,
                params: Dict[str, object],
                step: int,
                subs: Dict[str, object]) -> Dict[str, object]:
        rddl = compiled.rddl
        action_vars = {}
        for action in rddl.actions:
            states = {name: subs[name][0] for name in rddl.states}
            features = self.state_feature(action, states)
            action_value = params[f'bias_{action}'][0]
            for i, feature in enumerate(features):
                param = params[f'weight_{action}_{i}'][0]
                action_value += param * feature
            action_var = compiled._add_real_var(model)
            model.addConstr(action_var == action_value)
            action_vars[action] = (
                action_var, GRB.CONTINUOUS, -GRB.INFINITY, GRB.INFINITY, True)
        return action_vars
    
    def init_params(self, compiled: 'GurobiRDDLCompiler',
                    model: gurobipy.Model) -> Dict[str, object]:
        rddl = compiled.rddl
        params = {}
        for action in rddl.actions:
            params[f'bias_{action}'] = 0.0
            features = self.state_feature(action, rddl.states)
            for i in range(len(features)):
                params[f'weight_{action}_{i}'] = np.random.normal(scale=self.noise)
        return params
