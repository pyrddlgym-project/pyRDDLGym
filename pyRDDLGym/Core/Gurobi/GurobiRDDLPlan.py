import gurobipy
from gurobipy import GRB
from typing import Callable, Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from pyRDDLGym.Core.Gurobi.GurobiRDDLCompiler import GurobiRDDLCompiler


class GurobiRDDLPlan:
    
    def __init__(self, action_bounds: Dict[str, Tuple[float, float]]={}):
        self.action_bounds = action_bounds
    
    def _bounds(self, rddl, action):
        if rddl.actionsranges[action] == 'bool':
            return (0, 1)
        else:
            return self.action_bounds.get(
                action, (-GRB.INFINITY, +GRB.INFINITY))
                
    def params(self, compiled: 'GurobiRDDLCompiler',
               model: gurobipy.Model,
               values: Dict[str, object]=None) -> Dict[str, object]:
        '''Returns the parameters of this plan/policy to be optimized.
        
        :param compiled: A gurobi compiler where the current plan is initialized
        :param model: the gurobi model instance
        :param values: if None, freeze policy parameters to these values
        '''
        raise NotImplementedError
        
    def init_params(self, compiled: 'GurobiRDDLCompiler',
                    model: gurobipy.Model) -> Dict[str, object]:
        '''Return initial parameter values for the current policy class.
        
        :param compiled: A gurobi compiler where the current plan is initialized
        :param model: the gurobi model instance
        '''
        raise NotImplementedError

    def actions(self, compiled: 'GurobiRDDLCompiler',
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
    
    def evaluate(self, compiled: 'GurobiRDDLCompiler',
                 params: Dict[str, object],
                 step: int,
                 subs: Dict[str, object]) -> Dict[str, object]:
        '''Evaluates the current policy with state variables in subs.
        
        :param compiled: A gurobi compiler where the current plan is initialized
        :param params: parameter variables of the plan/policy
        :param step: the decision epoch
        :param subs: the set of fluent and non-fluent variables available at the
        current step
        '''
        raise NotImplementedError
        

class GurobiRDDLStraightLinePlan(GurobiRDDLPlan):
    
    def params(self, compiled: 'GurobiRDDLCompiler',
               model: gurobipy.Model,
               values: Dict[str, object]=None) -> Dict[str, object]:
        rddl = compiled.rddl
        action_vars = {}
        for (action, prange) in rddl.actionsranges.items():
            lb, ub = self._bounds(rddl, action)
            vtype = compiled.GUROBI_TYPES[prange]
            for step in range(compiled.horizon):
                var_name = f'{action}__{step}'
                if values is None:
                    var = compiled._add_var(model, vtype, lb, ub)
                    action_vars[var_name] = (var, vtype, lb, ub, True)
                else:
                    value = values[var_name]
                    action_vars[var_name] = (value, vtype, value, value, False)
        return action_vars
        
    def init_params(self, compiled: 'GurobiRDDLCompiler',
                    model: gurobipy.Model) -> Dict[str, object]:
        param_values = {}
        for action in compiled.rddl.actions:
            for step in range(compiled.horizon):
                param_values[f'{action}__{step}'] = compiled.init_values[action]
        return param_values

    def actions(self, compiled: 'GurobiRDDLCompiler',
                model: gurobipy.Model,
                params: Dict[str, object],
                step: int,
                subs: Dict[str, object]) -> Dict[str, object]:
        action_vars = {action: params[f'{action}__{step}'] 
                       for action in compiled.rddl.actions}
        return action_vars
    
    def evaluate(self, compiled: 'GurobiRDDLCompiler',
                 params: Dict[str, object],
                 step: int,
                 subs: Dict[str, object]) -> Dict[str, object]:
        rddl = compiled.rddl
        action_values = {}
        for (action, prange) in rddl.actionsranges.items():
            action_value = params[f'{action}__{step}'][0].X
            if prange == 'int':
                action_value = int(action_value)
            action_values[action] = action_value        
        return action_values
        

class GurobiLinearPolicy(GurobiRDDLPlan):
    
    def __init__(self, *args,
                 feature_map: Callable=(lambda s: [1.0] + list(s.values())),
                 noise: float=0.01,
                 **kwargs) -> None:
        super(GurobiLinearPolicy, self).__init__(*args, **kwargs)
        self.feature_map = feature_map
        self.noise = noise
        
    def params(self, compiled: 'GurobiRDDLCompiler',
               model: gurobipy.Model,
               values: Dict[str, object]=None) -> Dict[str, object]:
        rddl = compiled.rddl
        unbounded = (-GRB.INFINITY, +GRB.INFINITY)     
        n_features = len(self.feature_map(rddl.states))   
        param_vars = {}
        for action in rddl.actions:
            for i in range(n_features):
                var_name = f'weight__{action}__{i}'
                if values is None:
                    var = compiled._add_real_var(model)
                    param_vars[var_name] = (var, GRB.CONTINUOUS, *unbounded, True)
                else:
                    value = values[var_name]
                    param_vars[var_name] = (value, GRB.CONTINUOUS, value, value, False)
        return param_vars
    
    def init_params(self, compiled: 'GurobiRDDLCompiler',
                    model: gurobipy.Model) -> Dict[str, object]:
        rddl = compiled.rddl
        n_features = len(self.feature_map(rddl.states))
        param_values = {}
        for action in rddl.actions:
            param_values[f'weight__{action}__0'] = compiled.init_values[action]
            for i in range(1, n_features):
                param_values[f'weight__{action}__{i}'] = 0.0
        return param_values
    
    def actions(self, compiled: 'GurobiRDDLCompiler',
                model: gurobipy.Model,
                params: Dict[str, object],
                step: int,
                subs: Dict[str, object]) -> Dict[str, object]:
        rddl = compiled.rddl
        state_vars = {name: subs[name][0] for name in rddl.states}
        feature_vars = self.feature_map(state_vars)
        action_vars = {}
        for action in rddl.actions:            
            linexpr = 0.0
            for (i, feature_var) in enumerate(feature_vars):
                param_var = params[f'weight__{action}__{i}'][0]
                linexpr += param_var * feature_var
            lb, ub = self._bounds(rddl, action)
            var = compiled._add_real_var(model, lb, ub)
            model.addConstr(var == linexpr)
            action_vars[action] = (var, GRB.CONTINUOUS, lb, ub, True)
        return action_vars
    
    def evaluate(self, compiled: 'GurobiRDDLCompiler',
                 params: Dict[str, object],
                 step: int,
                 subs: Dict[str, object]) -> Dict[str, object]:
        rddl = compiled.rddl
        state_values = {name: subs[name] for name in rddl.states}
        feature_values = self.feature_map(state_values)
        action_values = {}
        for action in rddl.actions:            
            action_value = 0.0
            for (i, feature_value) in enumerate(feature_values):
                param_value = params[f'weight__{action}__{i}'][0].X
                action_value += param_value * feature_value
            action_values[action] = action_value
        return action_values
    

class GurobiFactoredPWSCPolicy(GurobiRDDLPlan):
    
    def params(self, compiled: 'GurobiRDDLCompiler',
               model: gurobipy.Model,
               values: Dict[str, object]=None) -> Dict[str, object]:
        rddl = compiled.rddl
        unbounded = (-GRB.INFINITY, +GRB.INFINITY)     
        param_vars = {}
        for ((state, srange), (action, arange)) in zip(
            rddl.statesranges.items(), rddl.actionsranges.items()):
            stype = compiled.GUROBI_TYPES[srange]
            atype = compiled.GUROBI_TYPES[arange]
            constr_name = f'threshold__{state}__{action}'
            value1_name = f'value1__{state}__{action}'
            value2_name = f'value2__{state}__{action}'
            lb, ub = self._bounds(rddl, action)
            if values is None:
                constr_var = compiled._add_var(model, stype)
                value1_var = compiled._add_var(model, atype, lb, ub)
                value2_var = compiled._add_var(model, atype, lb, ub)
                param_vars[constr_name] = (constr_var, stype, *unbounded, True)
                param_vars[value1_name] = (value1_var, atype, lb, ub, True)
                param_vars[value2_name] = (value2_var, atype, lb, ub, True)
            else:
                constr_val = values[constr_name]
                value1_val = values[value1_name]
                value2_val = values[value2_name]
                param_vars[constr_name] = (constr_val, stype, constr_val, constr_val, False)
                param_vars[value1_name] = (value1_val, atype, value1_val, value1_val, False)
                param_vars[value2_name] = (value2_val, atype, value2_val, value2_val, False)
        return param_vars
    
    def init_params(self, compiled: 'GurobiRDDLCompiler',
                    model: gurobipy.Model) -> Dict[str, object]:
        rddl = compiled.rddl
        param_values = {}
        for (state, action) in zip(rddl.states, rddl.actions):
            param_values[f'threshold__{state}__{action}'] = 0
            param_values[f'value1__{state}__{action}'] = compiled.init_values[action]
            param_values[f'value2__{state}__{action}'] = compiled.init_values[action]
        return param_values
    
    def actions(self, compiled: 'GurobiRDDLCompiler',
                model: gurobipy.Model,
                params: Dict[str, object],
                step: int,
                subs: Dict[str, object]) -> Dict[str, object]:
        rddl = compiled.rddl
        action_vars = {}
        for ((state, _), (action, arange)) in zip(
            rddl.statesranges.items(), rddl.actionsranges.items()):
            state_var = subs[state][0]
            constr_var = params[f'threshold__{state}__{action}'][0]
            diffexpr = state_var - constr_var
            constr_sat_var = compiled._add_bool_var(model)
            model.addConstr((constr_sat_var == 1) >> (diffexpr >= 0))
            model.addConstr((constr_sat_var == 0) >> (diffexpr <= 0))
            
            value1_var = params[f'value1__{state}__{action}'][0]
            value2_var = params[f'value2__{state}__{action}'][0]
            atype = compiled.GUROBI_TYPES[arange]
            lb, ub = self._bounds(rddl, action)
            action_var = compiled._add_var(model, atype, lb, ub)
            model.addConstr((constr_sat_var == 1) >> (action_var == value1_var))
            model.addConstr((constr_sat_var == 0) >> (action_var == value2_var))
            action_vars[action] = (action_var, atype, lb, ub, True)
        return action_vars
    
    def evaluate(self, compiled: 'GurobiRDDLCompiler',
                 params: Dict[str, object],
                 step: int,
                 subs: Dict[str, object]) -> Dict[str, object]:
        rddl = compiled.rddl
        action_values = {}
        for (state, (action, arange)) in zip(rddl.states, rddl.actionsranges.items()):
            if subs[state] >= params[f'threshold__{state}__{action}'][0].X:
                action_value = params[f'value1__{state}__{action}'][0].X
            else:
                action_value = params[f'value2__{state}__{action}'][0].X
            if arange == 'int':
                action_value = int(action_value)
            action_values[action] = action_value
        return action_values
    