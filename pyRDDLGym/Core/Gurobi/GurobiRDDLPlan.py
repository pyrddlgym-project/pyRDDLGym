import numpy as np
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
        params = {}
        for (action, prange) in rddl.actionsranges.items():
            lb, ub = self._bounds(rddl, action)
            vtype = compiled.GUROBI_TYPES[prange]
            
            # each decision epoch has a set of action variables in the model
            for step in range(compiled.horizon):
                var_name = f'{action}__{step}'
                if values is None:
                    var = compiled._add_var(model, vtype, lb, ub)
                    params[var_name] = (var, vtype, lb, ub, True)
                else:
                    value = values[var_name]
                    params[var_name] = (value, vtype, value, value, False)
        return params
        
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
            value = params[f'{action}__{step}'][0].X
            if prange == 'int':
                value = int(value)
            action_values[action] = value        
        return action_values
        

class GurobiLinearPolicy(GurobiRDDLPlan):
    
    def __init__(self, *args,
                 feature_map: Callable=(lambda a, s: s.values()),
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
        
        params = {}
        for action in rddl.actions:
            
            # bias parameter
            var_name = f'bias__{action}'
            if values is None: 
                var = compiled._add_real_var(model)
                params[var_name] = (var, GRB.CONTINUOUS, *unbounded, True)
            else:
                value = values[var_name]
                params[var_name] = (value, GRB.CONTINUOUS, value, value, False)
            
            # feature weight parameters
            n_features = len(self.feature_map(action, rddl.states))
            for i in range(n_features):
                var_name = f'weight__{action}__{i}'
                if values is None:
                    var = compiled._add_real_var(model)
                    params[var_name] = (var, GRB.CONTINUOUS, *unbounded, True)
                else:
                    value = values[var_name]
                    params[var_name] = (value, GRB.CONTINUOUS, value, value, False)
        return params
    
    def init_params(self, compiled: 'GurobiRDDLCompiler',
                    model: gurobipy.Model) -> Dict[str, object]:
        rddl = compiled.rddl
        param_values = {}
        for action in rddl.actions:
            param_values[f'bias__{action}'] = 0.0
            n_features = len(self.feature_map(action, rddl.states))
            for i in range(n_features):
                param_values[f'weight__{action}__{i}'] = np.random.normal(scale=self.noise)
        return param_values
    
    def actions(self, compiled: 'GurobiRDDLCompiler',
                model: gurobipy.Model,
                params: Dict[str, object],
                step: int,
                subs: Dict[str, object]) -> Dict[str, object]:
        rddl = compiled.rddl
        action_vars = {}
        for action in rddl.actions:
            
            # compute feature map
            states = {name: subs[name][0] for name in rddl.states}
            features = self.feature_map(action, states)
            
            # compute linear combination bias + feature.T * weight
            action_value = params[f'bias__{action}'][0]
            for (i, feature) in enumerate(features):
                param = params[f'weight__{action}__{i}'][0]
                action_value += param * feature
            lb, ub = self._bounds(rddl, action)
            action_var = compiled._add_real_var(model, lb, ub)
            model.addConstr(action_var == action_value)
            action_vars[action] = (action_var, GRB.CONTINUOUS, lb, ub, True)
        return action_vars
    
    def evaluate(self, compiled: 'GurobiRDDLCompiler',
                 params: Dict[str, object],
                 step: int,
                 subs: Dict[str, object]) -> Dict[str, object]:
        rddl = compiled.rddl
        action_values = {}
        for action in rddl.actions:
            states = {name: subs[name] for name in rddl.states}
            features = self.feature_map(action, states)
            action_value = params[f'bias__{action}'][0].X
            for (i, feature) in enumerate(features):
                param = params[f'weight__{action}__{i}'][0].X
                action_value += param * feature
            action_values[action] = action_value
        return action_values
    

class GurobiCornersPolicy(GurobiRDDLPlan):
    
    def params(self, compiled: 'GurobiRDDLCompiler',
               model: gurobipy.Model,
               values: Dict[str, object]=None) -> Dict[str, object]:
        rddl = compiled.rddl
        unbounded = (-GRB.INFINITY, +GRB.INFINITY)
        params = {}
        for action in rddl.actions:
            
            # for a constraint (s1 > C(a, s1)) & (s2 > C(a, s2)) ...
            for state in rddl.states:
                var_name = f'corner__{action}__{state}'
                if values is None:
                    var = compiled._add_real_var(model)
                    params[var_name] = (var, GRB.CONTINUOUS, *unbounded, True)
                else:
                    value = values[var_name]
                    params[var_name] = (value, GRB.CONTINUOUS, value, value, False)
            
            # action = action1 if constraint == True else action2
            lb, ub = self._bounds(rddl, action)
            if values is None:
                a1 = compiled._add_real_var(model, lb, ub)
                a2 = compiled._add_real_var(model, lb, ub)
                params[f'action1__{action}'] = (a1, GRB.CONTINUOUS, lb, ub, True)
                params[f'action2__{action}'] = (a2, GRB.CONTINUOUS, lb, ub, True)
            else:
                a1 = values[f'action1__{action}']
                a2 = values[f'action2__{action}']
                params[f'action1__{action}'] = (a1, GRB.CONTINUOUS, a1, a1, False)
                params[f'action2__{action}'] = (a2, GRB.CONTINUOUS, a2, a2, False)                
        return params

    def init_params(self, compiled: 'GurobiRDDLCompiler',
                    model: gurobipy.Model) -> Dict[str, object]:
        rddl = compiled.rddl
        params = {}
        for action in rddl.actions:
            for state in rddl.states:
                params[f'corner__{action}__{state}'] = 0
            params[f'action1__{action}'] = 0
            params[f'action2__{action}'] = 0
        return params

    def actions(self, compiled: 'GurobiRDDLCompiler',
                model: gurobipy.Model,
                params: Dict[str, object],
                step: int,
                subs: Dict[str, object]) -> Dict[str, object]:
        rddl = compiled.rddl
        action_vars = {}
        for action in rddl.actions:
            
            # calculate constraint (s1 > C(a, s1)) & (s2 > C(a, s2))
            args = []
            for state in rddl.states:
                var_name = f'corner__{action}__{state}'
                diff = subs[state][0] - params[var_name][0]
                satisfied = compiled._add_bool_var(model)
                model.addConstr((satisfied == 1) >> (diff >= compiled.epsilon))
                model.addConstr((satisfied == 0) >> (diff <= 0))
                args.append(satisfied)
            var_constr = compiled._add_bool_var(model)
            model.addGenConstrAnd(var_constr, args)
            
            # calculate action values
            a1 = params[f'action1__{action}'][0]
            a2 = params[f'action2__{action}'][0]
            lb, ub = self._bounds(rddl, action)
            action_var = compiled._add_real_var(model, lb, ub)
            model.addConstr((var_constr == 1) >> (action_var == a1))
            model.addConstr((var_constr == 0) >> (action_var == a2))
            action_vars[action] = (action_var, GRB.CONTINUOUS, lb, ub, True)
        return action_vars
    
    def evaluate(self, compiled: 'GurobiRDDLCompiler',
                 params: Dict[str, object],
                 step: int,
                 subs: Dict[str, object]) -> Dict[str, object]:
        rddl = compiled.rddl
        action_values = {}
        for (action, prange) in rddl.actionsranges.items():
            
            # evaluate constraint (s1 > C(a, s1)) & (s2 > C(a, s2))
            satisfied = True
            for state in rddl.states:
                var_name = f'corner__{action}__{state}'
                satisfied = satisfied and (subs[state] > params[var_name][0].X)
            
            # evaluate action values
            var_name = f'action{1 if satisfied else 2}__{action}'
            value = params[var_name][0].X
            if prange == 'int':
                value = int(value)
            action_values[action] = value
        return action_values
    

class GurobiInventoryPolicy(GurobiRDDLPlan):
        
    def params(self, compiled: 'GurobiRDDLCompiler',
               model: gurobipy.Model,
               values: Dict[str, object]=None) -> Dict[str, object]:
        rddl = compiled.rddl
        params = {}
        for (action, state) in zip(rddl.actions, rddl.states):
            var_name = f'threshold__{action}__{state}'
            if values is None:
                var = compiled._add_real_var(model)
                params[var_name] = (var, GRB.INTEGER, -GRB.INFINITY, GRB.INFINITY, True)
            else:
                value = values[var_name]
                params[var_name] = (value, GRB.INTEGER, value, value, False)
        return params
    
    def init_params(self, compiled: 'GurobiRDDLCompiler',
                    model: gurobipy.Model) -> Dict[str, object]:
        rddl = compiled.rddl
        param_values = {}
        for (action, state) in zip(rddl.actions, rddl.states): 
            param_values[f'threshold__{action}__{state}'] = 0.0
        return param_values
    
    def actions(self, compiled: 'GurobiRDDLCompiler',
                model: gurobipy.Model,
                params: Dict[str, object],
                step: int,
                subs: Dict[str, object]) -> Dict[str, object]:
        rddl = compiled.rddl
        action_vars = {}
        for (action, state) in zip(rddl.actions, rddl.states):
            var_name = f'threshold__{action}__{state}'
            var_diff = compiled._add_int_var(model)
            model.addConstr(var_diff == params[var_name][0] - subs[state][0])
            lb, ub = self._bounds(rddl, action)
            var_order = compiled._add_int_var(model, lb, ub)
            model.addGenConstrMax(var_order, [var_diff], 0)
            action_vars[action] = (var_order, GRB.INTEGER, lb, ub, True)
        return action_vars
    
    def evaluate(self, compiled: 'GurobiRDDLCompiler',
                 params: Dict[str, object],
                 step: int,
                 subs: Dict[str, object]) -> Dict[str, object]:
        rddl = compiled.rddl
        action_values = {}
        for (action, state) in zip(rddl.actions, rddl.states): 
            var_name = f'threshold__{action}__{state}'
            action_values[action] = max(int(params[var_name][0].X - subs[state]), 0)
        return action_values
    
