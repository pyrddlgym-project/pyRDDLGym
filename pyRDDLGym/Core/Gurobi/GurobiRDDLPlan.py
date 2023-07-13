import gurobipy
from gurobipy import GRB
from typing import Callable, Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from pyRDDLGym.Core.Gurobi.GurobiRDDLCompiler import GurobiRDDLCompiler

UNBOUNDED = (-GRB.INFINITY, +GRB.INFINITY)


class GurobiRDDLPlan:
    
    def __init__(self, action_bounds: Dict[str, Tuple[float, float]]={}):
        self.action_bounds = action_bounds
    
    def _bounds(self, rddl, action):
        if rddl.actionsranges[action] == 'bool':
            return (0, 1)
        else:
            return self.action_bounds.get(action, UNBOUNDED)
                
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
    
    def to_string(self, compiled: 'GurobiRDDLCompiler',
                  params: Dict[str, object]) -> str:
        '''Returns a string representation of the current policy.
        
        :param params: parameter variables of the plan/policy
        :param compiled: A gurobi compiler where the current plan is initialized
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
    
    def to_string(self, compiled: 'GurobiRDDLCompiler',
                  params: Dict[str, object]) -> str:
        rddl = compiled.rddl
        res = ''
        for step in range(compiled.horizon):
            values = []
            for action in rddl.actions:
                action_value = params[f'{action}__{step}'][0].X
                values.append(f'{action}_{step} = {action_value}')
            res += ', '.join(values) + '\n'
        return res

                
class GurobiLinearPolicy(GurobiRDDLPlan):
    
    def __init__(self, *args, 
                 n_features: int,
                 feature_map: Callable=(lambda model, s: [1.0] + list(s.values())),
                 feature_eval: Callable=(lambda s: [1.0] + list(s.values())),
                 **kwargs) -> None:
        super(GurobiLinearPolicy, self).__init__(*args, **kwargs)
        
        self.n_features = n_features
        self.feature_map = feature_map
        self.feature_eval = feature_eval
        
    def params(self, compiled: 'GurobiRDDLCompiler',
               model: gurobipy.Model,
               values: Dict[str, object]=None) -> Dict[str, object]:
        rddl = compiled.rddl   
        param_vars = {}
        for action in rddl.actions:
            for i in range(self.n_features):
                var_name = f'weight__{action}__{i}'
                if values is None:
                    var = compiled._add_real_var(model)
                    param_vars[var_name] = (var, GRB.CONTINUOUS, *UNBOUNDED, True)
                else:
                    value = values[var_name]
                    param_vars[var_name] = (value, GRB.CONTINUOUS, value, value, False)
        return param_vars
    
    def init_params(self, compiled: 'GurobiRDDLCompiler',
                    model: gurobipy.Model) -> Dict[str, object]:
        rddl = compiled.rddl
        param_values = {}
        for action in rddl.actions:
            param_values[f'weight__{action}__0'] = compiled.init_values[action]
            for i in range(1, self.n_features):
                param_values[f'weight__{action}__{i}'] = 0.0
        return param_values
    
    def actions(self, compiled: 'GurobiRDDLCompiler',
                model: gurobipy.Model,
                params: Dict[str, object],
                step: int,
                subs: Dict[str, object]) -> Dict[str, object]:
        rddl = compiled.rddl
        state_vars = {name: subs[name][0] for name in rddl.states}
        feature_vars = self.feature_map(model, state_vars)
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
        feature_values = self.feature_eval(state_values)
        action_values = {}
        for action in rddl.actions: 
            action_value = 0.0
            for (i, feature_value) in enumerate(feature_values):
                param_value = params[f'weight__{action}__{i}'][0].X
                action_value += param_value * feature_value
            action_values[action] = action_value
        return action_values

    def to_string(self, compiled: 'GurobiRDDLCompiler',
                  params: Dict[str, object]) -> str:
        rddl = compiled.rddl
        res = ''
        for action in rddl.actions:
            values = []
            for i in range(self.n_features):
                param_value = params[f'weight__{action}__{i}'][0].X
                values.append(f'{param_value} * f_{i}(s)')
            res += f'{action}(s) = ' + ' + '.join(values)
        return res


class GurobiFactoredPWSCPolicy(GurobiRDDLPlan):
    
    def __init__(self, *args,
                 state_bounds: Dict[str, Tuple[float, float]]={},
                 upper_bound: bool=False,
                 num_cases: int=1,
                 **kwargs) -> None:
        super(GurobiFactoredPWSCPolicy, self).__init__(*args, **kwargs)
        self.state_bounds = state_bounds
        self.upper_bound = upper_bound or num_cases > 1
        self.num_cases = num_cases
        
    def params(self, compiled: 'GurobiRDDLCompiler',
               model: gurobipy.Model,
               values: Dict[str, object]=None) -> Dict[str, object]:
        rddl = compiled.rddl  
        param_vars = {}
        for ((state, srange), (action, arange)) in zip(
            rddl.statesranges.items(), rddl.actionsranges.items()):
            atype = compiled.GUROBI_TYPES[arange]
            stype = compiled.GUROBI_TYPES[srange]
            lb, ub = self._bounds(rddl, action)
            lbs, ubs = self.state_bounds.get(state, UNBOUNDED)
            
            # initialize parameters of constraint s_i >= lb_i ^ s_i <= ub_i
            # initialize action parameter a_i if constraint is true
            for icase in range(self.num_cases):
                l_name = f'low__{icase}__{state}__{action}'
                h_name = f'high__{icase}__{state}__{action}'
                a_name = f'action__{icase}__{state}__{action}'
                if values is None:
                    l_var = compiled._add_var(model, stype, lbs, ubs)
                    param_vars[l_name] = (l_var, stype, lbs, ubs, True)
                    a_var = compiled._add_var(model, atype, lb, ub)
                    param_vars[a_name] = (a_var, atype, lb, ub, True)
                    if self.upper_bound:
                        h_var = compiled._add_var(model, stype, lbs, ubs)
                        model.addConstr(h_var >= l_var)
                        param_vars[h_name] = (h_var, stype, lbs, ubs, True)
                else:
                    l_val = values[l_name]
                    a_val = values[a_name]                    
                    param_vars[l_name] = (l_val, stype, l_val, l_val, False)
                    param_vars[a_name] = (a_val, atype, a_val, a_val, False)                    
                    if self.upper_bound:
                        h_val = values[h_name]
                        param_vars[h_name] = (h_val, stype, h_val, h_val, False)
            
            # initialize the action parameter a_i of the else constraint
            a_else_name = f'action__else__{state}__{action}'
            if values is None:
                a_else_var = compiled._add_var(model, atype, lb, ub)
                param_vars[a_else_name] = (a_else_var, atype, lb, ub, True)
            else:
                a_else_val = values[a_else_name]
                param_vars[a_else_name] = (a_else_val, atype, a_else_val, a_else_val, False)
                
        return param_vars
    
    def init_params(self, compiled: 'GurobiRDDLCompiler',
                    model: gurobipy.Model) -> Dict[str, object]:
        rddl = compiled.rddl
        param_values = {}
        for (state, action) in zip(rddl.states, rddl.actions):
            lbs, ubs = self.state_bounds.get(state, UNBOUNDED)
            
            # initialize bounds lb_i, ub_i in constraint to default state bounds
            # and action parameter a_i to no-op action
            for icase in range(self.num_cases):                
                l_name = f'low__{icase}__{state}__{action}'
                h_name = f'high__{icase}__{state}__{action}'
                a_name = f'action__{icase}__{state}__{action}'
                param_values[l_name] = lbs
                if self.upper_bound:
                    param_values[h_name] = ubs
                param_values[a_name] = compiled.init_values[action]
            
            # initial action parameter a_i in else constraint to no-op action
            a_else_name = f'action__else__{state}__{action}'
            param_values[a_else_name] = compiled.init_values[action]
            
        return param_values
    
    def actions(self, compiled: 'GurobiRDDLCompiler',
                model: gurobipy.Model,
                params: Dict[str, object],
                step: int,
                subs: Dict[str, object]) -> Dict[str, object]:
        rddl = compiled.rddl
        action_vars = {}
        for (state, (action, arange)) in zip(rddl.states, rddl.actionsranges.items()):
            atype = compiled.GUROBI_TYPES[arange]
            lb, ub = self._bounds(rddl, action)
            
            # check if s_i >= lb_i ^ s_i <= ub_i, and return action a_i 
            sat_vars = []
            res = compiled._add_var(model, atype, lb, ub)
            action_vars[action] = (res, atype, lb, ub, True)
            for icase in range(self.num_cases):
                l_name = f'low__{icase}__{state}__{action}'
                h_name = f'high__{icase}__{state}__{action}'
                a_name = f'action__{icase}__{state}__{action}'
            
                # check the constraint s_i >= l_i or s_i >= l_i ^ s_i <= u_i
                l_diff = subs[state][0] - params[l_name][0]
                l_sat_var = compiled._add_bool_var(model)
                model.addConstr((l_sat_var == 1) >> (l_diff >= 0))
                model.addConstr((l_sat_var == 0) >> (l_diff <= 0))
                if self.upper_bound:
                    h_diff = subs[state][0] - params[h_name][0]
                    h_sat_var = compiled._add_bool_var(model)
                    model.addConstr((h_sat_var == 1) >> (h_diff <= 0))
                    model.addConstr((h_sat_var == 0) >> (h_diff >= 0))
                    sat_var = compiled._add_bool_var(model)
                    model.addGenConstrAnd(sat_var, [l_sat_var, h_sat_var])
                else:
                    sat_var = l_sat_var
                sat_vars.append(sat_var)
                
                # assign action to a_i if constraint satisfied
                a_var = params[a_name][0]
                model.addConstr((sat_var == 1) >> (res == a_var))
            
            # if none of the cases hold then use the last case action
            or_sat_vars = compiled._add_bool_var(model)
            model.addGenConstrOr(or_sat_vars, sat_vars)
            a_else_name = f'action__else__{state}__{action}'
            a_else_var = params[a_else_name][0]
            model.addConstr((or_sat_vars == 0) >> (res == a_else_var))
            
        return action_vars
    
    def evaluate(self, compiled: 'GurobiRDDLCompiler',
                 params: Dict[str, object],
                 step: int,
                 subs: Dict[str, object]) -> Dict[str, object]:
        rddl = compiled.rddl
        action_values = {}
        for (state, (action, arange)) in zip(rddl.states, rddl.actionsranges.items()):
            
            # check if s_i >= lb_i ^ s_i <= ub_i, and return action a_i
            action_value = None
            for icase in range(self.num_cases):
                l_name = f'low__{icase}__{state}__{action}'
                h_name = f'high__{icase}__{state}__{action}'
                a_name = f'action__{icase}__{state}__{action}'
                l_val = params[l_name][0].X
                h_val = params[h_name][0].X if self.upper_bound else float('inf')                
                if subs[state] >= l_val and subs[state] <= h_val:
                    action_value = params[a_name][0].X
                    break
            
            # if none of the cases hold then use the last case action
            if action_value is None:
                a_else_name = f'action__else__{state}__{action}'
                action_value = params[a_else_name][0].X
            
            # cast action to appropriate type
            if arange == 'int':
                action_value = int(action_value)
            action_values[action] = action_value
            
        return action_values

    def to_string(self, compiled: 'GurobiRDDLCompiler',
                  params: Dict[str, object]) -> str:
        rddl = compiled.rddl
        res = ''
        for (state, action) in zip(rddl.states, rddl.actions):
            cases = []
            for icase in range(self.num_cases):
                l_name = f'low__{icase}__{state}__{action}'
                h_name = f'high__{icase}__{state}__{action}'
                a_name = f'action__{icase}__{state}__{action}'
                l_val = params[l_name][0].X  
                a_val = params[a_name][0].X
                if self.upper_bound:
                    h_val = params[h_name][0].X
                    case_val = f'{a_val}, if {state} >= {l_val} ^ {state} <= {h_val}'
                else:
                    case_val = f'{a_val}, if {state} >= {l_val}'
                cases.append(case_val)
             
            a_else_name = f'action__else__{state}__{action}'
            a_else_val = params[a_else_name][0].X
            cases.append(f'{a_else_val}, otherwise')
            
            res += f'{action}({state}) = ' + '\n'.join(cases) + '\n'
        return res
