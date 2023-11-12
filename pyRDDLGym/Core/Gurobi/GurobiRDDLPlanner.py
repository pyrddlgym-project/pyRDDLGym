from typing import Dict, List, Tuple
import warnings

import gurobipy
from gurobipy import GRB

from pyRDDLGym.Core.Compiler.RDDLLiftedModel import RDDLLiftedModel
from pyRDDLGym.Core.Gurobi.GurobiRDDLCompiler import GurobiRDDLCompiler
from pyRDDLGym.Core.Policies.Agents import BaseAgent

UNBOUNDED = (-GRB.INFINITY, +GRB.INFINITY)


# ***********************************************************************
# ALL VERSIONS OF GUROBI PLANS
# 
# - straight line plan
# - piecewise linear policy
# - quadratic policy
#
# ***********************************************************************
class GurobiRDDLPlan:
    '''Base class for all Gurobi compiled policies or plans.'''
    
    def __init__(self, action_bounds: Dict[str, Tuple[float, float]]={}):
        self.action_bounds = action_bounds
    
    def _bounds(self, rddl, action):
        if rddl.actionsranges[action] == 'bool':
            return (0, 1)
        else:
            return self.action_bounds.get(action, UNBOUNDED)
    
    def summarize_hyperparameters(self):
        pass
        
    def params(self, compiled: GurobiRDDLCompiler,
               model: gurobipy.Model,
               values: Dict[str, object]=None) -> Dict[str, object]:
        '''Returns the parameters of this plan/policy to be optimized.
        
        :param compiled: A gurobi compiler where the current plan is initialized
        :param model: the gurobi model instance
        :param values: if None, freeze policy parameters to these values
        '''
        raise NotImplementedError
        
    def init_params(self, compiled: GurobiRDDLCompiler,
                    model: gurobipy.Model) -> Dict[str, object]:
        '''Return initial parameter values for the current policy class.
        
        :param compiled: A gurobi compiler where the current plan is initialized
        :param model: the gurobi model instance
        '''
        raise NotImplementedError

    def actions(self, compiled: GurobiRDDLCompiler,
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
    
    def evaluate(self, compiled: GurobiRDDLCompiler,
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
    
    def to_string(self, compiled: GurobiRDDLCompiler,
                  params: Dict[str, object]) -> str:
        '''Returns a string representation of the current policy.
        
        :param params: parameter variables of the plan/policy
        :param compiled: A gurobi compiler where the current plan is initialized
        '''
        raise NotImplementedError


class GurobiStraightLinePlan(GurobiRDDLPlan):
    '''A straight-line open-loop plan in Gurobi.'''
    
    def params(self, compiled: GurobiRDDLCompiler,
               model: gurobipy.Model,
               values: Dict[str, object]=None) -> Dict[str, object]:
        rddl = compiled.rddl
        action_vars = {}
        for (action, prange) in rddl.actionsranges.items():
            bounds = self._bounds(rddl, action)
            atype = compiled.GUROBI_TYPES[prange]
            for step in range(compiled.horizon):
                name = f'{action}__{step}'
                if values is None:
                    var = compiled._add_var(model, atype, *bounds)
                    action_vars[name] = (var, atype, *bounds, True)
                else:
                    value = values[name]
                    action_vars[name] = (value, atype, value, value, False)
        return action_vars
        
    def init_params(self, compiled: GurobiRDDLCompiler,
                    model: gurobipy.Model) -> Dict[str, object]:
        param_values = {}
        for action in compiled.rddl.actions:
            for step in range(compiled.horizon):
                param_values[f'{action}__{step}'] = compiled.init_values[action]
        return param_values

    def actions(self, compiled: GurobiRDDLCompiler,
                model: gurobipy.Model,
                params: Dict[str, object],
                step: int,
                subs: Dict[str, object]) -> Dict[str, object]:
        action_vars = {action: params[f'{action}__{step}'] 
                       for action in compiled.rddl.actions}
        return action_vars
    
    def evaluate(self, compiled: GurobiRDDLCompiler,
                 params: Dict[str, object],
                 step: int,
                 subs: Dict[str, object]) -> Dict[str, object]:
        rddl = compiled.rddl
        action_values = {}
        for (action, prange) in rddl.actionsranges.items():
            name = f'{action}__{step}'
            action_value = params[name][0].X
            if prange == 'int':
                action_value = int(action_value)
            elif prange == 'bool':
                action_value = bool(action_value > 0.5)
            action_values[action] = action_value        
        return action_values
    
    def to_string(self, compiled: GurobiRDDLCompiler,
                  params: Dict[str, object]) -> str:
        rddl = compiled.rddl
        res = ''
        for step in range(compiled.horizon):
            values = []
            for action in rddl.actions:
                name = f'{action}__{step}'
                values.append(f'{action}[{step}] = {params[name][0].X}')
            res += ', '.join(values) + '\n'
        return res


class GurobiPiecewisePolicy(GurobiRDDLPlan):
    '''A piecewise linear policy in Gurobi.'''
    
    def __init__(self, *args,
                 state_bounds: Dict[str, Tuple[float, float]]={},
                 dependencies_constr: Dict[str, List[str]]={},
                 dependencies_values: Dict[str, List[str]]={},
                 num_cases: int=1,
                 **kwargs) -> None:
        super(GurobiPiecewisePolicy, self).__init__(*args, **kwargs)        
        self.state_bounds = state_bounds
        self.dependencies_constr = dependencies_constr
        if dependencies_values is None:
            dependencies_values = {}
        self.dependencies_values = dependencies_values
        self.num_cases = num_cases
    
    def summarize_hyperparameters(self):
        print(f'Gurobi policy hyper-params:\n'
              f'    num_cases     ={self.num_cases}\n'
              f'    state_bounds  ={self.state_bounds}\n'
              f'    constraint_dep={self.dependencies_constr}\n'
              f'    value_dep     ={self.dependencies_values}')
        
    def _get_states_for_constraints(self, rddl):
        if self.dependencies_constr:
            return self.dependencies_constr
        else:
            state_names = list(rddl.states.keys())
            return {action: state_names for action in rddl.actions}
    
    def params(self, compiled: GurobiRDDLCompiler,
               model: gurobipy.Model,
               values: Dict[str, object]=None) -> Dict[str, object]:
        rddl = compiled.rddl  
        states_in_constr = self._get_states_for_constraints(rddl)
        states_in_values = self.dependencies_values
        
        param_vars = {}
        for (action, arange) in rddl.actionsranges.items():
            
            # each case k
            for k in list(range(self.num_cases)) + ['else']:
                
                # constraint is a linear function of state variables
                if k != 'else':
                    
                    # constraint value assumes two cases:
                    # 1. PWL: S = {s1,... sK} --> w1 * s1 + ... + wK * sK
                    # 2. PWS: S = {s}         --> s
                    states = states_in_constr[action]
                    is_linear = len(states) > 1
                    if is_linear:
                        for feature in states:
                            name = f'constr_weight__{action}__{k}__{feature}'
                            if values is None:
                                var = compiled._add_real_var(model)
                                param_vars[name] = (var, GRB.CONTINUOUS, *UNBOUNDED, True)
                            else:
                                val = values[name]
                                param_vars[name] = (val, GRB.CONTINUOUS, val, val, False)
                        
                    # lower and upper bounds for constraint value
                    vtype = GRB.CONTINUOUS
                    lb_name = f'lb__{action}__{k}'
                    ub_name = f'ub__{action}__{k}'                    
                    if values is None:
                        lb, ub = self.state_bounds[states[0]]
                        var_bounds = UNBOUNDED if is_linear else (lb - 1, ub + 1)
                        lb_var = compiled._add_var(model, vtype, *var_bounds)                        
                        ub_var = compiled._add_var(model, vtype, *var_bounds)
                        model.addConstr(ub_var >= lb_var)
                        param_vars[lb_name] = (lb_var, vtype, *var_bounds, True)
                        param_vars[ub_name] = (ub_var, vtype, *var_bounds, True)
                    else:
                        lb_val = values[lb_name]       
                        ub_val = values[ub_name]                          
                        param_vars[lb_name] = (lb_val, vtype, lb_val, lb_val, False)
                        param_vars[ub_name] = (ub_val, vtype, ub_val, ub_val, False)
                    
                # action values are generally linear, but two cases:
                # C: only a bias term
                # S/L: has bias and weight parameters
                states = states_in_values.get(action, [])
                is_linear = len(states) > 0
                var_bounds = UNBOUNDED if is_linear else self._bounds(rddl, action)
                vtype = compiled.GUROBI_TYPES[arange]
                for feature in ['bias'] + states:
                    name = f'value_weight__{action}__{k}__{feature}'
                    if values is None:
                        var = compiled._add_var(model, vtype, *var_bounds)
                        param_vars[name] = (var, vtype, *var_bounds, True)
                    else:
                        val = values[name]
                        param_vars[name] = (val, vtype, val, val, False)          
                
        return param_vars
    
    def init_params(self, compiled: GurobiRDDLCompiler,
                    model: gurobipy.Model) -> Dict[str, object]:
        rddl = compiled.rddl
        states_in_constr = self._get_states_for_constraints(rddl)
        states_in_values = self.dependencies_values
        
        param_values = {}
        for action in rddl.actions:
            
            # each case k
            for k in list(range(self.num_cases)) + ['else']: 
                
                # constraint
                if k != 'else':
                    
                    # constraint value initialized to zero
                    states = states_in_constr[action]
                    is_linear = len(states) > 1
                    if is_linear:
                        for feature in states:
                            name = f'constr_weight__{action}__{k}__{feature}'
                            param_values[name] = 0
                    
                    # constraint bounds - make non-overlapping initial bounds
                    if is_linear:
                        lb, ub = -100, +100
                    else:
                        lb, ub = self.state_bounds[states[0]]
                    delta = (ub - lb) / self.num_cases
                    lbk = lb + delta * k + compiled.epsilon
                    ubk = lb + delta * (k + 1) - compiled.epsilon
                    lb_name = f'lb__{action}__{k}'
                    ub_name = f'ub__{action}__{k}'
                    param_values[lb_name] = lbk
                    param_values[ub_name] = ubk
                    
                # action value initialized to default action
                states = states_in_values.get(action, [])
                for feature in ['bias'] + states:
                    name = f'value_weight__{action}__{k}__{feature}'
                    if feature == 'bias':
                        param_values[name] = compiled.init_values[action]
                    else:
                        param_values[name] = 0
                
        return param_values
    
    def actions(self, compiled: GurobiRDDLCompiler,
                model: gurobipy.Model,
                params: Dict[str, object],
                step: int,
                subs: Dict[str, object]) -> Dict[str, object]:
        rddl = compiled.rddl
        states_in_constr = self._get_states_for_constraints(rddl)
        states_in_values = self.dependencies_values
        
        action_vars = {}
        for (action, arange) in rddl.actionsranges.items():
            
            # action variable
            atype = compiled.GUROBI_TYPES[arange]
            action_bounds = self.action_bounds[action]
            action_var = compiled._add_var(model, atype, *action_bounds)
            action_vars[action] = (action_var, atype, *action_bounds, True)
            
            # each case k
            constr_sat_vars = []
            for k in list(range(self.num_cases)) + ['else']:
                
                # constraint
                if k != 'else':
                    
                    # constraint value
                    states = states_in_constr[action]
                    is_linear = len(states) > 1
                    if is_linear:
                        constr_value = 0
                        for feature in states:
                            name = f'constr_weight__{action}__{k}__{feature}'
                            constr_value += params[name][0] * subs[feature][0]
                        constr_value_var = compiled._add_real_var(model)
                        model.addConstr(constr_value_var == constr_value)
                    else:
                        constr_value_var = subs[states[0]][0]
                        
                    # constraint bounds
                    lb_name = f'lb__{action}__{k}'
                    ub_name = f'ub__{action}__{k}'   
                    lb_var = params[lb_name][0]
                    ub_var = params[ub_name][0]
                    lb_sat_var = compiled._add_bool_var(model)
                    ub_sat_var = compiled._add_bool_var(model)
                    lb_diff = constr_value_var - lb_var
                    ub_diff = constr_value_var - ub_var
                    model.addConstr((lb_sat_var == 1) >> (lb_diff >= 0))
                    model.addConstr((lb_sat_var == 0) >> (lb_diff <= -compiled.epsilon))
                    model.addConstr((ub_sat_var == 1) >> (ub_diff <= 0))
                    model.addConstr((ub_sat_var == 0) >> (ub_diff >= +compiled.epsilon))
                    constr_sat_var = compiled._add_bool_var(model)
                    model.addGenConstrAnd(constr_sat_var, [lb_sat_var, ub_sat_var])
                    constr_sat_vars.append(constr_sat_var)
                
                # action value
                states = states_in_values.get(action, [])
                name = f'value_weight__{action}__{k}__bias'
                action_case_value = params[name][0]
                for feature in states:
                    name = f'value_weight__{action}__{k}__{feature}'
                    action_case_value += params[name][0] * subs[feature][0]
                action_case_var = compiled._add_var(model, atype, *action_bounds)
                model.addConstr(action_case_var == action_case_value)
                
                # if the current constraint is satisfied assign action value
                if k != 'else':
                    model.addConstr((constr_sat_var == 1) >> 
                                    (action_var == action_case_var))
            
            # at most one constraint satisfied - implies disjoint case conditions
            num_sat_vars = sum(constr_sat_vars)
            model.addConstr(num_sat_vars <= 1)
            
            # if no constraint is satisfied assign default action value
            any_sat_var = compiled._add_bool_var(model)
            model.addGenConstrOr(any_sat_var, constr_sat_vars)
            model.addConstr((any_sat_var == 0) >> (action_var == action_case_var))
                    
        return action_vars
    
    def evaluate(self, compiled: GurobiRDDLCompiler,
                 params: Dict[str, object],
                 step: int,
                 subs: Dict[str, object]) -> Dict[str, object]:
        rddl = compiled.rddl
        states_in_constr = self._get_states_for_constraints(rddl)
        states_in_values = self.dependencies_values
        
        action_values = {}
        for (action, arange) in rddl.actionsranges.items():
            
            # case k
            for k in list(range(self.num_cases)) + ['else']:
                
                # constraint   
                if k == 'else':
                    case_i_holds = True
                else:
                    # constraint value
                    states = states_in_constr[action]
                    is_linear = len(states) > 1
                    if is_linear:
                        constr_value = 0
                        for feature in states:
                            name = f'constr_weight__{action}__{k}__{feature}'
                            constr_value += params[name][0].X * subs[feature]
                    else:
                        constr_value = subs[states[0]]
                        
                    # constraint bounds
                    lb_name = f'lb__{action}__{k}'
                    ub_name = f'ub__{action}__{k}'
                    lb_val = params[lb_name][0].X
                    ub_val = params[ub_name][0].X
                    case_i_holds = (lb_val <= constr_value <= ub_val)
                
                # action value
                if case_i_holds:
                    states = states_in_values.get(action, [])
                    name = f'value_weight__{action}__{k}__bias'
                    action_value = params[name][0].X
                    for feature in states:
                        name = f'value_weight__{action}__{k}__{feature}'
                        action_value += params[name][0].X * subs[feature]
                    
                    # clip to valid range
                    lb, ub = self._bounds(rddl, action)
                    action_value = max(lb, min(ub, action_value))
                    
                    # cast action to appropriate type   
                    if arange == 'int':
                        action_value = int(action_value)
                    elif arange == 'bool':
                        action_value = bool(action_value > 0.5)
                    action_values[action] = action_value           
                    break
            
        return action_values

    def to_string(self, compiled: GurobiRDDLCompiler,
                  params: Dict[str, object]) -> str:
        rddl = compiled.rddl
        states_in_constr = self._get_states_for_constraints(rddl)
        states_in_values = self.dependencies_values
        
        values = []
        for action in rddl.actions:
            
            # case k
            action_case_values = []
            for k in list(range(self.num_cases)) + ['else']:
                
                # constraint
                if k != 'else':
                    
                    # constraint value
                    states = states_in_constr[action]
                    if len(states) > 1:
                        constr_values = []
                        for feature in states:
                            name = f'constr_weight__{action}__{k}__{feature}'
                            constr_values.append(f'{params[name][0].X} * {feature}')
                        constr_value = ' + '.join(constr_values)
                    else:
                        constr_value = f'{states[0]}'
                    
                    # constraint bounds
                    lb_name = f'lb__{action}__{k}'
                    ub_name = f'ub__{action}__{k}'
                    lb_val = params[lb_name][0].X
                    ub_val = params[ub_name][0].X
                    case_i_holds = f'{lb_val} <= {constr_value} <= {ub_val}'
                else:
                    case_i_holds = 'else'
                    
                # action case value
                states = states_in_values.get(action, [])
                name = f'value_weight__{action}__{k}__bias'
                action_case_terms = [f'{params[name][0].X}']
                for feature in states:
                    name = f'value_weight__{action}__{k}__{feature}'
                    action_case_terms.append(f'{params[name][0].X} * {feature}')
                action_case_value = ' + '.join(action_case_terms)
                
                # update expression for action value
                if k == 'else':
                    action_case_values.append(action_case_value)
                else:
                    action_case_values.append(f'{action_case_value} if {case_i_holds}') 
            values.append(f'{action} = ' + ' else '.join(action_case_values))
        
        return '\n'.join(values)


class GurobiQuadraticPolicy(GurobiRDDLPlan):
    '''A quadratic policy in Gurobi.'''
    
    def __init__(self, *args,
                 action_clip_value: float=100.,
                 **kwargs) -> None:
        super(GurobiQuadraticPolicy, self).__init__(*args, **kwargs)
        self.action_clip_value = action_clip_value
        
    def params(self, compiled: GurobiRDDLCompiler,
               model: gurobipy.Model,
               values: Dict[str, object]=None) -> Dict[str, object]:
        rddl = compiled.rddl
        states = list(rddl.states.keys())
        clip_range = (-self.action_clip_value, +self.action_clip_value)
        
        param_vars = {}
        for action in rddl.actions:
            
            # linear terms
            for state in ['bias'] + states:
                name = f'weight__{action}__{state}'
                if values is None:
                    var = compiled._add_real_var(model, *clip_range)
                    param_vars[name] = (var, GRB.CONTINUOUS, *clip_range, True)
                else:
                    val = values[name]
                    param_vars[name] = (val, GRB.CONTINUOUS, val, val, False)
            
            # quadratic terms
            for (i, s1) in enumerate(states):
                for s2 in states[i:]:
                    name = f'weight__{action}__{s1}__{s2}'
                    if values is None:
                        var = compiled._add_real_var(model, *clip_range)
                        param_vars[name] = (var, GRB.CONTINUOUS, *clip_range, True)
                    else:
                        val = values[name]
                        param_vars[name] = (val, GRB.CONTINUOUS, val, val, False)
                        
        return param_vars
        
    def init_params(self, compiled: GurobiRDDLCompiler,
                    model: gurobipy.Model) -> Dict[str, object]:
        rddl = compiled.rddl
        states = list(rddl.states.keys())
        
        param_values = {}
        for action in rddl.actions:
            
            # bias initialized to no-op action value
            name = f'weight__{action}__bias'
            param_values[name] = compiled.init_values[action]
            
            # linear and quadratic terms are zero
            for state in states:
                name = f'weight__{action}__{state}'
                param_values[name] = 0
            for (i, s1) in enumerate(states):
                for s2 in states[i:]:
                    name = f'weight__{action}__{s1}__{s2}'
                    param_values[name] = 0
                    
        return param_values
    
    def actions(self, compiled: GurobiRDDLCompiler,
                model: gurobipy.Model,
                params: Dict[str, object],
                step: int,
                subs: Dict[str, object]) -> Dict[str, object]:
        rddl = compiled.rddl
        states = list(rddl.states.keys())
        
        action_vars = {}        
        for action in rddl.actions:
            
            # linear terms
            name = f'weight__{action}__bias'
            action_value = params[name][0]
            for state in states:
                name = f'weight__{action}__{state}'
                action_value += params[name][0] * subs[state][0]
            
            # quadratic terms
            for (i, s1) in enumerate(states):
                for s2 in states[i:]:
                    name = f'weight__{action}__{s1}__{s2}'
                    var = compiled._add_real_var(model)
                    model.addConstr(var == subs[s1][0] * subs[s2][0])
                    action_value += params[name][0] * var
            
            # action variable
            bounds = self._bounds(rddl, action)
            action_var = compiled._add_real_var(model, *bounds)
            action_vars[action] = (action_var, GRB.CONTINUOUS, *bounds, True)
            model.addConstr(action_var == action_value)
            
        return action_vars
    
    def evaluate(self, compiled: GurobiRDDLCompiler,
                 params: Dict[str, object],
                 step: int,
                 subs: Dict[str, object]) -> Dict[str, object]:
        rddl = compiled.rddl
        states = list(rddl.states.keys())
        
        action_values = {}
        for action in rddl.actions:
            
            # linear terms
            name = f'weight__{action}__bias'
            action_value = params[name][0].X
            for state in states:
                name = f'weight__{action}__{state}'
                action_value += params[name][0].X * subs[state]
            
            # quadratic terms
            for (i, s1) in enumerate(states):
                for s2 in states[i:]:
                    name = f'weight__{action}__{s1}__{s2}'
                    action_value += params[name][0].X * subs[s1] * subs[s2]
            
            # bound to valid range
            lb, ub = self._bounds(rddl, action)
            action_values[action] = max(min(action_value, ub), lb)
            
        return action_values
        
    def to_string(self, compiled: GurobiRDDLCompiler,
                  params: Dict[str, object]) -> str:
        rddl = compiled.rddl
        states = list(rddl.states.keys())
        
        res = ''
        for action in rddl.actions:
            
            # linear terms
            terms = []
            for state in ['bias'] + states:
                name = f'weight__{action}__{state}'
                if state == 'bias':
                    terms.append(f'{params[name][0].X}')
                else:
                    terms.append(f'{params[name][0].X} * {state}')
            
            # quadratic terms
            for (i, s1) in enumerate(states):
                for s2 in states[i:]:
                    name = f'weight__{action}__{s1}__{s2}'
                    val = params[name][0].X
                    terms.append(f'{val} * {s1} * {s2}')
            
            res += f'{action} = ' + ' + '.join(terms) + '\n'
            
        return res


# ***********************************************************************
# ALL VERSIONS OF GUROBI POLICIES
# 
# - just simple determinized planner
#
# ***********************************************************************
class GurobiOfflineController(BaseAgent):
    '''A container class for a Gurobi policy trained offline.'''
    
    def __init__(self, rddl: RDDLLiftedModel,
                 plan: GurobiRDDLPlan,
                 env: gurobipy.Env=None,
                 **compiler_kwargs):
        '''Creates a new Gurobi control policy that is optimized offline in an 
        open-loop fashion.
        
        :param rddl: the RDDL model
        :param plan: the plan or policy to optimize
        :param env: an existing gurobi environment
        :param allow_synchronous_state: whether state-fluent can be synchronous
        :param rollout_horizon: length of the planning horizon (uses the RDDL
        defined horizon if None)
        :param epsilon: small positive constant used for comparing equality of
        real numbers in Gurobi constraints
        :param float_range: range of floating values that can be passed to 
        Gurobi to initialize fluents and non-fluents (values outside this range
        are clipped)
        :param model_params: dictionary of parameter name and values to
        pass to Gurobi model after compilation
        :param piecewise_options: a string of parameters to pass to Gurobi
        "options" parameter when creating constraints that contain piecewise
        linear approximations (e.g. cos, log, exp)
        :param logger: to log information about compilation to file
        '''
        self.rddl = rddl
        self.plan = plan
        self.compiler = GurobiRDDLCompiler(rddl=rddl, plan=plan, **compiler_kwargs)
        
        # optimize the plan or policy here
        self.reset()
        if env is None:
            env = gurobipy.Env()
        self.env = env
        model, _, params = self.compiler.compile(env=self.env)
        model.optimize()
        self.model = model
        self.params = params
            
        # check for existence of valid solution
        self.solved = model.SolCount > 0
        if not self.solved:
            warnings.warn(f'Gurobi failed to find a feasible solution '
                          f'in the given time limit: using no-op action.',
                          stacklevel=2)
    
    def sample_action(self, state):
        
        # inputs to the optimizer include all current fluent values
        subs = self.compiler._compile_init_subs()
        subs.update(self.compiler._compile_init_subs(state))
        subs = {name: value[0] for (name, value) in subs.items()}
        
        # check for existence of solution
        if not self.solved:
            return {}
        
        # evaluate policy at the current time step
        self.model.update()
        action_values = self.plan.evaluate(
            self.compiler, params=self.params, step=self.step, subs=subs)
        action_values = {name: value for (name, value) in action_values.items()
                         if value != self.compiler.noop_actions[name]}
        
        self.step += 1
        return action_values
                
    def reset(self):
        self.step = 0


class GurobiOnlineController(BaseAgent): 
    '''A container class for a Gurobi controller continuously updated using 
    state feedback.'''

    def __init__(self, rddl: RDDLLiftedModel,
                 plan: GurobiRDDLPlan,
                 env: gurobipy.Env=None,
                 **compiler_kwargs):
        '''Creates a new Gurobi control policy that is optimized online in a 
        closed-loop fashion.
        
        :param rddl: the RDDL model
        :param plan: the plan or policy to optimize
        :param env: an existing gurobi environment
        :param allow_synchronous_state: whether state-fluent can be synchronous
        :param rollout_horizon: length of the planning horizon (uses the RDDL
        defined horizon if None)
        :param epsilon: small positive constant used for comparing equality of
        real numbers in Gurobi constraints
        :param float_range: range of floating values that can be passed to 
        Gurobi to initialize fluents and non-fluents (values outside this range
        are clipped)
        :param model_params: dictionary of parameter name and values to
        pass to Gurobi model after compilation
        :param piecewise_options: a string of parameters to pass to Gurobi
        "options" parameter when creating constraints that contain piecewise
        linear approximations (e.g. cos, log, exp)
        :param logger: to log information about compilation to file
        '''
        self.rddl = rddl
        self.plan = plan
        self.compiler = GurobiRDDLCompiler(rddl=rddl, plan=plan, **compiler_kwargs)
        if env is None:
            env = gurobipy.Env()
        self.env = env
        self.reset()
    
    def sample_action(self, state):
        
        # inputs to the optimizer include all current fluent values
        subs = self.compiler._compile_init_subs()
        subs.update(self.compiler._compile_init_subs(state))
        subs = {name: value[0] for (name, value) in subs.items()}     
        
        # optimize the policy parameters at the current time step
        model, _, params = self.compiler.compile(subs, env=self.env)
        model.optimize()
        
        # check for existence of solution
        if not (model.SolCount > 0):
            warnings.warn(f'Gurobi failed to find a feasible solution '
                          f'in the given time limit: using no-op action.',
                          stacklevel=2)
            del model
            return {}
            
        # evaluate policy at the current time step with current inputs
        action_values = self.plan.evaluate(
            self.compiler, params=params, step=0, subs=subs)
        action_values = {name: value for (name, value) in action_values.items()
                         if value != self.compiler.noop_actions[name]}
        del model
        return action_values
        
    def reset(self):
        pass
