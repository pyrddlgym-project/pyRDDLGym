import gurobipy
from gurobipy import GRB
from typing import Dict, List, Tuple, TYPE_CHECKING

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


class GurobiStraightLinePlan(GurobiRDDLPlan):
    
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
            elif prange == 'bool':
                action_value = bool(action_value > 0.5)
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


class GurobiPiecewisePolicy(GurobiRDDLPlan):
    
    def __init__(self, *args,
                 state_bounds: Dict[str, Tuple[float, float]]={},
                 upper_bound: bool=True,
                 dependencies: Dict[str, List[str]]=None,
                 linear_value: bool=False,
                 num_cases: int=1,
                 **kwargs) -> None:
        super(GurobiPiecewisePolicy, self).__init__(*args, **kwargs)
        
        self.state_bounds = state_bounds
        self.upper_bound = upper_bound or num_cases > 1
        self.dependencies = dependencies
        self.linear_value = linear_value
        self.num_cases = num_cases
    
    def _get_states_for_constraints(self, rddl):
        states = {}
        if self.dependencies is not None:
            for action in rddl.actions:
                states[action] = list(self.dependencies[action])
        else:
            for action in rddl.actions:
                states[action] = list(rddl.states.keys())
        return states
    
    def params(self, compiled: 'GurobiRDDLCompiler',
               model: gurobipy.Model,
               values: Dict[str, object]=None) -> Dict[str, object]:
        rddl = compiled.rddl  
        states_in_constr = self._get_states_for_constraints(rddl)
        
        param_vars = {}
        for (action, arange) in rddl.actionsranges.items():
            atype = compiled.GUROBI_TYPES[arange]
            lb, ub = self._bounds(rddl, action)
            
            # each case i
            for icase in list(range(self.num_cases)) + ['else']:
                    
                # a constraint for an action is an intersection of constraints on
                # state of the form s_i >= lb_i ^ s_i <= ub_i
                if icase != 'else':
                    for state in states_in_constr[action]:
                        srange = rddl.statesranges[state]
                        stype = compiled.GUROBI_TYPES[srange]
                        lbs, ubs = self.state_bounds.get(state, UNBOUNDED)
                    
                        # initialize parameters of constraint s_i >= lb_i ^ s_i <= ub_i
                        # initialize action parameter a_i if constraint is true
                        lname = f'low__{icase}__{state}__{action}'
                        hname = f'high__{icase}__{state}__{action}'
                        if values is None:
                            lvar = compiled._add_var(model, stype, lbs, ubs)
                            param_vars[lname] = (lvar, stype, lbs, ubs, True)                        
                            if self.upper_bound:
                                hvar = compiled._add_var(model, stype, lbs, ubs)
                                model.addConstr(hvar >= lvar)
                                param_vars[hname] = (hvar, stype, lbs, ubs, True)
                        else:
                            lval = values[lname]                                       
                            param_vars[lname] = (lval, stype, lval, lval, False)
                            if self.upper_bound:
                                hval = values[hname]
                                param_vars[hname] = (hval, stype, hval, hval, False)
                
                # action parameters a_i for current case
                if self.linear_value:
                    for state in ['bias'] + states_in_constr[action]:
                        wname = f'weight__{state}__{icase}__{action}'
                        if values is None:
                            wvar = compiled._add_var(model, atype, lb, ub)
                            param_vars[wname] = (wvar, atype, lb, ub, True)
                        else:
                            wval = values[wname]
                            param_vars[wname] = (wval, atype, wval, wval, False)
                else:
                    aname = f'action__{icase}__{action}'
                    if values is None:
                        avar = compiled._add_var(model, atype, lb, ub)
                        param_vars[aname] = (avar, atype, lb, ub, True)
                    else:
                        aval = values[aname]     
                        param_vars[aname] = (aval, atype, aval, aval, False)              
                
        return param_vars
    
    def init_params(self, compiled: 'GurobiRDDLCompiler',
                    model: gurobipy.Model) -> Dict[str, object]:
        rddl = compiled.rddl
        states_in_constr = self._get_states_for_constraints(rddl)
        
        param_values = {}
        for action in rddl.actions:
            
            # each case i
            for icase in list(range(self.num_cases)) + ['else']: 
                
                # initialize bounds lb_i, ub_i in constraint to default state bounds
                if icase != 'else':
                    for state in states_in_constr[action]:
                        lbs, ubs = self.state_bounds.get(state, UNBOUNDED)            
                        lname = f'low__{icase}__{state}__{action}'
                        param_values[lname] = lbs
                        if self.upper_bound:
                            hname = f'high__{icase}__{state}__{action}'
                            param_values[hname] = ubs
            
                # initialize action parameter a_i for current case to no-op
                if self.linear_value:
                    aname = f'weight__bias__{icase}__{action}'
                    param_values[aname] = compiled.init_values[action]
                    for state in states_in_constr[action]:
                        aname = f'weight__{state}__{icase}__{action}'
                        param_values[aname] = 0
                else:
                    aname = f'action__{icase}__{action}'
                    param_values[aname] = compiled.init_values[action]
                
        return param_values
    
    def actions(self, compiled: 'GurobiRDDLCompiler',
                model: gurobipy.Model,
                params: Dict[str, object],
                step: int,
                subs: Dict[str, object]) -> Dict[str, object]:
        rddl = compiled.rddl
        states_in_constr = self._get_states_for_constraints(rddl)
        
        action_vars = {}
        for (action, arange) in rddl.actionsranges.items():
            
            # action variable a_i
            atype = compiled.GUROBI_TYPES[arange]
            lb, ub = self._bounds(rddl, action)
            res = compiled._add_var(model, atype, lb, ub)
            action_vars[action] = (res, atype, lb, ub, True)
            
            # each case i
            case_sat_vars = []
            for icase in list(range(self.num_cases)) + ['else']:
                
                # each constraint i is an intersection of constraints
                if icase != 'else':
                    
                    # check sub-constraints s_j >= lb_ij ^ s_j <= ub_ij
                    case_vars = []
                    for state in states_in_constr[action]:
                        l_name = f'low__{icase}__{state}__{action}'
                        h_name = f'high__{icase}__{state}__{action}'
                        
                        # assign s_j >= lb_ij ^ s_j <= ub_ij to a variable
                        ldiff = subs[state][0] - params[l_name][0]
                        lvar = compiled._add_bool_var(model)
                        model.addConstr((lvar == 1) >> (ldiff >= 0))
                        model.addConstr((lvar == 0) >> (ldiff <= 0))
                        case_vars.append(lvar)
                        if self.upper_bound:
                            hdiff = subs[state][0] - params[h_name][0]
                            hvar = compiled._add_bool_var(model)
                            model.addConstr((hvar == 1) >> (hdiff <= 0))
                            model.addConstr((hvar == 0) >> (hdiff >= 0))
                            case_vars.append(hvar)
                        
                    # check constraint of case i satisfied
                    case_var = compiled._add_bool_var(model)
                    model.addGenConstrAnd(case_var, case_vars)
                    case_sat_vars.append(case_var)
                
                # construct an action var constant or linear function of state
                if self.linear_value:
                    wname = f'weight__bias__{icase}__{action}'
                    aexpr = params[wname][0]
                    for state in states_in_constr[action]:
                        wname = f'weight__{state}__{icase}__{action}'
                        aexpr = aexpr + params[wname][0] * subs[state][0]
                    avar = compiled._add_var(model, atype, lb, ub)
                    model.addConstr(avar == aexpr)
                else:
                    aname = f'action__{icase}__{action}'
                    avar = params[aname][0]
                
                # assign action to a_i if constraint satisfied
                # if none of the state constraints hold, assign else value
                if icase == 'else':
                    any_sat_var = compiled._add_bool_var(model)
                    model.addGenConstrOr(any_sat_var, case_sat_vars)
                    model.addConstr((any_sat_var == 0) >> (res == avar))
                else:
                    model.addConstr((case_var == 1) >> (res == avar))
                    
        return action_vars
    
    def evaluate(self, compiled: 'GurobiRDDLCompiler',
                 params: Dict[str, object],
                 step: int,
                 subs: Dict[str, object]) -> Dict[str, object]:
        rddl = compiled.rddl
        states_in_constr = self._get_states_for_constraints(rddl)
        
        action_values = {}
        for (action, arange) in rddl.actionsranges.items():
            
            # for each case
            for icase in list(range(self.num_cases)) + ['else']:
                
                # check if case i constraint is satisfied
                case_i_holds = True                
                if icase != 'else':
                    for state in states_in_constr[action]:
                        lname = f'low__{icase}__{state}__{action}'
                        hname = f'high__{icase}__{state}__{action}'
                        lval = params[lname][0].X
                        hval = params[hname][0].X if self.upper_bound else float('inf')   
                        if not (lval <= subs[state] <= hval):
                            case_i_holds = False
                            break
                
                # evaluate the action as either constant or linear in the state
                if case_i_holds:
                    if self.linear_value:
                        wname = f'weight__bias__{icase}__{action}'  
                        aval = params[wname][0].X
                        for state in states_in_constr[action]:
                            wname = f'weight__{state}__{icase}__{action}'   
                            aval += params[wname][0].X * subs[state]
                    else:
                        aname = f'action__{icase}__{action}'
                        aval = params[aname][0].X    
                        
                    # cast action to appropriate type   
                    if arange == 'int':
                        aval = int(aval)
                    elif arange == 'bool':
                        aval = bool(aval > 0.5)
                    action_values[action] = aval               
                    break
            
        return action_values

    def to_string(self, compiled: 'GurobiRDDLCompiler',
                  params: Dict[str, object]) -> str:
        rddl = compiled.rddl
        states_in_constr = self._get_states_for_constraints(rddl)
        
        res = ''
        for action in rddl.actions:
            case_strs = []
            
            # for each case i
            for icase in list(range(self.num_cases)) + ['else']:
                
                # print the action value
                if self.linear_value:
                    a_vals = []
                    w_name = f'weight__bias__{icase}__{action}'               
                    w_val = params[w_name][0].X
                    a_vals.append(str(w_val))
                    for state in states_in_constr[action]:
                        w_name = f'weight__{state}__{icase}__{action}'   
                        w_val = params[w_name][0].X
                        a_vals.append(f'{w_val} * {state}')
                    a_val = ' + '.join(a_vals)
                else:
                    a_name = f'action__{icase}__{action}'
                    a_val = params[a_name][0].X
                    
                # print the intersection constraint for current case i
                if icase == 'else':
                    case_str = f'{a_val} otherwise'
                else:
                    case_constrs = []                    
                    for state in states_in_constr[action]:
                        l_name = f'low__{icase}__{state}__{action}'
                        l_val = params[l_name][0].X  
                        if self.upper_bound:
                            h_name = f'high__{icase}__{state}__{action}'
                            h_val = params[h_name][0].X
                            case_val = f'{state} >= {l_val} ^ {state} <= {h_val}'
                        else:
                            case_val = f'{state} >= {l_val}'
                        case_constrs.append(case_val)
                    case_str = f'{a_val} if ' + ' ^ '.join(case_constrs)
                case_strs.append(case_str)
                    
            res += f'{action} = ' + ', '.join(case_strs) + '\n'
            
        return res
