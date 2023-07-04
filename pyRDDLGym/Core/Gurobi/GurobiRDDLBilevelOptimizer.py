from typing import Dict, Iterable

import gurobipy
from gurobipy.gurobipy import GRB

from pyRDDLGym.Core.Compiler.RDDLLiftedModel import RDDLLiftedModel
from pyRDDLGym.Core.Gurobi.GurobiRDDLCompiler import GurobiRDDLCompiler
from pyRDDLGym.Core.Gurobi.GurobiRDDLPlan import GurobiRDDLPlan
from pyRDDLGym.Core.Gurobi.GurobiRDDLPlan import GurobiRDDLStraightLinePlan

        
class GurobiRDDLBilevelOptimizer:
    
    def __init__(self, rddl: RDDLLiftedModel,
                 policy: GurobiRDDLPlan,
                 state_bounds: Dict[str, object],
                 **compiler_kwargs) -> None:
        self.rddl = rddl
        self.policy = policy
        self.kwargs = compiler_kwargs
        
        self.state_bounds = {var: state_bounds[rddl.parse(var)[0]]
                             for var in rddl.groundstates()}
    
    def solve(self, max_iters: int, tol: float=1e-4) -> Iterable[Dict[str, object]]:
        
        # compile outer problem
        compiler, outer_model, params = self._compile_outer_problem()     
        self.compiler, self.params = compiler, params
        
        # initialize policy arbitrarily
        param_values = self.policy.init_params(compiler, outer_model)  
        
        # main optimization loop
        error = GRB.INFINITY
        error_hist = []
        for it in range(max_iters):
            print('\n=========================================================')
            print(f'iteration {it}:')
            print('=========================================================')
            
            # solve inner problem for worst-case state and plan
            print('\nSOLVING INNER PROBLEM:\n')
            worst_value, worst_action, worst_state = self._solve_inner_problem(param_values)
            
            # solve outer problem for policy
            print('\nADDING CONSTRAINT AND SOLVING OUTER PROBLEM:\n')
            param_values = self._resolve_outer_problem(
                worst_value, worst_state, compiler, outer_model, params)
            
            # check stopping condition
            new_error = outer_model.getVarByName('error').X
            error_hist.append(new_error)
            converged = abs(new_error - error) <= tol * abs(error)
            error = new_error            
            
            # callback
            yield {
                'it': it,
                'converged': converged,
                'worst_value': worst_value,
                'worst_action': worst_action,
                'worst_state': worst_state,
                'error': error,
                'error_hist': error_hist,
                'params': param_values
            }
            if converged:
                break

    def _compile_outer_problem(self):
    
        # model for policy optimization
        compiler = GurobiRDDLCompiler(self.rddl, plan=self.policy, **self.kwargs)
        model = compiler._create_model()
        params = self.policy.parameterize(compiler, model)
    
        # optimization objective for the outer problem is min_{policy} error
        # constraints on error will be added iteratively
        error = compiler._add_real_var(model, name='error')
        model.setObjective(error, GRB.MINIMIZE)
        model.update()
        return compiler, model, params
    
    def _solve_inner_problem(self, param_values):
        
        # model for straight line plan
        slp = GurobiRDDLStraightLinePlan()
        compiler = GurobiRDDLCompiler(self.rddl, plan=slp, **self.kwargs)
        model = compiler._create_model()
        
        # add variables for the initial states s0
        rddl = compiler.rddl
        subs = compiler._compile_init_subs()
        for name in rddl.states:
            prange = rddl.variable_ranges[name]
            vtype = compiler.GUROBI_TYPES[prange]
            if prange == 'bool':
                lb, ub = 0, 1
            else:
                lb, ub = self.state_bounds[name]
            var = compiler._add_var(model, vtype, lb, ub, name=name)
            subs[name] = (var, vtype, lb, ub, True)        

        # roll out from s0 using a_1, ... a_T
        slp_params = slp.parameterize(compiler, model)
        value_slp, action_vars = compiler._rollout(model, slp, slp_params, subs.copy())
        value_slp_var = compiler._add_real_var(model, name='value')
        model.addConstr(value_slp_var == value_slp)
        
        # roll out from s0 using a_t = policy(s_t), t = 1, 2, ... T
        # here the policy is frozen during optimization of the plan above
        pol_params = self.policy.parameterize(compiler, model, values=param_values)
        value_pol, _ = compiler._rollout(model, self.policy, pol_params, subs.copy())
        
        # optimization objective for the inner problem is
        # max_{a_1, ... a_T, s0} [V(a_1, ... a_T, s0) - V(policy, s0)]
        model.setObjective(value_slp - value_pol, GRB.MAXIMIZE)     
        model.optimize()    
        
        # read a_1, ... a_T, s0 and V(a_1, ... a_T, s0) from the optimized model
        worst_value = model.getVarByName('value').X
        worst_action = compiler._get_optimal_actions(action_vars)
        worst_state = {}
        for name in rddl.states:
            value = model.getVarByName(name).X
            vtype = subs[name][1]
            worst_state[name] = (value, vtype, value, value, False) 
        
        # release the model resources
        model.dispose()
        return worst_value, worst_action, worst_state
    
    def _resolve_outer_problem(self, worst_value: float,
                               worst_state: Dict[str, object],
                               compiler: GurobiRDDLCompiler,
                               model: gurobipy.Model,
                               policy_params: Dict[str, object]) -> Dict[str, object]:
        
        # roll out from worst-case s_0 using a_t = policy(s_t), t = 1, 2, ... T
        subs = compiler._compile_init_subs()
        subs.update(worst_state)
        value_pol, _ = compiler._rollout(model, self.policy, policy_params, subs)
                
        # add constraint on error to outer model
        error = model.getVarByName('error')
        model.addConstr(error >= worst_value - value_pol)
        
        # optimize error and return new policy parameter values
        model.optimize()
        param_values = {name: value[0].X for (name, value) in policy_params.items()}
        return param_values
    
