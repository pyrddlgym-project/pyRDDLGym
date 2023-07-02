from typing import Dict

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
    
    def solve(self, max_iters: int, tol: float=1e-4) -> None:
        compiler, outer_model, params = self._compile_outer_problem()  
        param_values = self.policy.init_params(compiler, outer_model)     
        self.compiler, self.outer_model, self.params = compiler, outer_model, params
        error = GRB.INFINITY
        error_hist = []
        
        for it in range(max_iters):
            print('\n=========================================================')
            print(f'iteration {it}:')
            print('=========================================================')
            
            print('\nSOLVING INNER PROBLEM:\n')
            inner_model = self._solve_inner_problem(param_values=param_values)
            
            print('\nADDING CONSTRAINT AND SOLVING OUTER PROBLEM:\n')
            param_values, _ = self._resolve_outer_problem(
                inner_model, outer_model, compiler, params)
            inner_model.dispose()
            
            # check stopping condition
            new_error = outer_model.getVarByName('error').X
            error_hist.append(new_error)
            if abs(new_error - error) < tol * abs(error):
                print(f'halting optimization with error {new_error}')
                break
            else:
                error = new_error
                
        return error_hist

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
        
        # add variables for the initial states
        grounded_rddl = compiler.rddl
        subs = compiler._compile_init_subs()
        for name in subs:
            if name in grounded_rddl.states:
                prange = grounded_rddl.variable_ranges[name]
                vtype = compiler.GUROBI_TYPES[prange]
                if prange == 'bool':
                    lb, ub = 0, 1
                else:
                    lb, ub = self.state_bounds[name]
                var = compiler._add_var(model, vtype, lb, ub, name=name)
                subs[name] = (var, vtype, lb, ub, True)        

        # roll out from s using a_1, ... a_T
        slp_params = slp.parameterize(compiler, model)
        value_slp, _ = compiler._rollout(model, slp, slp_params, subs.copy())
        value_slp_var = compiler._add_real_var(model, name='value')
        model.addConstr(value_slp_var == value_slp)
        
        # roll out from s using a_t = policy(s_t), t = 1, 2, ... T
        # here the policy is frozen during optimization of the plan above
        policy = self.policy
        policy_params = policy.parameterize(compiler, model, values=param_values)
        value_policy, _ = compiler._rollout(model, policy, policy_params, subs.copy())
        
        # optimization objective for the inner problem is
        # max_{a_1, ... a_T, s} [V(a_1, ... a_T, s) - V(policy, s)]
        objective = value_slp - value_policy
        model.setObjective(objective, GRB.MAXIMIZE)     
        model.optimize()
        return model
    
    def _resolve_outer_problem(self, inner_model: gurobipy.Model,
                               outer_model: gurobipy.Model,
                               compiler: GurobiRDDLCompiler,
                               policy_params: Dict[str, object]) -> None:
        grounded_rddl = compiler.rddl
        
        # get Value(a_1, ... a_T, s) from the solution to the inner problem
        value_slp = inner_model.getVarByName('value').X
        
        # get initial state s from the solution to the inner problem
        state_values = {}
        subs = compiler._compile_init_subs()
        for name in subs:
            if name in grounded_rddl.states:
                prange = grounded_rddl.variable_ranges[name]
                vtype = compiler.GUROBI_TYPES[prange]
                value = inner_model.getVarByName(name).X
                subs[name] = (value, vtype, value, value, False) 
                state_values[name] = value
        
        # roll out from above s using a_t = policy(s_t), t = 1, 2, ... T
        value_policy, _ = compiler._rollout(
            outer_model, self.policy, policy_params, subs)
                
        # add constraint to outer model
        error = outer_model.getVarByName('error')
        outer_model.addConstr(error >= value_slp - value_policy)
        
        # optimize then return new policy parameter values
        outer_model.optimize()
        param_values = {name: var.X for (name, (var, *_)) in policy_params.items()}
        return param_values, state_values
    
