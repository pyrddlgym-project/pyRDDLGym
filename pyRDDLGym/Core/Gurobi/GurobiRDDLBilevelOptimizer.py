import math
import scipy
import time
from typing import Dict, Iterable

import gurobipy
from gurobipy.gurobipy import GRB

from pyRDDLGym.Core.Compiler.RDDLLiftedModel import RDDLLiftedModel
from pyRDDLGym.Core.Gurobi.GurobiRDDLCompiler import GurobiRDDLCompiler
from pyRDDLGym.Core.Gurobi.GurobiRDDLPlan import GurobiRDDLPlan
from pyRDDLGym.Core.Gurobi.GurobiRDDLPlan import GurobiStraightLinePlan


class GurobiRDDLChanceConstrainedCompiler(GurobiRDDLCompiler):
    
    def __init__(self, *args, chance: float=0.99, **kwargs):
        super(GurobiRDDLChanceConstrainedCompiler, self).__init__(*args, **kwargs)
        self.chance = chance
        
    def _compile_init_subs(self, init_values=None) -> Dict[str, object]:
        subs = super(GurobiRDDLChanceConstrainedCompiler, self)._compile_init_subs(
            init_values)
        subs['noise__count'] = {'uniform': 0, 'normal': 0}
        subs['noise__var'] = {'uniform': {}, 'normal': {}}
        return subs
        
    def _gurobi_uniform(self, expr, model, subs):
        count = subs['noise__count']['uniform'] + 1
        subs['noise__count']['uniform'] = count
        uniform_vars = subs['noise__var']['uniform']
                
        # use cached noise variable
        if count in uniform_vars:
            return uniform_vars[count]
        
        # chance constraint for uniform
        arg1, arg2 = expr.args
        low, _, lbl, _, _ = self._gurobi(arg1, model, subs)
        high, _, _, ubh, _ = self._gurobi(arg2, model, subs)
        midpoint = (low + high) / 2
        interval = self.chance * (high - low) / 2
        lb, ub = GurobiRDDLCompiler._fix_bounds(lbl, ubh)
        noise = self._add_real_var(model, lb, ub)
        model.addConstr(noise >= midpoint - interval)
        model.addConstr(noise <= midpoint + interval)
        res = (noise, GRB.CONTINUOUS, lb, ub, True)    
        uniform_vars[count] = res
        return res
    
    def _gurobi_normal(self, expr, model, subs):
        count = subs['noise__count']['normal'] + 1
        subs['noise__count']['normal'] = count
        normal_vars = subs['noise__var']['normal']
        
        # use cached noise variable
        if count in normal_vars:
            return normal_vars[count]
        
        # standard deviation of normal
        arg1, arg2 = expr.args
        mean, _, lbm, ubm, _ = self._gurobi(arg1, model, subs)
        var, _, lbv, ubv, symb2 = self._gurobi(arg2, model, subs)
        if symb2:
            lbv, ubv = max(lbv, 0), max(ubv, 0)
            arg = self._add_real_var(model, lbv, ubv)
            model.addGenConstrMax(arg, [var], constant=0)
            lbs, ubs = GurobiRDDLCompiler._fix_bounds(math.sqrt(lbv), math.sqrt(ubv))
            std = self._add_real_var(model, lbs, ubs)
            model.addGenConstrPow(arg, std, 0.5, options=self.pw_options)
        else:
            lbs = ubs = std = math.sqrt(var)
        
        # chance constraint for normal   
        cil, ciu = scipy.stats.norm.interval(self.chance)
        lb, ub = GurobiRDDLCompiler._fix_bounds(lbm + ubs * cil, ubm + ubs * ciu)
        noise = self._add_real_var(model, lb, ub)
        model.addConstr(noise >= mean + std * cil)
        model.addConstr(noise <= mean + std * ciu)        
        res = (noise, GRB.CONTINUOUS, lb, ub, True)
        normal_vars[count] = res
        return res
        
    
class GurobiRDDLBilevelOptimizer:
    
    def __init__(self, rddl: RDDLLiftedModel,
                 policy: GurobiRDDLPlan,
                 state_bounds: Dict[str, object],
                 use_cc: bool=True,
                 **compiler_kwargs) -> None:
        self.rddl = rddl
        self.policy = policy
        self.state_bounds = state_bounds   
        self.use_cc = use_cc
        self.kwargs = compiler_kwargs
          
        self.action_bounds = policy.action_bounds
        self._compiler_cl = GurobiRDDLChanceConstrainedCompiler if use_cc \
                            else GurobiRDDLCompiler

    def solve(self, max_iters: int, tol: float=1e-4) -> Iterable[Dict[str, object]]:
        
        # compile outer problem
        compiler, outer_model, params = self._compile_outer_problem()     
        self.compiler, self.params = compiler, params
        
        # initialize policy arbitrarily
        param_values = self.policy.init_params(compiler, outer_model)  
        
        # main optimization loop
        error = GRB.INFINITY
        for it in range(max_iters):
            print('\n=========================================================')
            print(f'iteration {it}:')
            print('=========================================================')
            
            # solve inner problem for worst-case state and plan
            print('\nSOLVING INNER PROBLEM:\n')
            start_time = time.time()
            worst_val_slp, worst_val_pol, \
            worst_action_slp, worst_action_pol, \
            worst_state, worst_noise, \
            worst_next_states_slp, worst_next_states_pol, \
                inner_stats = self._solve_inner_problem(param_values)
            elapsed_time_inner = time.time() - start_time
            
            # solve outer problem for policy
            print('\nADDING CONSTRAINT AND SOLVING OUTER PROBLEM:\n')
            start_time = time.time()
            outer_value_pol, param_values, outer_stats = self._resolve_outer_problem(
                worst_val_slp, worst_state, worst_noise, 
                compiler, outer_model, params)
            elapsed_time_outer = time.time() - start_time
            
            # check stopping condition
            new_error = worst_val_slp - worst_val_pol
            converged = abs(new_error - error) <= tol * abs(error)
            error = new_error            
            
            # callback
            yield {
                'iteration': it,
                'converged': converged,
                'elapsed_time': {'inner': elapsed_time_inner, 
                                 'outer': elapsed_time_outer},
                'model_stats': {'inner': inner_stats, 
                                'outer': outer_stats},                
                'inner_value': {'plan': worst_val_slp, 
                                'policy': worst_val_pol, 
                                'epsilon': new_error},
                'inner_sol': {'actions': {'plan': worst_action_slp,
                                          'policy': worst_action_pol},
                              'init_state': worst_state,
                              'noises': worst_noise,
                              'states': {'plan': worst_next_states_slp,
                                         'policy': worst_next_states_pol}},
                'outer_value': {'policy': outer_value_pol},
                'parameters': param_values,
                'policy': self.policy.to_string(compiler, params),
            }
            if converged:
                break

    def _compile_outer_problem(self):
    
        # model for policy optimization
        compiler = self._compiler_cl(self.rddl, plan=self.policy, **self.kwargs)
        model = compiler._create_model()
        params = self.policy.params(compiler, model)
    
        # optimization objective for the outer problem is min_{policy} error
        # constraints on error will be added iteratively
        error = compiler._add_real_var(model, lb=0, name='error')
        model.setObjective(error, GRB.MINIMIZE)
        model.update()
        return compiler, model, params
    
    def _model_stats(self, model):        
        model_stats = {
            'variables': {'total': model.NumVars, 
                          'integer': model.NumIntVars, 
                          'binary': model.NumBinVars,
                          'piecewise': model.NumPWLObjVars},
            'constraints': {'linear': model.NumConstrs,
                            'SOS': model.NumSOS,
                            'quadratic': model.NumQConstrs, 
                            'general': model.NumGenConstrs},
            'sense': model.ModelSense,
            'objective': {'value': model.ObjVal,
                          'bound': model.ObjBound},
            'gap': model.MIPGap,
            'runtime': model.Runtime,
            'status': model.Status,
            'iters': model.IterCount
        }
        return model_stats
        
    def _solve_inner_problem(self, param_values):
        
        # model for straight line plan
        slp = GurobiStraightLinePlan(self.action_bounds)
        compiler = self._compiler_cl(self.rddl, plan=slp, **self.kwargs)
        model = compiler._create_model()
        
        # add variables for the initial states s0
        rddl = compiler.rddl
        init_state_vars = {}
        for name in rddl.states:
            prange = rddl.variable_ranges[name]
            vtype = compiler.GUROBI_TYPES[prange]
            lb, ub = (0, 1) if prange == 'bool' else self.state_bounds[name]
            var = compiler._add_var(model, vtype, lb, ub, name=name)
            init_state_vars[name] = (var, vtype, lb, ub, True)        

        # roll out from s0 using a_1, ... a_T
        slp_subs = compiler._compile_init_subs()
        slp_subs.update(init_state_vars)
        slp_params = slp.params(compiler, model)
        value_slp, action_vars_slp, next_state_vars_slp = compiler._rollout(
            model, slp, slp_params, slp_subs)
        
        # roll out from s0 using a_t = policy(s_t), t = 1, 2, ... T
        # here the policy is frozen during optimization of the plan above
        # noise variables are copied over from the above rollout
        pol_subs = compiler._compile_init_subs()
        pol_subs.update(init_state_vars)
        if self.use_cc:
            pol_subs['noise__count'] = {dt: 0 for dt in slp_subs['noise__count']}
            pol_subs['noise__var'] = slp_subs['noise__var']
        pol_params = self.policy.params(compiler, model, values=param_values)
        value_pol, action_vars_pol, next_state_vars_pol = compiler._rollout(
            model, self.policy, pol_params, pol_subs)
        
        # optimization objective for the inner problem is
        # max_{a_1, ... a_T, s0} [V(a_1, ... a_T, s0) - V(policy, s0)]
        model.setObjective(value_slp - value_pol, GRB.MAXIMIZE)     
        model.optimize()    
        
        # read a_1... a_T, V(a_1, ... a_T, s0), V(pi, s0) from the solved model
        worst_action_slp = compiler._get_optimal_actions(action_vars_slp)
        worst_action_pol = compiler._get_optimal_actions(action_vars_pol)
        worst_value_slp = value_slp.getValue()
        worst_value_pol = value_pol.getValue()
        
        # read worst state s0 from the optimized model
        worst_state = {}
        for name in rddl.states:
            value = init_state_vars[name][0].X
            vtype = init_state_vars[name][1]
            worst_state[name] = (value, vtype, value, value, False) 
        
        # read worst noise variables from the optimized model
        worst_noise = {}
        if self.use_cc:
            for (key, noise_vars) in pol_subs['noise__var'].items():
                worst_noise[key] = {}
                for (count, (var, vtype, *_)) in noise_vars.items():
                    value = var.X
                    worst_noise[key][count] = (value, vtype, value, value, False)
        
        # read worst next-state variables from the optimized model
        worst_next_states_slp = []
        for next_states_step in next_state_vars_slp:
            next_states_val = {key: var.X 
                               for (key, (var, *_)) in next_states_step.items()}
            worst_next_states_slp.append(next_states_val)
        worst_next_states_pol = []
        for next_states_step in next_state_vars_pol:
            next_states_val = {key: var.X 
                               for (key, (var, *_)) in next_states_step.items()}
            worst_next_states_pol.append(next_states_val)
        
        # read stats and release the model resources
        model_stats = self._model_stats(model)
        model.dispose()
        
        return worst_value_slp, worst_value_pol, \
            worst_action_slp, worst_action_pol, \
            worst_state, worst_noise, \
            worst_next_states_slp, worst_next_states_pol, model_stats
    
    def _resolve_outer_problem(self, worst_value_slp: float,
                               worst_state: Dict[str, object],
                               worst_noise: Dict[str, Dict[int, object]],
                               compiler: GurobiRDDLCompiler,
                               model: gurobipy.Model,
                               policy_params: Dict[str, object]) -> Dict[str, object]:
        
        # roll out from worst-case s_0 using a_t = policy(s_t), t = 1, 2, ... T
        subs = compiler._compile_init_subs()
        subs.update(worst_state)
        for (state, srange) in self.rddl.statesranges.items():
            (value, vtype, lb, ub, symb) = subs[state]
            if srange == 'int':
                subs[state] = (int(value), vtype, lb, ub, symb)
            elif srange == 'bool':
                subs[state] = (value > 0.5, vtype, lb, ub, symb)
        if self.use_cc:
            subs['noise__var'] = worst_noise
        value_pol, *_ = compiler._rollout(model, self.policy, policy_params, subs)
                
        # add constraint on error to outer model
        error = model.getVarByName('error')
        model.addConstr(error >= worst_value_slp - value_pol)
        
        # optimize error and return new policy parameter values
        model.optimize()
        param_values = {name: value[0].X for (name, value) in policy_params.items()}
        value_pol = value_pol.getValue()
        model_stats = self._model_stats(model)
        
        return value_pol, param_values, model_stats
    
