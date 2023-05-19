import gurobipy
from gurobipy import GRB
import math

from pyRDDLGym.Core.ErrorHandling.RDDLException import print_stack_trace
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLNotImplementedError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLUndefinedVariableError

from pyRDDLGym.Core.Compiler.RDDLLevelAnalysis import RDDLLevelAnalysis
from pyRDDLGym.Core.Compiler.RDDLLiftedModel import RDDLLiftedModel
from pyRDDLGym.Core.Compiler.RDDLObjectsTracer import RDDLObjectsTracer
from pyRDDLGym.Core.Compiler.RDDLValueInitializer import RDDLValueInitializer
from pyRDDLGym.Core.Debug.Logger import Logger
from pyRDDLGym.Core.Grounder.RDDLGrounder import RDDLGrounder


class GurobiRDDLCompiler:
    
    def __init__(self, rddl: RDDLLiftedModel,
                 allow_synchronous_state: bool=True,
                 rollout_horizon: int=None,
                 logger: Logger=None):
        if rollout_horizon is None:
            rollout_horizon = rddl.horizon
        self.horizon = rollout_horizon
        self.allow_synchronous_state = allow_synchronous_state
        self.logger = logger
        
        # type conversion to Gurobi
        self.GUROBI_TYPES = {
            'int': GRB.INTEGER,
            'real': GRB.CONTINUOUS,
            'bool': GRB.BINARY
        }
        
        # ground out the domain
        grounder = RDDLGrounder(rddl._AST)
        self.rddl = grounder.Ground()
        
        # compile initial values
        if self.logger is not None:
            self.logger.clear()
        initializer = RDDLValueInitializer(self.rddl, logger=self.logger)
        self.init_values = initializer.initialize()
        
        # compute dependency graph for CPFs and sort them by evaluation order
        sorter = RDDLLevelAnalysis(
            self.rddl, allow_synchronous_state, logger=self.logger)
        self.levels = sorter.compute_levels()     
        
        # trace expressions to cache information to be used later
        tracer = RDDLObjectsTracer(self.rddl, logger=self.logger)
        self.traced = tracer.trace()
    
    # ===========================================================================
    # main compilation subroutines
    # ===========================================================================
     
    def compile(self):
        model = gurobipy.Model()
        
        # initial variable substitution table
        subs = {}
        for (var, value) in self.init_values.items():
            prange = self.rddl.variable_ranges[var]
            vtype = self.GUROBI_TYPES[prange]
            lb, ub = GurobiRDDLCompiler._fix_bounds(value, value)
            subs[var] = (value, vtype, lb, ub, False)
        
        # compile forward model and objective
        objective = 0
        for step in range(self.horizon):
            reward = self._compile_step(step, model, subs)
            objective += reward
        model.setObjective(objective, GRB.MAXIMIZE)
        
        # set additional model settings here before optimization
        model.params.NonConvex = 2
        return model
    
    def _add_var(self, model, vtype, lb, ub, name=""):
        return model.addVar(vtype=vtype, lb=lb, ub=ub, name=name)
    
    def _add_bool_var(self, model, name=""):
        return self._add_var(model, GRB.BINARY, 0, 1, name=name)
    
    def _add_real_var(self, model, lb, ub, name=""):
        return self._add_var(model, GRB.CONTINUOUS, lb, ub, name=name)
    
    def _add_int_var(self, model, lb, ub, name=""):
        return self._add_var(model, GRB.INTEGER, lb, ub, name=name)
    
    def _compile_step(self, step, model, subs):
        rddl = self.rddl
        
        # add action fluent variables to model
        for (action, prange) in rddl.actionsranges.items():
            name = f'{action}___{step}'
            if prange == 'bool':
                lb, ub = 0, 1
            else:
                lb, ub = -GRB.INFINITY, +GRB.INFINITY
            vtype = self.GUROBI_TYPES[prange]
            var = self._add_var(model, vtype, lb, ub, name=name)
            subs[action] = (var, vtype, lb, ub, True)
        
        # evaluate CPFs
        for cpfs in self.levels.values():
            for cpf in cpfs:
                _, expr = rddl.cpfs[cpf]
                subs[cpf] = self._gurobi(expr, model, subs)
        
        # evaluate reward
        reward, *_ = self._gurobi(rddl.reward, model, subs)
        
        # update state
        for (state, next_state) in rddl.next_state.items():
            subs[state] = subs[next_state]
            
        return reward
    
    # ===========================================================================
    # start of compilation subroutines
    # ===========================================================================
    
    def _gurobi(self, expr, model, subs):
        etype, _ = expr.etype
        if etype == 'constant':
            return self._gurobi_constant(expr, model, subs)
        elif etype == 'pvar':
            return self._gurobi_pvar(expr, model, subs)
        elif etype == 'arithmetic':
            return self._gurobi_arithmetic(expr, model, subs)
        elif etype == 'relational':
            return self._gurobi_relational(expr, model, subs)
        elif etype == 'boolean':
            return self._gurobi_logical(expr, model, subs)
        elif etype == 'func':
            return self._gurobi_function(expr, model, subs)
        elif etype == 'control':
            return self._gurobi_control(expr, model, subs)
        else:
            raise RDDLNotImplementedError(
                f'Expression type {etype} is not supported in Gurobi compiler.\n' + 
                print_stack_trace(expr))
            
    # ===========================================================================
    # leaves
    # ===========================================================================
    
    def _gurobi_constant(self, expr, *_):
        
        # get the cached value of this constant
        value = self.traced.cached_sim_info(expr)
        
        # infer type of value and assign to Gurobi type
        if isinstance(value, bool):
            vtype = self.GUROBI_TYPES['bool']
        elif isinstance(value, int):
            vtype = self.GUROBI_TYPES['int']
        elif isinstance(value, float):
            vtype = self.GUROBI_TYPES['real']
        else:
            raise RDDLNotImplementedError(
                f'Range of {value} is not supported in Gurobi compiler.')
        
        # bounds form a singleton set containing the cached value
        lb, ub = GurobiRDDLCompiler._fix_bounds(value, value)
        return value, vtype, lb, ub, False

    def _gurobi_pvar(self, expr, model, subs):
        var, _ = expr.args
        
        # domain object converted to canonical index
        is_value, value = self.traced.cached_sim_info(expr)
        if is_value:
            return value, GRB.INTEGER, value, value, False
        
        # extract variable value
        value = subs.get(var, None)
        if value is None:
            raise RDDLUndefinedVariableError(
                f'Variable <{var}> is referenced before assignment.\n' + 
                print_stack_trace(expr))
        return value
    
    # ===========================================================================
    # arithmetic
    # ===========================================================================
    
    @staticmethod
    def _promote_vtype(vtype1, vtype2):
        if vtype1 == GRB.BINARY:
            return vtype2
        elif vtype2 == GRB.BINARY:
            return vtype1
        elif vtype1 == GRB.INTEGER:
            return vtype2
        elif vtype2 == GRB.INTEGER:
            return vtype1
        else:
            assert (vtype1 == vtype2 == GRB.CONTINUOUS)
            return vtype1
    
    @staticmethod
    def _at_least_int(vtype):
        return GurobiRDDLCompiler._promote_vtype(vtype, GRB.INTEGER)
    
    @staticmethod
    def _fix_bounds(lb, ub):
        assert (ub >= lb)
        lb = max(min(lb, GRB.INFINITY), -GRB.INFINITY)
        ub = max(min(ub, GRB.INFINITY), -GRB.INFINITY)
        return lb, ub
        
    def _gurobi_arithmetic(self, expr, model, subs):
        _, op = expr.etype
        args = expr.args        
        n = len(args)
        
        # unary negation
        if n == 1 and op == '-':
            arg, = args
            gterm, vtype, lb, ub, symb = self._gurobi(arg, model, subs)
            vtype = GurobiRDDLCompiler._at_least_int(vtype)
            lb, ub = GurobiRDDLCompiler._fix_bounds(-ub, -lb)
            
            # assign negative to a new variable
            if symb:    
                res = self._add_var(model, vtype, lb, ub)
                model.addConstr(res == -gterm)
            else:
                res = -gterm
                lb, ub = res, res
                
            return res, vtype, lb, ub, symb
        
        # binary operations
        elif n >= 2:
            
            # unwrap addition to binary operations
            if op == '+':
                arg, *rest = args
                res, vtype, lb, ub, symb = self._gurobi(arg, model, subs)
                vtype = GurobiRDDLCompiler._at_least_int(vtype)
                for arg in rest:
                    gterm, vtype1, lb2, ub2, symb2 = self._gurobi(arg, model, subs)
                    res = res + gterm
                    vtype = GurobiRDDLCompiler._promote_vtype(vtype, vtype1)
                    
                    # update bounds for sum
                    lb, ub = GurobiRDDLCompiler._fix_bounds(lb + lb2, ub + ub2)
                    symb = symb or symb2
                
                # assign sum to a new variable
                if symb:
                    newres = self._add_var(model, vtype, lb, ub)
                    model.addConstr(newres == res)
                else:
                    newres = res         
                    lb, ub = newres, newres
                    
                return newres, vtype, lb, ub, symb
            
            # unwrap multiplication to binary operations
            elif op == '*':
                arg, *rest = args
                res, vtype, lb, ub, symb = self._gurobi(arg, model, subs)
                vtype = GurobiRDDLCompiler._at_least_int(vtype)
                for arg in rest:
                    gterm, vtype1, lb2, ub2, symb2 = self._gurobi(arg, model, subs)
                    res = res * gterm
                    vtype = GurobiRDDLCompiler._promote_vtype(vtype, vtype1)
                    
                    # update bounds for product
                    cross_lb_ub = (lb * lb2, lb * ub2, ub * lb2, ub * ub2)
                    lb, ub = GurobiRDDLCompiler._fix_bounds(
                        min(cross_lb_ub), max(cross_lb_ub))
                    symb = symb or symb2
                    
                # assign product to a new variable
                if symb:    
                    newres = self._add_var(model, vtype, lb, ub)
                    model.addConstr(newres == res)
                else:
                    newres = res
                    lb, ub = newres, newres
                    
                return newres, vtype, lb, ub, symb
            
            # subtraction
            elif op == '-' and n == 2:
                arg1, arg2 = args
                gterm1, vtype1, lb1, ub1, symb1 = self._gurobi(arg1, model, subs)
                gterm2, vtype2, lb2, ub2, symb2 = self._gurobi(arg2, model, subs)
                vtype = GurobiRDDLCompiler._promote_vtype(vtype1, vtype2)
                vtype = GurobiRDDLCompiler._at_least_int(vtype)
                symb = symb1 or symb2
                
                if symb:
                    # compute bounds on the difference
                    lb, ub = GurobiRDDLCompiler._fix_bounds(lb1 - ub2, ub1 - lb2)
                    
                    # assign difference to a new variable
                    res = self._add_var(model, vtype, lb, ub)
                    model.addConstr(res == gterm1 - gterm2)
                else:
                    res = gterm1 - gterm2
                    lb, ub = res, res
                    
                return res, vtype, lb, ub, symb
            
            # quotient x / y
            elif op == '/' and n == 2:
                arg1, arg2 = args
                gterm1, _, lb1, ub1, symb1 = self._gurobi(arg1, model, subs)
                gterm2, _, lb2, ub2, symb2 = self._gurobi(arg2, model, subs)
                
                if symb2:                    
                    # compute interval containing 1 / y
                    if 0 > lb2 and 0 < ub2:
                        lb2, ub2 = -GRB.INFINITY, GRB.INFINITY
                    elif lb2 == 0 and ub2 == 0:
                        lb2, ub2 = GRB.INFINITY, GRB.INFINITY
                    elif lb2 == 0:
                        lb2, ub2 = 1 / ub2, GRB.INFINITY
                    elif ub2 == 0:
                        lb2, ub2 = -GRB.INFINITY, 1 / lb2
                    else:
                        lb2, ub2 = 1 / ub2, 1 / lb2
                    lb2, ub2 = GurobiRDDLCompiler._fix_bounds(lb2, ub2)
                    
                    # implement z = 1 / y as a constraint z * y = 1
                    recip2 = self._add_real_var(model, lb2, ub2)
                    model.addConstr(recip2 * gterm2 == 1)
                else:
                    recip2 = 1 / gterm2
                    lb2, ub2 = recip2, recip2
                
                symb = symb1 or symb2
                if symb:
                    # compute interval containing x / y
                    cross_lb_ub = (lb1 * lb2, lb1 * ub2, ub1 * lb2, ub1 * ub2)
                    lb, ub = GurobiRDDLCompiler._fix_bounds(
                        min(cross_lb_ub), max(cross_lb_ub))
                    
                    # finally compute x / y = x * (1 / y)
                    res = self._add_real_var(model, lb, ub)
                    model.addConstr(res == gterm1 * recip2)
                else:
                    res = gterm1 * recip2
                    lb, ub = res, res
                    
                return res, GRB.CONTINUOUS, lb, ub, symb
        
        raise RDDLNotImplementedError(
            f'Arithmetic operator {op} with {n} arguments is not '
            f'supported in Gurobi compiler.\n' + 
            print_stack_trace(expr))
    
    # ===========================================================================
    # boolean
    # ===========================================================================
    
    def _gurobi_relational(self, expr, model, subs):
        _, op = expr.etype
        args = expr.args        
        n = len(args)
        
        if n == 2:
            lhs, rhs = args
            glhs, *_, symb1 = self._gurobi(lhs, model, subs)
            grhs, *_, symb2 = self._gurobi(rhs, model, subs)
            
            # convert <= to >=, < to >, etc.
            if op == '<=' or op == '<':
                glhs, grhs = grhs, glhs
                op = '>=' if op == '<=' else '>'
            
            # assign comparison operator to binary variable
            diff = glhs - grhs
            symb = symb1 or symb2
            if op == '==':
                if symb:
                    res = self._add_bool_var(model)
                    model.addGenConstrIndicator(res, True, diff, GRB.EQUAL, 0)
                else:
                    res = glhs == grhs
                return res, GRB.BINARY, 0, 1, symb
            
            elif op == '>=':
                if symb:
                    res = self._add_bool_var(model)
                    model.addGenConstrIndicator(res, True, diff, GRB.GREATER_EQUAL, 0)
                else:
                    res = glhs >= grhs
                return res, GRB.BINARY, 0, 1, symb
            
            elif op == '~=':
                if symb:
                    res = self._add_bool_var(model)
                    model.addGenConstrIndicator(res, False, diff, GRB.EQUAL, 0)
                else:   
                    res = glhs != grhs
                return res, GRB.BINARY, 0, 1, symb
            
            elif op == '>':
                if symb:
                    res = self._add_bool_var(model)
                    model.addGenConstrIndicator(res, False, diff, GRB.LESS_EQUAL, 0)
                else:
                    res = glhs > grhs
                return res, GRB.BINARY, 0, 1, symb
            
        raise RDDLNotImplementedError(
            f'Relational operator {op} with {n} arguments is not '
            f'supported in Gurobi compiler.\n' + 
            print_stack_trace(expr))
    
    def _gurobi_logical(self, expr, model, subs):
        _, op = expr.etype
        if op == '&':
            op = '^'
        args = expr.args        
        n = len(args)
        
        # unary negation ~z of z is a variable y such that y + z = 1
        if n == 1 and op == '~':
            arg, = args
            gterm, *_, symb = self._gurobi(arg, model, subs)
            if symb:
                res = self._add_bool_var(model)
                model.addConstr(res + gterm == 1)
            else:
                res = not gterm
                
            return res, GRB.BINARY, 0, 1, symb
            
        # binary operations
        elif n >= 2:
            results = [self._gurobi(arg, model, subs) for arg in args]
            gterms = [result[0] for result in results]
            symbs = [result[-1] for result in results]
            symb = any(symbs)
            
            # any non-variables must be converted to variables
            if symb:
                for (i, gterm) in enumerate(gterms):
                    if not symbs[i]:
                        var = self._add_bool_var(model)
                        model.addConstr(var == gterm)
                        gterms[i] = var
                        symbs[i] = True
            
            # unwrap AND to binary operations
            if op == '^':
                if symb:
                    res = self._add_bool_var(model)
                    model.addGenConstrAnd(res, gterms)
                else:
                    res = all(gterms)
                    
                return res, GRB.BINARY, 0, 1, symb
            
            # unwrap OR to binary operations
            elif op == '|':
                if symb:
                    res = self._add_bool_var(model)
                    model.addGenConstrOr(res, gterms)
                else:
                    res = any(gterms)
                    
                return res, GRB.BINARY, 0, 1, symb
        
        raise RDDLNotImplementedError(
            f'Logical operator {op} with {n} arguments is not '
            f'supported in Gurobi compiler.\n' + 
            print_stack_trace(expr))
    
    # ===========================================================================
    # function
    # ===========================================================================

    @staticmethod
    def GRB_log(x):
        if x <= 0:
            return -GRB.INFINITY
        else:
            return math.log(x)
    
    def _gurobi_function(self, expr, model, subs):
        _, name = expr.etype
        args = expr.args
        n = len(args)
        
        # unary functions
        if n == 1:
            arg, = args
            gterm, vtype, lb, ub, symb = self._gurobi(arg, model, subs)
            vtype = GurobiRDDLCompiler._at_least_int(vtype)
            
            if name == 'abs':                
                if symb:
                    # assign abs to new variable
                    res = self._add_var(model, vtype, lb, ub)
                    model.addGenConstrAbs(res, gterm)
                    
                    # compute bounds for abs
                    if lb >= 0:
                        lb, ub = lb, ub
                    elif ub <= 0:
                        lb, ub = -ub, -lb
                    else:
                        lb, ub = 0, max(abs(lb), abs(ub))
                    lb, ub = GurobiRDDLCompiler._fix_bounds(lb, ub)
                else:
                    res = abs(gterm)
                    lb, ub = res, res
                    
                return res, vtype, lb, ub, symb
            
            elif name == 'cos':
                if symb:
                    lb, ub = -1.0, 1.0
                    res = self._add_real_var(model, -1.0, 1.0)
                    model.addGenConstrCos(gterm, res)
                else:
                    res = math.cos(gterm)
                    lb, ub = res, res
                    
                return res, GRB.CONTINUOUS, lb, ub, symb
            
            elif name == 'sin':
                if symb:
                    lb, ub = -1.0, 1.0
                    res = self._add_real_var(model, -1.0, 1.0)
                    model.addGenConstrSin(gterm, res)
                else:
                    res = math.sin(gterm)
                    lb, ub = res, res
                    
                return res, GRB.CONTINUOUS, lb, ub, symb
            
            elif name == 'tan':
                if symb:
                    lb, ub = -GRB.INFINITY, GRB.INFINITY
                    res = self._add_real_var(model, lb, ub)
                    model.addGenConstrTan(gterm, res)
                else:
                    res = math.tan(gterm)
                    lb, ub = res, res
                    
                return res, GRB.CONTINUOUS, lb, ub, symb
            
            elif name == 'exp':
                if symb:                
                    lb, ub = GurobiRDDLCompiler._fix_bounds(
                        math.exp(lb), math.exp(ub))
                    res = self._add_real_var(model, lb, ub)
                    model.addGenConstrExp(gterm, res)
                else:
                    res = math.exp(gterm)
                    lb, ub = res, res
                    
                return res, GRB.CONTINUOUS, lb, ub, symb
            
            elif name == 'ln':                
                if symb:                                   
                    # argument must be non-negative
                    lb, ub = max(lb, 0), max(ub, 0)
                    arg = self._add_var(model, vtype, lb, ub)
                    model.addGenConstrMax(arg, [gterm], constant=0)
                        
                    # compute bounds on log
                    lb, ub = GurobiRDDLCompiler._fix_bounds(
                        GurobiRDDLCompiler.GRB_log(lb),
                        GurobiRDDLCompiler.GRB_log(ub))
                    
                    # assign ln to new variable
                    res = self._add_real_var(model, lb, ub)
                    model.addGenConstrLog(arg, res)
                else:
                    res = math.log(gterm)
                    lb, ub = res, res
                    
                return res, GRB.CONTINUOUS, lb, ub, symb
            
            elif name == 'sqrt':
                if symb:                
                    # argument must be non-negative
                    lb, ub = max(lb, 0), max(ub, 0)
                    arg = self._add_var(model, vtype, lb, ub)
                    model.addGenConstrMax(arg, [gterm], constant=0)
                    
                    # compute bounds on sqrt
                    lb, ub = GurobiRDDLCompiler._fix_bounds(
                        math.sqrt(lb), math.sqrt(ub))
                    
                    # assign sqrt to new variable
                    res = self._add_real_var(model, lb, ub)
                    model.addGenConstrPow(arg, res, 0.5)
                else:
                    res = math.sqrt(gterm)
                    lb, ub = res, res
                    
                return res, GRB.CONTINUOUS, lb, ub, symb
        
        # binary functions
        elif n == 2:
            arg1, arg2 = args
            gterm1, vtype1, lb1, ub1, symb1 = self._gurobi(arg1, model, subs)
            gterm2, vtype2, lb2, ub2, symb2 = self._gurobi(arg2, model, subs)
            vtype = GurobiRDDLCompiler._promote_vtype(vtype1, vtype2)
            vtype = GurobiRDDLCompiler._at_least_int(vtype)
            symb = symb1 or symb2
            
            if name == 'min':                
                if symb:                    
                    # compute bounds on min
                    lb, ub = GurobiRDDLCompiler._fix_bounds(
                        min(lb1, lb2), min(ub1, ub2))
                
                    # assign min to new variable
                    res = self._add_var(model, vtype, lb, ub)
                    model.addGenConstrMin(res, [gterm1, gterm2])
                else:
                    res = min(gterm1, gterm2)
                    lb, ub = res, res
                    
                return res, vtype, lb, ub, symb
            
            elif name == 'max':
                if symb:
                    # compute bounds on max
                    lb, ub = GurobiRDDLCompiler._fix_bounds(
                        max(lb1, lb2), max(ub1, ub2))
                    
                    # assign max to new variable
                    res = self._add_var(model, vtype, lb, ub)
                    model.addGenConstrMax(res, [gterm1, gterm2])
                else:
                    res = max(gterm1, gterm2)
                    lb, ub = res, res
                    
                return res, vtype, lb, ub, symb
            
            elif name == 'pow':
                if symb:                    
                    # argument must be non-negative
                    lb1, ub1 = max(lb1, 0), max(ub1, 0)
                    base = self._add_var(model, vtype, lb1, ub1)
                    model.addGenConstrMax(base, [gterm1], constant=0)
                    
                    # TODO: compute bounds on pow
                    lb, ub = -GRB.INFINITY, GRB.INFINITY
                    
                    # assign pow to new variable
                    res = self._add_real_var(model, lb, ub)
                    model.addGenConstrPow(base, res, gterm2)
                else:
                    res = math.pow(gterm1, gterm2)   
                    lb, ub = res, res   
                                
                return res, GRB.CONTINUOUS, lb, ub, symb
        
        raise RDDLNotImplementedError(
            f'Function operator {name} with {n} arguments is not '
            f'supported in Gurobi compiler.\n' + 
            print_stack_trace(expr))

    # ===========================================================================
    # control flow
    # ===========================================================================
    
    def _gurobi_control(self, expr, model, subs):
        _, op = expr.etype
        args = expr.args
        n = len(args)
        
        if op == 'if' and n == 3:
            pred, arg1, arg2 = args
            gpred, *_, symbp = self._gurobi(pred, model, subs)
            gterm1, vtype1, lb1, ub1, symb1 = self._gurobi(arg1, model, subs)
            gterm2, vtype2, lb2, ub2, symb2 = self._gurobi(arg2, model, subs)
            vtype = GurobiRDDLCompiler._promote_vtype(vtype1, vtype2)
            
            # compute bounds on if
            lb, ub = GurobiRDDLCompiler._fix_bounds(min(lb1, lb2), max(ub1, ub2))
            
            # assign if to new variable
            if symbp:
                res = self._add_var(model, vtype, lb, ub)
                model.addConstr((gpred == 1) >> (res == gterm1))
                model.addConstr((gpred == 0) >> (res == gterm2))
                symb = True
            elif gpred:
                res, symb = gterm1, symb1
            else:
                res, symb = gterm2, symb2
            return res, vtype, lb, ub, symb
            
        raise RDDLNotImplementedError(
            f'Control flow {op} with {n} arguments is not '
            f'supported in Gurobi compiler.\n' + 
            print_stack_trace(expr))

    # ===========================================================================
    # random variables
    # ===========================================================================
    
    def _gurobi_random(self, expr, model, subs):
        _, name = expr.etype
        if name == 'KronDelta':
            return self._gurobi_random_kron(expr, model, subs)
        elif name == 'DiracDelta':
            return self._gurobi_random_dirac(expr, model, subs)
        else:
            raise RDDLNotImplementedError(
                f'Distribution {name} is not supported in Gurobi compiler.\n' + 
                print_stack_trace(expr))
    
    def _gurobi_random_kron(self, expr, model, subs):
        return self._gurobi(expr, model, subs)
    
    def _gurobi_random_dirac(self, expr, model, subs):
        return self._gurobi(expr, model, subs)
    
    