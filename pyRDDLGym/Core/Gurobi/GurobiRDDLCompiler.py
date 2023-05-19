import gurobipy
from gurobipy import GRB

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
        subs = self.init_values.copy() 
        objective = 0.0
        for step in range(self.horizon):
            reward = self._compile_step(step, model, subs)
            objective += reward
        model.setObjective(objective, GRB.MAXIMIZE)
        model.params.NonConvex = 2
        return model
            
    def _compile_step(self, step, model, subs):
        rddl = self.rddl
        
        # add action fluent variables to model
        for (action, prange) in rddl.actionsranges.items():
            vtype = self.GUROBI_TYPES[prange]
            name = f'{action}___{step}'
            subs[action] = model.addVar(
                vtype=vtype, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=name)
        
        # evaluate CPFs
        for cpfs in self.levels.values():
            for cpf in cpfs:
                _, expr = rddl.cpfs[cpf]
                subs[cpf], _ = self._gurobi(expr, model, subs)
        
        # evaluate reward
        reward, _ = self._gurobi(rddl.reward, model, subs)
        
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
        else:
            raise RDDLNotImplementedError(
                f'Expression type {etype} is not supported in Gurobi compiler.\n' + 
                print_stack_trace(expr))
            
    # ===========================================================================
    # leaves
    # ===========================================================================
    
    def _gurobi_constant(self, expr, *_):
        value = self.traced.cached_sim_info(expr)
        if isinstance(value, bool):
            vtype = self.GUROBI_TYPES['bool']
        elif isinstance(value, int):
            vtype = self.GUROBI_TYPES['int']
        elif isinstance(value, float):
            vtype = self.GUROBI_TYPES['real']
        else:
            raise RDDLNotImplementedError(
                f'Range of {value} is not supported in Gurobi compiler.')
        return value, vtype

    def _gurobi_pvar(self, expr, model, subs):
        var, _ = expr.args
        
        # domain object converted to canonical index
        is_value, cached_info = self.traced.cached_sim_info(expr)
        if is_value:
            return cached_info
        
        # extract variable value
        value = subs.get(var, None)
        if value is None:
            raise RDDLUndefinedVariableError(
                f'Variable <{var}> is referenced before assignment.\n' + 
                print_stack_trace(expr))
        
        # retrieve the pvariable type
        prange = self.rddl.variable_ranges[var]
        vtype = self.GUROBI_TYPES[prange]
        return value, vtype
    
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
    
    def _gurobi_arithmetic(self, expr, model, subs):
        _, op = expr.etype
        args = expr.args        
        n = len(args)
        
        # unary negation
        if n == 1 and op == '-':
            arg, = args
            gterm, vtype = self._gurobi(arg, model, subs)
            var = model.addVar(vtype=vtype, lb=-GRB.INFINITY, ub=GRB.INFINITY)
            model.addConstr(var == -gterm)
            return var, vtype
        
        # binary operations
        elif n >= 2:
            
            # unwrap addition to binary operations
            if op == '+':
                arg, *rest = args
                sumvar, vtype = self._gurobi(arg, model, subs)
                for arg in rest:
                    gterm, vtype1 = self._gurobi(arg, model, subs)
                    vtype = GurobiRDDLCompiler._promote_vtype(vtype, vtype1)
                    var = model.addVar(vtype=vtype, lb=-GRB.INFINITY, ub=GRB.INFINITY)
                    model.addConstr(var == sumvar + gterm)
                    sumvar = var
                return var, vtype
            
            # unwrap multiplication to binary operations
            elif op == '*':
                arg, *rest = args
                prodvar, vtype = self._gurobi(arg, model, subs)
                for arg in rest:
                    gterm, vtype1 = self._gurobi(arg, model, subs)
                    vtype = GurobiRDDLCompiler._promote_vtype(vtype, vtype1)
                    var = model.addVar(vtype=vtype, lb=-GRB.INFINITY, ub=GRB.INFINITY)
                    model.addConstr(var == prodvar * gterm)
                    prodvar = var
                return var, vtype
            
            # subtraction
            elif op == '-' and n == 2:
                arg1, arg2 = args
                gterm1, vtype1 = self._gurobi(arg1, model, subs)
                gterm2, vtype2 = self._gurobi(arg2, model, subs)
                vtype = GurobiRDDLCompiler._promote_vtype(vtype1, vtype2)
                var = model.addVar(vtype=vtype, lb=-GRB.INFINITY, ub=GRB.INFINITY)
                model.addConstr(var == gterm1 - gterm2)
                return var, vtype
            
            # implement x / y by modeling z = 1 / y as a constraint z * y = 1
            elif op == '/' and n == 2:
                arg1, arg2 = args
                gterm1, vtype1 = self._gurobi(arg1, model, subs)
                gterm2, vtype2 = self._gurobi(arg2, model, subs)
                inv_gterm2 = model.addVar(
                    vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY)
                model.addConstr(inv_gterm2 * gterm2 == 1)
                vtype = GurobiRDDLCompiler._promote_vtype(vtype1, vtype2)
                var = model.addVar(vtype=vtype, lb=-GRB.INFINITY, ub=GRB.INFINITY)
                model.addConstr(var == gterm1 * inv_gterm2)
                return var, vtype
        
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
            glhs, _ = self._gurobi(lhs, model, subs)
            grhs, _ = self._gurobi(rhs, model, subs)
            
            # convert <= to >=, < to >, etc.
            if op == '<=' or op == '<':
                glhs, grhs = grhs, glhs
                op = '>=' if op == '<=' else '>'
            
            # assign comparison operator to binary variable
            var = model.addVar(vtype=GRB.BINARY)
            gdif = glhs - grhs
            if op == '==':
                model.addGenConstrIndicator(var, True, gdif, GRB.EQUAL, 0)
                return var, GRB.BINARY
            elif op == '>=':
                model.addGenConstrIndicator(var, True, gdif, GRB.GREATER_EQUAL, 0)
                return var, GRB.BINARY
            elif op == '~=':
                model.addGenConstrIndicator(var, False, gdif, GRB.EQUAL, 0)           
                return var, GRB.BINARY
            elif op == '>':
                model.addGenConstrIndicator(var, False, gdif, GRB.LESS_EQUAL, 0)
                return var, GRB.BINARY
            
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
            gterm, _ = self._gurobi(arg, model, subs)
            var = model.addVar(vtype=GRB.BINARY)
            model.addConstr(var + gterm == 1)
            return var, GRB.BINARY
            
        # binary operations
        elif n >= 2:
            
            # unwrap AND to binary operations
            if op == '^':
                gterms = [self._gurobi(arg, model, subs)[0] for arg in args]
                var = model.addVar(vtype=GRB.BINARY)
                model.addGenConstrAnd(var, gterms)
                return var, GRB.BINARY
            
            # unwrap OR to binary operations
            elif op == '|':
                gterms = [self._gurobi(arg, model, subs)[0] for arg in args]
                var = model.addVar(vtype=GRB.BINARY)
                model.addGenConstrOr(var, gterms)
                return var, GRB.BINARY
        
        raise RDDLNotImplementedError(
            f'Logical operator {op} with {n} arguments is not '
            f'supported in Gurobi compiler.\n' + 
            print_stack_trace(expr))
    
    # ===========================================================================
    # function
    # ===========================================================================
    
    def _gurobi_function(self, expr, model, subs):
        _, name = expr.etype
        args = expr.args
        n = len(args)
        
        # unary functions
        if n == 1:
            arg, = args
            gterm, vtype = self._gurobi(arg, model, subs)
            if name == 'abs':
                var = model.addVar(vtype=vtype, lb=0, ub=GRB.INFINITY)
                model.addGenConstrAbs(var, gterm)
                return var, vtype
            elif name == 'cos':
                var = model.addVar(vtype=vtype, lb=-1.0, ub=1.0)
                model.addGenConstrCos(gterm, var)
                return var, GRB.CONTINUOUS
            elif name == 'sin':
                var = model.addVar(vtype=vtype, lb=-1.0, ub=1.0)
                model.addGenConstrSin(gterm, var)
                return var, GRB.CONTINUOUS
            elif name == 'tan':
                var = model.addVar(vtype=vtype, lb=-GRB.INFINITY, ub=GRB.INFINITY)
                model.addGenConstrTan(gterm, var)
                return var, GRB.CONTINUOUS
            elif name == 'exp':
                var = model.addVar(vtype=vtype, lb=0, ub=GRB.INFINITY)
                model.addGenConstrExp(gterm, var)
                return var, GRB.CONTINUOUS
            elif name == 'ln':
                pos = model.addVar(vtype=vtype, lb=0, ub=GRB.INFINITY)
                model.addGenConstrMax(pos, [gterm], constant=0)
                var = model.addVar(vtype=vtype, lb=-GRB.INFINITY, ub=GRB.INFINITY)
                model.addGenConstrLog(pos, var)
                return var, GRB.CONTINUOUS
            elif name == 'sqrt':
                pos = model.addVar(vtype=vtype, lb=0, ub=GRB.INFINITY)
                model.addGenConstrMax(pos, [gterm], constant=0)
                var = model.addVar(vtype=vtype, lb=0, ub=GRB.INFINITY)
                model.addGenConstrPow(pos, var, 0.5)
                return var, GRB.CONTINUOUS
        
        # binary functions
        elif n == 2:
            arg1, arg2 = args
            gterm1, vtype1 = self._gurobi(arg1, model, subs)
            gterm2, vtype2 = self._gurobi(arg2, model, subs)
            vtype = GurobiRDDLCompiler._promote_vtype(vtype1, vtype2)
            
            if name == 'min':
                var = model.addVar(vtype=vtype, lb=-GRB.INFINITY, ub=GRB.INFINITY)
                model.addGenConstrMin(var, [gterm1, gterm2])
                return var, vtype
            elif name == 'max':
                var = model.addVar(vtype=vtype, lb=-GRB.INFINITY, ub=GRB.INFINITY)
                model.addGenConstrMax(var, [gterm1, gterm2])
                return var, vtype
            elif name == 'pow':
                pos = model.addVar(vtype=vtype, lb=0, ub=GRB.INFINITY)
                model.addGenConstrMax(pos, [gterm1], constant=0)
                var = model.addVar(vtype=vtype, lb=-GRB.INFINITY, ub=GRB.INFINITY)
                model.addGenConstrPow(pos, var, gterm2)
                return var, GRB.CONTINUOUS
        
        raise RDDLNotImplementedError(
            f'Function operator {name} with {n} arguments is not '
            f'supported in Gurobi compiler.\n' + 
            print_stack_trace(expr))
    