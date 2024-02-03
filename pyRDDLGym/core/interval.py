import numpy as np
from typing import Dict, List, Set, Tuple, Union

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.compiler.tracer import RDDLObjectsTracer, RDDLTracedObjects
from pyRDDLGym.core.debug.exception import (
    print_stack_trace as PST,
    RDDLInvalidNumberOfArgumentsError,
    RDDLInvalidObjectError,
    RDDLNotImplementedError,
    RDDLRepeatedVariableError,
    RDDLTypeError,
    RDDLUndefinedVariableError
)
from pyRDDLGym.core.debug.logger import Logger
from pyRDDLGym.core.parser.expr import Expression


class RDDLIntervalAnalysis:
    
    def __init__(self, rddl: RDDLPlanningModel, trace: RDDLTracedObjects,
                 cpf_levels: Dict[str, List[str]],
                 logger: Logger=None):
        '''Creates a new interval analysis object for the given RDDL domain.
        
        :param rddl: the RDDL domain to analyze
        :param trace: pre-computed tracing of the RDDL
        :param cpf_levels: order of evaluation of CPFs
        :param logger: to log compilation information during tracing to file
        '''
        self.rddl = rddl
        self.trace = trace
        self.cpf_levels = cpf_levels
        self.logger = logger
        
        self.NUMPY_PROD_FUNC = np.frompyfunc(self._bound_product_scalar, nin=2, nout=1)    
        self.NUMPY_OR_FUNC = np.frompyfunc(self._bound_or_scalar, nin=2, nout=1)
    
    def bound(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        rddl = self.rddl
        intervals = self._bound_initial_values()
        for _ in range(rddl.horizon):
            self._bound_next_epoch(intervals)
        return intervals
    
    def _bound_initial_values(self):
        rddl = self.rddl 
        
        # initially all bounds are calculated based on the initial values
        init_values = {}
        init_values.update(rddl.non_fluents)
        init_values.update(rddl.state_fluents)
        init_values.update(rddl.action_fluents)
        init_values.update(rddl.interm_fluents)
        init_values.update(rddl.derived_fluents)
        init_values.update(rddl.observ_fluents)
        
        # calculate initial values
        intervals = {}
        for (name, values) in init_values.items():
            params = rddl.variable_params[name]
            shape = rddl.object_counts(params)
            values = np.reshape(values, newshape=shape)
            intervals[name] = (values, values)
        return intervals
            
    def _bound_next_epoch(self, intervals):
        rddl = self.rddl 
        
        # trace and bound constraints
        for (i, expr) in enumerate(rddl.invariants):
            self._bound(expr, intervals)
        for (i, expr) in enumerate(rddl.preconditions):
            self._bound(expr, intervals)
            
        # trace and bound CPFs
        for cpfs in self.cpf_levels.values():
            for cpf in cpfs:
                _, expr = rddl.cpfs[cpf]
                lb0, ub0 = intervals.get(cpf, (None, None))
                lb1, ub1 = self._bound(expr, intervals)
                if cpf in rddl.interm_fluents or cpf in rddl.derived_fluents \
                or cpf in rddl.prev_state:
                    intervals[cpf] = (lb1, ub1)
                else:
                    intervals[cpf] = (np.minimum(lb0, lb1), np.maximum(ub0, ub1))
        
        # compute bounds on reward
        reward_bounds = self._bound(rddl.reward, intervals)
        intervals['reward'] = reward_bounds
        
        # update state bounds from next-state fluent bounds
        for (state, next_state) in rddl.next_state.items():
            intervals[state] = intervals[next_state]
                
    # ===========================================================================
    # start of bound calculation subroutines
    # ===========================================================================
    
    def _bound(self, expr, intervals):
        etype, _ = expr.etype
        if etype == 'constant':
            return self._bound_constant(expr, intervals)
        elif etype == 'pvar':
            return self._bound_pvar(expr, intervals)
        elif etype == 'arithmetic':
            return self._bound_arithmetic(expr, intervals)
        elif etype == 'relational':
            return self._bound_relational(expr, intervals)
        elif etype == 'boolean':
            return self._bound_logical(expr, intervals)
        elif etype == 'aggregation':
            return self._bound_aggregation(expr, intervals)
        elif etype == 'func':
            return self._bound_func(expr, intervals)
        elif etype == 'control':
            return self._bound_control(expr, intervals)
        elif etype == 'randomvar':
            return self._bound_random(expr, intervals)
        elif etype == 'randomvector':
            return self._bound_random_vector(expr, intervals)
        elif etype == 'matrix':
            return self._bound_matrix(expr, intervals)
        else:
            raise RDDLNotImplementedError(
                f'Internal error: expression type {etype} is not supported.\n' + 
                PST(expr, out._current_root))
    
    # ===========================================================================
    # leaves
    # ===========================================================================
        
    def _bound_constant(self, expr, intervals):
        lower = upper = self.trace.cached_sim_info(expr)
        return (lower, upper)
    
    def _bound_pvar(self, expr, intervals):
        var, args = expr.args
        
        # free variable (e.g., ?x) and object converted to canonical index
        is_value, cached_info = self.trace.cached_sim_info(expr)
        if is_value:
            lower = upper = cached_info
            return (lower, upper)
        
        # extract variable bounds
        bounds = intervals.get(var, None)
        if bounds is None:
            raise RDDLUndefinedVariableError(
                f'Variable <{var}> is referenced before assignment.\n' + PST(expr))
        lower, upper = bounds
        
        # propagate the bounds forward
        if cached_info is not None:
            slices, axis, shape, op_code, op_args = cached_info
            if slices: 
                raise RDDLNotImplementedError('Nested pvariable not supported.')
            if axis:
                lower = np.expand_dims(lower, axis=axis)
                lower = np.broadcast_to(lower, shape=shape)
                upper = np.expand_dims(upper, axis=axis)
                upper = np.broadcast_to(upper, shape=shape)
            if op_code == RDDLObjectsTracer.NUMPY_OP_CODE.EINSUM:
                lower = np.einsum(lower, *op_args)
                upper = np.einsum(upper, *op_args)
            elif op_code == RDDLObjectsTracer.NUMPY_OP_CODE.TRANSPOSE:
                lower = np.transpose(lower, axes=op_args)
                upper = np.transpose(upper, axes=op_args)
        return (lower, upper)
    
    # ===========================================================================
    # arithmetic
    # ===========================================================================
    
    def _mask_assign(self, dest, mask, value, mask_value=False):
        assert (np.shape(dest) == np.shape(mask))
        if np.shape(dest):
            if mask_value:
                value = value[mask]
            dest[mask] = value
        elif mask:
            dest = value
        return dest
        
    def _bound_arithmetic_expr(self, int1, int2, op):
        (l1, u1), (l2, u2) = int1, int2
        if op == '+':
            return (l1 + l2, u1 + u2)
        elif op == '-':
            return (l1 - u2, u1 - l2)
        elif op == '*':
            parts = [l1 * l2, l1 * u2, u1 * l2, u1 * u2]
            return (np.minimum.reduce(parts), np.maximum.reduce(parts))
        elif op == '/':
            zero_in_open = (0 > l2) & (0 < u2)             
            l2_inv = 1. / u2
            u2_inv = 1. / l2
            l2_inv = self._mask_assign(l2_inv, u2 == 0, -np.inf)
            u2_inv = self._mask_assign(u2_inv, l2 == 0, +np.inf)
            l2_inv = self._mask_assign(l2_inv, zero_in_open, -np.inf)
            u2_inv = self._mask_assign(u2_inv, zero_in_open, +np.inf)
            int2_inv = (l2_inv, u2_inv)    
            return self._bound_arithmetic_expr(int1, int2_inv, '*')
        else:
            raise RDDLNotImplementedError(
                f'Arithmetic operator {op} is not supported.')
        
    def _bound_arithmetic(self, expr, intervals):
        _, op = expr.etype
        args = expr.args        
        n = len(args)
        
        # unary negation
        if n == 1 and op == '-':
            arg, = args
            lower, upper = self._bound(arg, intervals)
            return (-upper, -lower)
        
        # binary operator
        elif n == 2:
            lhs, rhs = args
            int1 = self._bound(lhs, intervals)
            int2 = self._bound(rhs, intervals)
            return self._bound_arithmetic_expr(int1, int2, op)
        
        # ternary and higher order operator
        elif n >= 2 and op in ('+', '*'):
            int_res = self._bound(args[0], intervals)
            for arg in args[1:]:
                int_arg = self._bound(arg, intervals)
                int_res = self._bound_arithmetic_expr(int_res, int_arg, op)
            return int_res
        
        else:
            raise RDDLInvalidNumberOfArgumentsError(
                f'Arithmetic operator {op} does not have the required '
                f'number of arguments.\n' + PST(expr))
    
    # ===========================================================================
    # boolean
    # ===========================================================================
    
    def _bound_relational_expr(self, int1, int2, op):
        (l1, u1), (l2, u2) = int1, int2
        lower = np.zeros(np.shape(l1), dtype=int)
        upper = np.ones(np.shape(u1), dtype=int)
        if op == '>=':
            lower = self._mask_assign(lower, l1 >= u2, 1)
            upper = self._mask_assign(upper, u1 < l2, 1)
            return lower, upper
        elif op == '>':
            lower = self._mask_assign(lower, l1 > u2, 1)
            upper = self._mask_assign(upper, u1 <= l2, 0)
            return lower, upper
        elif op == '==':
            lower = self._mask_assign(lower, (l1 == u1) & (l2 == u2) & (l1 == u2), 1)
            upper = self._mask_assign(upper, (u1 < l2) | (l1 > u2), 0)
            return lower, upper
        elif op == '~=':
            lower = self._mask_assign(lower, (u1 < l2) | (l1 > u2), 1)
            upper = self._mask_assign(upper, (l1 == u1) & (l2 == u2) & (l1 == u2), 0)
            return lower, upper
        elif op == '<':
            return self._bound_relational_expr(int2, int1, '>')
        elif op == '<=':
            return self._bound_relational_expr(int2, int1, '>=')
        else:
            raise RDDLNotImplementedError(
                f'Relational operator {op} is not supported.')
         
    def _bound_relational(self, expr, intervals):
        _, op = expr.etype
        args = expr.args
        lhs, rhs = args
        int1 = self._bound(lhs, intervals)
        int2 = self._bound(rhs, intervals)
        return self._bound_relational_expr(int1, int2, op)
    
    def _bound_logical_expr(self, int1, int2, op):
        if op == '^':
            return self._bound_arithmetic_expr(int1, int2, '*')
        elif op == '|':
            not1 = self._bound_logical_expr(int1, None, '~')
            not2 = self._bound_logical_expr(int2, None, '~')
            not12 = self._bound_logical_expr(not1, not2, '^')
            return self._bound_logical_expr(not12, None, '~')
        elif op == '~':
            if int2 is None:
                l1, u1 = int1
                return (1 - u1, 1 - l1)
            else:
                or12 = self._bound_logical_expr(int1, int2, '|')
                and12 = self._bound_logical_expr(int1, int2, '^')
                not12 = self._bound_logical_expr(and12, None, '~')
                return self._bound_logical_expr(or12, not12, '^')
        elif op == '=>':
            not1 = self._bound_logical_expr(int1, None, '~')
            return self._bound_logical_expr(not1, int2, '|')
        elif op == '<=>':
            return self._bound_relational_expr(int1, int2, '==')
        else:
            raise RDDLNotImplementedError(
                f'Logical operator {op} is not supported.')
           
    def _bound_logical(self, expr, intervals):
        _, op = expr.etype
        if op == '&':
            op = '^'
        args = expr.args
        n = len(args)
        
        # unary logical negation
        if n == 1:
            arg, = args
            int1 = self._bound(arg, intervals)
            return self._bound_logical_expr(int1, None, op)
        
        # binary operator
        elif n == 2:
            lhs, rhs = args
            int1 = self._bound(lhs, intervals)
            int2 = self._bound(rhs, intervals)
            return self._bound_logical_expr(int1, int2, op)
        
        # ternary and higher order operator
        elif n >= 2 and op in ('^', '|'):
            int_res = self._bound(args[0], intervals)
            for arg in args[1:]:
                int_arg = self._bound(arg, intervals)
                int_res = self._bound_logical_expr(int_res, int_arg, op)
            return int_res
        
        else:
            raise RDDLInvalidNumberOfArgumentsError(
                f'Logical operator {op} does not have the required '
                f'number of arguments.\n' + PST(expr))
        
    # ===========================================================================
    # aggregation
    # ===========================================================================
    
    @staticmethod
    def _bound_product_scalar(int1, int2):
        (l1, u1), (l2, u2) = int1, int2
        parts = [l1 * l2, l1 * u2, u1 * l2, u1 * u2]
        return (min(parts), max(parts))
        
    @staticmethod
    def _bound_or_scalar(int1, int2):
        (l1, u1), (l2, u2) = int1, int2
        l1, u1 = (1 - u1, 1 - l1)
        l2, u2 = (1 - u2, 1 - l2)
        parts = [l1 * l2, l1 * u2, u1 * l2, u1 * u2]
        notl, notu = min(parts), max(parts)
        return (1 - notu, 1 - notl)        
    
    @staticmethod
    def _zip_bounds_to_single_array(lower, upper):
        shape = np.shape(lower)
        lower = np.ravel(lower, order='C')
        upper = np.ravel(upper, order='C')
        bounds = np.empty(lower.size, dtype=object)
        bounds[:] = list(zip(lower, upper))
        bounds = np.reshape(bounds, newshape=shape, order='C')
        return bounds
    
    @staticmethod
    def _unzip_single_array_to_bounds(array):
        if isinstance(array, tuple):
            return array
        shape = np.shape(array)
        array = np.ravel(array, order='C')
        lower = np.reshape([p[0] for p in array], newshape=shape, order='C')
        upper = np.reshape([p[1] for p in array], newshape=shape, order='C')
        return (lower, upper)
    
    def _bound_aggregation_func(self, lower, upper, axes, numpyfunc):
        array = self._zip_bounds_to_single_array(lower, upper)
        for axis in axes[::-1]:
            array = numpyfunc.reduce(array, axis=axis)
        return self._unzip_single_array_to_bounds(array)
        
    def _bound_aggregation(self, expr, intervals):
        _, op = expr.etype
        * _, arg = expr.args        
        lower, upper = self._bound(arg, intervals)
        
        # aggregate bound calculation over axes
        _, axes = self.trace.cached_sim_info(expr)
        if op == 'sum':
            return (np.sum(lower, axis=axes), np.sum(upper, axis=axes))        
        elif op == 'avg':
            return (np.mean(lower, axis=axes), np.mean(upper, axis=axes))        
        elif op == 'prod':
            return self._bound_aggregation_func(lower, upper, axes, self.NUMPY_PROD_FUNC)        
        elif op == 'minimum':
            return (np.min(lower, axis=axes), np.min(upper, axis=axes))        
        elif op == 'maximum':
            return (np.max(lower, axis=axes), np.max(upper, axis=axes))        
        elif op == 'forall':
            return self._bound_aggregation_func(lower, upper, axes, self.NUMPY_PROD_FUNC)        
        elif op == 'exists':
            return self._bound_aggregation_func(lower, upper, axes, self.NUMPY_OR_FUNC)
        else:        
            raise RDDLNotImplementedError(
                f'Aggregation operator {op} is not supported.\n' + PST(expr))
    
    # ===========================================================================
    # function
    # ===========================================================================
    
    UNARY_MONOTONE = {
        'sgn': lambda x: np.sign(x).astype(int),
        'round': lambda x: np.round(x).astype(int),
        'floor': lambda x: np.floor(x).astype(int),
        'ceil': lambda x: np.ceil(x).astype(int),
        'acos': np.arccos,
        'asin': np.arcsin,
        'atan': np.arctan,
        'sinh': np.sinh,
        'tanh': np.tanh,
        'exp': np.exp,
        'ln': np.log,
        'sqrt': np.sqrt
    }
    
    UNARY_U_SHAPED = {
        'abs': (np.abs, 0),
        'cosh': (np.cosh, 0)
    }
    
    UNARY_GENERAL = {
        # 'cos': np.cos,
        # 'sin': np.sin,
        # 'tan': np.tan,
        # 'gamma': 
        # 'lngamma':
    }
    
    def _bound_func_monotone(self, l, u, func):
        fl, fu = func(l), func(u)
        return (np.minimum(fl, fu), np.maximum(fl, fu))
    
    def _bound_func_u_shaped(self, l, u, x_crit, func):
        fl, fu = func(l), func(u)
        f_crit = func(x_crit)
        lower = np.full(shape=np.shape(l), fill_value=f_crit)
        upper = np.maximum(fl, fu)
        lower = self._mask_assign(lower, l >= x_crit, fl, True)
        upper = self._mask_assign(upper, l >= x_crit, fu, True)
        lower = self._mask_assign(lower, u <= x_crit, fu, True)
        upper = self._mask_assign(upper, u <= x_crit, fl, True)
        return (lower, upper)
    
    def _bound_func_unary(self, expr, intervals):
        _, name = expr.etype
        arg, = expr.args
        l, u = self._bound(arg, intervals)
        
        if name in self.UNARY_MONOTONE:
            return self._bound_func_monotone(l, u, self.UNARY_MONOTONE[name])
            
        elif name in self.UNARY_U_SHAPED:
            func, x_crit = self.UNARY_U_SHAPED[name]
            return self._bound_func_u_shaped(l, u, x_crit, func)
        
        # TODO
        elif name in self.UNARY_GENERAL:
            pass
        
        else:
            raise RDDLNotImplementedError(
                f'Unary function {name} is not supported.\n' + PST(expr))
    
    def _bound_func_binary(self, expr, intervals):
        _, name = expr.etype
        arg1, arg2 = expr.args
        int1 = (l1, u1) = self._bound(arg1, intervals)
        int2 = (l2, u2) = self._bound(arg2, intervals)
        
        if name == 'div': 
            l, u = self._bound_arithmetic_expr(int1, int2, '/')
            return self._bound_func_monotone(l, u, np.floor)
            
        elif name == 'min':
            return (np.minimum(l1, l2), np.minimum(u1, u2))
            
        elif name == 'max':
            return (np.maximum(l1, l2), np.maximum(u1, u2))
        
        # TODO
        elif name == 'mod':
            pass
        
        # TODO
        elif name == 'fmod':
            pass
        
        elif name == 'pow':
            lower = np.ones(shape=np.shape(l1)) * np.nan
            upper = np.ones(shape=np.shape(u1)) * np.nan
            
            # positive base, means well defined for any real power
            log1 = self._bound_func_monotone(l1, u1, np.log)
            pow = self._bound_arithmetic_expr(log1, int2, '*')
            l1pos, u1pos = self._bound_func_monotone(*pow, np.exp)
            lower = self._mask_assign(lower, l1 > 0, l1pos, True)
            upper = self._mask_assign(upper, l1 > 0, u1pos, True)
            
            # otherwise, defined if the power is an integer point
            l2_is_int = np.equal(np.mod(l2, 1), 0)
            u2_is_int = np.equal(np.mod(u2, 1), 0)
            pow_valid = l2_is_int & u2_is_int & (l2 >= 0) & (l2 == u2)
            pow = l2.astype(int) if np.shape(l2) else int(l2)
            pow_even = (np.mod(pow, 2) == 0)
            
            # if the base interval contains 0
            case1 = pow_valid & (0 >= l1) & (0 <= u1)
            lower = self._mask_assign(lower, case1 & pow_even, 0)
            upper = self._mask_assign(upper, case1 & (pow == 0), 1)
            upper = self._mask_assign(upper, case1 & (pow > 0) & pow_even, 
                                      np.maximum(l1 ** pow, l2 ** pow), True)
            lower = self._mask_assign(lower, case1 & ~pow_even, l1 ** pow, True)
            upper = self._mask_assign(upper, case1 & ~pow_even, l2 ** pow, True)
            
            # if the base is strictly negative
            case2 = pow_valid & (u1 < 0)
            lower = self._mask_assign(lower, case2 & (pow == 0), 1)
            upper = self._mask_assign(upper, case2 & (pow == 0), 1)
            lower = self._mask_assign(lower, case2 & pow_even, u1 ** pow, True)
            upper = self._mask_assign(upper, case2 & pow_even, l1 ** pow, True)
            lower = self._mask_assign(lower, case2 & ~pow_even, l1 ** pow, True)
            upper = self._mask_assign(upper, case2 & ~pow_even, u1 ** pow, True)
            return (lower, upper)
        
        elif name == 'log':
            if np.any(l2 <= 0):
                raise RDDLNotImplementedError(
                    f'Function {name} with base <= 0 is not supported.')
            if np.any(l1 <= 0):
                raise RDDLNotImplementedError(
                    f'Function {name} with argument <= 0 is not supported.')                    
            log1 = self._bound_func_monotone(l1, u1, np.log)
            log2 = self._bound_func_monotone(l2, u2, np.log)
            return self._bound_arithmetic_expr(log1, log2, '/')
        
        elif name == 'hypot': 
            pow1 = self._bound_func_u_shaped(l1, u1, 0, np.square)
            pow2 = self._bound_func_u_shaped(l2, u2, 0, np.square)
            l, u = self._bound_arithmetic_expr(pow1, pow2, '+')
            return self._bound_func_monotone(l, u, np.sqrt)
        
        else:
            raise RDDLNotImplementedError(
                f'Binary function {name} is not supported.\n' + PST(expr))
        
    def _bound_func(self, expr, intervals):
        _, name = expr.etype
        args = expr.args
        n = len(args)
        
        if n == 1:
            return self._bound_func_unary(expr, intervals)
        elif n == 2:
            return self._bound_func_binary(expr, intervals)
        else:
            raise RDDLNotImplementedError(
                f'Function {name} with {n} arguments is not supported.\n' + PST(expr))
            
    # ===========================================================================
    # control flow
    # ===========================================================================
    
    def _bound_control(self, expr, intervals):
        _, op = expr.etype
        
        if op == 'if':
            return self._bound_if(expr, intervals)
        else:
            return self._bound_switch(expr, intervals) 
    
    def _bound_if(self, expr, intervals):
        args = expr.args
        pred, arg1, arg2 = args
        intp = self._bound(pred, intervals)
        int1 = (l1, u1) = self._bound(arg1, intervals)
        int2 = (l2, u2) = self._bound(arg2, intervals)        
        return (np.minimum(l1, l2), np.maximum(u1, u2))
    
    def _bound_switch(self, expr, intervals):
        raise RDDLNotImplementedError(
            f'Function switch is not supported.\n' + PST(expr))
    
    # ===========================================================================
    # random variables
    # ===========================================================================
    
    def _bound_random(self, expr, intervals):
        _, name = expr.etype
        if name == 'KronDelta':
            return self._bound_random_kron(expr, intervals)
        elif name == 'DiracDelta':
            return self._bound_random_dirac(expr, intervals)
        elif name == 'Uniform':
            return self._bound_uniform(expr, intervals)            
        elif name == 'Bernoulli':
            return self._bound_bernoulli(expr, intervals)
        elif name == 'Normal':
            return self._bound_normal(expr, intervals)
        elif name == 'Poisson':
            return self._bound_poisson(expr, intervals)
        elif name == 'Exponential':
            return self._bound_exponential(expr, intervals)
        elif name == 'Weibull':
            return self._bound_weibull(expr, intervals)        
        elif name == 'Gamma':
            return self._bound_gamma(expr, intervals)
        elif name == 'Binomial':
            return self._bound_binomial(expr, intervals)
        elif name == 'NegativeBinomial':
            return self._bound_negative_binomial(expr, intervals)
        elif name == 'Beta':
            return self._bound_beta(expr, intervals)
        elif name == 'Geometric':
            return self._bound_geometric(expr, intervals)
        elif name == 'Pareto':
            return self._bound_pareto(expr, intervals)
        elif name == 'Student':
            return self._bound_student(expr, intervals)
        elif name == 'Gumbel':
            return self._bound_gumbel(expr, intervals)
        elif name == 'Laplace':
            return self._bound_laplace(expr, intervals)
        elif name == 'Cauchy':
            return self._bound_cauchy(expr, intervals)
        elif name == 'Gompertz':
            return self._bound_gompertz(expr, intervals)
        elif name == 'ChiSquare':
            return self._bound_chisquare(expr, intervals)
        elif name == 'Kumaraswamy':
            return self._bound_kumaraswamy(expr, intervals)
        elif name == 'Discrete':
            return self._bound_discrete(expr, intervals, False)
        elif name == 'UnnormDiscrete':
            return self._bound_discrete(expr, intervals, True)
        elif name == 'Discrete(p)':
            return self._bound_discrete_pvar(expr, intervals, False)
        elif name == 'UnnormDiscrete(p)':
            return self._bound_discrete_pvar(expr, intervals, True)
        else:
            raise RDDLNotImplementedError(
                f'Distribution {name} is not supported.\n' + PST(expr))
    
    def _bound_random_kron(self, expr, intervals):
        args = expr.args
        arg, = args
        return self._bound(arg, intervals)
    
    def _bound_random_dirac(self, expr, intervals):
        args = expr.args
        arg, = args
        return self._bound(arg, intervals)
    
    def _bound_uniform(self, expr, intervals):
        args = expr.args
        lb, ub = args
        intl = (ll, ul) = self._bound(lb, intervals)
        intu = (lu, uu) = self._bound(ub, intervals)
        
        u01 = (np.zeros(shape=np.shape(ll)), np.ones(shape=np.shape(ul)))
        diff = self._bound_arithmetic_expr(intu, intl, '-')
        return self._bound_arithmetic_expr(
            self._bound_arithmetic_expr(diff, u01, '*'), intl, '+')
    
    def _bound_bernoulli(self, expr, intervals):
        args = expr.args
        p, = args
        intp = (lp, up) = self._bound(p, intervals)
        
        lower = np.zeros(shape=np.shape(lp), dtype=bool)
        upper = np.ones(shape=np.shape(up), dtype=bool)
        lower = self._mask_assign(lower, lp >= 1, 1)
        upper = self._mask_assign(upper, up <= 0, 0)
        return (lower, upper)
    
    def _bound_normal(self, expr, intervals):
        args = expr.args
        mean, var = args
        intm = (lm, um) = self._bound(mean, intervals)
        intv = (lv, uv) = self._bound(var, intervals)
        
        lower = np.ones(shape=np.shape(lm)) * -np.inf
        upper = np.ones(shape=np.shape(um)) * np.inf
        lower = self._mask_assign(lower, (lv == 0) & (uv == 0), lm, True)
        upper = self._mask_assign(upper, (lv == 0) & (uv == 0), um, True)
        return (lower, upper)
    
    def _bound_poisson(self, expr, intervals):
        args = expr.args
        p, = args
        intp = (lp, up) = self._bound(p, intervals)
        
        lower = np.zeros(shape=np.shape(lp), dtype=int)
        upper = np.ones(shape=np.shape(up), dtype=int) * np.inf
        return (lower, upper)
    
    def _bound_exponential(self, expr, intervals):
        args = expr.args
        scale, = args
        ints = (ls, us) = self._bound(scale, intervals)
        
        lower = np.zeros(shape=np.shape(ls))
        upper = np.ones(shape=np.shape(us)) * np.inf
        return (lower, upper)
    
    def _bound_weibull(self, expr, intervals):
        args = expr.args
        shape, scale = args
        intsh = (lsh, ush) = self._bound(shape, intervals)
        intsc = (lsc, usc) = self._bound(scale, intervals)
        
        lower = np.zeros(shape=np.shape(lsh))
        upper = np.ones(shape=np.shape(ush)) * np.inf
        return (lower, upper)
    
    def _bound_gamma(self, expr, intervals):
        args = expr.args
        shape, scale = args
        intsh = (lsh, ush) = self._bound(shape, intervals)
        intsc = (lsc, usc) = self._bound(scale, intervals)
        
        lower = np.zeros(shape=np.shape(ls))
        upper = np.ones(shape=np.shape(us)) * np.inf
        return (lower, upper)
    
    def _bound_binomial(self, expr, intervals):
        args = expr.args
        n, p = args
        intn = (ln, un) = self._bound(n, intervals)
        intp = (lp, up) = self._bound(p, intervals)
        
        lower = np.zeros(shape=np.shape(ln), dtype=int)
        upper = un
        return (lower, upper)
    
    def _bound_beta(self, expr, intervals):
        args = expr.args
        shape, rate = args
        ints = (ls, us) = self._bound(shape, intervals)
        intr = (lr, ur) = self._bound(rate, intervals)
        
        lower = np.zeros(shape=np.shape(ls))
        upper = np.ones(shape=np.shape(us))
        return (lower, upper)
    
    def _bound_geometric(self, expr, intervals):
        args = expr.args
        p, = args
        
        intp = (lp, up) = self._bound(p, intervals)
        lower = np.ones(shape=np.shape(lp), dtype=int)
        upper = np.ones(shape=np.shape(up)) * np.inf
        return (lower, upper)
    
    def _bound_pareto(self, expr, intervals):
        args = expr.args
        shape, scale = args
        intsh = (lsh, ush) = self._bound(shape, intervals)
        intsc = (lsc, usc) = self._bound(scale, intervals)
        
        lower = lsc
        upper = np.ones(shape=np.shape(usc)) * np.inf
        return (lower, upper)
    
    def _bound_student(self, expr, intervals):
        args = expr.args
        df, = args
        intd = (ld, ud) = self._bound(df, intervals)
        
        lower = np.ones(shape=np.shape(ld)) * -np.inf
        upper = np.ones(shape=np.shape(ud)) * np.inf
        return (lower, upper)
    
    def _bound_gumbel(self, expr, intervals):
        args = expr.args
        mean, scale = args
        intm = (lm, um) = self._bound(mean, intervals)
        ints = (ls, us) = self._bound(scale, intervals)
        
        lower = np.ones(shape=np.shape(lm)) * -np.inf
        upper = np.ones(shape=np.shape(um)) * np.inf
        return (lower, upper)
    
    def _bound_cauchy(self, expr, intervals):
        args = expr.args
        mean, scale = args
        intm = (lm, um) = self._bound(mean, intervals)
        ints = (ls, us) = self._bound(scale, intervals)
        
        lower = np.ones(shape=np.shape(lm)) * -np.inf
        upper = np.ones(shape=np.shape(um)) * np.inf
        return (lower, upper)
    
    def _bound_gompertz(self, expr, intervals):
        args = expr.args
        shape, scale = args
        intsh = (lsh, ush) = self._bound(shape, intervals)
        intsc = (lsc, usc) = self._bound(scale, intervals)
        
        lower = np.zeros(shape=np.shape(lsh))
        upper = np.ones(shape=np.shape(ush)) * np.inf
        return (lower, upper)
    
    def _bound_chisquare(self, expr, intervals):
        args = expr.args
        df, = args
        intd = (ld, ud) = self._bound(df, intervals)
        
        lower = np.zeros(shape=np.shape(ld))
        upper = np.ones(shape=np.shape(ud)) * np.inf
        return (lower, upper)
    
    def _bound_kumaraswamy(self, expr, intervals):
        args = expr.args
        a, b = args
        inta = (la, ua) = self._bound(a, intervals)
        intb = (lb, ub) = self._bound(b, intervals)
        
        lower = np.zeros(shape=np.shape(la))
        upper = np.ones(shape=np.shape(ua))
        return (lower, upper)
