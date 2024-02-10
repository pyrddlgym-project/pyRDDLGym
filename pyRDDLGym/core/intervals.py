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
from pyRDDLGym.core.simulator import lngamma


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
            result = self._bound_constant(expr, intervals)
        elif etype == 'pvar':
            result = self._bound_pvar(expr, intervals)
        elif etype == 'arithmetic':
            result = self._bound_arithmetic(expr, intervals)
        elif etype == 'relational':
            result = self._bound_relational(expr, intervals)
        elif etype == 'boolean':
            result = self._bound_logical(expr, intervals)
        elif etype == 'aggregation':
            result = self._bound_aggregation(expr, intervals)
        elif etype == 'func':
            result = self._bound_func(expr, intervals)
        elif etype == 'control':
            result = self._bound_control(expr, intervals)
        elif etype == 'randomvar':
            result = self._bound_random(expr, intervals)
        elif etype == 'randomvector':
            result = self._bound_random_vector(expr, intervals)
        elif etype == 'matrix':
            result = self._bound_matrix(expr, intervals)
        else:
            raise RDDLNotImplementedError(
                f'Internal error: expression type {etype} is not supported.\n' + 
                PST(expr))
        
        # check valid bounds
        lower, upper = result
        if np.any(lower > upper):
            raise RuntimeError(
                f'Internal error: lower bound {lower} > upper bound {upper}.\n' +
                PST(expr))
        
        return result
    
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
                raise RDDLNotImplementedError('Nested pvariable not supported.\n' + PST(expr))
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
    
    @staticmethod
    def _mask_assign(dest, mask, value, mask_value=False):
        assert (np.shape(dest) == np.shape(mask))
        if np.shape(dest):
            if mask_value:
                value = value[mask]
            dest[mask] = value
        elif mask:
            dest = value
        return dest
    
    @staticmethod
    def _bound_arithmetic_expr(int1, int2, op):
        (l1, u1), (l2, u2) = int1, int2
        
        # [a, b] + [c, d] = [a + c, b + d]
        if op == '+':
            lower = l1 + l2
            upper = u1 + u2
            return (lower, upper)
        
        # [a, b] - [c, d] = [a, d] - [b, c]
        elif op == '-':
            lower = l1 - u2
            upper = u1 - l2
            return (lower, upper)
        
        # [a, b] * [c, d] 
        elif op == '*':
            parts = [l1 * l2, l1 * u2, u1 * l2, u1 * u2]
            lower = np.minimum.reduce(parts)
            upper = np.maximum.reduce(parts)
            return (lower, upper)
        
        # [a, b] / [c, d] = [a, b] * (1 / [c, d])
        elif op == '/':
            mask_fn = RDDLIntervalAnalysis._mask_assign
            zero_in_open = (0 > l2) & (0 < u2)             
            l2_inv = 1. / u2
            u2_inv = 1. / l2
            l2_inv = mask_fn(l2_inv, u2 == 0, -np.inf)
            u2_inv = mask_fn(u2_inv, l2 == 0, +np.inf)
            l2_inv = mask_fn(l2_inv, zero_in_open, -np.inf)
            u2_inv = mask_fn(u2_inv, zero_in_open, +np.inf)
            int2_inv = (l2_inv, u2_inv)    
            return RDDLIntervalAnalysis._bound_arithmetic_expr(int1, int2_inv, '*')
        
        else:
            raise RDDLNotImplementedError(
                f'Arithmetic operator {op} is not supported.\n' + PST(expr))
        
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
    
    @staticmethod
    def _bound_relational_expr(int1, int2, op):
        (l1, u1), (l2, u2) = int1, int2
        lower = np.zeros(np.shape(l1), dtype=int)
        upper = np.ones(np.shape(u1), dtype=int)
        mask_fn = RDDLIntervalAnalysis._mask_assign
        
        if op == '>=':
            lower = mask_fn(lower, l1 >= u2, 1)
            upper = mask_fn(upper, u1 < l2, 1)
            return lower, upper
        elif op == '>':
            lower = mask_fn(lower, l1 > u2, 1)
            upper = mask_fn(upper, u1 <= l2, 0)
            return lower, upper
        elif op == '==':
            lower = mask_fn(lower, (l1 == u1) & (l2 == u2) & (l1 == u2), 1)
            upper = mask_fn(upper, (u1 < l2) | (l1 > u2), 0)
            return lower, upper
        elif op == '~=':
            lower = mask_fn(lower, (u1 < l2) | (l1 > u2), 1)
            upper = mask_fn(upper, (l1 == u1) & (l2 == u2) & (l1 == u2), 0)
            return lower, upper
        elif op == '<':
            return RDDLIntervalAnalysis._bound_relational_expr(int2, int1, '>')
        elif op == '<=':
            return RDDLIntervalAnalysis._bound_relational_expr(int2, int1, '>=')
        else:
            raise RDDLNotImplementedError(
                f'Relational operator {op} is not supported.\n' + PST(expr))
         
    def _bound_relational(self, expr, intervals):
        _, op = expr.etype
        args = expr.args
        lhs, rhs = args
        int1 = self._bound(lhs, intervals)
        int2 = self._bound(rhs, intervals)
        return self._bound_relational_expr(int1, int2, op)
    
    @staticmethod
    def _bound_logical_expr(int1, int2, op):
        if op == '^':
            return RDDLIntervalAnalysis._bound_arithmetic_expr(int1, int2, '*')
        
        # x | y = ~(~x & ~y)
        elif op == '|':
            not1 = RDDLIntervalAnalysis._bound_logical_expr(int1, None, '~')
            not2 = RDDLIntervalAnalysis._bound_logical_expr(int2, None, '~')
            not12 = RDDLIntervalAnalysis._bound_logical_expr(not1, not2, '^')
            return RDDLIntervalAnalysis._bound_logical_expr(not12, None, '~')
        
        # x ~ y = (x | y) ^ ~(x & y)
        elif op == '~':
            if int2 is None:
                l1, u1 = int1
                lower = 1 - u1
                upper = 1 - l1
                return (lower, upper)
            else:
                or12 = RDDLIntervalAnalysis._bound_logical_expr(int1, int2, '|')
                and12 = RDDLIntervalAnalysis._bound_logical_expr(int1, int2, '^')
                not12 = RDDLIntervalAnalysis._bound_logical_expr(and12, None, '~')
                return RDDLIntervalAnalysis._bound_logical_expr(or12, not12, '^')
        
        # x => y = ~x | y
        elif op == '=>':
            not1 = RDDLIntervalAnalysis._bound_logical_expr(int1, None, '~')
            return RDDLIntervalAnalysis._bound_logical_expr(not1, int2, '|')
        
        # x <=> y = x == y
        elif op == '<=>':
            return RDDLIntervalAnalysis._bound_relational_expr(int1, int2, '==')
        
        else:
            raise RDDLNotImplementedError(
                f'Logical operator {op} is not supported.\n' + PST(expr))
           
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
        lower = min(parts)
        upper = max(parts)
        return (lower, upper)
        
    @staticmethod
    def _bound_or_scalar(int1, int2):
        (l1, u1), (l2, u2) = int1, int2
        l1, u1 = (1 - u1, 1 - l1)
        l2, u2 = (1 - u2, 1 - l2)        
        parts = [l1 * l2, l1 * u2, u1 * l2, u1 * u2]
        not_l, not_u = min(parts), max(parts)        
        lower = 1 - not_u
        upper = 1 - not_l
        return (lower, upper)        
    
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
    
    @staticmethod
    def _bound_aggregation_func(lower, upper, axes, numpyfunc):
        array = RDDLIntervalAnalysis._zip_bounds_to_single_array(lower, upper)
        for axis in sorted(axes, reverse=True):
            array = numpyfunc.reduce(array, axis=axis)
        return RDDLIntervalAnalysis._unzip_single_array_to_bounds(array)
        
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
        'cosh': (np.cosh, 0),
        'lngamma': (lngamma, 1.4616321449683623),
        'gamma': ((lambda x: np.exp(lngamma(x))), 1.4616321449683623)
    }
    
    @staticmethod
    def _bound_func_monotone(l, u, func):
        fl, fu = func(l), func(u)
        lower = np.minimum(fl, fu)
        upper = np.maximum(fl, fu)
        return (lower, upper)
    
    @staticmethod
    def _next_multiple(start, offset, period=2 * np.pi):
        multiples = -np.floor_divide(start - offset, -period)
        next_up = period * multiples + offset
        return next_up
        
    @staticmethod
    def _bound_func_u_shaped(l, u, x_crit, func):
        fl, fu = func(l), func(u)
        f_crit = func(x_crit)
        lower = np.full(shape=np.shape(l), fill_value=f_crit)
        upper = np.maximum(fl, fu)
        lower = RDDLIntervalAnalysis._mask_assign(lower, l >= x_crit, fl, True)
        upper = RDDLIntervalAnalysis._mask_assign(upper, l >= x_crit, fu, True)
        lower = RDDLIntervalAnalysis._mask_assign(lower, u <= x_crit, fu, True)
        upper = RDDLIntervalAnalysis._mask_assign(upper, u <= x_crit, fl, True)
        return (lower, upper)
    
    def _bound_func_unary(self, expr, intervals):
        _, name = expr.etype
        arg, = expr.args
        l, u = self._bound(arg, intervals)
        
        # f is monotone, x in [a, b] => f(x) in [f(a), f(b)]
        if name in self.UNARY_MONOTONE:
            return self._bound_func_monotone(l, u, self.UNARY_MONOTONE[name])
        
        # f is a u-shaped function
        elif name in self.UNARY_U_SHAPED:
            func, x_crit = self.UNARY_U_SHAPED[name]
            return self._bound_func_u_shaped(l, u, x_crit, func)
        
        # special functions sin
        elif name == 'sin':
            
            # by default evaluate at end points
            sin_l, sin_u = np.sin(l), np.sin(u)
            lower = np.minimum(sin_l, sin_u)
            upper = np.maximum(sin_l, sin_u)
            
            # any infinity results in [-1, 1]
            has_inf = (l == -np.inf) | (u == np.inf)
            lower = self._mask_assign(lower, has_inf, -1.0)
            upper = self._mask_assign(upper, has_inf, +1.0)
            
            # find critical points
            next_max = self._next_multiple(l, np.pi / 2)
            next_min = self._next_multiple(l, 3 * np.pi / 2)
            lower = self._mask_assign(lower, next_min <= u, -1.0)
            upper = self._mask_assign(upper, next_max <= u, +1.0)
            return lower, upper
        
        # special functions cos
        elif name == 'cos':
            
            # by default evaluate at end points
            cos_l, cos_u = np.cos(l), np.cos(u)
            lower = np.minimum(cos_l, cos_u)
            upper = np.maximum(cos_l, cos_u)
            
            # any infinity results in [-1, 1]
            has_inf = (l == -np.inf) | (u == np.inf)
            lower = self._mask_assign(lower, has_inf, -1.0)
            upper = self._mask_assign(upper, has_inf, +1.0)
            
            # find critical points
            next_max = self._next_multiple(l, 0)
            next_min = self._next_multiple(l, np.pi)
            lower = self._mask_assign(lower, next_min <= u, -1.0)
            upper = self._mask_assign(upper, next_max <= u, +1.0)
            return lower, upper
        
        # special functions tan
        elif name == 'tan':
            
            # by default evaluate at end points
            tan_l, tan_u = np.tan(l), np.tan(u)
            lower = np.minimum(tan_l, tan_u)
            upper = np.maximum(tan_l, tan_u)
            
            # any infinity results in [-inf, inf]
            has_inf = (l == -np.inf) | (u == np.inf)
            lower = self._mask_assign(lower, has_inf, -np.inf)
            upper = self._mask_assign(upper, has_inf, +np.inf)
            
            # an asymptote is inside the open interval, set to [-inf, inf]
            next_asymptote = self._next_multiple(l, np.pi / 2, np.pi)
            asymptote_in_open = (l < next_asymptote) & (u > next_asymptote)
            lower = self._mask_assign(lower, asymptote_in_open, -np.inf)
            upper = self._mask_assign(upper, asymptote_in_open, +np.inf)
            
            # an asymptote is on an edge
            lower = self._mask_assign(lower, l == next_asymptote, -np.inf)
            upper = self._mask_assign(upper, u == next_asymptote, +np.inf)
            upper = self._mask_assign(upper, u == next_asymptote + np.pi, +np.inf)            
            return lower, upper
        
        else:
            raise RDDLNotImplementedError(
                f'Unary function {name} is not supported.\n' + PST(expr))
    
    @staticmethod
    def _bound_func_power(int1, int2):
        (l1, u1), (l2, u2) = int1, int2        
        lower = np.full(shape=np.shape(l1), fill_value=-np.inf)
        upper = np.full(shape=np.shape(u1), fill_value=+np.inf)
        mask_fn = RDDLIntervalAnalysis._mask_assign
        
        # positive base, means well defined for any real power
        log1 = RDDLIntervalAnalysis._bound_func_monotone(l1, u1, np.log)
        pow = RDDLIntervalAnalysis._bound_arithmetic_expr(log1, int2, '*')
        l1pos, u1pos = RDDLIntervalAnalysis._bound_func_monotone(*pow, np.exp)
        lower = mask_fn(lower, l1 > 0, l1pos, True)
        upper = mask_fn(upper, l1 > 0, u1pos, True)
            
        # otherwise, defined if the power is an integer point
        l2_is_int = np.equal(np.mod(l2, 1), 0)
        u2_is_int = np.equal(np.mod(u2, 1), 0)
        pow_valid = l2_is_int & u2_is_int & (l2 >= 0) & (l2 == u2)
        pow = l2.astype(int) if np.shape(l2) else int(l2)
        pow_even = (np.mod(pow, 2) == 0)
            
        # if the base interval contains 0
        case1 = pow_valid & (0 >= l1) & (0 <= u1)
        lower = mask_fn(lower, case1 & pow_even, 0)
        upper = mask_fn(upper, case1 & (pow == 0), 1)
        upper = mask_fn(upper, case1 & (pow > 0) & pow_even, 
                        np.maximum(l1 ** pow, l2 ** pow), True)
        lower = mask_fn(lower, case1 & ~pow_even, l1 ** pow, True)
        upper = mask_fn(upper, case1 & ~pow_even, l2 ** pow, True)
            
        # if the base is strictly negative
        case2 = pow_valid & (u1 < 0)
        lower = mask_fn(lower, case2 & (pow == 0), 1)
        upper = mask_fn(upper, case2 & (pow == 0), 1)
        lower = mask_fn(lower, case2 & pow_even, u1 ** pow, True)
        upper = mask_fn(upper, case2 & pow_even, l1 ** pow, True)
        lower = mask_fn(lower, case2 & ~pow_even, l1 ** pow, True)
        upper = mask_fn(upper, case2 & ~pow_even, u1 ** pow, True)
        return lower, upper
    
    def _bound_func_binary(self, expr, intervals):
        _, name = expr.etype
        arg1, arg2 = expr.args
        int1 = (l1, u1) = self._bound(arg1, intervals)
        int2 = (l2, u2) = self._bound(arg2, intervals)
        
        # div(x, y) = floor(x / y)
        if name == 'div': 
            l, u = self._bound_arithmetic_expr(int1, int2, '/')
            return self._bound_func_monotone(l, u, np.floor)
        
        # min([a, b], [c, d]) = (min[a, c], min[b, d])
        elif name == 'min':
            return (np.minimum(l1, l2), np.minimum(u1, u2))
        
        # max([a, b], [c, d]) = (max[a, c], max[b, d])
        elif name == 'max':
            return (np.maximum(l1, l2), np.maximum(u1, u2))
        
        # TODO
        elif name == 'mod':
            raise RDDLNotImplementedError(
                f'Binary function {name} is not supported.\n' + PST(expr))
        
        # TODO
        elif name == 'fmod':
            raise RDDLNotImplementedError(
                f'Binary function {name} is not supported.\n' + PST(expr))
        
        elif name == 'pow':
            return self._bound_func_power(int1, int2)
        
        # log[x, b] = log[x] / log[b]
        elif name == 'log':
            if np.any(l2 <= 0):
                raise RDDLNotImplementedError(
                    f'Function {name} with base <= 0 is not supported.\n' + PST(expr))
            if np.any(l1 <= 0):
                raise RDDLNotImplementedError(
                    f'Function {name} with argument <= 0 is not supported.\n' + PST(expr))                    
            log1 = self._bound_func_monotone(l1, u1, np.log)
            log2 = self._bound_func_monotone(l2, u2, np.log)
            return self._bound_arithmetic_expr(log1, log2, '/')
        
        # hypot[x, y] = sqrt[x * x + y * y]
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
                f'Function {name} with {n} arguments is not supported.\n' + 
                PST(expr))
            
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
        intp = (lp, up) = self._bound(pred, intervals)
        int1 = (l1, u1) = self._bound(arg1, intervals)
        int2 = (l2, u2) = self._bound(arg2, intervals)      
        lower = np.minimum(l1, l2)
        upper = np.maximum(u1, u2)
        
        # reduce range if the predicate is known with certainty
        lower = self._mask_assign(lower, lp >= 1, l1, True)
        upper = self._mask_assign(upper, lp >= 1, u1, True)
        lower = self._mask_assign(lower, up <= 0, l2, True)
        upper = self._mask_assign(upper, up <= 0, u2, True)
        return (lower, upper)
            
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
        a, b = args
        intl = (la, ua) = self._bound(a, intervals)
        intu = (lb, ub) = self._bound(b, intervals)
        lower = np.minimum(la, ua)
        upper = np.maximum(lb, ub)
        return (lower, upper)
    
    def _bound_bernoulli(self, expr, intervals):
        args = expr.args
        p, = args
        intp = (lp, up) = self._bound(p, intervals)
        
        lower = np.zeros(shape=np.shape(lp), dtype=int)
        upper = np.ones(shape=np.shape(up), dtype=int)
        lower = self._mask_assign(lower, lp >= 1, 1)
        upper = self._mask_assign(upper, up <= 0, 0)
        return (lower, upper)
    
    def _bound_normal(self, expr, intervals):
        args = expr.args
        mean, var = args
        intm = (lm, um) = self._bound(mean, intervals)
        intv = (lv, uv) = self._bound(var, intervals)
        
        lower = np.full(shape=np.shape(lm), fill_value=-np.inf)
        upper = np.full(shape=np.shape(um), fill_value=+np.inf)
        lower = self._mask_assign(lower, (lv == 0) & (uv == 0), lm, True)
        upper = self._mask_assign(upper, (lv == 0) & (uv == 0), um, True)
        return (lower, upper)
    
    def _bound_poisson(self, expr, intervals):
        args = expr.args
        p, = args
        intp = (lp, up) = self._bound(p, intervals)
        
        lower = np.zeros(shape=np.shape(lp), dtype=int)
        upper = np.full(shape=np.shape(up), fill_value=np.inf)
        return (lower, upper)
    
    def _bound_exponential(self, expr, intervals):
        args = expr.args
        scale, = args
        ints = (ls, us) = self._bound(scale, intervals)
        
        lower = np.zeros(shape=np.shape(ls))
        upper = np.full(shape=np.shape(us), fill_value=np.inf)
        return (lower, upper)
    
    def _bound_weibull(self, expr, intervals):
        args = expr.args
        shape, scale = args
        intsh = (lsh, ush) = self._bound(shape, intervals)
        intsc = (lsc, usc) = self._bound(scale, intervals)
        
        lower = np.zeros(shape=np.shape(lsh))
        upper = np.full(shape=np.shape(ush), fill_value=np.inf)
        return (lower, upper)
    
    def _bound_gamma(self, expr, intervals):
        args = expr.args
        shape, scale = args
        intsh = (lsh, ush) = self._bound(shape, intervals)
        intsc = (lsc, usc) = self._bound(scale, intervals)
        
        lower = np.zeros(shape=np.shape(ls))
        upper = np.full(shape=np.shape(us), fill_value=np.inf)
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
        upper = np.full(shape=np.shape(up), fill_value=np.inf)
        return (lower, upper)
    
    def _bound_pareto(self, expr, intervals):
        args = expr.args
        shape, scale = args
        intsh = (lsh, ush) = self._bound(shape, intervals)
        intsc = (lsc, usc) = self._bound(scale, intervals)
        
        lower = lsc
        upper = np.full(shape=np.shape(usc), fill_value=np.inf)
        return (lower, upper)
    
    def _bound_student(self, expr, intervals):
        args = expr.args
        df, = args
        intd = (ld, ud) = self._bound(df, intervals)
        
        lower = np.full(shape=np.shape(ld), fill_value=-np.inf)
        upper = np.full(shape=np.shape(ud), fill_value=+np.inf)
        return (lower, upper)
    
    def _bound_gumbel(self, expr, intervals):
        args = expr.args
        mean, scale = args
        intm = (lm, um) = self._bound(mean, intervals)
        ints = (ls, us) = self._bound(scale, intervals)
        
        lower = np.full(shape=np.shape(lm), fill_value=-np.inf)
        upper = np.full(shape=np.shape(um), fill_value=+np.inf)
        return (lower, upper)
    
    def _bound_cauchy(self, expr, intervals):
        args = expr.args
        mean, scale = args
        intm = (lm, um) = self._bound(mean, intervals)
        ints = (ls, us) = self._bound(scale, intervals)
        
        lower = np.full(shape=np.shape(lm), fill_value=-np.inf)
        upper = np.full(shape=np.shape(um), fill_value=+np.inf)
        return (lower, upper)
    
    def _bound_gompertz(self, expr, intervals):
        args = expr.args
        shape, scale = args
        intsh = (lsh, ush) = self._bound(shape, intervals)
        intsc = (lsc, usc) = self._bound(scale, intervals)
        
        lower = np.zeros(shape=np.shape(lsh))
        upper = np.full(shape=np.shape(ush), fill_value=np.inf)
        return (lower, upper)
    
    def _bound_chisquare(self, expr, intervals):
        args = expr.args
        df, = args
        intd = (ld, ud) = self._bound(df, intervals)
        
        lower = np.zeros(shape=np.shape(ld))
        upper = np.full(shape=np.shape(ud), fill_value=np.inf)
        return (lower, upper)
    
    def _bound_kumaraswamy(self, expr, intervals):
        args = expr.args
        a, b = args
        inta = (la, ua) = self._bound(a, intervals)
        intb = (lb, ub) = self._bound(b, intervals)
        
        lower = np.zeros(shape=np.shape(la))
        upper = np.ones(shape=np.shape(ua))
        return (lower, upper)
