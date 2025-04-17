import numpy as np
import traceback
from typing import Dict, Optional, Tuple, Union

Interval = Tuple[np.ndarray, np.ndarray]
Bounds = Dict[str, Interval]

from pyRDDLGym.core.compiler.levels import RDDLLevelAnalysis
from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.compiler.tracer import RDDLObjectsTracer
from pyRDDLGym.core.debug.exception import (
    raise_warning,
    print_stack_trace as PST,
    RDDLInvalidNumberOfArgumentsError,
    RDDLNotImplementedError,
    RDDLUndefinedVariableError
)
from pyRDDLGym.core.debug.logger import Logger
from pyRDDLGym.core.simulator import lngamma

# try to load scipy
try:
    import scipy
    import scipy.stats as stats
except Exception:
    raise_warning('failed to import scipy: '
                  'some interval arithmetic operations will fail.', 'red')
    traceback.print_exc()
    scipy = None
    stats = None


class RDDLIntervalAnalysis:
    
    def __init__(self, rddl: RDDLPlanningModel, logger: Optional[Logger]=None) -> None:
        '''Creates a new interval analysis object for the given RDDL domain.
        Bounds on probability distributions are calculated using their exact
        support, which can be unbounded intervals.
        
        :param rddl: the RDDL domain to analyze
        :param logger: to log compilation information during tracing to file
        '''
        self.rddl = rddl
        self.logger = logger
        
        sorter = RDDLLevelAnalysis(rddl, allow_synchronous_state=True, logger=self.logger)
        self.cpf_levels = sorter.compute_levels()
          
        tracer = RDDLObjectsTracer(rddl, logger=self.logger, cpf_levels=self.cpf_levels)
        self.trace = tracer.trace()
        
        self.NUMPY_LITERAL_TO_INT = np.vectorize(self.rddl.object_to_index.__getitem__)
        
    def bound(self, action_bounds: Optional[Bounds]=None, 
              per_epoch: bool=False,
              state_bounds: Optional[Bounds]=None) -> Bounds:
        '''Computes intervals on all fluents and reward for the planning problem.
        
        :param action_bounds: optional bounds on action fluents (defaults to
        a "random" policy otherwise)
        :param per_epoch: if True, the returned bounds are tensors with leading
        dimension indicating the decision epoch; if False, the returned bounds
        are valid across all decision epochs
        :param state_bounds: optional bounds on state fluents (defaults to
        the initial state values otherwise).
        '''
        
        # get initial values as bounds
        intervals = self._bound_initial_values(state_bounds)
        if per_epoch:
            result = {}
        
        # propagate bounds across time
        for _ in range(self.rddl.horizon):
            self._bound_next_epoch(
                intervals, action_bounds=action_bounds, per_epoch=per_epoch)
            if per_epoch:
                for (name, (lower, upper)) in intervals.items():
                    lower_all, upper_all = result.setdefault(name, ([], []))
                    lower_all.append(lower)
                    upper_all.append(upper)
        
        # concatenate bounds across time dimension
        if per_epoch:
            result = {name: (np.asarray(lower), np.asarray(upper))
                      for (name, (lower, upper)) in result.items()}
            return result
        else:
            return intervals
    
    def _bound_initial_values(self, state_bounds=None):
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
            if state_bounds is not None and name in state_bounds:
                intervals[name] = state_bounds[name]
            else:
                intervals[name] = (values, values)
        return intervals
            
    def _bound_next_epoch(self, intervals, action_bounds=None, per_epoch=False):
        rddl = self.rddl 
        
        # update action bounds from user
        if action_bounds is not None:
            for (name, (lower, upper)) in action_bounds.items():
                if name in rddl.action_fluents:
                    params = rddl.variable_params[name]
                    shape = rddl.object_counts(params)
                    if not np.shape(lower):
                        lower = np.full(shape=shape, fill_value=lower)
                    if not np.shape(upper):
                        upper = np.full(shape=shape, fill_value=upper)
                    intervals[name] = (lower, upper)
        
        # trace and bound constraints
        for (i, expr) in enumerate(rddl.invariants):
            self._bound(expr, intervals)
        for (i, expr) in enumerate(rddl.preconditions):
            self._bound(expr, intervals)
            
        # trace and bound CPFs
        for cpfs in self.cpf_levels.values():
            for cpf in cpfs:
                _, expr = rddl.cpfs[cpf]
                lb1, ub1 = self._bound(expr, intervals)
                if per_epoch or cpf in rddl.interm_fluents \
                or cpf in rddl.derived_fluents or cpf in rddl.prev_state:
                    intervals[cpf] = (lb1, ub1)
                else:
                    lb0, ub0 = intervals[cpf]
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
        # TODO: complete randomvector and matrix
        # elif etype == 'randomvector':
        #     result = self._bound_random_vector(expr, intervals)
        # elif etype == 'matrix':
        #     result = self._bound_matrix(expr, intervals)
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
    
    def _cast_enum_values_to_int(self, var, values):
        if self.rddl.variable_ranges[var] in self.rddl.enum_types \
        and not np.issubdtype(np.atleast_1d(values).dtype, np.number):
            return self.NUMPY_LITERAL_TO_INT(values).astype(np.int64)
        else:
            return values
        
    def _bound_constant(self, expr, intervals):
        lower = upper = self.trace.cached_sim_info(expr)
        return (lower, upper)
    
    def _bound_pvar(self, expr, intervals):
        var, _ = expr.args
        
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
        
        # enum literals need to be converted to integers
        lower = self._cast_enum_values_to_int(var, lower)
        upper = self._cast_enum_values_to_int(var, upper)
        
        # propagate the bounds forward
        if cached_info is not None:
            slices, axis, shape, op_code, op_args = cached_info
            if slices:
                for slice in slices:
                    if slice is None:
                        raise RDDLNotImplementedError(
                            'Nested pvariables are not supported.\n' + PST(expr))
                lower = lower[slices]
                upper = upper[slices]
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
        '''Assings a value to a destination array based on a mask.
        
        :param dest: the destination array to assign to
        :param mask: the mask array to determine where to assign
        :param value: the value to assign
        :param mask_value: if True, the value is also masked
        '''
        assert (np.shape(dest) == np.shape(mask))
        if np.shape(dest):
            if mask_value:
                value = value[mask]
            dest[mask] = value
        elif mask:
            dest = value
        return dest
    
    @staticmethod
    def _op_without_overflow(x, y, op):
        if op == '+':
            value = 1. * x + 1. * y
        elif op == '-':
            value = 1. * x - 1. * y
        elif op == '*':
            value = (1. * x) * y
        else:
            raise RDDLNotImplementedError(f'Safe operation {op} is not supported.')
        if np.issubdtype(np.result_type(x), np.integer) \
        and np.issubdtype(np.result_type(y), np.integer) \
        and np.any(np.abs(value) >= 2 ** 64 - 1):
            raise_warning(f'Arguments of operation {op} are integer, '
                          f'but the operation would result in overflow: '
                          f'casting arguments to float.')
            return value
        elif op == '+':
            return x + y
        elif op == '-':
            return x - y
        elif op == '*':
            return x * y
        
    @staticmethod
    def bound_arithmetic_expr(int1: Interval, int2: Interval, op: str) -> Interval:
        (l1, u1), (l2, u2) = int1, int2
        op_fn = RDDLIntervalAnalysis._op_without_overflow
        
        # [a, b] + [c, d] = [a + c, b + d]
        if op == '+':
            lower = op_fn(l1, l2, '+')
            upper = op_fn(u1, u2, '+')
            return (lower, upper)
        
        # [a, b] - [c, d] = [a, d] - [b, c]
        elif op == '-':
            lower = op_fn(l1, u2, '-')
            upper = op_fn(u1, l2, '-')
            return (lower, upper)
        
        # [a, b] * [c, d] 
        elif op == '*':
            parts = [op_fn(l1, l2, '*'), op_fn(l1, u2, '*'), 
                     op_fn(u1, l2, '*'), op_fn(u1, u2, '*')]
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
            return RDDLIntervalAnalysis.bound_arithmetic_expr(int1, (l2_inv, u2_inv), '*')
        
        else:
            raise RDDLNotImplementedError(f'Arithmetic operator {op} is not supported.\n')
        
    def _bound_arithmetic(self, expr, intervals):
        _, op = expr.etype
        args = expr.args        
        n = len(args)
        
        # unary negation
        if n == 1 and op == '-':
            arg, = args
            lower, upper = self._bound(arg, intervals)
            return (-1 * upper, -1 * lower)
        
        # binary operator
        elif n == 2:
            lhs, rhs = args
            int1 = self._bound(lhs, intervals)
            int2 = self._bound(rhs, intervals)
            return self.bound_arithmetic_expr(int1, int2, op)
        
        # ternary and higher order operator
        elif n >= 2 and op in ('+', '*'):
            int_res = self._bound(args[0], intervals)
            for arg in args[1:]:
                int_arg = self._bound(arg, intervals)
                int_res = self.bound_arithmetic_expr(int_res, int_arg, op)
            return int_res
        
        else:
            raise RDDLInvalidNumberOfArgumentsError(
                f'Arithmetic operator {op} does not have the required '
                f'number of arguments.\n' + PST(expr))
    
    # ===========================================================================
    # relational
    # ===========================================================================
    
    @staticmethod
    def bound_relational_expr(int1: Interval, int2: Interval, op: str) -> Interval:
        (l1, u1), (l2, u2) = int1, int2
        lower = np.zeros(np.shape(l1), dtype=np.int64)
        upper = np.ones(np.shape(u1), dtype=np.int64)
        mask_fn = RDDLIntervalAnalysis._mask_assign
        
        if op == '>=':
            lower = mask_fn(lower, l1 >= u2, 1)
            upper = mask_fn(upper, u1 < l2, 0)
            return (lower, upper)
        elif op == '>':
            lower = mask_fn(lower, l1 > u2, 1)
            upper = mask_fn(upper, u1 <= l2, 0)
            return (lower, upper)
        elif op == '==':
            lower = mask_fn(lower, (l1 == u1) & (l2 == u2) & (l1 == u2), 1)
            upper = mask_fn(upper, (u1 < l2) | (l1 > u2), 0)
            return (lower, upper)
        elif op == '~=':
            lower = mask_fn(lower, (u1 < l2) | (l1 > u2), 1)
            upper = mask_fn(upper, (l1 == u1) & (l2 == u2) & (l1 == u2), 0)
            return (lower, upper)
        elif op == '<':
            return RDDLIntervalAnalysis.bound_relational_expr(int2, int1, '>')
        elif op == '<=':
            return RDDLIntervalAnalysis.bound_relational_expr(int2, int1, '>=')
        else:
            raise RDDLNotImplementedError(f'Relational operator {op} is not supported.\n')
         
    def _bound_relational(self, expr, intervals):
        _, op = expr.etype
        args = expr.args
        lhs, rhs = args
        int1 = self._bound(lhs, intervals)
        int2 = self._bound(rhs, intervals)
        return self.bound_relational_expr(int1, int2, op)
    
    # ===========================================================================
    # logical
    # ===========================================================================
    
    @staticmethod
    def bound_logical_expr(int1: Interval, int2: Optional[Interval], op: str) -> Interval:
        if op == '&':
            op = '^'

        if op == '^':
            return RDDLIntervalAnalysis.bound_arithmetic_expr(int1, int2, '*')
        
        # x | y = ~(~x & ~y)
        elif op == '|':
            not1 = RDDLIntervalAnalysis.bound_logical_expr(int1, None, '~')
            not2 = RDDLIntervalAnalysis.bound_logical_expr(int2, None, '~')
            not12 = RDDLIntervalAnalysis.bound_logical_expr(not1, not2, '^')
            return RDDLIntervalAnalysis.bound_logical_expr(not12, None, '~')
        
        # x ~ y = (x | y) ^ ~(x & y)
        elif op == '~':
            if int2 is None:
                l1, u1 = int1
                return (1 - u1, 1 - l1)
            else:
                or12 = RDDLIntervalAnalysis.bound_logical_expr(int1, int2, '|')
                and12 = RDDLIntervalAnalysis.bound_logical_expr(int1, int2, '^')
                not12 = RDDLIntervalAnalysis.bound_logical_expr(and12, None, '~')
                return RDDLIntervalAnalysis.bound_logical_expr(or12, not12, '^')
        
        # x => y = ~x | y
        elif op == '=>':
            not1 = RDDLIntervalAnalysis.bound_logical_expr(int1, None, '~')
            return RDDLIntervalAnalysis.bound_logical_expr(not1, int2, '|')
        
        # x <=> y = x == y
        elif op == '<=>':
            return RDDLIntervalAnalysis.bound_relational_expr(int1, int2, '==')
        
        else:
            raise RDDLNotImplementedError(f'Logical operator {op} is not supported.\n')
           
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
            return self.bound_logical_expr(int1, None, op)
        
        # binary operator
        elif n == 2:
            lhs, rhs = args
            int1 = self._bound(lhs, intervals)
            int2 = self._bound(rhs, intervals)
            return self.bound_logical_expr(int1, int2, op)
        
        # ternary and higher order operator
        elif n >= 2 and op in ('^', '|'):
            int_res = self._bound(args[0], intervals)
            for arg in args[1:]:
                int_arg = self._bound(arg, intervals)
                int_res = self.bound_logical_expr(int_res, int_arg, op)
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
        op_fn = RDDLIntervalAnalysis._op_without_overflow
        parts = [op_fn(l1, l2, '*'), op_fn(l1, u2, '*'), 
                 op_fn(u1, l2, '*'), op_fn(u1, u2, '*')]
        lower = min(parts)
        upper = max(parts)
        return (lower, upper)
        
    @staticmethod
    def _bound_or_scalar(int1, int2):
        (l1, u1), (l2, u2) = int1, int2
        l1, u1 = (1 - u1, 1 - l1)
        l2, u2 = (1 - u2, 1 - l2)        
        parts = [l1 * l2, l1 * u2, u1 * l2, u1 * u2] 
        lower = 1 - max(parts)
        upper = 1 - min(parts)
        return (lower, upper)        
    
    NUMPY_PROD_FUNC = np.frompyfunc(_bound_product_scalar, nin=2, nout=1)    
    NUMPY_OR_FUNC = np.frompyfunc(_bound_or_scalar, nin=2, nout=1)
        
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
    
    @staticmethod
    def bound_aggregation_expr(interval: Interval, op: str, 
                               axes: Union[int, Tuple[int, ...]]) -> Interval:
        lower, upper = interval
        if op == 'sum':
            return (np.sum(lower, axis=axes), np.sum(upper, axis=axes))
        elif op == 'avg':
            return (np.mean(lower, axis=axes), np.mean(upper, axis=axes))
        elif op == 'prod':
            return RDDLIntervalAnalysis._bound_aggregation_func(
                lower, upper, axes, RDDLIntervalAnalysis.NUMPY_PROD_FUNC)
        elif op == 'minimum':
            return (np.min(lower, axis=axes), np.min(upper, axis=axes))        
        elif op == 'maximum':
            return (np.max(lower, axis=axes), np.max(upper, axis=axes))        
        elif op == 'forall':
            return RDDLIntervalAnalysis._bound_aggregation_func(
                lower, upper, axes, RDDLIntervalAnalysis.NUMPY_PROD_FUNC)        
        elif op == 'exists':
            return RDDLIntervalAnalysis._bound_aggregation_func(
                lower, upper, axes, RDDLIntervalAnalysis.NUMPY_OR_FUNC)
        else:        
            raise RDDLNotImplementedError(
                f'Aggregation operator {op} is not supported.\n')

    def _bound_aggregation(self, expr, intervals):
        _, op = expr.etype
        *_, arg = expr.args        
        interval = self._bound(arg, intervals)
        _, axes = self.trace.cached_sim_info(expr)
        return self.bound_aggregation_expr(interval, op, axes)
    
    # ===========================================================================
    # function
    # ===========================================================================
    
    UNARY_MONOTONE = {
        'sgn': lambda x: np.sign(x).astype(np.int64),
        'round': lambda x: np.round(x).astype(np.int64),
        'floor': lambda x: np.floor(x).astype(np.int64),
        'ceil': lambda x: np.ceil(x).astype(np.int64),
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
    def _bound_func_monotone(interval, func):
        l, u = interval
        fl, fu = func(l), func(u)
        lower = np.minimum(fl, fu)
        upper = np.maximum(fl, fu)
        return (lower, upper)
    
    @staticmethod
    def _bound_func_u_shaped(interval, x_crit, func):
        l, u = interval
        fl, fu = func(l), func(u)
        f_crit = func(x_crit)
        lower = np.full(shape=np.shape(l), fill_value=f_crit)
        upper = np.maximum(fl, fu)
        mask_fn = RDDLIntervalAnalysis._mask_assign
        lower = mask_fn(lower, l >= x_crit, fl, True)
        upper = mask_fn(upper, l >= x_crit, fu, True)
        lower = mask_fn(lower, u <= x_crit, fu, True)
        upper = mask_fn(upper, u <= x_crit, fl, True)
        return (lower, upper)
    
    @staticmethod
    def _next_multiple(start, offset, period=2 * np.pi):
        multiples = -np.floor_divide(start - offset, -period)
        next_up = period * multiples + offset
        return next_up
    
    @staticmethod
    def bound_unary_func(interval: Interval, name: str) -> Interval:
        mask_fn = RDDLIntervalAnalysis._mask_assign
        l, u = interval

        # f is monotone, x in [a, b] => f(x) in [f(a), f(b)]
        if name in RDDLIntervalAnalysis.UNARY_MONOTONE:
            return RDDLIntervalAnalysis._bound_func_monotone(
                interval, RDDLIntervalAnalysis.UNARY_MONOTONE[name])
        
        # f is a u-shaped function
        elif name in RDDLIntervalAnalysis.UNARY_U_SHAPED:
            func, x_crit = RDDLIntervalAnalysis.UNARY_U_SHAPED[name]
            return RDDLIntervalAnalysis._bound_func_u_shaped(interval, x_crit, func)
        
        # special functions sin
        elif name == 'sin':
            
            # by default evaluate at end points
            sin_l, sin_u = np.sin(l), np.sin(u)
            lower = np.minimum(sin_l, sin_u)
            upper = np.maximum(sin_l, sin_u)
            
            # any infinity results in [-1, 1]
            has_inf = (l == -np.inf) | (u == np.inf)
            lower = mask_fn(lower, has_inf, -1.0)
            upper = mask_fn(upper, has_inf, +1.0)
            
            # find critical points
            next_max = RDDLIntervalAnalysis._next_multiple(l, np.pi / 2)
            next_min = RDDLIntervalAnalysis._next_multiple(l, 3 * np.pi / 2)
            lower = mask_fn(lower, next_min <= u, -1.0)
            upper = mask_fn(upper, next_max <= u, +1.0)
            return lower, upper
        
        # special functions cos
        elif name == 'cos':
            
            # by default evaluate at end points
            cos_l, cos_u = np.cos(l), np.cos(u)
            lower = np.minimum(cos_l, cos_u)
            upper = np.maximum(cos_l, cos_u)
            
            # any infinity results in [-1, 1]
            has_inf = (l == -np.inf) | (u == np.inf)
            lower = mask_fn(lower, has_inf, -1.0)
            upper = mask_fn(upper, has_inf, +1.0)
            
            # find critical points
            next_max = RDDLIntervalAnalysis._next_multiple(l, 0)
            next_min = RDDLIntervalAnalysis._next_multiple(l, np.pi)
            lower = mask_fn(lower, next_min <= u, -1.0)
            upper = mask_fn(upper, next_max <= u, +1.0)
            return lower, upper
        
        # special functions tan
        elif name == 'tan':
            
            # by default evaluate at end points
            tan_l, tan_u = np.tan(l), np.tan(u)
            lower = np.minimum(tan_l, tan_u)
            upper = np.maximum(tan_l, tan_u)
            
            # any infinity results in [-inf, inf]
            has_inf = (l == -np.inf) | (u == np.inf)
            lower = mask_fn(lower, has_inf, -np.inf)
            upper = mask_fn(upper, has_inf, +np.inf)
            
            # an asymptote is inside the open interval, set to [-inf, inf]
            next_asymptote = RDDLIntervalAnalysis._next_multiple(l, np.pi / 2, np.pi)
            asymptote_in_open = (l < next_asymptote) & (u > next_asymptote)
            lower = mask_fn(lower, asymptote_in_open, -np.inf)
            upper = mask_fn(upper, asymptote_in_open, +np.inf)
            
            # an asymptote is on an edge
            lower = mask_fn(lower, l == next_asymptote, -np.inf)
            upper = mask_fn(upper, u == next_asymptote, +np.inf)
            upper = mask_fn(upper, u == next_asymptote + np.pi, +np.inf)            
            return lower, upper
        
        else:
            raise RDDLNotImplementedError(f'Unary function {name} is not supported.\n')

    # TODO: fix overflow
    @staticmethod
    def _bound_func_power(int1, int2):
        (l1, u1), (l2, u2) = int1, int2        
        lower = np.full(shape=np.shape(l1), fill_value=-np.inf, dtype=np.float64)
        upper = np.full(shape=np.shape(u1), fill_value=+np.inf, dtype=np.float64)
        mask_fn = RDDLIntervalAnalysis._mask_assign
        
        # positive base, means well defined for any real power
        log1 = RDDLIntervalAnalysis.bound_unary_func(int1, 'ln')
        pow_ = RDDLIntervalAnalysis.bound_arithmetic_expr(log1, int2, '*')
        l1pos, u1pos = RDDLIntervalAnalysis.bound_unary_func(pow_, 'exp')
        lower = mask_fn(lower, l1 > 0, l1pos, True)
        upper = mask_fn(upper, l1 > 0, u1pos, True)
            
        # otherwise, defined if the power is an integer point
        l2_is_int = np.equal(np.mod(l2, 1), 0)
        u2_is_int = np.equal(np.mod(u2, 1), 0)
        pow_valid = l2_is_int & u2_is_int & (l2 >= 0) & (l2 == u2)
        pow_ = l2.astype(np.int64) if np.shape(l2) else int(l2)
        pow_even = (np.mod(pow_, 2) == 0)
            
        # if the base interval contains 0
        case1 = pow_valid & (0 >= l1) & (0 <= u1)
        lower = mask_fn(lower, case1 & pow_even, 0)
        upper = mask_fn(upper, case1 & (pow_ == 0), 1)
        upper = mask_fn(upper, case1 & (pow_ > 0) & pow_even, 
                        np.maximum(l1 ** pow_, l2 ** pow_), True)
        lower = mask_fn(lower, case1 & ~pow_even, l1 ** pow_, True)
        upper = mask_fn(upper, case1 & ~pow_even, l2 ** pow_, True)
            
        # if the base is strictly negative
        case2 = pow_valid & (u1 < 0)
        lower = mask_fn(lower, case2 & (pow_ == 0), 1)
        upper = mask_fn(upper, case2 & (pow_ == 0), 1)
        lower = mask_fn(lower, case2 & pow_even, u1 ** pow_, True)
        upper = mask_fn(upper, case2 & pow_even, l1 ** pow_, True)
        lower = mask_fn(lower, case2 & ~pow_even, l1 ** pow_, True)
        upper = mask_fn(upper, case2 & ~pow_even, u1 ** pow_, True)
        return (lower, upper)
    
    @staticmethod
    def bound_binary_func(int1: Interval, int2: Interval, name: str) -> Interval:
        (l1, u1), (l2, u2) = int1, int2

        # div(x, y) = floor(x / y)
        if name == 'div': 
            lu = RDDLIntervalAnalysis.bound_arithmetic_expr(int1, int2, '/')
            return RDDLIntervalAnalysis.bound_unary_func(lu, 'floor')
        
        # min([a, b], [c, d]) = (min[a, c], min[b, d])
        elif name == 'min':
            return (np.minimum(l1, l2), np.minimum(u1, u2))
        
        # max([a, b], [c, d]) = (max[a, c], max[b, d])
        elif name == 'max':
            return (np.maximum(l1, l2), np.maximum(u1, u2))
        
        # power
        elif name == 'pow':
            return RDDLIntervalAnalysis._bound_func_power(int1, int2)
        
        # log[x, b] = log[x] / log[b]
        elif name == 'log':
            if np.any(l2 <= 0):
                raise ValueError(f'Function {name} with base <= 0 is not supported.\n')
            if np.any(l1 <= 0):
                raise ValueError(f'Function {name} with argument <= 0 is not supported.\n')                    
            log1 = RDDLIntervalAnalysis.bound_unary_func(int1, 'ln')
            log2 = RDDLIntervalAnalysis.bound_unary_func(int2, 'ln')
            return RDDLIntervalAnalysis.bound_arithmetic_expr(log1, log2, '/')
        
        # hypot[x, y] = sqrt[x * x + y * y]
        elif name == 'hypot': 
            pow1 = RDDLIntervalAnalysis._bound_func_u_shaped(int1, 0, np.square)
            pow2 = RDDLIntervalAnalysis._bound_func_u_shaped(int2, 0, np.square)
            lu = RDDLIntervalAnalysis.bound_arithmetic_expr(pow1, pow2, '+')
            return RDDLIntervalAnalysis.bound_unary_func(lu, 'sqrt')
        
        else:
            raise RDDLNotImplementedError(f'Binary function {name} is not supported.\n')        

    def _bound_func(self, expr, intervals):
        _, name = expr.etype
        args = expr.args
        n = len(args)
        
        # unary function
        if n == 1:
            _, name = expr.etype
            arg, = expr.args
            lu = self._bound(arg, intervals)        
            return self.bound_unary_func(lu, name)
        
        # binary function
        elif n == 2:
            _, name = expr.etype
            arg1, arg2 = expr.args
            int1 = self._bound(arg1, intervals)
            int2 = self._bound(arg2, intervals)
            return self.bound_binary_func(int1, int2, name)
        
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
        (lp, up) = self._bound(pred, intervals)
        (l1, u1) = self._bound(arg1, intervals)
        (l2, u2) = self._bound(arg2, intervals)      
        lower = np.minimum(l1, l2)
        upper = np.maximum(u1, u2)
        
        # reduce range if the predicate is known with certainty
        mask_fn = RDDLIntervalAnalysis._mask_assign
        lower = mask_fn(lower, lp >= 1, l1, True)
        upper = mask_fn(upper, lp >= 1, u1, True)
        lower = mask_fn(lower, up <= 0, l2, True)
        upper = mask_fn(upper, up <= 0, u2, True)
        return (lower, upper)
            
    def _bound_switch(self, expr, intervals):
        
        # compute bounds on predicate
        pred, *_ = expr.args
        lower_pred, upper_pred = self._bound(pred, intervals)  
        
        # compute bounds on case expressions        
        cases, default = self.trace.cached_sim_info(expr)  
        def_bounds = None if default is None else self._bound(default, intervals)
        bounds = [
            (def_bounds if arg is None else self._bound(arg, intervals))
            for arg in cases
        ]
        
        # (pred == @o1) * expr1 + (pred == @o2) * expr2 ...
        mask_fn = RDDLIntervalAnalysis._mask_assign
        final_bounds = None
        for (i, (lower_expr, upper_expr)) in enumerate(bounds):
            lower_eq = np.zeros(np.shape(lower_expr), dtype=np.int64)
            upper_eq = np.ones(np.shape(upper_expr), dtype=np.int64)
            lower_eq = mask_fn(lower_eq, (lower_pred == upper_pred) & (lower_pred == i), 1)
            upper_eq = mask_fn(upper_eq, (upper_pred < i) | (lower_pred > i), 0)
            bounds_obj = RDDLIntervalAnalysis.bound_arithmetic_expr(
                (lower_eq, upper_eq), (lower_expr, upper_expr), '*')      
            if final_bounds is None:
                final_bounds = bounds_obj
            else:
                final_bounds = RDDLIntervalAnalysis.bound_arithmetic_expr(
                    final_bounds, bounds_obj, '+')
        return final_bounds
    
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
        (lower, _) = self._bound(a, intervals)
        (_, upper) = self._bound(b, intervals)
        return (lower, upper)
    
    def _bound_bernoulli(self, expr, intervals):
        args = expr.args
        p, = args
        (lp, up) = self._bound(p, intervals)
        
        lower = np.zeros(shape=np.shape(lp), dtype=np.int64)
        upper = np.ones(shape=np.shape(up), dtype=np.int64)
        lower = self._mask_assign(lower, lp >= 1, 1)
        upper = self._mask_assign(upper, up <= 0, 0)
        return (lower, upper)
    
    def _bound_normal(self, expr, intervals):
        args = expr.args
        mean, var = args
        (lm, um) = self._bound(mean, intervals)
        (lv, uv) = self._bound(var, intervals)
        
        lower = np.full(shape=np.shape(lm), fill_value=-np.inf, dtype=np.float64)
        upper = np.full(shape=np.shape(um), fill_value=+np.inf, dtype=np.float64)
        lower = self._mask_assign(lower, (lv == 0) & (uv == 0), lm, True)
        upper = self._mask_assign(upper, (lv == 0) & (uv == 0), um, True)
        return (lower, upper)
    
    def _bound_poisson(self, expr, intervals):
        args = expr.args
        p, = args
        (lp, up) = self._bound(p, intervals)
        
        lower = np.zeros(shape=np.shape(lp), dtype=np.int64)
        upper = np.full(shape=np.shape(up), fill_value=np.inf, dtype=np.float64)
        return (lower, upper)
    
    def _bound_exponential(self, expr, intervals):
        args = expr.args
        scale, = args
        (ls, us) = self._bound(scale, intervals)
        
        lower = np.zeros(shape=np.shape(ls), dtype=np.float64)
        upper = np.full(shape=np.shape(us), fill_value=np.inf, dtype=np.float64)
        return (lower, upper)
    
    def _bound_weibull(self, expr, intervals):
        args = expr.args
        shape, scale = args
        (lsh, ush) = self._bound(shape, intervals)
        _ = self._bound(scale, intervals)
        
        lower = np.zeros(shape=np.shape(lsh), dtype=np.float64)
        upper = np.full(shape=np.shape(ush), fill_value=np.inf, dtype=np.float64)
        return (lower, upper)
    
    def _bound_gamma(self, expr, intervals):
        args = expr.args
        shape, scale = args
        (lsh, ush) = self._bound(shape, intervals)
        _ = self._bound(scale, intervals)
        
        lower = np.zeros(shape=np.shape(lsh), dtype=np.float64)
        upper = np.full(shape=np.shape(ush), fill_value=np.inf, dtype=np.float64)
        return (lower, upper)
    
    def _bound_binomial(self, expr, intervals):
        args = expr.args
        n, p = args
        (ln, un) = self._bound(n, intervals)
        (lp, up) = self._bound(p, intervals)
        
        lower = np.zeros(shape=np.shape(ln), dtype=np.int64)
        upper = np.copy(un)
        lower = self._mask_assign(lower, (lp >= 1) & (ln > 0), ln, True)
        upper = self._mask_assign(upper, (up <= 0), 0)
        return (lower, upper)
    
    def _bound_negative_binomial(self, expr, intervals):
        args = expr.args
        n, p = args
        _ = self._bound(n, intervals)
        (lp, up) = self._bound(p, intervals)

        lower = np.ones(shape=np.shape(lp), dtype=np.int64)
        upper = np.full(shape=np.shape(up), fill_value=np.inf, dtype=np.float64)
        lower = self._mask_assign(lower, (up <= 0), np.inf)
        upper = self._mask_assign(upper, (lp >= 1), 0)
        return (lower, upper)

    def _bound_beta(self, expr, intervals):
        args = expr.args
        shape, rate = args
        (ls, us) = self._bound(shape, intervals)
        _ = self._bound(rate, intervals)
        
        lower = np.zeros(shape=np.shape(ls), dtype=np.float64)
        upper = np.ones(shape=np.shape(us), dtype=np.float64)
        return (lower, upper)
    
    def _bound_geometric(self, expr, intervals):
        args = expr.args
        p, = args
        
        (lp, up) = self._bound(p, intervals)
        lower = np.ones(shape=np.shape(lp), dtype=np.int64)
        upper = np.full(shape=np.shape(up), fill_value=np.inf, dtype=np.float64)
        return (lower, upper)
    
    def _bound_pareto(self, expr, intervals):
        args = expr.args
        shape, scale = args
        _ = self._bound(shape, intervals)
        (lsc, usc) = self._bound(scale, intervals)
        
        lower = lsc
        upper = np.full(shape=np.shape(usc), fill_value=np.inf, dtype=np.float64)
        return (lower, upper)
    
    def _bound_student(self, expr, intervals):
        args = expr.args
        df, = args
        (ld, ud) = self._bound(df, intervals)
        
        lower = np.full(shape=np.shape(ld), fill_value=-np.inf, dtype=np.float64)
        upper = np.full(shape=np.shape(ud), fill_value=+np.inf, dtype=np.float64)
        return (lower, upper)
    
    def _bound_gumbel(self, expr, intervals):
        args = expr.args
        mean, scale = args
        (lm, um) = self._bound(mean, intervals)
        _ = self._bound(scale, intervals)
        
        lower = np.full(shape=np.shape(lm), fill_value=-np.inf, dtype=np.float64)
        upper = np.full(shape=np.shape(um), fill_value=+np.inf, dtype=np.float64)
        return (lower, upper)
    
    def _bound_laplace(self, expr, intervals):
        args = expr.args
        mean, scale = args
        (lm, um) = self._bound(mean, intervals)
        _ = self._bound(scale, intervals)
        
        lower = np.full(shape=np.shape(lm), fill_value=-np.inf, dtype=np.float64)
        upper = np.full(shape=np.shape(um), fill_value=+np.inf, dtype=np.float64)
        return (lower, upper)
    
    def _bound_cauchy(self, expr, intervals):
        args = expr.args
        mean, scale = args
        (lm, um) = self._bound(mean, intervals)
        _ = self._bound(scale, intervals)
        
        lower = np.full(shape=np.shape(lm), fill_value=-np.inf, dtype=np.float64)
        upper = np.full(shape=np.shape(um), fill_value=+np.inf, dtype=np.float64)
        return (lower, upper)
    
    def _bound_gompertz(self, expr, intervals):
        args = expr.args
        shape, scale = args
        (lsh, ush) = self._bound(shape, intervals)
        _ = self._bound(scale, intervals)
        
        lower = np.zeros(shape=np.shape(lsh), dtype=np.float64)
        upper = np.full(shape=np.shape(ush), fill_value=np.inf, dtype=np.float64)
        return (lower, upper)
    
    def _bound_chisquare(self, expr, intervals):
        args = expr.args
        df, = args
        (ld, ud) = self._bound(df, intervals)
        
        lower = np.zeros(shape=np.shape(ld), dtype=np.float64)
        upper = np.full(shape=np.shape(ud), fill_value=np.inf, dtype=np.float64)
        return (lower, upper)
    
    def _bound_kumaraswamy(self, expr, intervals):
        args = expr.args
        a, b = args
        (la, ua) = self._bound(a, intervals)
        _ = self._bound(b, intervals)
        
        lower = np.zeros(shape=np.shape(la), dtype=np.float64)
        upper = np.ones(shape=np.shape(ua), dtype=np.float64)
        return (lower, upper)
    
    # ===========================================================================
    # random variables with enum support
    # ===========================================================================
    
    def _bound_discrete_helper(self, bounds):
        lower = np.full(shape=np.shape(bounds[0][0]), fill_value=len(bounds), dtype=np.int64)
        upper = np.full(shape=np.shape(bounds[0][1]), fill_value=-1, dtype=np.int64)
        for (index, (lower_prob, upper_prob)) in enumerate(bounds):
            nonzero = upper_prob > 0
            lower[nonzero] = np.minimum(lower[nonzero], index)
            upper[nonzero] = np.maximum(upper[nonzero], index)
        return lower, upper
        
    def _bound_discrete(self, expr, intervals, unnorm):
        sorted_args = self.trace.cached_sim_info(expr)
        bounds = [self._bound(arg, intervals) for arg in sorted_args]
        return self._bound_discrete_helper(bounds) 
    
    def _bound_discrete_pvar(self, expr, intervals, unnorm):
        _, args = expr.args
        arg, = args
        lower_prob, upper_prob = self._bound(arg, intervals)
        bounds = [(lower_prob[..., i], upper_prob[..., i])
                  for i in range(lower_prob.shape[-1])]
        return self._bound_discrete_helper(bounds)


class RDDLIntervalAnalysisMean(RDDLIntervalAnalysis):
    '''Interval analysis that replaces distributions with their mean.
    '''
    
    def _bound_uniform(self, expr, intervals):
        args = expr.args
        a, b = args
        lua = self._bound(a, intervals)
        lub = self._bound(b, intervals)
        lsum, usum = self.bound_arithmetic_expr(lua, lub, '+')
        return (lsum / 2, usum / 2)
    
    def _bound_bernoulli(self, expr, intervals):
        args = expr.args
        p, = args
        lp, up = self._bound(p, intervals)
        return (np.clip(lp, 0., 1.), np.clip(up, 0., 1.))
    
    def _bound_normal(self, expr, intervals):
        args = expr.args
        mean, var = args
        lum = self._bound(mean, intervals)
        _ = self._bound(var, intervals)
        return lum
    
    def _bound_poisson(self, expr, intervals):
        args = expr.args
        p, = args
        lp, up = self._bound(p, intervals)
        return (np.maximum(lp, 0.), np.maximum(up, 0.))
    
    def _bound_exponential(self, expr, intervals):
        args = expr.args
        scale, = args
        ls, us = self._bound(scale, intervals)
        return (np.maximum(ls, 0.), np.maximum(us, 0.))
    
    def _bound_weibull(self, expr, intervals):
        args = expr.args
        shape, scale = args
        (lsh, ush) = lush = self._bound(shape, intervals)
        lusc = self._bound(scale, intervals)
        
        # scale * gamma(1 + 1 / shape)
        one = (np.ones_like(lsh), np.ones_like(ush))
        shinv = self.bound_arithmetic_expr(one, lush, '/')
        shinvp1 = self.bound_arithmetic_expr(one, shinv, '+')
        lugam = self.bound_unary_func(shinvp1, 'gamma')
        lower, upper = self.bound_arithmetic_expr(lusc, lugam, '*')
        lower, upper = np.maximum(lower, 0.), np.maximum(upper, 0.)
        return (lower, upper)
    
    def _bound_gamma(self, expr, intervals):
        args = expr.args
        shape, scale = args
        lush = self._bound(shape, intervals)
        lusc = self._bound(scale, intervals)
        
        # shape * scale
        lower, upper = RDDLIntervalAnalysis.bound_arithmetic_expr(lush, lusc, '*')
        lower, upper = np.maximum(lower, 0.), np.maximum(upper, 0.)
        return (lower, upper)
    
    def _bound_binomial(self, expr, intervals):
        args = expr.args
        n, p = args
        lun = self._bound(n, intervals)
        lup = self._bound(p, intervals)
        
        # n * p
        lower, upper = RDDLIntervalAnalysis.bound_arithmetic_expr(lun, lup, '*')
        lower, upper = np.maximum(lower, 0.), np.maximum(upper, 0.)
        return (lower, upper)
    
    def _bound_negative_binomial(self, expr, intervals):
        args = expr.args
        n, p = args
        lun = self._bound(n, intervals)
        lup = (lp, up) = self._bound(p, intervals)

        # n * (1 - p) / p
        numer = RDDLIntervalAnalysis.bound_arithmetic_expr(lun, (1 - up, 1 - lp), '*')
        lower, upper = RDDLIntervalAnalysis.bound_arithmetic_expr(numer, lup, '/')
        lower, upper = np.maximum(lower, 0), np.maximum(upper, 0)
        return (lower, upper)

    def _bound_beta(self, expr, intervals):
        args = expr.args
        shape, rate = args
        lush = self._bound(shape, intervals)
        lur = self._bound(rate, intervals)
        
        # a / (a + b)
        lusum = self.bound_arithmetic_expr(lush, lur, '+')
        lower, upper = self.bound_arithmetic_expr(lush, lusum, '/')
        lower, upper = np.clip(lower, 0., 1.), np.clip(upper, 0., 1.)
        return (lower, upper)
    
    def _bound_geometric(self, expr, intervals):
        args = expr.args
        p, = args        
        (lp, up) = self._bound(p, intervals)
        
        # 1 / p
        ones = (np.ones_like(lp), np.ones_like(up))
        lower, upper = RDDLIntervalAnalysis.bound_arithmetic_expr(ones, (lp, up), '/')
        lower, upper = np.maximum(lower, 1.), np.maximum(upper, 1.)
        return (lower, upper)
    
    def _bound_pareto(self, expr, intervals):
        args = expr.args
        shape, scale = args
        (lsh, ush) = self._bound(shape, intervals)
        lusc = self._bound(scale, intervals)
        
        # shape * scale / (shape - 1)
        lush = np.maximum(lsh, 1.), np.maximum(ush, 1.)
        ones = (np.ones_like(lsh), np.ones_like(ush))
        numer = self.bound_arithmetic_expr(lush, lusc, '*')
        denom = self.bound_arithmetic_expr(lush, ones, '-')
        return self.bound_arithmetic_expr(numer, denom, '/')
    
    def _bound_student(self, expr, intervals):
        args = expr.args
        df, = args
        (ld, ud) = self._bound(df, intervals)
        lower = np.zeros_like(ld)
        upper = np.zeros_like(ud)
        return (lower, upper)
    
    def _bound_gumbel(self, expr, intervals):
        args = expr.args
        mean, scale = args
        lum = self._bound(mean, intervals)
        lusc = self._bound(scale, intervals)
        
        # mean + euler-mascheroni * scale
        euler_masch = (0.577215664901532, 0.577215664901532)
        scaled_gumbel = self.bound_arithmetic_expr(euler_masch, lusc, '*')
        return self.bound_arithmetic_expr(scaled_gumbel, lum, '+')
    
    def _bound_laplace(self, expr, intervals):
        args = expr.args
        mean, scale = args
        lum = self._bound(mean, intervals)
        _ = self._bound(scale, intervals)
        return lum
    
    def _bound_cauchy(self, expr, intervals):
        args = expr.args
        mean, scale = args
        lum = self._bound(mean, intervals)
        _ = self._bound(scale, intervals)
        return lum
    
    def _bound_gompertz(self, expr, intervals):
        args = expr.args
        shape, scale = args
        (lsh, ush) = self._bound(shape, intervals)
        lusc = self._bound(scale, intervals)

        # exp(shape) * Ei(-shape) / scale
        ei = self._bound_func_monotone((-ush, -lsh), scipy.special.expi)
        exp = self.bound_unary_func((lsh, ush), 'exp')
        prod = self.bound_arithmetic_expr(exp, ei, '*')
        return self.bound_arithmetic_expr(prod, lusc, '/')
    
    def _bound_chisquare(self, expr, intervals):
        args = expr.args
        df, = args
        ldf, udf = self._bound(df, intervals)
        lower, upper = np.maximum(ldf, 0.), np.maximum(udf, 0.)
        return (lower, upper)
    
    def _bound_kumaraswamy(self, expr, intervals):
        args = expr.args
        a, b = args
        (la, ua) = self._bound(a, intervals)
        lub = self._bound(b, intervals)

        one = (np.ones_like(la), np.ones_like(ua))
        ainv = self.bound_arithmetic_expr(one, (la, ua), '/')
        ainvp1 = self.bound_arithmetic_expr(one, ainv, '+')
        sumab = self.bound_arithmetic_expr(ainvp1, lub, '+')
        agam = self.bound_unary_func(ainvp1, 'gamma')
        bgam = self.bound_unary_func(lub, 'gamma')
        sumgam = self.bound_unary_func(sumab, 'gamma')
        numer = self.bound_arithmetic_expr(
            lub, self.bound_arithmetic_expr(agam, bgam, '*'), '*')
        return self.bound_arithmetic_expr(numer, sumgam, '/')


class RDDLIntervalAnalysisPercentile(RDDLIntervalAnalysis):
    '''Interval analysis that replaces distributions with their lower and upper
    percentiles. This is a middle ground, since intervals with normally unbounded
    support can produce bounded intervals, with the percentiles being 
    configurable by the user.
    '''
    
    def __init__(self, rddl: RDDLPlanningModel, 
                 percentiles: Tuple[float, float], 
                 logger: Optional[Logger]=None) -> None:
        '''Creates a new interval analysis object for the given RDDL domain.
        
        :param rddl: the RDDL domain to analyze
        :param percentiles: percentiles used to compute bounds
        :param logger: to log compilation information during tracing to file
        '''
        self.percentiles = percentiles
        lower, upper = self.percentiles
        if lower < 0 or lower > 1 or upper < 0 or upper > 1 or lower > upper:
            raise ValueError('Percentiles must be in the range [0, 1] and lower <= upper.')
    
        super().__init__(rddl, logger=logger)
    
    def _bound_location_scale(self, percentiles, mean, scale):
        '''For a location scale member X with given percentiles, computes
        the interval for mean + scale * X.'''
        scaled_percentiles = self.bound_arithmetic_expr(percentiles, scale, '*')
        bounds = self.bound_arithmetic_expr(mean, scaled_percentiles, '+')
        return bounds
        
    def _bound_uniform(self, expr, intervals):
        args = expr.args
        a, b = args
        (la, ua) = self._bound(a, intervals)
        lub = self._bound(b, intervals)
        
        # a + (b - a) * U, where U is percentile of Uniform(0, 1)
        lower_percentile, upper_percentile = self.percentiles
        lower_u01 = np.ones_like(la) * lower_percentile
        upper_u01 = np.ones_like(ua) * upper_percentile
        lu_u01_inv = (1 - upper_u01, 1 - lower_u01)
        scaled_a = self.bound_arithmetic_expr((la, ua), lu_u01_inv, '*')
        scaled_b = self.bound_arithmetic_expr(lub, (lower_u01, upper_u01), '*')
        return self.bound_arithmetic_expr(scaled_a, scaled_b, '+')
    
    def _bound_bernoulli(self, expr, intervals):
        args = expr.args
        p, = args
        (lp, up) = self._bound(p, intervals)
        
        # TODO: check Bernoulli percentile
        lower_percentile, upper_percentile = self.percentiles
        lower = np.zeros(shape=np.shape(lp), dtype=np.int64)
        upper = np.ones(shape=np.shape(up), dtype=np.int64)
        lower = self._mask_assign(lower, lower_percentile > (1 - lp), 1)
        upper = self._mask_assign(upper, upper_percentile <= (1 - up), 0)
        return (lower, upper)
    
    def _bound_normal(self, expr, intervals):
        args = expr.args
        mean, var = args
        lum = self._bound(mean, intervals)
        luv = self._bound(var, intervals)
        
        # mean + std * Z, where Z is percentile of Normal(0, 1)
        lower_pctl, upper_pctl = self.percentiles
        normal_01 = (stats.norm.ppf(lower_pctl), stats.norm.ppf(upper_pctl))
        scale = self.bound_unary_func(luv, 'sqrt')
        return self._bound_location_scale(normal_01, lum, scale)
    
    def _bound_poisson(self, expr, intervals):
        args = expr.args
        p, = args
        (lp, up) = self._bound(p, intervals)
        lp, up = np.maximum(lp, 0), np.maximum(up, 0)   
        
        # percentiles of Poisson distribution at lower, upper of rate
        lower_pctl, upper_pctl = self.percentiles
        lower = stats.poisson.ppf(lower_pctl, lp)
        upper = stats.poisson.ppf(upper_pctl, up)
        lower = self._mask_assign(lower, lp <= 0, 0.0)
        upper = self._mask_assign(upper, up <= 0, 0.0)
        lower = self._mask_assign(lower, ~np.isfinite(lp), np.inf)
        upper = self._mask_assign(upper, ~np.isfinite(up), np.inf)
        return (lower, upper)
    
    def _bound_exponential(self, expr, intervals):
        args = expr.args
        scale, = args
        lusc = self._bound(scale, intervals)
        
        # scale * Exp1, where Exp1 is percentile of Exponential(1)
        lower_pctl, upper_pctl = self.percentiles
        exp1 = (-np.log(1 - lower_pctl), -np.log(1 - upper_pctl))
        lower, upper = self.bound_arithmetic_expr(lusc, exp1, '*')
        lower, upper = np.maximum(lower, 0.0), np.maximum(upper, 0.0)
        return (lower, upper)
     
    def _bound_weibull(self, expr, intervals):
        args = expr.args
        shape, scale = args
        (lsh, ush) = self._bound(shape, intervals)
        lusc = self._bound(scale, intervals)
        
        # scale * (-ln(1 - p))^(1 / shape)
        lower_pctl, upper_pctl = self.percentiles
        weibull_01 = (-np.log(1 - lower_pctl), -np.log(1 - upper_pctl))
        one = (np.ones_like(lsh), np.ones_like(ush))
        inv_shape = self.bound_arithmetic_expr(one, (lsh, ush), '/')
        shaped_weibull = self.bound_binary_func(weibull_01, inv_shape, 'pow')
        lower, upper = self.bound_arithmetic_expr(lusc, shaped_weibull, '*')
        lower, upper = np.maximum(lower, 0.0), np.maximum(upper, 0.0)
        return (lower, upper)
    
    def _bound_gamma(self, expr, intervals):
        # TODO: implement percentile Gamma interval
        raise NotImplementedError("Percentile strategy is not implemented for Gamma distribution yet.")
    
    def _bound_binomial(self, expr, intervals):
        # TODO: implement percentile Binomial interval
        raise NotImplementedError("Percentile strategy is not implemented for Binomial distribution yet.")
    
    def _bound_negative_binomial(self, expr, intervals):
        # TODO: implement percentile NegativeBinomial interval
        raise NotImplementedError("Percentile strategy is not implemented for NegativeBinomial distribution yet.")
    
    def _bound_beta(self, expr, intervals):
        # TODO: implement percentile Beta interval
        raise NotImplementedError("Percentile strategy is not implemented for Beta distribution yet.")
    
    def _bound_geometric(self, expr, intervals):
        # TODO: implement percentile Geometric interval
        raise NotImplementedError("Percentile strategy is not implemented for Geometric distribution yet.")
    
    def _bound_pareto(self, expr, intervals):
        args = expr.args
        shape, scale = args
        (lsh, ush) = self._bound(shape, intervals)
        (lsc, usc) = self._bound(scale, intervals)
        
        # scale * (1 - percentile) ** (-1 / shape)
        lower_pctl, upper_pctl = self.percentiles
        pareto_01 = (1. / (1. - lower_pctl), 1. / (1. - upper_pctl))
        one = (np.ones_like(lsh), np.ones_like(ush))
        inv_shape = self.bound_arithmetic_expr(one, (lsh, ush), '/')
        shaped_pareto = self.bound_binary_func(pareto_01, inv_shape, 'pow')
        lower, upper = self.bound_arithmetic_expr((lsc, usc), shaped_pareto, '*')
        lower = np.maximum(lower, lsc)
        upper = np.maximum(upper, lower)
        return (lower, upper)
    
    def _bound_student(self, expr, intervals):
        args = expr.args
        df, = args
        (ld, ud) = self._bound(df, intervals)
        
        # student_inverted_cdf(df) at the lowest degree of freedom
        lower_pctl, upper_pctl = self.percentiles
        lower = stats.t.ppf(lower_pctl, df=ld)
        upper = stats.t.ppf(upper_pctl, df=ld)
        lower = self._mask_assign(lower, ld <= 0, -np.inf)
        upper = self._mask_assign(upper, ld <= 0, +np.inf)
        return (lower, upper)
    
    def _bound_gumbel(self, expr, intervals):
        args = expr.args
        mean, scale = args
        lum = self._bound(mean, intervals)
        lusc = self._bound(scale, intervals)
        
        # mean - scale * ln(-ln(percentiles))
        lower_pctl, upper_pctl = self.percentiles
        gumbel_01 = (-np.log(-np.log(lower_pctl)), -np.log(-np.log(upper_pctl)))
        return self._bound_location_scale(gumbel_01, lum, lusc)
    
    def _bound_laplace(self, expr, intervals):
        args = expr.args
        mean, scale = args
        lum = self._bound(mean, intervals)
        lusc = self._bound(scale, intervals)
        
        # if percentile <= 0.5 then mean + scale * ln(2 percentile)
        # otherwise mean - scale * ln(2 - 2 percentile)
        lower_pctl, upper_pctl = self.percentiles
        if lower_pctl <= 0.5:
            lower_lap01 = np.log(2 * lower_pctl)
        else:
            lower_lap01 = -np.log(2 - 2 * lower_pctl)
        if upper_pctl <= 0.5:
            upper_lap01 = np.log(2 * upper_pctl)
        else:
            upper_lap01 = -np.log(2 - 2 * upper_pctl)
        laplace_01 = (lower_lap01, upper_lap01)
        return self._bound_location_scale(laplace_01, lum, lusc)
    
    def _bound_cauchy(self, expr, intervals):
        args = expr.args
        mean, scale = args
        lum = self._bound(mean, intervals)
        lusc = self._bound(scale, intervals)
        
        # scale * C01 + mean, where C01 are the percentiles of Cauchy(0, 1)
        lower_pctl, upper_pctl = self.percentiles
        lower_pctl = np.pi * (lower_pctl - 0.5)
        upper_pctl = np.pi * (upper_pctl - 0.5)
        lower_c01 = np.minimum(np.tan(lower_pctl), np.tan(upper_pctl))
        upper_c01 = np.maximum(np.tan(lower_pctl), np.tan(upper_pctl))
        cauchy_01 = (lower_c01, upper_c01)
        return self._bound_location_scale(cauchy_01, lum, lusc)
    
    def _bound_gompertz(self, expr, intervals):
        args = expr.args
        shape, scale = args
        lush = self._bound(shape, intervals)
        lusc = self._bound(scale, intervals)
        
        # (1/scale) * ln(1 - (1/shape) * ln(1 - G)) where G is standard Gompertz
        lower_pctl, upper_pctl = self.percentiles
        percentiles = (-np.log(1 - lower_pctl), -np.log(1 - upper_pctl))
        lower_shaped, upper_shaped = self.bound_arithmetic_expr(percentiles, lush, '/')
        percentiles2 = (np.log(1 + lower_shaped), np.log(1 + upper_shaped))
        lower, upper = self.bound_arithmetic_expr(percentiles2, lusc, '/')
        lower, upper = np.maximum(lower, 0.0), np.maximum(upper, 0.0)
        return (lower, upper)
    
    def _bound_chisquare(self, expr, intervals):
        args = expr.args
        df, = args
        ldf, udf = self._bound(df, intervals)
        ldf, udf = np.maximum(ldf, 0), np.maximum(udf, 0)   

        # evaluate inverse quantiles at lower and upper degree of freedom
        lower_pctl, upper_pctl = self.percentiles
        lower = stats.chi2.ppf(lower_pctl, ldf)
        upper = stats.chi2.ppf(upper_pctl, udf)
        lower = self._mask_assign(lower, ldf <= 0, 0.0)
        upper = self._mask_assign(upper, udf <= 0, 0.0)
        lower = self._mask_assign(lower, ~np.isfinite(ldf), np.inf)
        upper = self._mask_assign(upper, ~np.isfinite(udf), np.inf)
        return (lower, upper)        
    
    def _bound_kumaraswamy(self, expr, intervals):
        # TODO: implement percentile Kumaraswamy interval
        raise NotImplementedError("Percentile strategy is not implemented for Kumaraswamy distribution yet.")
    