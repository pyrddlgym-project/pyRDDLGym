import numpy as np
np.seterr(all='raise')
from typing import Dict, Union
import warnings

from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLActionPreconditionNotSatisfiedError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidActionError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidNumberOfArgumentsError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLNotImplementedError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLStateInvariantNotSatisfiedError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLTypeError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLUndefinedVariableError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLValueOutOfRangeError

from pyRDDLGym.Core.Compiler.RDDLDecompiler import RDDLDecompiler
from pyRDDLGym.Core.Compiler.RDDLLevelAnalysis import RDDLLevelAnalysis
from pyRDDLGym.Core.Compiler.RDDLModel import PlanningModel
from pyRDDLGym.Core.Compiler.RDDLObjectsTracer import RDDLObjectsTracer
from pyRDDLGym.Core.Compiler.RDDLValueInitializer import RDDLValueInitializer
from pyRDDLGym.Core.Debug.Logger import Logger
from pyRDDLGym.Core.Parser.expr import Expression, Value

Args = Dict[str, Value]

        
class RDDLSimulator:
    
    def __init__(self, rddl: PlanningModel,
                 allow_synchronous_state: bool=True,
                 rng: np.random.Generator=np.random.default_rng(),
                 logger: Logger=None) -> None:
        '''Creates a new simulator for the given RDDL model.
        
        :param rddl: the RDDL model
        :param allow_synchronous_state: whether state-fluent can be synchronous
        :param rng: the random number generator
        :param logger: to log information about compilation to file
        '''
        self.rddl = rddl
        self.rng = rng
        self.logger = logger
        
        # compile initial values
        if self.logger is not None:
            self.logger.clear()
        initializer = RDDLValueInitializer(rddl, logger=self.logger)
        self.init_values = initializer.initialize()
        
        # compute dependency graph for CPFs and sort them by evaluation order
        sorter = RDDLLevelAnalysis(rddl, allow_synchronous_state)
        levels = sorter.compute_levels()      
        self.cpfs = []  
        for cpfs in levels.values():
            for cpf in cpfs:
                _, expr = rddl.cpfs[cpf]
                prange = rddl.variable_ranges[cpf]
                dtype = RDDLValueInitializer.NUMPY_TYPES.get(
                    prange, RDDLValueInitializer.INT)
                self.cpfs.append((cpf, expr, dtype))
                
        # log dependency graph information to file
        if self.logger is not None: 
            levels_info = '\n\t'.join(f"{k}: {{{', '.join(v)}}}"
                                      for (k, v) in levels.items())
            message = (f'computed order of CPF evaluation:\n' 
                       f'\t{levels_info}\n')
            self.logger.log(message)
        
        # trace expressions to cache information to be used later
        tracer = RDDLObjectsTracer(rddl, logger=self.logger)
        self.traced = tracer.trace()
        
        # initialize all fluent and non-fluent values        
        self.subs = self.init_values.copy()
        self.state = None  
        self.noop_actions = {var: values
                             for (var, values) in self.init_values.items()
                             if rddl.variable_types[var] == 'action-fluent'}
        self._pomdp = bool(rddl.observ)
        
        # basic operations
        self.ARITHMETIC_OPS = {
            '+': np.add,
            '-': np.subtract,
            '*': np.multiply,
            '/': np.divide
        }    
        self.RELATIONAL_OPS = {
            '>=': np.greater_equal,
            '<=': np.less_equal,
            '<': np.less,
            '>': np.greater,
            '==': np.equal,
            '~=': np.not_equal
        }
        self.LOGICAL_OPS = {
            '^': np.logical_and,
            '&': np.logical_and,
            '|': np.logical_or,
            '~': np.logical_xor,
            '=>': lambda x, y: np.logical_or(np.logical_not(x), y),
            '<=>': np.equal
        }
        self.AGGREGATION_OPS = {
            'sum': np.sum,
            'avg': np.mean,
            'prod': np.prod,
            'minimum': np.min,
            'maximum': np.max,
            'forall': np.all,
            'exists': np.any,
            'argmin': np.argmin,
            'argmax': np.argmax
        }
        self.UNARY = {        
            'abs': np.abs,
            'sgn': lambda x: np.sign(x).astype(RDDLValueInitializer.INT),
            'round': lambda x: np.round(x).astype(RDDLValueInitializer.INT),
            'floor': lambda x: np.floor(x).astype(RDDLValueInitializer.INT),
            'ceil': lambda x: np.ceil(x).astype(RDDLValueInitializer.INT),
            'cos': np.cos,
            'sin': np.sin,
            'tan': np.tan,
            'acos': np.arccos,
            'asin': np.arcsin,
            'atan': np.arctan,
            'cosh': np.cosh,
            'sinh': np.sinh,
            'tanh': np.tanh,
            'exp': np.exp,
            'ln': np.log,
            'sqrt': np.sqrt,
            'lngamma': lngamma,
            'gamma': lambda x: np.exp(lngamma(x))
        }        
        self.BINARY = {
            'div': lambda x, y: np.floor_divide(x, y).astype(RDDLValueInitializer.INT),
            'mod': lambda x, y: np.mod(x, y).astype(RDDLValueInitializer.INT),
            'min': np.minimum,
            'max': np.maximum,
            'pow': np.power,
            'log': lambda x, y: np.log(x) / np.log(y)
        }
        self.CONTROL_OPS = {'if': np.where,
                            'switch': np.select}
    
    @property
    def states(self) -> Args:
        return self.state.copy()

    @property
    def isPOMDP(self) -> bool:
        return self._pomdp

    # ===========================================================================
    # error checks
    # ===========================================================================
    
    @staticmethod
    def _print_stack_trace(expr):
        if isinstance(expr, Expression):
            trace = RDDLDecompiler().decompile_expr(expr)
        else:
            trace = str(expr)
        return '>> ' + trace
    
    @staticmethod
    def _check_type(value, valid, msg, expr, arg=None):
        if not np.can_cast(value, valid):
            dtype = getattr(value, 'dtype', type(value))
            if arg is None:
                raise RDDLTypeError(
                    f'{msg} must evaluate to {valid}, '
                    f'got {value} of type {dtype}.\n' + 
                    RDDLSimulator._print_stack_trace(expr))
            else:
                raise RDDLTypeError(
                    f'Argument {arg} of {msg} must evaluate to {valid}, '
                    f'got {value} of type {dtype}.\n' + 
                    RDDLSimulator._print_stack_trace(expr))
    
    @staticmethod
    def _check_types(value, valid, msg, expr):
        for valid_type in valid:
            if np.can_cast(value, valid_type):
                return
        dtype = getattr(value, 'dtype', type(value))
        raise RDDLTypeError(
            f'{msg} must evaluate to one of {valid}, '
            f'got {value} of type {dtype}.\n' + 
            RDDLSimulator._print_stack_trace(expr))
    
    @staticmethod
    def _check_op(op, valid, msg, expr):
        numpy_op = valid.get(op, None)
        if numpy_op is None:
            raise RDDLNotImplementedError(
                f'{msg} operator {op} is not supported: '
                f'must be in {set(valid.keys())}.\n' + 
                RDDLSimulator._print_stack_trace(expr))
        return numpy_op
        
    @staticmethod
    def _check_arity(args, required, msg, expr):
        if len(args) != required:
            raise RDDLInvalidNumberOfArgumentsError(
                f'{msg} requires {required} arguments, got {len(args)}.\n' + 
                RDDLSimulator._print_stack_trace(expr))
    
    @staticmethod
    def _check_positive(value, strict, msg, expr):
        if strict and not np.all(value > 0):
            raise RDDLValueOutOfRangeError(
                f'{msg} must be positive, got {value}.\n' + 
                RDDLSimulator._print_stack_trace(expr))
        elif not strict and not np.all(value >= 0):
            raise RDDLValueOutOfRangeError(
                f'{msg} must be non-negative, got {value}.\n' + 
                RDDLSimulator._print_stack_trace(expr))
    
    @staticmethod
    def _check_bounds(lb, ub, msg, expr):
        if not np.all(lb <= ub):
            raise RDDLValueOutOfRangeError(
                f'Bounds of {msg} are invalid:' 
                f'max value {ub} must be >= min value {lb}.\n' + 
                RDDLSimulator._print_stack_trace(expr))
            
    @staticmethod
    def _check_range(value, lb, ub, msg, expr):
        if not np.all(np.logical_and(value >= lb, value <= ub)):
            raise RDDLValueOutOfRangeError(
                f'{msg} must be in the range [{lb}, {ub}], got {value}.\n' + 
                RDDLSimulator._print_stack_trace(expr))
    
    # ===========================================================================
    # main sampling routines
    # ===========================================================================
    
    def _process_actions(self, actions):
        new_actions = {action: np.copy(value) 
                       for (action, value) in self.noop_actions.items()}
        
        # override new_actions with any new actions
        rddl = self.rddl
        for (action, value) in actions.items(): 
            
            # enum literals are converted to their canonical int indices
            if value in rddl.enum_literals:
                value = rddl.index_of_object[value]
            
            # parse action string and assign to the correct coordinates
            if action in new_actions:
                new_actions[action] = value
            else:
                var, objects = rddl.parse(action)
                tensor = new_actions.get(var, None)
                if tensor is None: 
                    raise RDDLInvalidActionError(
                        f'<{action}> is not a valid action-fluent, ' 
                        f'must be one of {set(new_actions.keys())}.')
                RDDLSimulator._check_type(value, tensor.dtype, action, expr='')            
                tensor[rddl.indices(objects)] = value
                
        return new_actions
    
    def check_state_invariants(self) -> None:
        '''Throws an exception if the state invariants are not satisfied.'''
        for (i, invariant) in enumerate(self.rddl.invariants):
            sample = self._sample(invariant, self.subs)
            RDDLSimulator._check_type(sample, bool, 'Invariant', invariant)
            if not bool(sample):
                raise RDDLStateInvariantNotSatisfiedError(
                    f'Invariant {i + 1} is not satisfied.\n' + 
                    RDDLSimulator._print_stack_trace(invariant))
    
    def check_action_preconditions(self, actions: Args) -> None:
        '''Throws an exception if the action preconditions are not satisfied.'''        
        actions = self._process_actions(actions)
        self.subs.update(actions)
        
        for (i, precond) in enumerate(self.rddl.preconditions):
            sample = self._sample(precond, self.subs)
            RDDLSimulator._check_type(sample, bool, 'Precondition', precond)
            if not bool(sample):
                raise RDDLActionPreconditionNotSatisfiedError(
                    f'Precondition {i + 1} is not satisfied.\n' + 
                    RDDLSimulator._print_stack_trace(precond))
    
    def check_terminal_states(self) -> bool:
        '''Return True if a terminal state has been reached.'''
        for terminal in self.rddl.terminals:
            sample = self._sample(terminal, self.subs)
            RDDLSimulator._check_type(sample, bool, 'Termination', terminal)
            if bool(sample):
                return True
        return False
    
    def sample_reward(self) -> float:
        '''Samples the current reward given the current state and action.'''
        return float(self._sample(self.rddl.reward, self.subs))
    
    def reset(self) -> Union[Dict[str, None], Args]:
        '''Resets the state variables to their initial values.'''
        subs = self.subs = self.init_values.copy()
        
        # update state
        rddl = self.rddl
        self.state = {}
        for state in rddl.states:
            self.state.update(rddl.ground_values(state, subs[state]))
        
        # update observation
        if self._pomdp:
            obs = {var: None for var in rddl.observ}
        else:
            obs = self.state
        
        done = self.check_terminal_states()
        return obs, done
    
    def step(self, actions: Args) -> Args:
        '''Samples and returns the next state from the CPF expressions.
        
        :param actions: a dict mapping current action fluent to their values
        '''
        actions = self._process_actions(actions)
        subs = self.subs
        subs.update(actions)
        
        # evaluate CPFs in topological order
        rddl = self.rddl
        for (cpf, expr, dtype) in self.cpfs:
            sample = self._sample(expr, subs)
            RDDLSimulator._check_type(sample, dtype, cpf, expr)
            subs[cpf] = sample
        
        # evaluate reward
        reward = self.sample_reward()
        
        # update state
        self.state = {}
        for (state, next_state) in rddl.next_state.items():
            subs[state] = subs[next_state]
            self.state.update(rddl.ground_values(state, subs[state]))
        
        # update observation
        if self._pomdp: 
            obs = {}
            for var in rddl.observ:
                obs.update(rddl.ground_values(var, subs[var]))
        else:
            obs = self.state
        
        done = self.check_terminal_states()        
        return obs, reward, done
        
    # ===========================================================================
    # start of sampling subroutines
    # ===========================================================================
    
    def _sample(self, expr, subs):
        etype, _ = expr.etype
        if etype == 'constant':
            return self._sample_constant(expr, subs)
        elif etype == 'pvar':
            return self._sample_pvar(expr, subs)
        elif etype == 'arithmetic':
            return self._sample_arithmetic(expr, subs)
        elif etype == 'relational':
            return self._sample_relational(expr, subs)
        elif etype == 'boolean':
            return self._sample_logical(expr, subs)
        elif etype == 'aggregation':
            return self._sample_aggregation(expr, subs)
        elif etype == 'func':
            return self._sample_func(expr, subs)
        elif etype == 'control':
            return self._sample_control(expr, subs)
        elif etype == 'randomvar':
            return self._sample_random(expr, subs)
        else:
            raise RDDLNotImplementedError(
                f'Internal error: expression type {etype} is not supported.\n' + 
                RDDLSimulator._print_stack_trace(expr))
                
    # ===========================================================================
    # leaves
    # ===========================================================================
        
    def _sample_constant(self, expr, _):
        return self.traced.cached_sim_info(expr)
    
    def _sample_pvar(self, expr, subs):
        var, args = expr.args
        
        # literal of enumerated type is treated as integer
        if not args and self.rddl.is_literal(var):
            return self.traced.cached_sim_info(expr)
        
        # extract variable value
        sample = subs.get(var, None)
        if sample is None:
            raise RDDLUndefinedVariableError(
                f'Variable <{var}> is referenced before assignment.\n' + 
                RDDLSimulator._print_stack_trace(expr))
        
        # lifted domain must slice and/or reshape value tensor
        shape_info = self.traced.cached_sim_info(expr)
        if shape_info is not None:
            slices, axis, shape, op_code, op_args = shape_info
            if slices: 
                if op_code == -1:
                    slices = tuple(
                        (self._sample(arg, subs) if _slice is None else _slice)
                        for (arg, _slice) in zip(args, slices))
                sample = sample[slices]
            if axis:
                sample = np.expand_dims(sample, axis=axis)
                sample = np.broadcast_to(sample, shape=shape)
            if op_code == 0:
                sample = np.einsum(sample, *op_args)
            elif op_code == 1:
                sample = np.transpose(sample, axes=op_args)
        return sample
    
    # ===========================================================================
    # arithmetic
    # ===========================================================================
    
    def _sample_arithmetic(self, expr, subs):
        _, op = expr.etype
        numpy_op = RDDLSimulator._check_op(
            op, self.ARITHMETIC_OPS, 'Arithmetic', expr)
        
        args = expr.args        
        n = len(args)
        
        # unary negation
        if n == 1 and op == '-':
            arg, = args
            return -1 * self._sample(arg, subs)
        
        # binary operator: for * try to short-circuit if possible
        elif n == 2:
            lhs, rhs = args
            if op == '*':
                return self._sample_product(lhs, rhs, subs)
            else:
                sample_lhs = 1 * self._sample(lhs, subs)
                sample_rhs = 1 * self._sample(rhs, subs)
                try:
                    return numpy_op(sample_lhs, sample_rhs)
                except:
                    raise ArithmeticError(
                        f'Cannot execute arithmetic operation {op} '
                        f'with arguments {sample_lhs} and {sample_rhs}.\n' + 
                        RDDLSimulator._print_stack_trace(expr))
        
        # for a grounded domain can short-circuit * and +
        elif n > 0 and not self.traced.cached_objects_in_scope(expr):
            if op == '*':
                return self._sample_product_grounded(args, subs)
            elif op == '+':
                return sum(1 * self._sample(arg, subs) for arg in args)
        
        raise RDDLInvalidNumberOfArgumentsError(
            f'Arithmetic operator {op} does not have the required '
            f'number of arguments.\n' + 
            RDDLSimulator._print_stack_trace(expr))
    
    def _sample_product(self, lhs, rhs, subs):
        
        # prioritize simple expressions
        if rhs.is_constant_expression() or rhs.is_pvariable_expression():
            lhs, rhs = rhs, lhs
            
        sample_lhs = 1 * self._sample(lhs, subs)
        
        # short circuit if all zero
        if not np.any(sample_lhs):
            return sample_lhs
            
        sample_rhs = self._sample(rhs, subs)
        return sample_lhs * sample_rhs
    
    def _sample_product_grounded(self, args, subs):
        prod = 1
        
        # go through simple expressions first
        for arg in args: 
            if arg.is_constant_expression() or arg.is_pvariable_expression():
                sample = self._sample(arg, subs)
                prod *= sample
                if prod == 0:
                    return prod
        
        # go through complex expressions last
        for arg in args: 
            if not (arg.is_constant_expression() or arg.is_pvariable_expression()):
                sample = self._sample(arg, subs)
                prod *= sample
                if prod == 0:
                    return prod
                
        return prod
        
    # ===========================================================================
    # boolean
    # ===========================================================================
    
    def _sample_relational(self, expr, subs):
        _, op = expr.etype
        numpy_op = RDDLSimulator._check_op(
            op, self.RELATIONAL_OPS, 'Relational', expr)
        
        args = expr.args
        RDDLSimulator._check_arity(args, 2, op, expr)
        
        lhs, rhs = args
        sample_lhs = 1 * self._sample(lhs, subs)
        sample_rhs = 1 * self._sample(rhs, subs)
        return numpy_op(sample_lhs, sample_rhs)
    
    def _sample_logical(self, expr, subs):
        _, op = expr.etype
        if op == '&':
            op = '^'
        numpy_op = RDDLSimulator._check_op(op, self.LOGICAL_OPS, 'Logical', expr)
        
        args = expr.args
        n = len(args)
        
        if n == 1 and op == '~':
            arg, = args
            sample = self._sample(arg, subs)
            RDDLSimulator._check_type(sample, bool, op, expr, arg='')
            return np.logical_not(sample)
        
        # try to short-circuit ^ and | if possible
        elif n == 2:
            lhs, rhs = args
            if op == '^' or op == '|':
                return self._sample_and_or(lhs, rhs, op, expr, subs)
            else:
                sample_lhs = self._sample(lhs, subs)
                sample_rhs = self._sample(rhs, subs)
                RDDLSimulator._check_type(sample_lhs, bool, op, expr, arg=1)
                RDDLSimulator._check_type(sample_rhs, bool, op, expr, arg=2)
                return numpy_op(sample_lhs, sample_rhs)
        
        # for a grounded domain, we can short-circuit ^ and |
        elif n > 0 and (op == '^' or op == '|') \
        and not self.traced.cached_objects_in_scope(expr):
            return self._sample_and_or_grounded(args, op, expr, subs)
            
        raise RDDLInvalidNumberOfArgumentsError(
            f'Logical operator {op} does not have the required '
            f'number of arguments.\n' + 
            RDDLSimulator._print_stack_trace(expr))
    
    def _sample_and_or(self, lhs, rhs, op, expr, subs):
        
        # prioritize simple expressions
        if rhs.is_constant_expression() or rhs.is_pvariable_expression():
            lhs, rhs = rhs, lhs 
            
        sample_lhs = self._sample(lhs, subs)
        RDDLSimulator._check_type(sample_lhs, bool, op, expr, arg=1)
        
        # short-circuit if all True/False
        if (op == '^' and not np.any(sample_lhs)) \
        or (op == '|' and np.all(sample_lhs)):
            return sample_lhs
            
        sample_rhs = self._sample(rhs, subs)
        RDDLSimulator._check_type(sample_rhs, bool, op, expr, arg=2)
        
        if op == '^':
            return np.logical_and(sample_lhs, sample_rhs)
        else:
            return np.logical_or(sample_lhs, sample_rhs)
    
    def _sample_and_or_grounded(self, args, op, expr, subs): 
        
        # go through simple expressions first
        for (i, arg) in enumerate(args):
            if arg.is_constant_expression() or arg.is_pvariable_expression():
                sample = self._sample(arg, subs)
                RDDLSimulator._check_type(sample, bool, op, expr, arg=i + 1)
                sample = bool(sample)
                if not sample and op == '^':
                    return False
                elif sample and op == '|':
                    return True
        
        # go through complex expressions last
        for (i, arg) in enumerate(args):
            if not (arg.is_constant_expression() or arg.is_pvariable_expression()):
                sample = self._sample(arg, subs)
                RDDLSimulator._check_type(sample, bool, op, expr, arg=i + 1)
                sample = bool(sample)
                if not sample and op == '^':
                    return False
                elif sample and op == '|':
                    return True
            
        return op == '^'
            
    # ===========================================================================
    # aggregation
    # ===========================================================================
    
    def _sample_aggregation(self, expr, subs):
        _, op = expr.etype
        numpy_op = RDDLSimulator._check_op(
            op, self.AGGREGATION_OPS, 'Aggregation', expr)
        
        # sample the argument and aggregate over the reduced axes
        * _, arg = expr.args
        sample = self._sample(arg, subs)                
        if op == 'forall' or op == 'exists':
            RDDLSimulator._check_type(sample, bool, op, expr, arg='')
        else:
            sample = 1 * sample
        _, axes = self.traced.cached_sim_info(expr)
        return numpy_op(sample, axis=axes)
     
    # ===========================================================================
    # function
    # ===========================================================================
    
    def _sample_func(self, expr, subs):
        _, name = expr.etype
        args = expr.args
        
        # unary function
        unary_op = self.UNARY.get(name, None)
        if unary_op is not None:
            RDDLSimulator._check_arity(args, 1, name, expr)
            arg, = args
            sample = 1 * self._sample(arg, subs)
            return unary_op(sample)
        
        # binary function
        binary_op = self.BINARY.get(name, None)
        if binary_op is not None:
            RDDLSimulator._check_arity(args, 2, name, expr)
            lhs, rhs = args
            sample_lhs = 1 * self._sample(lhs, subs)
            sample_rhs = 1 * self._sample(rhs, subs)
            return binary_op(sample_lhs, sample_rhs)
        
        raise RDDLNotImplementedError(
            f'Function {name} is not supported.\n' + 
            RDDLSimulator._print_stack_trace(expr))
    
    # ===========================================================================
    # control flow
    # ===========================================================================
    
    def _sample_control(self, expr, subs):
        _, op = expr.etype
        RDDLSimulator._check_op(op, self.CONTROL_OPS, 'Control', expr)
        
        if op == 'if':
            return self._sample_if(expr, subs)
        else:
            return self._sample_switch(expr, subs)    
        
    def _sample_if(self, expr, subs):
        args = expr.args
        RDDLSimulator._check_arity(args, 3, 'If then else', expr)
        
        pred, arg1, arg2 = args
        sample_pred = self._sample(pred, subs)
        RDDLSimulator._check_type(sample_pred, bool, 'If predicate', expr)
        
        # can short circuit if all elements of predicate tensor equal
        first_elem = bool(sample_pred.flat[0] 
                          if self.traced.cached_objects_in_scope(expr) 
                          else sample_pred)
        all_equal = np.all(sample_pred == first_elem)
        
        if all_equal:
            arg = arg1 if first_elem else arg2
            return self._sample(arg, subs)
        else:
            sample_then = self._sample(arg1, subs)
            sample_else = self._sample(arg2, subs)
            return np.where(sample_pred, sample_then, sample_else)
    
    def _sample_switch(self, expr, subs):
        pred, *_ = expr.args             
        sample_pred = self._sample(pred, subs)
        RDDLSimulator._check_type(
            sample_pred, RDDLValueInitializer.INT, 'Switch predicate', expr)
        
        # can short circuit if all elements of predicate tensor equal
        cases, default = self.traced.cached_sim_info(expr)  
        first_elem = bool(sample_pred.flat[0] 
                          if self.traced.cached_objects_in_scope(expr) 
                          else sample_pred)
        all_equal = np.all(sample_pred == first_elem)
        
        if all_equal:
            arg = cases[first_elem]
            if arg is None:
                arg = default
            return self._sample(arg, subs)        
        else: 
            sample_default = None if default is None else self._sample(default, subs)
            sample_cases = np.asarray([
                (sample_default if arg is None else self._sample(arg, subs))
                for arg in cases
            ])
            sample_pred = np.expand_dims(sample_pred, axis=0)
            sample = np.take_along_axis(sample_cases, sample_pred, axis=0)   
            assert sample.shape[0] == 1
            return sample[0, ...]
        
    # ===========================================================================
    # random variables
    # ===========================================================================
    
    def _sample_random(self, expr, subs):
        _, name = expr.etype
        if name == 'KronDelta':
            return self._sample_kron_delta(expr, subs)        
        elif name == 'DiracDelta':
            return self._sample_dirac_delta(expr, subs)
        elif name == 'Uniform':
            return self._sample_uniform(expr, subs)
        elif name == 'Bernoulli':
            return self._sample_bernoulli(expr, subs)
        elif name == 'Normal':
            return self._sample_normal(expr, subs)
        elif name == 'Poisson':
            return self._sample_poisson(expr, subs)
        elif name == 'Exponential':
            return self._sample_exponential(expr, subs)
        elif name == 'Weibull':
            return self._sample_weibull(expr, subs)        
        elif name == 'Gamma':
            return self._sample_gamma(expr, subs)
        elif name == 'Binomial':
            return self._sample_binomial(expr, subs)
        elif name == 'NegativeBinomial':
            return self._sample_negative_binomial(expr, subs)
        elif name == 'Beta':
            return self._sample_beta(expr, subs)
        elif name == 'Geometric':
            return self._sample_geometric(expr, subs)
        elif name == 'Pareto':
            return self._sample_pareto(expr, subs)
        elif name == 'Student':
            return self._sample_student(expr, subs)
        elif name == 'Gumbel':
            return self._sample_gumbel(expr, subs)
        elif name == 'Laplace':
            return self._sample_laplace(expr, subs)
        elif name == 'Cauchy':
            return self._sample_cauchy(expr, subs)
        elif name == 'Gompertz':
            return self._sample_gompertz(expr, subs)
        elif name == 'Discrete':
            return self._sample_discrete(expr, subs, unnorm=False)
        elif name == 'UnnormDiscrete':
            return self._sample_discrete(expr, subs, unnorm=True)
        else:
            raise RDDLNotImplementedError(
                f'Distribution {name} is not supported.\n' + 
                RDDLSimulator._print_stack_trace(expr))

    def _sample_kron_delta(self, expr, subs):
        args = expr.args
        RDDLSimulator._check_arity(args, 1, 'KronDelta', expr)
        
        arg, = args
        sample = self._sample(arg, subs)
        RDDLSimulator._check_types(
            sample, {bool, RDDLValueInitializer.INT}, 'Argument of KronDelta', expr)
        return sample
    
    def _sample_dirac_delta(self, expr, subs):
        args = expr.args
        RDDLSimulator._check_arity(args, 1, 'DiracDelta', expr)
        
        arg, = args
        sample = self._sample(arg, subs)
        RDDLSimulator._check_type(
            sample, RDDLValueInitializer.REAL, 'Argument of DiracDelta', expr)        
        return sample
    
    def _sample_uniform(self, expr, subs):
        args = expr.args
        RDDLSimulator._check_arity(args, 2, 'Uniform', expr)

        lb, ub = args
        sample_lb = self._sample(lb, subs)
        sample_ub = self._sample(ub, subs)
        RDDLSimulator._check_bounds(sample_lb, sample_ub, 'Uniform', expr)
        return self.rng.uniform(sample_lb, sample_ub)      
    
    def _sample_bernoulli(self, expr, subs):
        args = expr.args
        RDDLSimulator._check_arity(args, 1, 'Bernoulli', expr)
        
        pr, = args
        sample_pr = self._sample(pr, subs)
        RDDLSimulator._check_range(sample_pr, 0, 1, 'Bernoulli p', expr)
        size = sample_pr.shape if self.traced.cached_objects_in_scope(expr) else None
        return self.rng.uniform(size=size) <= sample_pr
    
    def _sample_normal(self, expr, subs):
        args = expr.args
        RDDLSimulator._check_arity(args, 2, 'Normal', expr)
        
        mean, var = args
        sample_mean = self._sample(mean, subs)
        sample_var = self._sample(var, subs)
        RDDLSimulator._check_positive(sample_var, False, 'Normal variance', expr)  
        sample_std = np.sqrt(sample_var)
        return self.rng.normal(sample_mean, sample_std)
    
    def _sample_poisson(self, expr, subs):
        args = expr.args
        RDDLSimulator._check_arity(args, 1, 'Poisson', expr)
        
        rate, = args
        sample_rate = self._sample(rate, subs)
        RDDLSimulator._check_positive(sample_rate, False, 'Poisson rate', expr)        
        return self.rng.poisson(sample_rate)
    
    def _sample_exponential(self, expr, subs):
        args = expr.args
        RDDLSimulator._check_arity(args, 1, 'Exponential', expr)
        
        scale, = expr.args
        sample_scale = self._sample(scale, subs)
        RDDLSimulator._check_positive(sample_scale, True, 'Exponential rate', expr)
        return self.rng.exponential(sample_scale)
    
    def _sample_weibull(self, expr, subs):
        args = expr.args
        RDDLSimulator._check_arity(args, 2, 'Weibull', expr)
        
        shape, scale = args
        sample_shape = self._sample(shape, subs)
        sample_scale = self._sample(scale, subs)
        RDDLSimulator._check_positive(sample_shape, True, 'Weibull shape', expr)
        RDDLSimulator._check_positive(sample_scale, True, 'Weibull scale', expr)
        return sample_scale * self.rng.weibull(sample_shape)
    
    def _sample_gamma(self, expr, subs):
        args = expr.args
        RDDLSimulator._check_arity(args, 2, 'Gamma', expr)
        
        shape, scale = args
        sample_shape = self._sample(shape, subs)
        sample_scale = self._sample(scale, subs)
        RDDLSimulator._check_positive(sample_shape, True, 'Gamma shape', expr)            
        RDDLSimulator._check_positive(sample_scale, True, 'Gamma scale', expr)        
        return self.rng.gamma(sample_shape, sample_scale)
    
    def _sample_binomial(self, expr, subs):
        args = expr.args
        RDDLSimulator._check_arity(args, 2, 'Binomial', expr)
        
        count, pr = args
        sample_count = self._sample(count, subs)
        sample_pr = self._sample(pr, subs)
        RDDLSimulator._check_type(sample_count, RDDLValueInitializer.INT, 'Binomial count', expr)
        RDDLSimulator._check_positive(sample_count, False, 'Binomial count', expr)
        RDDLSimulator._check_range(sample_pr, 0, 1, 'Binomial p', expr)
        return self.rng.binomial(sample_count, sample_pr)
    
    def _sample_negative_binomial(self, expr, subs):
        args = expr.args
        RDDLSimulator._check_arity(args, 2, 'NegativeBinomial', expr)
        
        count, pr = args
        sample_count = self._sample(count, subs)
        sample_pr = self._sample(pr, subs)
        RDDLSimulator._check_positive(sample_count, True, 'NegativeBinomial r', expr)
        RDDLSimulator._check_range(sample_pr, 0, 1, 'NegativeBinomial p', expr)        
        return self.rng.negative_binomial(sample_count, sample_pr)
    
    def _sample_beta(self, expr, subs):
        args = expr.args
        RDDLSimulator._check_arity(args, 2, 'Beta', expr)
        
        shape, rate = args
        sample_shape = self._sample(shape, subs)
        sample_rate = self._sample(rate, subs)
        RDDLSimulator._check_positive(sample_shape, True, 'Beta shape', expr)
        RDDLSimulator._check_positive(sample_rate, True, 'Beta rate', expr)        
        return self.rng.beta(sample_shape, sample_rate)

    def _sample_geometric(self, expr, subs):
        args = expr.args
        RDDLSimulator._check_arity(args, 1, 'Geometric', expr)
        
        pr, = args
        sample_pr = self._sample(pr, subs)
        RDDLSimulator._check_range(sample_pr, 0, 1, 'Geometric p', expr)        
        return self.rng.geometric(sample_pr)
    
    def _sample_pareto(self, expr, subs):
        args = expr.args
        RDDLSimulator._check_arity(args, 2, 'Pareto', expr)
        
        shape, scale = args
        sample_shape = self._sample(shape, subs)
        sample_scale = self._sample(scale, subs)
        RDDLSimulator._check_positive(sample_shape, True, 'Pareto shape', expr)        
        RDDLSimulator._check_positive(sample_scale, True, 'Pareto scale', expr)        
        return sample_scale * self.rng.pareto(sample_shape)
    
    def _sample_student(self, expr, subs):
        args = expr.args
        RDDLSimulator._check_arity(args, 1, 'Student', expr)
        
        df, = args
        sample_df = self._sample(df, subs)
        RDDLSimulator._check_positive(sample_df, True, 'Student df', expr)            
        return self.rng.standard_t(sample_df)

    def _sample_gumbel(self, expr, subs):
        args = expr.args
        RDDLSimulator._check_arity(args, 2, 'Gumbel', expr)
        
        mean, scale = args
        sample_mean = self._sample(mean, subs)
        sample_scale = self._sample(scale, subs)
        RDDLSimulator._check_positive(sample_scale, True, 'Gumbel scale', expr)
        return self.rng.gumbel(sample_mean, sample_scale)
    
    def _sample_laplace(self, expr, subs):
        args = expr.args
        RDDLSimulator._check_arity(args, 2, 'Laplace', expr)
        
        mean, scale = args
        sample_mean = self._sample(mean, subs)
        sample_scale = self._sample(scale, subs)
        RDDLSimulator._check_positive(sample_scale, True, 'Laplace scale', expr)
        return self.rng.laplace(sample_mean, sample_scale)
    
    def _sample_cauchy(self, expr, subs):
        args = expr.args
        RDDLSimulator._check_arity(args, 2, 'Cauchy', expr)
        
        mean, scale = args
        sample_mean = self._sample(mean, subs)
        sample_scale = self._sample(scale, subs)
        RDDLSimulator._check_positive(sample_scale, True, 'Cauchy scale', expr)
        size = sample_mean.shape if self.traced.cached_objects_in_scope(expr) else None
        cauchy01 = self.rng.standard_cauchy(size=size)
        return sample_mean + sample_scale * cauchy01
    
    def _sample_gompertz(self, expr, subs):
        args = expr.args
        RDDLSimulator._check_arity(args, 2, 'Gompertz', expr)
        
        shape, scale = args
        sample_shape = self._sample(shape, subs)
        sample_scale = self._sample(scale, subs)
        RDDLSimulator._check_positive(sample_shape, True, 'Gompertz shape', expr)
        RDDLSimulator._check_positive(sample_scale, True, 'Gompertz scale', expr)
        size = sample_shape.shape if self.traced.cached_objects_in_scope(expr) else None
        U = self.rng.uniform(size=size)
        return np.log(1.0 - np.log1p(-U) / sample_shape) / sample_scale
    
    def _sample_discrete(self, expr, subs, unnorm):
        
        # calculate the CDF
        pdf = np.asarray([self._sample(arg, subs) 
                          for arg in self.traced.cached_sim_info(expr)])
        cdf = np.cumsum(pdf, axis=0)
        
        # must be non-negative
        if not np.all(pdf >= 0):
            raise RDDLValueOutOfRangeError(
                f'Discrete probabilities must be non-negative, got {pdf}.\n' + 
                RDDLSimulator._print_stack_trace(expr))  
          
        # CDF must sum to one
        if unnorm:
            cdf = cdf / cdf[-1:, ...]
        if not np.allclose(cdf[-1, ...], 1.0):
            raise RDDLValueOutOfRangeError(
                f'Discrete probabilities must sum to 1, got {cdf[-1, ...]}.\n' + 
                RDDLSimulator._print_stack_trace(expr))     
        
        # use inverse CDF sampling                  
        U = self.rng.random(size=(1,) + cdf.shape[1:])
        return np.argmax(U < cdf, axis=0)


def lngamma(x):
    xmin = np.min(x)
    if not (xmin > 0):
        raise ValueError(f'Cannot evaluate log-gamma at {xmin}.')
    
    # small x: use lngamma(x) = lngamma(x + m) - ln(x + m - 1)... - ln(x)
    # large x: use asymptotic expansion OEIS:A046969
    if xmin < 7:
        return lngamma(x + 2) - np.log(x) - np.log(x + 1)        
    x_squared = x * x
    return (x - 0.5) * np.log(x) - x + 0.5 * np.log(2 * np.pi) + \
        1 / (12 * x) * (
            1 + 1 / (30 * x_squared) * (
                -1 + 1 / (7 * x_squared / 2) * (
                    1 + 1 / (4 * x_squared / 3) * (
                        -1 + 1 / (99 * x_squared / 140) * (
                            1 + 1 / (910 * x_squared / 3))))))


class RDDLSimulatorWConstraints(RDDLSimulator):

    def __init__(self, *args, max_bound: float=np.inf, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.BigM = max_bound
        
        self.epsilon = 0.001
        
        self._bounds = {}
        for (var, vtype) in self.rddl.variable_types.items():
            if vtype in {'state-fluent', 'observ-fluent', 'action-fluent'}:
                ptypes = self.rddl.param_types[var]
                for gname in self.rddl.ground_names(var, ptypes):
                    self._bounds[gname] = [-self.BigM, +self.BigM]

        # actions and states bounds extraction for gym's action and state spaces
        # currently supports only linear inequality constraints
        for precond in self.rddl.preconditions:
            self._parse_bounds(precond, [], self.rddl.actions)
            
        for invariant in self.rddl.invariants:
            self._parse_bounds(invariant, [], self.rddl.states)

        for (name, bounds) in self._bounds.items():
            RDDLSimulator._check_bounds(*bounds, f'Variable <{name}>', bounds)
            
        # log bounds to file
        if self.logger is not None:
            bounds_info = '\n\t'.join(f'{k}: {v}' 
                                      for (k, v) in self._bounds.items())
            message = (f'computed simulation bounds:\n' 
                       f'\t{bounds_info}\n')
            self.logger.log(message)
        
    def _parse_bounds(self, expr, objects, search_vars):
        etype, op = expr.etype
        
        if etype == 'aggregation' and op == 'forall':
            * pvars, arg = expr.args
            new_objects = objects + [pvar for (_, pvar) in pvars]
            self._parse_bounds(arg, new_objects, search_vars)
            
        elif etype == 'boolean' and op == '^':
            for arg in expr.args:
                self._parse_bounds(arg, objects, search_vars)
                
        elif etype == 'relational':
            var, lim, loc, active = self._parse_bounds_relational(
                expr, objects, search_vars)
            if var is not None and loc is not None: 
                if objects:
                    ptypes = [ptype for (_, ptype) in objects]
                    variations = self.rddl.variations(ptypes)
                    lims = np.ravel(lim, order='C')
                    for (args, lim) in zip(variations, lims):
                        active_args = [args[i] for i in active]
                        key = self.rddl.ground_name(var, active_args)
                        self._update_bound(key, loc, lim)
                else:
                    self._update_bound(var, loc, lim)
    
    def _update_bound(self, key, loc, lim):
        if loc == 1:
            if self._bounds[key][loc] > lim:
                self._bounds[key][loc] = lim
        else:
            if self._bounds[key][loc] < lim:
                self._bounds[key][loc] = lim
        
    def _parse_bounds_relational(self, expr, objects, search_vars):
        left, right = expr.args    
        _, op = expr.etype
        is_left_pvar = left.is_pvariable_expression() and left.args[0] in search_vars
        is_right_pvar = right.is_pvariable_expression() and right.args[0] in search_vars
        
        if (is_left_pvar and is_right_pvar) or op not in ['<=', '<', '>=', '>']:
            warnings.warn(
                f'Constraint does not have a structure of '
                f'<action or state fluent> <op> <rhs>, where:' 
                    f'\n<op> is one of {{<=, <, >=, >}}'
                    f'\n<rhs> is a deterministic function of '
                    f'non-fluents or constants only.\n' + 
                RDDLSimulator._print_stack_trace(expr))
            return None, 0.0, None, []
            
        elif not is_left_pvar and not is_right_pvar:
            return None, 0.0, None, []
        
        else:
            if is_left_pvar:
                var, args = left.args
                const_expr = right
            else:
                var, args = right.args
                const_expr = left
            if args is None:
                args = []
                
            if not self.rddl.is_non_fluent_expression(const_expr):
                warnings.warn(
                    f'Bound must be a deterministic function of '
                    f'non-fluents or constants only.\n' + 
                    RDDLSimulator._print_stack_trace(const_expr))
                return None, 0.0, None, []
            
            const = self._sample(const_expr, self.subs)
            eps, loc = self._get_op_code(op, is_left_pvar)
            lim = const + eps
            
            arg_to_index = {obj[0]: i for (i, obj) in enumerate(objects)}
            active = [arg_to_index[arg] for arg in args if arg in arg_to_index]

            return var, lim, loc, active
            
    def _get_op_code(self, op, is_right):
        eps = 0.0
        if is_right:
            if op in ['<=', '<']:
                loc = 1
                if op == '<':
                    eps = -self.epsilon
            elif op in ['>=', '>']:
                loc = 0
                if op == '>':
                    eps = self.epsilon
        else:
            if op in ['<=', '<']:
                loc = 0
                if op == '<':
                    eps = self.epsilon
            elif op in ['>=', '>']:
                loc = 1
                if op == '>':
                    eps = -self.epsilon
        return eps, loc

    @property
    def bounds(self):
        return self._bounds

    @bounds.setter
    def bounds(self, value):
        self._bounds = value
