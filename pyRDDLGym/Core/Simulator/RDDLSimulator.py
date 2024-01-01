import numpy as np
np.seterr(all='raise')
from typing import Dict, Set, Union

from pyRDDLGym.Core.ErrorHandling.RDDLException import print_stack_trace
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLActionPreconditionNotSatisfiedError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidActionError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidNumberOfArgumentsError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLNotImplementedError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLStateInvariantNotSatisfiedError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLTypeError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLUndefinedVariableError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLValueOutOfRangeError

from pyRDDLGym.Core.Compiler.RDDLLevelAnalysis import RDDLLevelAnalysis
from pyRDDLGym.Core.Compiler.RDDLModel import PlanningModel
from pyRDDLGym.Core.Compiler.RDDLObjectsTracer import RDDLObjectsTracer
from pyRDDLGym.Core.Compiler.RDDLValueInitializer import RDDLValueInitializer
from pyRDDLGym.Core.Debug.Logger import Logger
from pyRDDLGym.Core.Parser.expr import Value

Args = Dict[str, Value]

        
class RDDLSimulator:
    
    def __init__(self, rddl: PlanningModel,
                 allow_synchronous_state: bool=True,
                 rng: np.random.Generator=np.random.default_rng(),
                 logger: Logger=None,
                 keep_tensors: bool=False) -> None:
        '''Creates a new simulator for the given RDDL model.
        
        :param rddl: the RDDL model
        :param allow_synchronous_state: whether state-fluent can be synchronous
        :param rng: the random number generator
        :param logger: to log information about compilation to file
        :param keep_tensors: whether the sampler takes actions and
        returns state in numpy array form
        '''
        self.rddl = rddl
        self.allow_synchronous_state = allow_synchronous_state
        self.rng = rng
        self.logger = logger
        self.keep_tensors = keep_tensors
        
        self._compile()
        
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
        self.AGGREGATION_BOOL = {'forall', 'exists'}
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
            'fmod': lambda x, y: np.mod(x, y),
            'min': np.minimum,
            'max': np.maximum,
            'pow': np.power,
            'log': lambda x, y: np.log(x) / np.log(y),
            'hypot': np.hypot
        }
        self.BINARY_REQUIRES_INT = {'div', 'mod'}
        self.CONTROL_OPS = {'if': np.where,
                            'switch': np.select}
    
    def seed(self, seed: int) -> None:
        '''Sets the pseudo-random RNG seed for generating random numbers.
        
        :param seed: seed value to use for all future simulations
        '''
        self.rng = np.random.default_rng(seed)
        
    def _compile(self):
        rddl = self.rddl
        
        # compile initial values
        if self.logger is not None:
            self.logger.clear()
        initializer = RDDLValueInitializer(rddl, logger=self.logger)
        self.init_values = initializer.initialize()
        
        # compute dependency graph for CPFs and sort them by evaluation order
        sorter = RDDLLevelAnalysis(rddl, 
                                   allow_synchronous_state=self.allow_synchronous_state, 
                                   logger=self.logger)
        levels = sorter.compute_levels()      
        self.cpfs = []  
        for cpfs in levels.values():
            for cpf in cpfs:
                _, expr = rddl.cpfs[cpf]
                prange = rddl.variable_ranges[cpf]
                dtype = RDDLValueInitializer.NUMPY_TYPES.get(
                    prange, RDDLValueInitializer.INT)
                self.cpfs.append((cpf, expr, dtype))
                
        # trace expressions to cache information to be used later
        tracer = RDDLObjectsTracer(rddl, logger=self.logger, cpf_levels=levels)
        self.traced = tracer.trace()
        
        # initialize all fluent and non-fluent values        
        self.subs = self.init_values.copy()
        self.state = None  
        self.noop_actions = {var: values
                             for (var, values) in self.init_values.items()
                             if rddl.variable_types[var] == 'action-fluent'}
        self._pomdp = bool(rddl.observ)
        
        # cached for performance
        self.invariant_names = [f'Invariant {i}' for i in range(len(rddl.invariants))]        
        self.precond_names = [f'Precondition {i}' for i in range(len(rddl.preconditions))]
        self.terminal_names = [f'Termination {i}' for i in range(len(rddl.terminals))]
        
        self.grounded_actionsranges = rddl.groundactionsranges()
        self.grounded_noop_actions = rddl.ground_values_from_dict(self.noop_actions)
        
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
    def _check_type(value, valid, msg, expr, arg=None):
        if not np.can_cast(value, valid):
            dtype = getattr(value, 'dtype', type(value))
            if arg is None:
                raise RDDLTypeError(
                    f'{msg} must evaluate to {valid}, '
                    f'got {value} of type {dtype}.\n' + print_stack_trace(expr))
            else:
                raise RDDLTypeError(
                    f'Argument {arg} of {msg} must evaluate to {valid}, '
                    f'got {value} of type {dtype}.\n' + print_stack_trace(expr))
    
    @staticmethod
    def _check_types(value, valid, msg, expr):
        for valid_type in valid:
            if np.can_cast(value, valid_type):
                return
        dtype = getattr(value, 'dtype', type(value))
        raise RDDLTypeError(
            f'{msg} must evaluate to one of {valid}, '
            f'got {value} of type {dtype}.\n' + print_stack_trace(expr))
    
    @staticmethod
    def _check_op(op, valid, msg, expr):
        numpy_op = valid.get(op, None)
        if numpy_op is None:
            raise RDDLNotImplementedError(
                f'{msg} operator {op} is not supported: '
                f'must be in {set(valid.keys())}.\n' + print_stack_trace(expr))
        return numpy_op
        
    @staticmethod
    def _check_arity(args, required, msg, expr):
        if len(args) != required:
            raise RDDLInvalidNumberOfArgumentsError(
                f'{msg} requires {required} argument(s), got {len(args)}.\n' + 
                print_stack_trace(expr))
    
    @staticmethod
    def _check_positive(value, strict, msg, expr):
        if strict and not np.all(value > 0):
            raise RDDLValueOutOfRangeError(
                f'{msg} must be positive, got {value}.\n' + 
                print_stack_trace(expr))
        elif not strict and not np.all(value >= 0):
            raise RDDLValueOutOfRangeError(
                f'{msg} must be non-negative, got {value}.\n' + 
                print_stack_trace(expr))
    
    @staticmethod
    def _check_bounds(lb, ub, msg, expr):
        if not np.all(lb <= ub):
            raise RDDLValueOutOfRangeError(
                f'Bounds of {msg} are invalid:' 
                f'max value {ub} must be >= min value {lb}.\n' + 
                print_stack_trace(expr))
            
    @staticmethod
    def _check_range(value, lb, ub, msg, expr):
        if not np.all(np.logical_and(value >= lb, value <= ub)):
            raise RDDLValueOutOfRangeError(
                f'{msg} must be in the range [{lb}, {ub}], got {value}.\n' + 
                print_stack_trace(expr))
    
    # ===========================================================================
    # main sampling routines
    # ===========================================================================
    
    def _process_actions(self, actions):
        rddl = self.rddl
        
        # override new_actions with any new actions
        if self.keep_tensors:
            new_actions = self.noop_actions.copy()
            for (action, value) in actions.items(): 
                if action in new_actions:
                    new_actions[action] = value
        else:            
            new_actions = {action: np.copy(value) 
                           for (action, value) in self.noop_actions.items()}
            for (action, value) in actions.items(): 
                value = rddl.index_of_object.get(value, value)
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
    
    def check_state_invariants(self, silent: bool=False) -> bool:
        '''Throws an exception if the state invariants are not satisfied.'''
        for (i, invariant) in enumerate(self.rddl.invariants):
            loc = self.invariant_names[i]
            sample = self._sample(invariant, self.subs)
            RDDLSimulator._check_type(sample, bool, loc, invariant)
            if not bool(sample):
                if not silent:
                    raise RDDLStateInvariantNotSatisfiedError(
                        f'{loc} is not satisfied.\n' + print_stack_trace(invariant))
                return False
        return True
    
    def check_default_action_count(self, actions: Args, 
                                   enforce_for_non_bool: bool=True) -> None:
        '''Throws an exception if the actions do not satisfy max-nondef-actions.'''     
        if self.keep_tensors:
            action_ranges = self.rddl.actionsranges
            noop_actions = self.noop_actions
        else:
            action_ranges = self.grounded_actionsranges
            noop_actions = self.grounded_noop_actions
            
        total_non_default = 0
        for (var, values) in actions.items():
            
            # check that action is valid
            prange = action_ranges.get(var, None)
            if prange is None:
                raise RDDLInvalidActionError(
                    f'<{var}> is not a valid action fluent, '
                    f'must be one of {set(action_ranges.keys())}.')
            
            # check that action shape is valid
            default_values = noop_actions[var]
            if np.shape(values) != np.shape(default_values):
                raise RDDLInvalidActionError(
                    f'Value array for action <{var}> must be of shape '
                    f'{np.shape(default_values)}, got array of shape '
                    f'{np.shape(values)}.')
            
            # accumulate count of non-default actions
            if enforce_for_non_bool or prange == 'bool':
                total_non_default += np.count_nonzero(values != default_values)
        
        if total_non_default > self.rddl.max_allowed_actions:
            raise RDDLInvalidActionError(
                f'Expected at most {self.rddl.max_allowed_actions} '
                f'non-default actions, got {total_non_default}.')
        
    def check_action_preconditions(self, actions: Args, silent: bool=False) -> bool:
        '''Throws an exception if the action preconditions are not satisfied.'''     
        actions = self._process_actions(actions)
        self.subs.update(actions)
        
        for (i, precond) in enumerate(self.rddl.preconditions):
            loc = self.precond_names[i]
            sample = self._sample(precond, self.subs)
            RDDLSimulator._check_type(sample, bool, loc, precond)
            if not bool(sample):
                if not silent:
                    raise RDDLActionPreconditionNotSatisfiedError(
                        f'{loc} is not satisfied for actions {actions}.\n' + 
                        print_stack_trace(precond))
                return False
        return True
    
    def check_terminal_states(self) -> bool:
        '''Return True if a terminal state has been reached.'''
        for (i, terminal) in enumerate(self.rddl.terminals):
            loc = self.terminal_names[i]
            sample = self._sample(terminal, self.subs)
            RDDLSimulator._check_type(sample, bool, loc, terminal)
            if bool(sample):
                return True
        return False
    
    def sample_reward(self) -> float:
        '''Samples the current reward given the current state and action.'''
        return float(self._sample(self.rddl.reward, self.subs))
    
    def reset(self) -> Union[Dict[str, None], Args]:
        '''Resets the state variables to their initial values.'''
        rddl = self.rddl
        subs = self.subs = self.init_values.copy()
        keep_tensors = self.keep_tensors
        
        # update state
        self.state = {}
        for state in rddl.states:
            if keep_tensors:
                self.state[state] = subs[state]
            else:
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
        rddl = self.rddl
        keep_tensors = self.keep_tensors
        actions = self._process_actions(actions)
        subs = self.subs
        subs.update(actions)
        
        # evaluate CPFs in topological order
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
            if keep_tensors:
                self.state[state] = subs[state]
            else:
                self.state.update(rddl.ground_values(state, subs[state]))
        
        # update observation
        if self._pomdp: 
            obs = {}
            for var in rddl.observ:
                if keep_tensors:
                    obs[var] = subs[var]
                else:
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
        elif etype == 'randomvector':
            return self._sample_random_vector(expr, subs)
        elif etype == 'matrix':
            return self._sample_matrix(expr, subs)
        else:
            raise RDDLNotImplementedError(
                f'Internal error: expression type {etype} is not supported.\n' + 
                print_stack_trace(expr))
                
    # ===========================================================================
    # leaves
    # ===========================================================================
        
    def _sample_constant(self, expr, _):
        return self.traced.cached_sim_info(expr)
    
    def _sample_pvar(self, expr, subs):
        var, args = expr.args
        
        # free variable (e.g., ?x) and object converted to canonical index
        is_value, cached_info = self.traced.cached_sim_info(expr)
        if is_value:
            return cached_info
        
        # extract variable value
        sample = subs.get(var, None)
        if sample is None:
            raise RDDLUndefinedVariableError(
                f'Variable <{var}> is referenced before assignment.\n' + 
                print_stack_trace(expr))
        
        # lifted domain must slice and/or reshape value tensor
        if cached_info is not None:
            slices, axis, shape, op_code, op_args = cached_info
            if slices: 
                if op_code == RDDLObjectsTracer.NUMPY_OP_CODE.NESTED_SLICE:
                    slices = tuple(
                        (self._sample(arg, subs) if _slice is None else _slice)
                        for (arg, _slice) in zip(args, slices)
                    )
                sample = sample[slices]
            if axis:
                sample = np.expand_dims(sample, axis=axis)
                sample = np.broadcast_to(sample, shape=shape)
            if op_code == RDDLObjectsTracer.NUMPY_OP_CODE.EINSUM:
                sample = np.einsum(sample, *op_args)
            elif op_code == RDDLObjectsTracer.NUMPY_OP_CODE.TRANSPOSE:
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
                        f'Cannot evaluate arithmetic operation {op} '
                        f'at {sample_lhs} and {sample_rhs}.\n' + 
                        print_stack_trace(expr))
        
        # for a grounded domain can short-circuit * and +
        elif n > 0 and not self.traced.cached_objects_in_scope(expr):
            if op == '*':
                return self._sample_product_grounded(args, subs)
            elif op == '+':
                return sum(1 * self._sample(arg, subs) for arg in args)
        
        raise RDDLInvalidNumberOfArgumentsError(
            f'Arithmetic operator {op} does not have the required '
            f'number of arguments.\n' + print_stack_trace(expr))
    
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
            f'number of arguments.\n' + print_stack_trace(expr))
    
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
        use_and = op == '^'
        
        # go through simple expressions first
        for (i, arg) in enumerate(args):
            if arg.is_constant_expression() or arg.is_pvariable_expression():
                sample = self._sample(arg, subs)
                RDDLSimulator._check_type(sample, bool, op, expr, arg=i + 1)
                sample = bool(sample)
                if use_and and not sample:
                    return False
                elif not use_and and sample:
                    return True
        
        # go through complex expressions last
        for (i, arg) in enumerate(args):
            if not (arg.is_constant_expression() or arg.is_pvariable_expression()):
                sample = self._sample(arg, subs)
                RDDLSimulator._check_type(sample, bool, op, expr, arg=i + 1)
                sample = bool(sample)
                if use_and and not sample:
                    return False
                elif not use_and and sample:
                    return True
            
        return use_and
            
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
        if op in self.AGGREGATION_BOOL:
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
            try:
                return unary_op(sample)
            except:
                raise ArithmeticError(
                    f'Cannot evaluate unary function {name} at {sample}.\n' + 
                    print_stack_trace(expr))
        
        # binary function
        binary_op = self.BINARY.get(name, None)
        if binary_op is not None:
            RDDLSimulator._check_arity(args, 2, name, expr)
            lhs, rhs = args
            sample_lhs = 1 * self._sample(lhs, subs)
            sample_rhs = 1 * self._sample(rhs, subs)
            if name in self.BINARY_REQUIRES_INT:
                RDDLSimulator._check_type(
                    sample_lhs, RDDLValueInitializer.INT, name, expr, arg=1)
                RDDLSimulator._check_type(
                    sample_rhs, RDDLValueInitializer.INT, name, expr, arg=2)
            try:
                return binary_op(sample_lhs, sample_rhs)
            except:
                raise ArithmeticError(
                    f'Cannot evaluate binary function {name} at '
                    f'{sample_lhs} and {sample_rhs}.\n' + print_stack_trace(expr))
        
        raise RDDLNotImplementedError(
            f'Function {name} is not supported.\n' + print_stack_trace(expr))
    
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
            sample_def = None if default is None else self._sample(default, subs)
            sample_cases = np.asarray([
                (sample_def if arg is None else self._sample(arg, subs))
                for arg in cases
            ])
            sample_pred = sample_pred[np.newaxis, ...]
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
        elif name == 'ChiSquare':
            return self._sample_chisquare(expr, subs)
        elif name == 'Kumaraswamy':
            return self._sample_kumaraswamy(expr, subs)
        elif name == 'Discrete':
            return self._sample_discrete(expr, subs, unnorm=False)
        elif name == 'UnnormDiscrete':
            return self._sample_discrete(expr, subs, unnorm=True)
        elif name == 'Discrete(p)':
            return self._sample_discrete_pvar(expr, subs, unnorm=False)
        elif name == 'UnnormDiscrete(p)':
            return self._sample_discrete_pvar(expr, subs, unnorm=True)
        else:
            raise RDDLNotImplementedError(
                f'Distribution {name} is not supported.\n' + 
                print_stack_trace(expr))

    def _sample_kron_delta(self, expr, subs):
        args = expr.args
        RDDLSimulator._check_arity(args, 1, 'KronDelta', expr)
        
        arg, = args
        sample = self._sample(arg, subs)
        RDDLSimulator._check_types(
            sample, (bool, RDDLValueInitializer.INT), 'Argument of KronDelta', expr)
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
        return self.rng.uniform(low=sample_lb, high=sample_ub)      
    
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
        return self.rng.normal(loc=sample_mean, scale=sample_std)
    
    def _sample_poisson(self, expr, subs):
        args = expr.args
        RDDLSimulator._check_arity(args, 1, 'Poisson', expr)
        
        rate, = args
        sample_rate = self._sample(rate, subs)
        RDDLSimulator._check_positive(sample_rate, False, 'Poisson rate', expr)        
        return self.rng.poisson(lam=sample_rate)
    
    def _sample_exponential(self, expr, subs):
        args = expr.args
        RDDLSimulator._check_arity(args, 1, 'Exponential', expr)
        
        scale, = expr.args
        sample_scale = self._sample(scale, subs)
        RDDLSimulator._check_positive(sample_scale, True, 'Exponential rate', expr)
        return self.rng.exponential(scale=sample_scale)
    
    def _sample_weibull(self, expr, subs):
        args = expr.args
        RDDLSimulator._check_arity(args, 2, 'Weibull', expr)
        
        shape, scale = args
        sample_shape = self._sample(shape, subs)
        sample_scale = self._sample(scale, subs)
        RDDLSimulator._check_positive(sample_shape, True, 'Weibull shape', expr)
        RDDLSimulator._check_positive(sample_scale, True, 'Weibull scale', expr)
        return sample_scale * self.rng.weibull(a=sample_shape)
    
    def _sample_gamma(self, expr, subs):
        args = expr.args
        RDDLSimulator._check_arity(args, 2, 'Gamma', expr)
        
        shape, scale = args
        sample_shape = self._sample(shape, subs)
        sample_scale = self._sample(scale, subs)
        RDDLSimulator._check_positive(sample_shape, True, 'Gamma shape', expr)            
        RDDLSimulator._check_positive(sample_scale, True, 'Gamma scale', expr)        
        return self.rng.gamma(shape=sample_shape, scale=sample_scale)
    
    def _sample_binomial(self, expr, subs):
        args = expr.args
        RDDLSimulator._check_arity(args, 2, 'Binomial', expr)
        
        count, pr = args
        sample_count = self._sample(count, subs)
        sample_pr = self._sample(pr, subs)
        RDDLSimulator._check_type(sample_count, RDDLValueInitializer.INT, 'Binomial count', expr)
        RDDLSimulator._check_positive(sample_count, False, 'Binomial count', expr)
        RDDLSimulator._check_range(sample_pr, 0, 1, 'Binomial p', expr)
        return self.rng.binomial(n=sample_count, p=sample_pr)
    
    def _sample_negative_binomial(self, expr, subs):
        args = expr.args
        RDDLSimulator._check_arity(args, 2, 'NegativeBinomial', expr)
        
        count, pr = args
        sample_count = self._sample(count, subs)
        sample_pr = self._sample(pr, subs)
        RDDLSimulator._check_positive(sample_count, True, 'NegativeBinomial r', expr)
        RDDLSimulator._check_range(sample_pr, 0, 1, 'NegativeBinomial p', expr)        
        return self.rng.negative_binomial(n=sample_count, p=sample_pr)
    
    def _sample_beta(self, expr, subs):
        args = expr.args
        RDDLSimulator._check_arity(args, 2, 'Beta', expr)
        
        shape, rate = args
        sample_shape = self._sample(shape, subs)
        sample_rate = self._sample(rate, subs)
        RDDLSimulator._check_positive(sample_shape, True, 'Beta shape', expr)
        RDDLSimulator._check_positive(sample_rate, True, 'Beta rate', expr)        
        return self.rng.beta(a=sample_shape, b=sample_rate)

    def _sample_geometric(self, expr, subs):
        args = expr.args
        RDDLSimulator._check_arity(args, 1, 'Geometric', expr)
        
        pr, = args
        sample_pr = self._sample(pr, subs)
        RDDLSimulator._check_range(sample_pr, 0, 1, 'Geometric p', expr)        
        return self.rng.geometric(p=sample_pr)
    
    def _sample_pareto(self, expr, subs):
        args = expr.args
        RDDLSimulator._check_arity(args, 2, 'Pareto', expr)
        
        shape, scale = args
        sample_shape = self._sample(shape, subs)
        sample_scale = self._sample(scale, subs)
        RDDLSimulator._check_positive(sample_shape, True, 'Pareto shape', expr)        
        RDDLSimulator._check_positive(sample_scale, True, 'Pareto scale', expr)        
        return sample_scale * self.rng.pareto(a=sample_shape)
    
    def _sample_student(self, expr, subs):
        args = expr.args
        RDDLSimulator._check_arity(args, 1, 'Student', expr)
        
        df, = args
        sample_df = self._sample(df, subs)
        RDDLSimulator._check_positive(sample_df, True, 'Student df', expr)            
        return self.rng.standard_t(df=sample_df)

    def _sample_gumbel(self, expr, subs):
        args = expr.args
        RDDLSimulator._check_arity(args, 2, 'Gumbel', expr)
        
        mean, scale = args
        sample_mean = self._sample(mean, subs)
        sample_scale = self._sample(scale, subs)
        RDDLSimulator._check_positive(sample_scale, True, 'Gumbel scale', expr)
        return self.rng.gumbel(loc=sample_mean, scale=sample_scale)
    
    def _sample_laplace(self, expr, subs):
        args = expr.args
        RDDLSimulator._check_arity(args, 2, 'Laplace', expr)
        
        mean, scale = args
        sample_mean = self._sample(mean, subs)
        sample_scale = self._sample(scale, subs)
        RDDLSimulator._check_positive(sample_scale, True, 'Laplace scale', expr)
        return self.rng.laplace(loc=sample_mean, scale=sample_scale)
    
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
    
    def _sample_chisquare(self, expr, subs):
        args = expr.args
        RDDLSimulator._check_arity(args, 1, 'ChiSquare', expr)
        
        df, = args
        sample_df = self._sample(df, subs)
        RDDLSimulator._check_positive(sample_df, True, 'ChiSquare df', expr)
        return self.rng.chisquare(df=sample_df)
    
    def _sample_kumaraswamy(self, expr, subs):
        args = expr.args
        RDDLSimulator._check_arity(args, 2, 'Kumaraswamy', expr)
        
        a, b = args
        sample_a = self._sample(a, subs)
        sample_b = self._sample(b, subs)
        RDDLSimulator._check_positive(sample_a, True, 'Kumaraswamy a', expr)
        RDDLSimulator._check_positive(sample_b, True, 'Kumaraswamy b', expr)
        size = sample_a.shape if self.traced.cached_objects_in_scope(expr) else None
        U = self.rng.uniform(size=size)
        return (1.0 - U ** (1.0 / sample_b)) ** (1.0 / sample_a)
    
    # ===========================================================================
    # random variables with enum support
    # ===========================================================================
    
    def _sample_discrete_helper(self, pdf, unnorm, expr):
        RDDLSimulator._check_positive(pdf, False, 'Discrete probabilities', expr)
        
        # calculate CDF       
        cdf = np.cumsum(pdf, axis=-1)
        if unnorm:
            cdf = cdf / cdf[..., -1:]
            
        # check valid CDF - still do this for unnorm to reject nan values
        if not np.allclose(cdf[..., -1], 1.0):
            raise RDDLValueOutOfRangeError(
                f'Discrete probabilities must sum to 1, got {cdf[..., -1]}.\n' + 
                print_stack_trace(expr))     
        
        # use inverse CDF sampling                  
        U = self.rng.random(size=cdf.shape[:-1] + (1,))
        return np.argmax(U < cdf, axis=-1)
        
    def _sample_discrete(self, expr, subs, unnorm):
        sorted_args = self.traced.cached_sim_info(expr)
        samples = [self._sample(arg, subs) for arg in sorted_args]
        pdf = np.stack(samples, axis=-1)
        return self._sample_discrete_helper(pdf, unnorm, expr)
    
    def _sample_discrete_pvar(self, expr, subs, unnorm):
        _, args = expr.args
        arg, = args
        pdf = self._sample(arg, subs)
        return self._sample_discrete_helper(pdf, unnorm, expr)
               
    # ===========================================================================
    # random vectors
    # ===========================================================================
    
    def _sample_random_vector(self, expr, subs):
        _, name = expr.etype
        if name == 'MultivariateNormal':
            return self._sample_multivariate_normal(expr, subs)
        elif name == 'MultivariateStudent':
            return self._sample_multivariate_student(expr, subs)
        elif name == 'Dirichlet':
            return self._sample_dirichlet(expr, subs)
        elif name == 'Multinomial':
            return self._sample_multinomial(expr, subs)
        else:
            raise RDDLNotImplementedError(
                f'Multivariate distribution {name} is not supported.\n' + 
                print_stack_trace(expr))
    
    def _sample_multivariate_normal(self, expr, subs):
        _, args = expr.args
        RDDLSimulator._check_arity(args, 2, 'MultivariateNormal', expr)
        
        mean, cov = args
        sample_mean = self._sample(mean, subs)
        sample_cov = self._sample(cov, subs)
        
        # reparameterization trick MN(m, LL') = LZ + m, where Z ~ Normal(0, 1)
        L = np.linalg.cholesky(sample_cov)
        Z = self.rng.standard_normal(
            size=sample_mean.shape + (1,),
            dtype=RDDLValueInitializer.REAL)
        sample = np.matmul(L, Z)[..., 0] + sample_mean

        # since the sampling is done in the last dimension we need to move it
        # to match the order of the CPF variables
        index, = self.traced.cached_sim_info(expr)
        sample = np.moveaxis(sample, source=-1, destination=index)
        return sample
    
    def _sample_multivariate_student(self, expr, subs):
        _, args = expr.args
        RDDLSimulator._check_arity(args, 3, 'MultivariateStudent', expr)
        
        mean, cov, df = args
        sample_mean = self._sample(mean, subs)
        sample_cov = self._sample(cov, subs)
        sample_df = self._sample(df, subs)
        RDDLSimulator._check_positive(sample_df, True, 'MultivariateStudent df', expr)
        
        # reparameterization trick MN(m, LL') = LZ + m, where Z ~ StudentT(0, 1)
        sample_df = sample_df[..., np.newaxis, np.newaxis]
        sample_df = np.broadcast_to(sample_df, shape=sample_mean.shape + (1,))
        L = np.linalg.cholesky(sample_cov)
        Z = self.rng.standard_t(df=sample_df)
        sample = np.matmul(L, Z)[..., 0] + sample_mean
        
        # since the sampling is done in the last dimension we need to move it
        # to match the order of the CPF variables
        index, = self.traced.cached_sim_info(expr)
        sample = np.moveaxis(sample, source=-1, destination=index)
        return sample
    
    def _sample_dirichlet(self, expr, subs):
        _, args = expr.args
        RDDLSimulator._check_arity(args, 1, 'Dirichlet', expr)
        
        alpha, = args
        sample_alpha = self._sample(alpha, subs)
        
        # sample Gamma(alpha_i, 1) and normalize across i
        RDDLSimulator._check_positive(sample_alpha, True, 'Dirichlet alpha', expr)
        Gamma = self.rng.gamma(shape=sample_alpha, scale=1.0)
        sample = Gamma / np.sum(Gamma, axis=-1, keepdims=True)
        
        # since the sampling is done in the last dimension we need to move it
        # to match the order of the CPF variables
        index, = self.traced.cached_sim_info(expr)
        sample = np.moveaxis(sample, source=-1, destination=index)
        return sample
    
    def _sample_multinomial(self, expr, subs):
        _, args = expr.args
        RDDLSimulator._check_arity(args, 2, 'Multinomial', expr)
        
        trials, prob = args
        sample_trials = self._sample(trials, subs) 
        sample_prob = self._sample(prob, subs)       
        RDDLSimulator._check_type(sample_trials, RDDLValueInitializer.INT, 'Multinomial trials', expr)
        RDDLSimulator._check_positive(sample_trials, False, 'Multinomial trials', expr)
        RDDLSimulator._check_positive(sample_prob, False, 'Discrete probabilities', expr)
        
        # check valid PMF
        cum_prob = np.sum(sample_prob, axis=-1)
        if not np.allclose(cum_prob, 1.0):
            raise RDDLValueOutOfRangeError(
                f'Multinomial probabilities must sum to 1, got {cum_prob}.\n' + 
                print_stack_trace(expr))    
            
        # sample from the multinomial
        sample = self.rng.multinomial(n=sample_trials, pvals=sample_prob)
        
        # since the sampling is done in the last dimension we need to move it
        # to match the order of the CPF variables
        index, = self.traced.cached_sim_info(expr)
        sample = np.moveaxis(sample, source=-1, destination=index)
        return sample
        
    # ===========================================================================
    # matrix algebra
    # ===========================================================================
    
    def _sample_matrix(self, expr, subs):
        _, op = expr.etype
        if op == 'det':
            return self._sample_matrix_det(expr, subs)
        elif op == 'inverse':
            return self._sample_matrix_inv(expr, subs, pseudo=False)
        elif op == 'pinverse':
            return self._sample_matrix_inv(expr, subs, pseudo=True)
        elif op == 'cholesky':
            return self._sample_matrix_cholesky(expr, subs)
        else:
            raise RDDLNotImplementedError(
                f'Matrix operator {op} is not supported.\n' + 
                print_stack_trace(expr))
    
    def _sample_matrix_det(self, expr, subs):
        * _, arg = expr.args
        sample_arg = self._sample(arg, subs)
        return np.linalg.det(sample_arg)
    
    def _sample_matrix_inv(self, expr, subs, pseudo):
        _, arg = expr.args
        sample_arg = self._sample(arg, subs)
        op = np.linalg.pinv if pseudo else np.linalg.inv
        sample = op(sample_arg)
        
        # matrix dimensions are last two axes, move them to the correct position
        indices = self.traced.cached_sim_info(expr)
        sample = np.moveaxis(sample, source=(-2, -1), destination=indices)
        return sample        
    
    def _sample_matrix_cholesky(self, expr, subs):
        _, arg = expr.args
        sample_arg = self._sample(arg, subs)
        op = np.linalg.cholesky
        sample = op(sample_arg)
        
        # matrix dimensions are last two axes, move them to the correct position
        indices = self.traced.cached_sim_info(expr)
        sample = np.moveaxis(sample, source=(-2, -1), destination=indices)
        return sample  
    
    
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


# A container class for compiling a simulator but from external info
class RDDLSimulatorPrecompiled(RDDLSimulator):
    
    def __init__(self, rddl: PlanningModel, 
                 init_values: Args, 
                 levels: Dict[int, Set[str]], 
                 trace_info: object,
                 rng: np.random.Generator=np.random.default_rng(),
                 keep_tensors: bool=False) -> None:
        self.init_values = init_values
        self.levels = levels
        self.traced = trace_info
        
        super(RDDLSimulatorPrecompiled, self).__init__(
            rddl=rddl, 
            allow_synchronous_state=True,
            rng=rng,
            logger=None,
            keep_tensors=keep_tensors)        
    
    def _compile(self):
        rddl = self.rddl
        
        # compute dependency graph for CPFs and sort them by evaluation order   
        self.cpfs = []  
        for cpfs in self.levels.values():
            for cpf in cpfs:
                _, expr = rddl.cpfs[cpf]
                prange = rddl.variable_ranges[cpf]
                dtype = RDDLValueInitializer.NUMPY_TYPES.get(
                    prange, RDDLValueInitializer.INT)
                self.cpfs.append((cpf, expr, dtype))
                
        # initialize all fluent and non-fluent values        
        self.subs = self.init_values.copy()
        self.state = None  
        self.noop_actions = {var: values
                             for (var, values) in self.init_values.items()
                             if rddl.variable_types[var] == 'action-fluent'}
        self._pomdp = bool(rddl.observ)
        
        # cached for performance
        self.invariant_names = [f'Invariant {i}' for i in range(len(rddl.invariants))]        
        self.precond_names = [f'Precondition {i}' for i in range(len(rddl.preconditions))]
        self.terminal_names = [f'Termination {i}' for i in range(len(rddl.terminals))]
        
        self.grounded_actionsranges = rddl.groundactionsranges()
        self.grounded_noop_actions = rddl.ground_values_from_dict(self.noop_actions)
        