import numpy as np
np.seterr(all='raise')
from typing import Dict, Union
import warnings

from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLActionPreconditionNotSatisfiedError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidActionError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidNumberOfArgumentsError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidObjectError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLNotImplementedError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLStateInvariantNotSatisfiedError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLTypeError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLUndefinedVariableError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLValueOutOfRangeError

from pyRDDLGym.Core.Compiler.RDDLDecompiler import RDDLDecompiler
from pyRDDLGym.Core.Compiler.RDDLLevelAnalysis import RDDLLevelAnalysis
from pyRDDLGym.Core.Compiler.RDDLModel import RDDLModel
from pyRDDLGym.Core.Parser.expr import Expression, Value
from pyRDDLGym.Core.Simulator.RDDLTensors import RDDLTensors

Args = Dict[str, Value]

        
class RDDLSimulator:
    
    def __init__(self, rddl: RDDLModel,
                 allow_synchronous_state: bool=True,
                 rng: np.random.Generator=np.random.default_rng(),
                 debug: bool=False) -> None:
        '''Creates a new simulator for the given RDDL model.
        
        :param rddl: the RDDL model
        :param allow_synchronous_state: whether state-fluent can be synchronous
        :param rng: the random number generator
        :param debug: whether to print compiler information
        '''
        self.rddl = rddl
        self.rng = rng
        self.debug = debug
        
        # static analysis and compilation
        self.static = RDDLLevelAnalysis(rddl, allow_synchronous_state)
        self.levels = self.static.compute_levels()
        self.tensors = RDDLTensors(rddl, debug=debug)
        
        # initialize all fluent and non-fluent values
        self.init_values = self.tensors.init_values
        self.subs = self.init_values.copy()
        self.state = None
        
        self.noop_actions, self.next_states, self.observ_fluents = {}, {}, []
        for name, value in self.init_values.items():
            var = rddl.parse(name)[0]
            vtype = rddl.variable_types[var]
            if vtype == 'action-fluent':
                self.noop_actions[name] = value
            elif vtype == 'state-fluent':
                self.next_states[name + '\''] = name
            elif vtype == 'observ-fluent':
                self.observ_fluents.append(name)
        self._pomdp = bool(self.observ_fluents)
        
        # enumerated types are converted to integers internally
        self._cpf_dtypes = {}
        for cpfs in self.levels.values():
            for cpf in cpfs:
                var = rddl.parse(cpf)[0]
                prange = rddl.variable_ranges[var]
                if prange in rddl.enum_types:
                    prange = 'int'
                self._cpf_dtypes[cpf] = RDDLTensors.NUMPY_TYPES[prange]
        
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
            '|': np.logical_or,
            '~': np.logical_xor,
            '=>': lambda e1, e2: np.logical_or(np.logical_not(e1), e2),
            '<=>': np.equal
        }
        self.AGGREGATION_OPS = {
            'sum': np.sum,
            'avg': np.mean,
            'prod': np.prod,
            'min': np.min,
            'max': np.max,
            'forall': np.all,
            'exists': np.any  
        }
        self.UNARY = {        
            'abs': np.abs,
            'sgn': lambda x: np.sign(x).astype(RDDLTensors.INT),
            'round': lambda x: np.round(x).astype(RDDLTensors.INT),
            'floor': lambda x: np.floor(x).astype(RDDLTensors.INT),
            'ceil': lambda x: np.ceil(x).astype(RDDLTensors.INT),
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
            'div': lambda x, y: np.floor_divide(x, y).astype(RDDLTensors.INT),
            'mod': lambda x, y: np.mod(x, y).astype(RDDLTensors.INT),
            'min': np.minimum,
            'max': np.maximum,
            'pow': np.power,
            'log': lambda x, y: np.log(x) / np.log(y)
        }
        self.CONTROL_OPS = {'if', 'switch'}
    
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
    def _check_type(value, valid, msg, expr):
        if not np.can_cast(value, valid):
            dtype = getattr(value, 'dtype', type(value))
            raise RDDLTypeError(
                f'{msg} must evaluate to {valid}, got {value} of type {dtype}.\n' + 
                RDDLSimulator._print_stack_trace(expr))
    
    @staticmethod
    def _check_type_in(value, valid, msg, expr):
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
        if op not in valid:
            raise RDDLNotImplementedError(
                f'{msg} operator {op} is not supported: must be in {valid}.\n' + 
                RDDLSimulator._print_stack_trace(expr))
    
    @staticmethod
    def _check_arity(args, required, msg, expr):
        actual = len(args)
        if actual != required:
            raise RDDLInvalidNumberOfArgumentsError(
                f'{msg} requires {required} arguments, got {actual}.\n' + 
                RDDLSimulator._print_stack_trace(expr))
    
    @staticmethod
    def _check_positive(value, strict, msg, expr):
        if strict:
            failed = not np.all(value > 0)
            message = f'{msg} must be positive, got {value}.\n'
        else:
            failed = not np.all(value >= 0)
            message = f'{msg} must be non-negative, got {value}.\n'
        if failed:
            raise RDDLValueOutOfRangeError(
                message + RDDLSimulator._print_stack_trace(expr))
    
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
    
    @staticmethod
    def _raise_unsupported(msg, expr):
        raise RDDLNotImplementedError(
            f'{msg} is not supported in the current RDDL version.\n' + 
            RDDLSimulator._print_stack_trace(expr))
    
    @staticmethod
    def _raise_no_args(op, msg, expr):
        raise RDDLInvalidNumberOfArgumentsError(
            f'{msg} operator {op} has no arguments.\n' + 
            RDDLSimulator._print_stack_trace(expr))
    
    # ===========================================================================
    # main sampling routines
    # ===========================================================================
    
    def _process_actions(self, actions):
        new_actions = {action: np.copy(value) 
                       for action, value in self.noop_actions.items()}
        
        for action, value in actions.items(): 
            if action not in self.rddl.actions:
                raise RDDLInvalidActionError(
                    f'<{action}> is not a valid action-fluent.')
            
            if self.rddl.is_grounded:
                new_actions[action] = value                
            else:
                var, objects = self.rddl.parse(action)
                tensor = new_actions[var]                
                RDDLSimulator._check_type(
                    value, tensor.dtype, f'Action-fluent <{action}>', '')            
                tensor[self.tensors.coordinates(objects, '')] = value
         
        return new_actions
    
    def check_state_invariants(self) -> None:
        '''Throws an exception if the state invariants are not satisfied.'''
        for i, invariant in enumerate(self.rddl.invariants):
            sample = self._sample(invariant, [], self.subs)
            RDDLSimulator._check_type(
                sample, bool, f'Invariant {i + 1}', invariant)
            if not bool(sample):
                raise RDDLStateInvariantNotSatisfiedError(
                    f'Invariant {i + 1} is not satisfied.\n' + 
                    RDDLSimulator._print_stack_trace(invariant))
    
    def check_action_preconditions(self, actions: Args) -> None:
        '''Throws an exception if the action preconditions are not satisfied.'''        
        actions = self._process_actions(actions)
        self.subs.update(actions)
        
        for i, precond in enumerate(self.rddl.preconditions):
            sample = self._sample(precond, [], self.subs)
            RDDLSimulator._check_type(
                sample, bool, f'Precondition {i + 1}', precond)
            if not bool(sample):
                raise RDDLActionPreconditionNotSatisfiedError(
                    f'Precondition {i + 1} is not satisfied.\n' + 
                    RDDLSimulator._print_stack_trace(precond))
    
    def check_terminal_states(self) -> bool:
        '''Return True if a terminal state has been reached.'''
        for i, terminal in enumerate(self.rddl.terminals):
            sample = self._sample(terminal, [], self.subs)
            RDDLSimulator._check_type(
                sample, bool, f'Termination {i + 1}', terminal)
            if bool(sample):
                return True
        return False
    
    def sample_reward(self) -> float:
        '''Samples the current reward given the current state and action.'''
        sample = self._sample(self.rddl.reward, [], self.subs)
        return float(sample)    
    
    def reset(self) -> Union[Dict[str, None], Args]:
        '''Resets the state variables to their initial values.'''
        subs = self.subs = self.init_values.copy()
        
        if self.rddl.is_grounded:
            self.state = {var: subs[var] for var in self.next_states.values()}
        else:
            self.state = {}
            for var in self.next_states.values():
                self.state.update(self.tensors.expand(var, subs[var]))
            
        if self._pomdp:
            obs = {var: None for var in self.observ_fluents}
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
        
        tensors, rddl = self.tensors, self.rddl
        for cpfs in self.levels.values():
            for cpf in cpfs:
                objects, expr = rddl.cpfs[cpf]
                sample = self._sample(expr, objects, subs)
                dtype = self._cpf_dtypes[cpf]
                RDDLSimulator._check_type(sample, dtype, f'CPF <{cpf}>', expr)
                subs[cpf] = sample
        reward = self.sample_reward()
        
        if rddl.is_grounded:
            for next_state, state in self.next_states.items():
                subs[state] = subs[next_state]
            self.state = {var: subs[var] for var in self.next_states.values()}
        else:
            self.state = {}
            for next_state, state in self.next_states.items():
                subs[state] = subs[next_state]
                self.state.update(tensors.expand(state, subs[state]))
            
        if self._pomdp: 
            if rddl.is_grounded:
                obs = {var: subs[var] for var in self.observ_fluents}
            else:
                obs = {}
                for var in self.observ_fluents:
                    obs.update(tensors.expand(var, subs[var]))
        else:
            obs = self.state
        
        done = self.check_terminal_states()        
        return obs, reward, done
        
    # ===========================================================================
    # start of sampling subroutines
    # ===========================================================================
    
    def _sample(self, expr, objects, subs):
        etype, _ = expr.etype
        if etype == 'constant':
            return self._sample_constant(expr, objects, subs)
        elif etype == 'pvar':
            return self._sample_pvar(expr, objects, subs)
        elif etype == 'arithmetic':
            return self._sample_arithmetic(expr, objects, subs)
        elif etype == 'relational':
            return self._sample_relational(expr, objects, subs)
        elif etype == 'boolean':
            return self._sample_logical(expr, objects, subs)
        elif etype == 'aggregation':
            return self._sample_aggregation(expr, objects, subs)
        elif etype == 'func':
            return self._sample_func(expr, objects, subs)
        elif etype == 'control':
            return self._sample_control(expr, objects, subs)
        elif etype == 'randomvar':
            return self._sample_random(expr, objects, subs)
        else:
            raise RDDLNotImplementedError(
                f'Internal error: expression {expr} is not recognized.')
                
    # ===========================================================================
    # leaves
    # ===========================================================================
        
    def _sample_constant(self, expr, objects, _):
        
        # grounded domain only returns a scalar
        if self.rddl.is_grounded:
            return np.asarray(expr.args)
        
        # argument is reshaped to match the free variables "objects"        
        cached_value = expr.cached_sim_info
        if cached_value is None:
            shape = self.tensors.shape([pobj[1] for pobj in objects])
            cached_value = np.full(shape=shape, fill_value=expr.args)
            expr.cached_sim_info = cached_value
        return cached_value
    
    def _sample_pvar(self, expr, objects, subs):
        _, name = expr.etype
        args = expr.args
        RDDLSimulator._check_arity(args, 2, f'Variable <{name}>', expr)
        var, pvars = args
        
        # literal of enumerated type is treated as integer
        if var in self.rddl.enum_literals:
            enum_index = self.tensors.index_of_object[var]

            # grounded domain only returns a scalar
            if self.rddl.is_grounded:
                return np.asarray(enum_index)
                    
            # argument is reshaped to match the free variables "objects"
            cached_value = expr.cached_sim_info
            if cached_value is None:
                shape = self.tensors.shape([pobj[1] for pobj in objects])
                cached_value = np.full(shape=shape, fill_value=enum_index)
                expr.cached_sim_info = cached_value
            return cached_value
        
        # extract variable value
        arg = subs.get(var, None)
        if arg is None:
            raise RDDLUndefinedVariableError(
                f'Variable <{var}> is referenced before assignment.\n' + 
                RDDLSimulator._print_stack_trace(expr))
        
        # grounded domain only returns a scalar
        if self.rddl.is_grounded:
            return np.asarray(arg)
        
        # argument is reshaped to match the free variables "objects"
        cached_transform = expr.cached_sim_info
        if cached_transform is None:
            cached_transform = self.tensors.map(
                var, pvars, objects,
                msg=RDDLSimulator._print_stack_trace(expr))            
            expr.cached_sim_info = cached_transform        
        return cached_transform(arg)
    
    # ===========================================================================
    # arithmetic
    # ===========================================================================
    
    def _sample_arithmetic(self, expr, objects, subs):
        _, op = expr.etype
        valid_ops = self.ARITHMETIC_OPS
        RDDLSimulator._check_op(op, valid_ops, 'Arithmetic', expr)
        
        args = expr.args        
        n = len(args)        
        if n == 1 and op == '-':
            arg, = args
            return -1 * self._sample(arg, objects, subs)
        
        elif n == 2:
            if op == '*':
                return self._sample_product(args, objects, subs)
            else:
                lhs, rhs = args
                lhs = 1 * self._sample(lhs, objects, subs)
                rhs = 1 * self._sample(rhs, objects, subs)
                return valid_ops[op](lhs, rhs)
        
        elif self.rddl.is_grounded and n > 0:
            if op == '*':
                return self._sample_product_grounded(args, objects, subs)
            elif op == '+':
                samples = [self._sample(arg, objects, subs) for arg in args]
                return np.sum(samples, axis=0)                
            
        RDDLSimulator._check_arity(args, 2, 'Arithmetic operator', expr)
    
    def _sample_product(self, args, objects, subs):
        lhs, rhs = args
        if rhs.is_constant_expression() or rhs.is_pvariable_expression():
            lhs, rhs = rhs, lhs
            
        lhs = 1 * self._sample(lhs, objects, subs)
        if not np.any(lhs):
            return lhs
            
        rhs = self._sample(rhs, objects, subs)
        return lhs * rhs
    
    def _sample_product_grounded(self, args, objects, subs):
        prod = 1
        for arg in args:  # go through simple expressions first
            if arg.is_constant_expression() or arg.is_pvariable_expression():
                sample = self._sample(arg, objects, subs)
                prod *= sample.item()
                if prod == 0:
                    return np.asarray(prod)
                
        for arg in args:  # go through complex expressions last
            if not (arg.is_constant_expression() or arg.is_pvariable_expression()):
                sample = self._sample(arg, objects, subs)
                prod *= sample.item()
                if prod == 0:
                    return np.asarray(prod)
                
        return np.asarray(prod)
        
    # ===========================================================================
    # boolean
    # ===========================================================================
    
    def _sample_relational(self, expr, objects, subs):
        _, op = expr.etype
        args = expr.args
        valid_ops = self.RELATIONAL_OPS
        RDDLSimulator._check_op(op, valid_ops, 'Relational', expr)
        RDDLSimulator._check_arity(args, 2, f'Relational operator {op}', expr)
        
        lhs, rhs = args
        lhs = 1 * self._sample(lhs, objects, subs)
        rhs = 1 * self._sample(rhs, objects, subs)
        return valid_ops[op](lhs, rhs)
    
    def _sample_logical(self, expr, objects, subs):
        _, op = expr.etype
        args = expr.args
        valid_ops = self.LOGICAL_OPS
        RDDLSimulator._check_op(op, valid_ops, 'Logical', expr)
        
        n = len(args)
        if n == 1 and op == '~':
            arg, = args
            arg = self._sample(arg, objects, subs)
            RDDLSimulator._check_type(
                arg, bool, f'Argument of logical operator {op}', expr)
            return np.logical_not(arg)
        
        elif n == 2:
            if op == '^' or op == '|':
                return self._sample_and_or(args, op, expr, objects, subs)
            else:
                lhs, rhs = args
                lhs = self._sample(lhs, objects, subs)
                rhs = self._sample(rhs, objects, subs)
                RDDLSimulator._check_type(
                    lhs, bool, f'Argument 1 of logical operator {op}', expr)
                RDDLSimulator._check_type(
                    rhs, bool, f'Argument 2 of logical operator {op}', expr)
                return valid_ops[op](lhs, rhs)
        
        elif self.rddl.is_grounded and n > 0 and (op == '^' or op == '|'):
            return self._sample_and_or_grounded(args, op, expr, objects, subs)
            
        RDDLSimulator._check_arity(args, 2, 'Logical operator', expr)
    
    def _sample_and_or(self, args, op, expr, objects, subs):
        lhs, rhs = args
        if rhs.is_constant_expression() or rhs.is_pvariable_expression():
            lhs, rhs = rhs, lhs  # prioritize simple expressions
            
        lhs = self._sample(lhs, objects, subs)
        RDDLSimulator._check_type(
            lhs, bool, f'Argument 1 of logical operator {op}', expr)
            
        if (op == '^' and not np.any(lhs)) or (op == '|' and np.all(lhs)):
            return lhs
            
        rhs = self._sample(rhs, objects, subs)
        RDDLSimulator._check_type(
            rhs, bool, f'Argument 2 of logical operator {op}', expr)
        
        if op == '^':
            return np.logical_and(lhs, rhs)
        else:
            return np.logical_or(lhs, rhs)
    
    def _sample_and_or_grounded(self, args, op, expr, objects, subs): 
        for i, arg in enumerate(args):  # go through simple expressions first
            if arg.is_constant_expression() or arg.is_pvariable_expression():
                sample = self._sample(arg, objects, subs)
                RDDLSimulator._check_type(
                    sample, bool, f'Argument {i + 1} of logical operator {op}', expr)
                sample = bool(sample)
                if op == '^' and not sample:
                    return np.asarray(False)
                elif op == '|' and sample:
                    return np.asarray(True)
            
        for i, arg in enumerate(args):  # go through complex expressions last
            if not (arg.is_constant_expression() or arg.is_pvariable_expression()):
                sample = self._sample(arg, objects, subs)
                RDDLSimulator._check_type(
                    sample, bool, f'Argument {i + 1} of logical operator {op}', expr)
                sample = bool(sample)
                if op == '^' and not sample:
                    return np.asarray(False)
                elif op == '|' and sample:
                    return np.asarray(True)
            
        return np.asarray(op == '^')
            
    # ===========================================================================
    # aggregation
    # ===========================================================================
    
    def _sample_aggregation(self, expr, objects, subs):
        if self.rddl.is_grounded:
            raise Exception(
                f'Internal error: aggregation in grounded domain {expr}.')
        
        _, op = expr.etype
        args = expr.args
        valid_ops = self.AGGREGATION_OPS
        RDDLSimulator._check_op(op, valid_ops, 'Aggregation', expr)

        # cache and read reduced axes tensor info for the aggregation
        * pvars, arg = args
        cached_objects = expr.cached_sim_info
        if cached_objects is None:
            new_objects = objects + [p[1] for p in pvars]
            reduced_axes = tuple(range(len(objects), len(new_objects)))             
            cached_objects = (new_objects, reduced_axes)
            expr.cached_sim_info = cached_objects
            
            # check for undefined types
            bad_types = {p for _, p in new_objects if p not in self.rddl.objects}
            if bad_types:
                raise RDDLInvalidObjectError(
                    f'Type(s) {bad_types} are not defined, '
                    f'must be one of {set(self.rddl.objects.keys())}.\n' + 
                    RDDLSimulator._print_stack_trace(expr))
            
            # check for duplicated iteration variables
            for _, (free_new, _) in pvars:
                for free_old, _ in objects:
                    if free_new == free_old:
                        raise RDDLInvalidObjectError(
                            f'Iteration variable <{free_new}> is already defined '
                            f'in outer scope.\n' + 
                            RDDLSimulator._print_stack_trace(expr))
            
            # debug compiler info
            self.tensors.write_debug_message(
                f'computing object info for aggregation:'
                    f'\n\toperator       ={op} {pvars}'
                    f'\n\tinput objects  ={new_objects}'
                    f'\n\toutput objects ={objects}'
                    f'\n\toperation      ={valid_ops[op]}, axes={reduced_axes}\n'
            )                            
        new_objects, axis = cached_objects
        
        # sample the argument and aggregate over the reduced axes
        arg = self._sample(arg, new_objects, subs)                
        if op == 'forall' or op == 'exists':
            RDDLSimulator._check_type(
                arg, bool, f'Argument of aggregation {op}', expr)
        else:
            arg = 1 * arg
        return valid_ops[op](arg, axis=axis)
    
    # ===========================================================================
    # function
    # ===========================================================================
    
    def _sample_func(self, expr, objects, subs):
        _, name = expr.etype
        args = expr.args
        
        if name in self.UNARY:
            RDDLSimulator._check_arity(args, 1, f'Unary function {name}', expr)
            arg, = args
            arg = 1 * self._sample(arg, objects, subs)
            return self.UNARY[name](arg)
        
        elif name in self.BINARY:
            RDDLSimulator._check_arity(args, 2, f'Binary function {name}', expr)
            lhs, rhs = args
            lhs = 1 * self._sample(lhs, objects, subs)
            rhs = 1 * self._sample(rhs, objects, subs)
            temp = self.BINARY[name](lhs, rhs)
            return temp
        
        RDDLSimulator._raise_unsupported(f'Function {name}', expr)
    
    # ===========================================================================
    # control flow
    # ===========================================================================
    
    def _sample_control(self, expr, objects, subs):
        _, op = expr.etype
        RDDLSimulator._check_op(op, self.CONTROL_OPS, 'Control', expr)
        if op == 'if':
            return self._sample_if(expr, objects, subs)
        else:
            return self._sample_switch(expr, objects, subs)    
        
    def _sample_if(self, expr, objects, subs):
        args = expr.args
        RDDLSimulator._check_arity(args, 3, 'If then else', expr)
        
        pred, arg1, arg2 = args
        pred = self._sample(pred, objects, subs)
        RDDLSimulator._check_type(pred, bool, 'If predicate', expr)
        
        count_true = np.sum(pred)
        if count_true == pred.size:  # all elements of pred are true
            return self._sample(arg1, objects, subs)
        elif count_true == 0:  # all elements of pred are false
            return self._sample(arg2, objects, subs)
        else:
            arg1 = self._sample(arg1, objects, subs)
            arg2 = self._sample(arg2, objects, subs)
            return np.where(pred, arg1, arg2)
    
    def _sample_switch(self, expr, objects, subs):
        pred, *cases = expr.args
        
        # cache the sorted expressions by enum index on the first pass
        cached_expr = expr.cached_sim_info
        if cached_expr is None:
            
            # must be a pvar
            etype, _ = pred.etype
            if etype != 'pvar':
                raise RDDLNotImplementedError(
                    f'Switch predicate can only be a variable, '
                    f'not a complex expression of type <{etype}>.\n' + 
                    RDDLSimulator._print_stack_trace(expr))
            
            # type in pvariables scope must be an enum
            name, _ = pred.args
            var = self.rddl.parse(name)[0]
            enum_type = self.rddl.variable_ranges[var]
            if enum_type not in self.rddl.enum_types:
                raise RDDLTypeError(
                    f'Type <{enum_type}> of switch predicate <{name}> is not an '
                    f'enum type, must be one of {self.rddl.enum_types}.\n' + 
                    RDDLSimulator._print_stack_trace(expr))
            
            # default statement becomes ("default", expr)
            case_dict = dict((cvalue if ctype == 'case' else (ctype, cvalue)) 
                             for ctype, cvalue in cases)
            if len(case_dict) != len(cases):
                raise RDDLInvalidNumberOfArgumentsError(
                    f'Duplicated literal or default cases.\n' + 
                    RDDLSimulator._print_stack_trace(expr))
            
            cached_expr = self._order_enum_cases(enum_type, case_dict, expr)
            expr.cached_sim_info = cached_expr
            
        cached_cases, cached_default = cached_expr
        pred = self._sample(pred, objects, subs)
        RDDLSimulator._check_type(pred, RDDLTensors.INT, 'Switch predicate', expr)
        
        first_elem = pred.flat[0]
        if np.all(pred == first_elem): 
            
            # short circuit if all switches equal
            arg = cached_cases[first_elem]
            if arg is None:
                arg = cached_default
            return self._sample(arg, objects, subs)
        else:
            
            # cannot short circuit, branch on case value
            default_values = None
            if cached_default is not None:
                default_values = self._sample(cached_default, objects, subs)
            case_values = [(default_values if arg is None 
                            else self._sample(arg, objects, subs))
                           for arg in cached_cases]
            case_values = np.asarray(case_values)
            pred = np.expand_dims(pred, axis=0)
            return np.take_along_axis(case_values, pred, axis=0)        
        
    def _order_enum_cases(self, enum_type, case_dict, expr): 
        enum_values = self.rddl.objects[enum_type]
        
        # check that all literals belong to enum_type
        for literal in case_dict.keys():
            if literal != 'default' \
            and self.rddl.objects_rev.get(literal, None) != enum_type:
                raise RDDLUndefinedVariableError(
                    f'Literal <{literal}> does not belong to enum type '
                    f'<{enum_type}>, must be one of {set(enum_values)}.\n' + 
                    RDDLSimulator._print_stack_trace(expr))
        
        # store expressions in order of canonical literal index
        expressions = [None] * len(enum_values)
        for literal in enum_values:
            arg = case_dict.get(literal, None)
            if arg is not None:                
                index = self.tensors.index_of_object[literal]
                expressions[index] = arg
        
        # if default statement is missing, cases must be comprehensive
        default_expr = case_dict.get('default', None)
        if default_expr is None:
            for i, arg in enumerate(expressions):
                if arg is None:
                    raise RDDLUndefinedVariableError(
                        f'Enum literal <{enum_values[i]}> of type <{enum_type}> '
                        f'is missing in case list.\n' + 
                        RDDLSimulator._print_stack_trace(expr))
            
        return (expressions, default_expr)
                
    # ===========================================================================
    # random variables
    # ===========================================================================
    
    def _sample_random(self, expr, objects, subs):
        _, name = expr.etype
        if name == 'KronDelta':
            return self._sample_kron_delta(expr, objects, subs)        
        elif name == 'DiracDelta':
            return self._sample_dirac_delta(expr, objects, subs)
        elif name == 'Uniform':
            return self._sample_uniform(expr, objects, subs)
        elif name == 'Bernoulli':
            return self._sample_bernoulli(expr, objects, subs)
        elif name == 'Normal':
            return self._sample_normal(expr, objects, subs)
        elif name == 'Poisson':
            return self._sample_poisson(expr, objects, subs)
        elif name == 'Exponential':
            return self._sample_exponential(expr, objects, subs)
        elif name == 'Weibull':
            return self._sample_weibull(expr, objects, subs)        
        elif name == 'Gamma':
            return self._sample_gamma(expr, objects, subs)
        elif name == 'Binomial':
            return self._sample_binomial(expr, objects, subs)
        elif name == 'NegativeBinomial':
            return self._sample_negative_binomial(expr, objects, subs)
        elif name == 'Beta':
            return self._sample_beta(expr, objects, subs)
        elif name == 'Geometric':
            return self._sample_geometric(expr, objects, subs)
        elif name == 'Pareto':
            return self._sample_pareto(expr, objects, subs)
        elif name == 'Student':
            return self._sample_student(expr, objects, subs)
        elif name == 'Gumbel':
            return self._sample_gumbel(expr, objects, subs)
        elif name == 'Laplace':
            return self._sample_laplace(expr, objects, subs)
        elif name == 'Cauchy':
            return self._sample_cauchy(expr, objects, subs)
        elif name == 'Gompertz':
            return self._sample_gompertz(expr, objects, subs)
        elif name == 'Discrete':
            return self._sample_discrete(expr, objects, subs)
        else:
            RDDLSimulator._raise_unsupported(f'Distribution {name}', expr)

    def _sample_kron_delta(self, expr, objects, subs):
        args = expr.args
        RDDLSimulator._check_arity(args, 1, 'KronDelta', expr)
        
        arg, = args
        arg = self._sample(arg, objects, subs)
        RDDLSimulator._check_type_in(
            arg, {bool, RDDLTensors.INT}, 'Argument of KronDelta', expr)
        return arg
    
    def _sample_dirac_delta(self, expr, objects, subs):
        args = expr.args
        RDDLSimulator._check_arity(args, 1, 'DiracDelta', expr)
        
        arg, = args
        arg = self._sample(arg, objects, subs)
        RDDLSimulator._check_type(
            arg, RDDLTensors.REAL, 'Argument of DiracDelta', expr)        
        return arg
    
    def _sample_uniform(self, expr, objects, subs):
        args = expr.args
        RDDLSimulator._check_arity(args, 2, 'Uniform', expr)

        lb, ub = args
        lb = self._sample(lb, objects, subs)
        ub = self._sample(ub, objects, subs)
        RDDLSimulator._check_bounds(lb, ub, 'Uniform', expr)
        return self.rng.uniform(lb, ub)      
    
    def _sample_bernoulli(self, expr, objects, subs):
        args = expr.args
        RDDLSimulator._check_arity(args, 1, 'Bernoulli', expr)
        
        pr, = args
        pr = self._sample(pr, objects, subs)
        RDDLSimulator._check_range(pr, 0, 1, 'Bernoulli p', expr)
        return self.rng.uniform(size=pr.shape) <= pr
    
    def _sample_normal(self, expr, objects, subs):
        args = expr.args
        RDDLSimulator._check_arity(args, 2, 'Normal', expr)
        
        mean, var = args
        mean = self._sample(mean, objects, subs)
        var = self._sample(var, objects, subs)
        RDDLSimulator._check_positive(var, False, 'Normal variance', expr)  
        std = np.sqrt(var)
        return self.rng.normal(mean, std)
    
    def _sample_poisson(self, expr, objects, subs):
        args = expr.args
        RDDLSimulator._check_arity(args, 1, 'Poisson', expr)
        
        rate, = args
        rate = self._sample(rate, objects, subs)
        RDDLSimulator._check_positive(rate, False, 'Poisson rate', expr)        
        return self.rng.poisson(rate)
    
    def _sample_exponential(self, expr, objects, subs):
        args = expr.args
        RDDLSimulator._check_arity(args, 1, 'Exponential', expr)
        
        scale, = expr.args
        scale = self._sample(scale, objects, subs)
        RDDLSimulator._check_positive(scale, True, 'Exponential rate', expr)
        return self.rng.exponential(scale)
    
    def _sample_weibull(self, expr, objects, subs):
        args = expr.args
        RDDLSimulator._check_arity(args, 2, 'Weibull', expr)
        
        shape, scale = args
        shape = self._sample(shape, objects, subs)
        scale = self._sample(scale, objects, subs)
        RDDLSimulator._check_positive(shape, True, 'Weibull shape', expr)
        RDDLSimulator._check_positive(scale, True, 'Weibull scale', expr)
        return scale * self.rng.weibull(shape)
    
    def _sample_gamma(self, expr, objects, subs):
        args = expr.args
        RDDLSimulator._check_arity(args, 2, 'Gamma', expr)
        
        shape, scale = args
        shape = self._sample(shape, objects, subs)
        scale = self._sample(scale, objects, subs)
        RDDLSimulator._check_positive(shape, True, 'Gamma shape', expr)            
        RDDLSimulator._check_positive(scale, True, 'Gamma scale', expr)        
        return self.rng.gamma(shape, scale)
    
    def _sample_binomial(self, expr, objects, subs):
        args = expr.args
        RDDLSimulator._check_arity(args, 2, 'Binomial', expr)
        
        count, pr = args
        count = self._sample(count, objects, subs)
        pr = self._sample(pr, objects, subs)
        RDDLSimulator._check_type(count, RDDLTensors.INT, 'Binomial count', expr)
        RDDLSimulator._check_positive(count, False, 'Binomial count', expr)
        RDDLSimulator._check_range(pr, 0, 1, 'Binomial p', expr)
        return self.rng.binomial(count, pr)
    
    def _sample_negative_binomial(self, expr, objects, subs):
        args = expr.args
        RDDLSimulator._check_arity(args, 2, 'NegativeBinomial', expr)
        
        count, pr = args
        count = self._sample(count, objects, subs)
        pr = self._sample(pr, objects, subs)
        RDDLSimulator._check_positive(count, True, 'NegativeBinomial r', expr)
        RDDLSimulator._check_range(pr, 0, 1, 'NegativeBinomial p', expr)        
        return self.rng.negative_binomial(count, pr)
    
    def _sample_beta(self, expr, objects, subs):
        args = expr.args
        RDDLSimulator._check_arity(args, 2, 'Beta', expr)
        
        shape, rate = args
        shape = self._sample(shape, objects, subs)
        rate = self._sample(rate, objects, subs)
        RDDLSimulator._check_positive(shape, True, 'Beta shape', expr)
        RDDLSimulator._check_positive(rate, True, 'Beta rate', expr)        
        return self.rng.beta(shape, rate)

    def _sample_geometric(self, expr, objects, subs):
        args = expr.args
        RDDLSimulator._check_arity(args, 1, 'Geometric', expr)
        
        pr, = args
        pr = self._sample(pr, objects, subs)
        RDDLSimulator._check_range(pr, 0, 1, 'Geometric p', expr)        
        return self.rng.geometric(pr)
    
    def _sample_pareto(self, expr, objects, subs):
        args = expr.args
        RDDLSimulator._check_arity(args, 2, 'Pareto', expr)
        
        shape, scale = args
        shape = self._sample(shape, objects, subs)
        scale = self._sample(scale, objects, subs)
        RDDLSimulator._check_positive(shape, True, 'Pareto shape', expr)        
        RDDLSimulator._check_positive(scale, True, 'Pareto scale', expr)        
        return scale * self.rng.pareto(shape)
    
    def _sample_student(self, expr, objects, subs):
        args = expr.args
        RDDLSimulator._check_arity(args, 1, 'Student', expr)
        
        df, = args
        df = self._sample(df, objects, subs)
        RDDLSimulator._check_positive(df, True, 'Student df', expr)            
        return self.rng.standard_t(df)

    def _sample_gumbel(self, expr, objects, subs):
        args = expr.args
        RDDLSimulator._check_arity(args, 2, 'Gumbel', expr)
        
        mean, scale = args
        mean = self._sample(mean, objects, subs)
        scale = self._sample(scale, objects, subs)
        RDDLSimulator._check_positive(scale, True, 'Gumbel scale', expr)
        return self.rng.gumbel(mean, scale)
    
    def _sample_laplace(self, expr, objects, subs):
        args = expr.args
        RDDLSimulator._check_arity(args, 2, 'Laplace', expr)
        
        mean, scale = args
        mean = self._sample(mean, objects, subs)
        scale = self._sample(scale, objects, subs)
        RDDLSimulator._check_positive(scale, True, 'Laplace scale', expr)
        return self.rng.laplace(mean, scale)
    
    def _sample_cauchy(self, expr, objects, subs):
        args = expr.args
        RDDLSimulator._check_arity(args, 2, 'Cauchy', expr)
        
        mean, scale = args
        mean = self._sample(mean, objects, subs)
        scale = self._sample(scale, objects, subs)
        RDDLSimulator._check_positive(scale, True, 'Cauchy scale', expr)
        sample = self.rng.standard_cauchy(size=mean.shape)
        sample = mean + scale * sample
        return sample
    
    def _sample_gompertz(self, expr, objects, subs):
        args = expr.args
        RDDLSimulator._check_arity(args, 2, 'Gompertz', expr)
        
        shape, scale = args
        shape = self._sample(shape, objects, subs)
        scale = self._sample(scale, objects, subs)
        RDDLSimulator._check_positive(shape, True, 'Gompertz shape', expr)
        RDDLSimulator._check_positive(scale, True, 'Gompertz scale', expr)
        U = self.rng.uniform(size=shape.shape)
        sample = np.log(1.0 - np.log1p(-U) / shape) / scale
        return sample
    
    def _sample_discrete(self, expr, objects, subs):
        
        # cache the sorted expressions by enum index on the first pass
        cached_expr = expr.cached_sim_info
        if cached_expr is None:
            (_, enum_type), *cases = expr.args
            
            # enum type must be a valid enum type
            if enum_type not in self.rddl.enum_types:
                raise RDDLTypeError(
                    f'Type <{enum_type}> in Discrete distribution is not an '
                    f'enum, must be one of {self.rddl.enum_types}.\n' + 
                    RDDLSimulator._print_stack_trace(expr))
            
            # no duplicate cases are allowed
            case_dict = dict(c for _, c in cases)
            if len(case_dict) != len(cases):
                raise RDDLInvalidNumberOfArgumentsError(
                    f'Duplicated literal or default cases.\n' + 
                    RDDLSimulator._print_stack_trace(expr))
            
            cached_expr, _ = self._order_enum_cases(enum_type, case_dict, expr)
            expr.cached_sim_info = cached_expr
        
        # calculate the CDF and check sum to one
        pdfs = [self._sample(arg, objects, subs) for arg in cached_expr]
        cdfs = np.cumsum(pdfs, axis=0)
        if not np.allclose(cdfs[-1, ...], 1.0):
            raise RDDLValueOutOfRangeError(
                f'Discrete probabilities must sum to 1, got {cdfs[-1, ...]}.\n' + 
                RDDLSimulator._print_stack_trace(expr))     
        
        # use inverse CDF sampling                  
        U = self.rng.random(size=(1,) + cdfs.shape[1:])
        return np.argmax(U < cdfs, axis=0)


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
        
        self._bounds, states, actions = {}, set(), set()
        for var, vtype in self.rddl.variable_types.items():
            if vtype in {'state-fluent', 'observ-fluent', 'action-fluent'}:
                ptypes = self.rddl.param_types[var]
                for name in self.rddl.grounded_names(var, ptypes):
                    self._bounds[name] = [-self.BigM, +self.BigM]
                    if self.rddl.is_grounded:
                        if vtype == 'action-fluent':
                            actions.add(name)
                        elif vtype == 'state-fluent':
                            states.add(name)
                if not self.rddl.is_grounded:
                    if vtype == 'action-fluent':
                        actions.add(var)
                    elif vtype == 'state-fluent':
                        states.add(var)

        # actions and states bounds extraction for gym's action and state spaces
        # currently supports only linear inequality constraints
        for precond in self.rddl.preconditions:
            self._parse_bounds(precond, [], actions)
            
        for invariant in self.rddl.invariants:
            self._parse_bounds(invariant, [], states)

        for name, bounds in self._bounds.items():
            RDDLSimulator._check_bounds(*bounds, f'Variable <{name}>', bounds)

    def _parse_bounds(self, expr, objects, search_vars):
        etype, op = expr.etype
        
        if etype == 'aggregation' and op == 'forall':
            if not self.rddl.is_grounded:
                * pvars, arg = expr.args
                new_objects = objects + [p[1] for p in pvars]
                self._parse_bounds(arg, new_objects, search_vars)
            
        elif etype == 'boolean' and op == '^':
            for arg in expr.args:
                self._parse_bounds(arg, objects, search_vars)
                
        elif etype == 'relational':
            var, lim, loc, active = self._parse_bounds_relational(
                expr, objects, search_vars)
            if var is not None and loc is not None: 
                if self.rddl.is_grounded:
                    self._update_bound(var, loc, lim)
                else: 
                    variations = self.rddl.variations([o[1] for o in objects])
                    lims = np.ravel(lim)
                    for args, lim in zip(variations, lims):
                        key = self.rddl.ground_name(var, [args[i] for i in active])
                        self._update_bound(key, loc, lim)
    
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
            
            const = self._sample(const_expr, objects, self.subs)
            eps, loc = self._get_op_code(op, is_left_pvar)
            lim = const + eps
            
            arg_to_index = {o[0]: i for i, o in enumerate(objects)}
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
