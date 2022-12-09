import numpy as np
np.seterr(all='raise')
import re
from typing import Dict, Union
import warnings

from pyRDDLGym.Core.Debug.decompiler import RDDLDecompiler
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLActionPreconditionNotSatisfiedError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidActionError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidNumberOfArgumentsError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLNotImplementedError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLStateInvariantNotSatisfiedError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLTypeError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLUndefinedVariableError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLValueOutOfRangeError
from pyRDDLGym.Core.Parser.expr import Expression, Value
from pyRDDLGym.Core.Parser.rddl import RDDL
from pyRDDLGym.Core.Simulator.LiftedRDDLLevelAnalysis import LiftedRDDLLevelAnalysis
from pyRDDLGym.Core.Simulator.LiftedRDDLTypeAnalysis import LiftedRDDLTypeAnalysis

Args = Dict[str, Value]

        
class LiftedRDDLSimulator:
    
    INT = np.int32
    REAL = np.float64
        
    NUMPY_TYPES = {
        'int': INT,
        'real': REAL,
        'bool': bool
    }
    
    DEFAULT_VALUES = {
        'int': 0,
        'real': 0.0,
        'bool': False
    }
    
    VALID_SYMBOLS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    def __init__(self,
                 rddl: RDDL,
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
        
        self.domain = rddl.domain
        self.instance = rddl.instance
        self.non_fluents = rddl.non_fluents
        
        # perform a dependency analysis for fluent variables
        self.static = LiftedRDDLLevelAnalysis(rddl, allow_synchronous_state)
        self.levels = self.static.compute_levels()
        self.next_states = self.static.next_states
        
        # extract information about types and their objects in the domain
        self.types = LiftedRDDLTypeAnalysis(rddl, debug=debug)
        
        self._compile_initial_values()
        self.action_pattern = re.compile(r'(\S*?)\((\S.*?)\)', re.VERBOSE)
        
        # is a POMDP
        self.observ_fluents = [pvar.name 
                               for pvar in self.domain.pvariables 
                               if pvar.is_observ_fluent()]
        self._pomdp = bool(self.observ_fluents)
        
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
            'sgn': np.sign,
            'round': np.round,
            'floor': np.floor,
            'ceil': np.ceil,
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
            'div': np.floor_divide,
            'mod': np.mod,
            'min': np.minimum,
            'max': np.maximum,
            'pow': np.power,
            'log': lambda x, y: np.log(x) / np.log(y)
        }
    
    # ===========================================================================
    # compilation time
    # ===========================================================================
    
    def _compile_initial_values(self):
        
        # get default values from domain
        self.init_values = {}
        self.noop_actions = {}
        for pvar in self.domain.pvariables:
            name = pvar.name                             
            ptypes = pvar.param_types
            prange = pvar.range
            dtype = LiftedRDDLSimulator.NUMPY_TYPES[prange]
            
            value = pvar.default
            if value is None:
                value = LiftedRDDLSimulator.DEFAULT_VALUES[prange]
            self.init_values[name] = self.types.tensor(ptypes, value, dtype)
            
            if pvar.is_action_fluent():
                self.noop_actions[name] = self.init_values[name]
        
        # override default values with instance
        blocks = {}
        if hasattr(self.instance, 'init_state'):
            blocks['init-state'] = self.instance.init_state
        if hasattr(self.non_fluents, 'init_non_fluent'):
            blocks['non-fluents'] = self.non_fluents.init_non_fluent
        
        for block_name, block_content in blocks.items():
            for (name, objects), value in block_content:
                init_values = self.init_values.get(name, None)
                if init_values is None:
                    raise RDDLUndefinedVariableError(
                        f'Variable <{name}> in {block_name} is not valid.')
                    
                if self.types.count_type_args(name):
                    self.types.put(name, objects, value, self.init_values[name])
                else:
                    self.init_values[name] = value
                
        if self.debug:
            val = ''.join(f'\n\t\t{k}: {v}' for k, v in self.init_values.items())
            warnings.warn(
                f'compiling initial value info:'
                f'\n\tvalues ={val}\n'
            )

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
        value = value.dtype
        if value != valid:
            raise RDDLTypeError(
                f'{msg} must evaluate to {valid}, got {value}.\n' + 
                LiftedRDDLSimulator._print_stack_trace(expr))
    
    @staticmethod
    def _check_types(value1, value2, valid, msg, expr):
        value1 = value1.dtype
        value2 = value2.dtype
        if value1 != valid and value2 != valid:
            raise RDDLTypeError(
                f'{msg} must evaluate to {valid}, got {value1} and {value2}.\n' + 
                LiftedRDDLSimulator._print_stack_trace(expr))
            
    @staticmethod
    def _check_op(op, valid, msg, expr):
        if op not in valid:
            raise RDDLNotImplementedError(
                f'{msg} operator {op} is not supported: must be in {valid}.\n' + 
                LiftedRDDLSimulator._print_stack_trace(expr))
    
    @staticmethod
    def _check_arity(args, required, msg, expr):
        actual = len(args)
        if actual != required:
            raise RDDLInvalidNumberOfArgumentsError(
                f'{msg} requires {required} arguments, got {actual}.\n' + 
                LiftedRDDLSimulator._print_stack_trace(expr))
    
    @staticmethod
    def _check_positive(value, strict, msg, expr):
        if strict:
            failed = not np.all(value > 0).item()
            message = f'{msg} must be strictly positive, got {value}.\n'
        else:
            failed = not np.all(value >= 0).item()
            message = f'{msg} must be non-negative, got {value}.\n'
        if failed:
            raise RDDLValueOutOfRangeError(
                message + LiftedRDDLSimulator._print_stack_trace(expr))
    
    @staticmethod
    def _check_bounds(lb, ub, msg, expr):
        if not np.all(lb <= ub).item():
            raise RDDLValueOutOfRangeError(
                f'Bounds of {msg} are invalid:' 
                f'max value {ub} must be >= min value {lb}.\n' + 
                LiftedRDDLSimulator._print_stack_trace(expr))
            
    @staticmethod
    def _check_range(value, lb, ub, msg, expr):
        if not np.all((value >= lb) & (value <= ub)).item():
            raise RDDLValueOutOfRangeError(
                f'{msg} must be in the range [{lb}, {ub}], got {value}.\n' + 
                LiftedRDDLSimulator._print_stack_trace(expr))
    
    @staticmethod
    def _raise_unsupported(msg, expr):
        raise RDDLNotImplementedError(
            f'{msg} is not supported in the current RDDL version.\n' + 
            LiftedRDDLSimulator._print_stack_trace(expr))
    
    @staticmethod
    def _raise_no_args(op, msg, expr):
        raise RDDLInvalidNumberOfArgumentsError(
            f'{msg} operator {op} has no arguments.\n' + 
            LiftedRDDLSimulator._print_stack_trace(expr))
    
    # ===========================================================================
    # main sampling routines
    # ===========================================================================
    
    def _actions_to_tensors(self, actions: Dict[str, Union[Value, np.ndarray]]) -> Args:
        new_actions = {action: np.copy(value) 
                       for action, value in self.noop_actions.items()}
        
        for action, value in actions.items():
            
            # parse the specified action
            name, objects = action, ''
            if '(' in action:
                parsed_action = self.action_pattern.match(action)
                if not parsed_action:
                    raise RDDLInvalidActionError(
                        f'Action fluent <{action}> is not valid, '
                        f'must be <name> or <name>(<type1>,<type2>...).')
                name, objects = parsed_action.groups()
            if name not in new_actions:
                raise RDDLInvalidActionError(
                    f'Action fluent <{name}> is not valid, '
                    f'must be one of {set(new_actions.keys())}.')
            
            # update the initial actions       
            objects = [obj.strip() for obj in objects.split(',')]
            objects = [obj for obj in objects if obj != '']
            self.types.put(name, objects, value, new_actions[name])
            
        return new_actions
    
    def check_state_invariants(self) -> None:
        '''Throws an exception if the state invariants are not satisfied.'''
        for i, invariant in enumerate(self.domain.invariants):
            sample = self._sample(invariant, [], self.subs)
            LiftedRDDLSimulator._check_type(
                sample, bool, f'Invariant {i}', invariant)
            if not sample.item():
                raise RDDLStateInvariantNotSatisfiedError(
                    f'Invariant {i} is not satisfied.\n' + 
                    LiftedRDDLSimulator._print_stack_trace(invariant))
    
    def check_action_preconditions(self, actions: Args) -> None:
        '''Throws an exception if the action preconditions are not satisfied.'''        
        actions = self._actions_to_tensors(actions)
        subs = self.subs
        subs.update(actions)
        
        for i, precond in enumerate(self.domain.preconds):
            sample = self._sample(precond, [], subs)
            LiftedRDDLSimulator._check_type(
                sample, bool, f'Precondition {i}', precond)
            if not sample.item():
                raise RDDLActionPreconditionNotSatisfiedError(
                    f'Precondition {i} is not satisfied.\n' + 
                    LiftedRDDLSimulator._print_stack_trace(precond))
    
    def check_terminal_states(self) -> bool:
        '''Return True if a terminal state has been reached.'''
        for i, terminal in enumerate(self.domain.terminals):
            sample = self._sample(terminal, [], self.subs)
            LiftedRDDLSimulator._check_type(
                sample, bool, f'Termination {i}', terminal)
            if sample.item():
                return True
        return False
    
    def sample_reward(self) -> float:
        '''Samples the current reward given the current state and action.'''
        sample = self._sample(self.domain.reward, [], self.subs)
        return float(sample.item())
    
    def reset(self) -> Union[Dict[str, None], Args]:
        '''Resets the state variables to their initial values.'''
        self.subs = self.init_values.copy()
        names = self.observ_fluents if self._pomdp \
                    else self.next_states.values()
        obs = {}
        for var in names:
            obs.update(self.types.expand(var, self.subs[var]))
        if self._pomdp:
            obs = {o: None for o in obs.keys()}
        done = self.check_terminal_states()
        return obs, done
    
    def step(self, actions: Args) -> Args:
        '''Samples and returns the next state from the CPF expressions.
        
        :param actions: a dict mapping current action fluent to their values
        '''
        actions = self._actions_to_tensors(actions)
        subs = self.subs
        subs.update(actions)
        
        for _, cpfs in self.levels.items():
            for cpf in cpfs:
                ptypes = self.types.cpf_types[cpf]
                expr = self.static.cpfs[cpf]
                subs[cpf] = self._sample(expr, ptypes, subs)
        reward = self.sample_reward()
        
        for next_state, state in self.next_states.items():
            subs[state] = subs[next_state]
        
        names = self.observ_fluents if self._pomdp \
                    else self.next_states.values()
        obs = {}
        for var in names:
            obs.update(self.types.expand(var, subs[var]))
        
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
        if not hasattr(expr, 'cached_value'):
            *_, shape = self.types.map(
                '', [], objects,
                str(expr), LiftedRDDLSimulator._print_stack_trace(expr))  
            expr.cached_value = np.full(shape=shape, fill_value=expr.args)            
            if self.debug:
                warnings.warn(f'caching constant <{expr.args}>')
                
        return expr.cached_value
    
    def _sample_pvar(self, expr, objects, subs):
        _, name = expr.etype
        args = expr.args
        LiftedRDDLSimulator._check_arity(args, 2, f'Variable <{name}>', expr)
        
        var, pvars = args
        if var not in subs:
            raise RDDLUndefinedVariableError(
                f'Variable <{var}> is not defined in the instance.\n' + 
                LiftedRDDLSimulator._print_stack_trace(expr))
            
        arg = subs[var]
        if arg is None:
            raise RDDLUndefinedVariableError(
                f'Variable <{var}> is referenced before assignment.\n' + 
                LiftedRDDLSimulator._print_stack_trace(expr))
        
        if not hasattr(expr, 'cached_sub_map'):
            expr.cached_sub_map = self.types.map(
                var, pvars, objects,
                str(expr), LiftedRDDLSimulator._print_stack_trace(expr))       
        permute, identity, new_dims = expr.cached_sub_map
        
        arg = np.asarray(arg)
        sample = arg
        if new_dims:
            new_axes = (1,) * len(new_dims)
            sample = np.reshape(arg, newshape=arg.shape + new_axes) 
            sample = np.broadcast_to(sample, shape=arg.shape + new_dims)
        if not identity:
            sample = np.einsum(permute, sample)
        return sample
    
    # ===========================================================================
    # mathematical
    # ===========================================================================
    
    def _sample_arithmetic(self, expr, objects, subs):
        _, op = expr.etype
        valid_ops = self.ARITHMETIC_OPS
        LiftedRDDLSimulator._check_op(op, valid_ops, 'Arithmetic', expr)
        
        args = expr.args        
        n = len(args)        
        if n == 1 and op == '-':
            arg, = args
            return -1 * self._sample(arg, objects, subs)
        
        elif n == 2:
            lhs, rhs = args
            lhs = 1 * self._sample(lhs, objects, subs)
            rhs = 1 * self._sample(rhs, objects, subs)
            return valid_ops[op](lhs, rhs)
        
        LiftedRDDLSimulator._check_arity(args, 2, 'Arithmetic operator', expr)
    
    def _sample_relational(self, expr, objects, subs):
        _, op = expr.etype
        args = expr.args
        valid_ops = self.RELATIONAL_OPS
        LiftedRDDLSimulator._check_op(op, valid_ops, 'Relational', expr)
        LiftedRDDLSimulator._check_arity(args, 2, f'Relational operator {op}', expr)
        
        lhs, rhs = args
        lhs = 1 * self._sample(lhs, objects, subs)
        rhs = 1 * self._sample(rhs, objects, subs)
        return valid_ops[op](lhs, rhs)
    
    def _sample_logical(self, expr, objects, subs):
        _, op = expr.etype
        args = expr.args
        valid_ops = self.LOGICAL_OPS
        LiftedRDDLSimulator._check_op(op, valid_ops, 'Logical', expr)
        
        n = len(args)
        if n == 1 and op == '~':
            arg, = args
            arg = self._sample(arg, objects, subs)
            LiftedRDDLSimulator._check_type(
                arg, bool, f'Argument of logical operator {op}', expr)
            return np.logical_not(arg)
        
        elif n == 2:
            lhs, rhs = args
            lhs = self._sample(lhs, objects, subs)
            rhs = self._sample(rhs, objects, subs)
            LiftedRDDLSimulator._check_types(
                lhs, rhs, bool, f'Arguments of logical operator {op}', expr)
            return valid_ops[op](lhs, rhs)
        
        LiftedRDDLSimulator._check_arity(args, 2, 'Logical operator', expr)
        
    def _sample_aggregation(self, expr, objects, subs):
        _, op = expr.etype
        args = expr.args
        valid_ops = self.AGGREGATION_OPS
        LiftedRDDLSimulator._check_op(op, valid_ops, 'Aggregation', expr)

        * pvars, arg = args
        if not hasattr(expr, 'cached_objects'):
            new_objects = objects + [p[1] for p in pvars]
            axis = tuple(range(len(objects), len(new_objects)))
            self.types.validate_types(
                new_objects, LiftedRDDLSimulator._print_stack_trace(expr))
            
            if self.debug:
                warnings.warn(
                    f'computing object info for aggregation:'
                    f'\n\toperator       ={op} {pvars}'
                    f'\n\tinput objects  ={new_objects}'
                    f'\n\toutput objects ={objects}'
                    f'\n\treduction axis ={axis}'
                    f'\n\treduction op   ={valid_ops[op]}\n'
                )
                
            expr.cached_objects = (new_objects, axis)
        new_objects, axis = expr.cached_objects
        
        arg = self._sample(arg, new_objects, subs)
                
        if op == 'forall' or op == 'exists':
            LiftedRDDLSimulator._check_type(
                arg, bool, f'Argument of aggregation {op}', expr)
        else:
            arg = 1 * arg
        return valid_ops[op](arg, axis=axis)
    
    def _sample_func(self, expr, objects, subs):
        _, name = expr.etype
        args = expr.args
        if isinstance(args, Expression):
            args = (args,)
            
        if name in self.UNARY:
            LiftedRDDLSimulator._check_arity(
                args, 1, f'Unary function {name}', expr)
            arg, = args
            arg = 1 * self._sample(arg, objects, subs)
            return self.UNARY[name](arg)
        
        elif name in self.BINARY:
            LiftedRDDLSimulator._check_arity(
                args, 2, f'Binary function {name}', expr)
            lhs, rhs = args
            lhs = 1 * self._sample(lhs, objects, subs)
            rhs = 1 * self._sample(rhs, objects, subs)
            return self.BINARY[name](lhs, rhs)
        
        LiftedRDDLSimulator._raise_unsupported(f'Function {name}', expr)
    
    # ===========================================================================
    # control flow
    # ===========================================================================
    
    def _sample_control(self, expr, objects, subs):
        _, op = expr.etype
        args = expr.args
        LiftedRDDLSimulator._check_op(op, {'if'}, 'Control', expr)
        LiftedRDDLSimulator._check_arity(args, 3, 'If then else', expr)
        
        pred, arg1, arg2 = args
        pred = self._sample(pred, objects, subs)
        LiftedRDDLSimulator._check_type(pred, bool, 'Predicate', expr)
        
        if pred.size == 1:  # can short-circuit
            arg = arg1 if pred.item() else arg2
            return self._sample(arg, objects, subs)
        else:
            arg1 = self._sample(arg1, objects, subs)
            arg2 = self._sample(arg2, objects, subs)
            return np.where(pred, arg1, arg2)
        
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
        else:  # no support for enum
            LiftedRDDLSimulator._raise_unsupported(f'Distribution {name}', expr)

    def _sample_kron_delta(self, expr, objects, subs):
        args = expr.args
        LiftedRDDLSimulator._check_arity(args, 1, 'KronDelta', expr)
        
        arg, = args
        arg = self._sample(arg, objects, subs)
        LiftedRDDLSimulator._check_type(
            arg, bool, 'Argument of KronDelta', expr)
        return arg
    
    def _sample_dirac_delta(self, expr, objects, subs):
        args = expr.args
        LiftedRDDLSimulator._check_arity(args, 1, 'DiracDelta', expr)
        
        arg, = args
        arg = self._sample(arg, objects, subs)
        LiftedRDDLSimulator._check_type(
            arg, LiftedRDDLSimulator.REAL, 'Argument of DiracDelta', expr)        
        return arg
    
    def _sample_uniform(self, expr, objects, subs):
        args = expr.args
        LiftedRDDLSimulator._check_arity(args, 2, 'Uniform', expr)

        lb, ub = args
        lb = self._sample(lb, objects, subs)
        ub = self._sample(ub, objects, subs)
        LiftedRDDLSimulator._check_bounds(lb, ub, 'Uniform', expr)
        return self.rng.uniform(lb, ub)      
    
    def _sample_bernoulli(self, expr, objects, subs):
        args = expr.args
        LiftedRDDLSimulator._check_arity(args, 1, 'Bernoulli', expr)
        
        pr, = args
        pr = self._sample(pr, objects, subs)
        LiftedRDDLSimulator._check_range(pr, 0, 1, 'Bernoulli p', expr)
        return self.rng.uniform(size=pr.shape) <= pr
    
    def _sample_normal(self, expr, objects, subs):
        args = expr.args
        LiftedRDDLSimulator._check_arity(args, 2, 'Normal', expr)
        
        mean, var = args
        mean = self._sample(mean, objects, subs)
        var = self._sample(var, objects, subs)
        LiftedRDDLSimulator._check_positive(var, False, 'Normal variance', expr)  
        std = np.sqrt(var)
        return self.rng.normal(mean, std)
    
    def _sample_poisson(self, expr, objects, subs):
        args = expr.args
        LiftedRDDLSimulator._check_arity(args, 1, 'Poisson', expr)
        
        rate, = args
        rate = self._sample(rate, objects, subs)
        LiftedRDDLSimulator._check_positive(rate, False, 'Poisson rate', expr)        
        return self.rng.poisson(rate)
    
    def _sample_exponential(self, expr, objects, subs):
        args = expr.args
        LiftedRDDLSimulator._check_arity(args, 1, 'Exponential', expr)
        
        scale, = expr.args
        scale = self._sample(scale, objects, subs)
        LiftedRDDLSimulator._check_positive(scale, True, 'Exponential rate', expr)
        return self.rng.exponential(scale)
    
    def _sample_weibull(self, expr, objects, subs):
        args = expr.args
        LiftedRDDLSimulator._check_arity(args, 2, 'Weibull', expr)
        
        shape, scale = args
        shape = self._sample(shape, objects, subs)
        scale = self._sample(scale, objects, subs)
        LiftedRDDLSimulator._check_positive(shape, True, 'Weibull shape', expr)
        LiftedRDDLSimulator._check_positive(scale, True, 'Weibull scale', expr)
        return scale * self.rng.weibull(shape)
    
    def _sample_gamma(self, expr, objects, subs):
        args = expr.args
        LiftedRDDLSimulator._check_arity(args, 2, 'Gamma', expr)
        
        shape, scale = args
        shape = self._sample(shape, objects, subs)
        scale = self._sample(scale, objects, subs)
        LiftedRDDLSimulator._check_positive(shape, True, 'Gamma shape', expr)            
        LiftedRDDLSimulator._check_positive(scale, True, 'Gamma scale', expr)        
        return self.rng.gamma(shape, scale)
    
    def _sample_binomial(self, expr, objects, subs):
        args = expr.args
        LiftedRDDLSimulator._check_arity(args, 2, 'Binomial', expr)
        
        count, pr = args
        count = self._sample(count, objects, subs)
        pr = self._sample(pr, objects, subs)
        LiftedRDDLSimulator._check_type(
            count, LiftedRDDLSimulator.INT, 'Binomial count', expr)
        LiftedRDDLSimulator._check_positive(count, False, 'Binomial count', expr)
        LiftedRDDLSimulator._check_range(pr, 0, 1, 'Binomial p', expr)
        return self.rng.binomial(count, pr)
    
    def _sample_negative_binomial(self, expr, objects, subs):
        args = expr.args
        LiftedRDDLSimulator._check_arity(args, 2, 'NegativeBinomial', expr)
        
        count, pr = args
        count = self._sample(count, objects, subs)
        pr = self._sample(pr, objects, subs)
        LiftedRDDLSimulator._check_positive(
            count, True, 'NegativeBinomial r', expr)
        LiftedRDDLSimulator._check_range(
            pr, 0, 1, 'NegativeBinomial p', expr)        
        return self.rng.negative_binomial(count, pr)
    
    def _sample_beta(self, expr, objects, subs):
        args = expr.args
        LiftedRDDLSimulator._check_arity(args, 2, 'Beta', expr)
        
        shape, rate = args
        shape = self._sample(shape, objects, subs)
        rate = self._sample(rate, objects, subs)
        LiftedRDDLSimulator._check_positive(shape, True, 'Beta shape', expr)
        LiftedRDDLSimulator._check_positive(rate, True, 'Beta rate', expr)        
        return self.rng.beta(shape, rate)

    def _sample_geometric(self, expr, objects, subs):
        args = expr.args
        LiftedRDDLSimulator._check_arity(args, 1, 'Geometric', expr)
        
        pr, = args
        pr = self._sample(pr, objects, subs)
        LiftedRDDLSimulator._check_range(pr, 0, 1, 'Geometric p', expr)        
        return self.rng.geometric(pr)
    
    def _sample_pareto(self, expr, objects, subs):
        args = expr.args
        LiftedRDDLSimulator._check_arity(args, 2, 'Pareto', expr)
        
        shape, scale = args
        shape = self._sample(shape, objects, subs)
        scale = self._sample(scale, objects, subs)
        LiftedRDDLSimulator._check_positive(shape, True, 'Pareto shape', expr)        
        LiftedRDDLSimulator._check_positive(scale, True, 'Pareto scale', expr)        
        return scale * self.rng.pareto(shape)
    
    def _sample_student(self, expr, objects, subs):
        args = expr.args
        LiftedRDDLSimulator._check_arity(args, 1, 'Student', expr)
        
        df, = args
        df = self._sample(df, objects, subs)
        LiftedRDDLSimulator._check_positive(df, True, 'Student df', expr)            
        return self.rng.standard_t(df)

    def _sample_gumbel(self, expr, objects, subs):
        args = expr.args
        LiftedRDDLSimulator._check_arity(args, 2, 'Gumbel', expr)
        
        mean, scale = args
        mean = self._sample(mean, objects, subs)
        scale = self._sample(scale, objects, subs)
        LiftedRDDLSimulator._check_positive(scale, True, 'Gumbel scale', expr)
        return self.rng.gumbel(mean, scale)
    
    def _sample_laplace(self, expr, objects, subs):
        args = expr.args
        LiftedRDDLSimulator._check_arity(args, 2, 'Laplace', expr)
        
        mean, scale = args
        mean = self._sample(mean, objects, subs)
        scale = self._sample(scale, objects, subs)
        LiftedRDDLSimulator._check_positive(scale, True, 'Laplace scale', expr)
        return self.rng.laplace(mean, scale)
    
    def _sample_cauchy(self, expr, objects, subs):
        args = expr.args
        LiftedRDDLSimulator._check_arity(args, 2, 'Cauchy', expr)
        
        mean, scale = args
        mean = self._sample(mean, objects, subs)
        scale = self._sample(scale, objects, subs)
        LiftedRDDLSimulator._check_positive(scale, True, 'Cauchy scale', expr)
        sample = self.rng.standard_cauchy(size=mean.shape)
        sample = mean + scale * sample
        return sample
    
    def _sample_gompertz(self, expr, objects, subs):
        args = expr.args
        LiftedRDDLSimulator._check_arity(args, 2, 'Gompertz', expr)
        
        shape, scale = args
        shape = self._sample(shape, objects, subs)
        scale = self._sample(scale, objects, subs)
        LiftedRDDLSimulator._check_positive(shape, True, 'Gompertz shape', expr)
        LiftedRDDLSimulator._check_positive(scale, True, 'Gompertz scale', expr)
        U = self.rng.uniform(size=shape.shape)
        sample = np.log(1.0 - np.log1p(-U) / shape) / scale
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
