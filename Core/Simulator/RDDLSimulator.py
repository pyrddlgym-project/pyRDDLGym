from collections.abc import Sized
import math
import numpy as np
from typing import Dict, Tuple

from Core.Grounder.RDDLModel import PlanningModel
from Core.ErrorHandling.RDDLException import RDDLActionPreconditionNotSatisfiedError
from Core.ErrorHandling.RDDLException import RDDLInvalidNumberOfArgumentsError
from Core.ErrorHandling.RDDLException import RDDLNotImplementedError
from Core.ErrorHandling.RDDLException import RDDLStateInvariantNotSatisfiedError
from Core.ErrorHandling.RDDLException import RDDLTypeError
from Core.ErrorHandling.RDDLException import RDDLUndefinedCPFError
from Core.ErrorHandling.RDDLException import RDDLUndefinedVariableError
from Core.ErrorHandling.RDDLException import RDDLValueOutOfRangeError
from Core.Parser.expr import Expression, Value
from Core.Simulator.RDDLDependencyAnalysis import RDDLDependencyAnalysis

Args = Dict[str, Value]


VALID_ARITHMETIC_OPS = {'+', '-', '*', '/'}
VALID_RELATIONAL_OPS = {'>=', '>', '<=', '<', '==', '~='}
VALID_LOGICAL_OPS = {'|', '^', '~', '=>', '<=>'}
VALID_AGGREGATE_OPS = {'minimum', 'maximum'}
VALID_CONTROL_OPS = {'if'}


class RDDLSimulator:
    
    def __init__(self,
                 model: PlanningModel,
                 rng: np.random.Generator=np.random.default_rng()) -> None:
        '''Creates a new simulator for the given RDDL model.
        
        :param model: the RDDL model
        :param rng: the random number generator
        '''
        self._model = model        
        self._rng = rng
        
        # perform a dependency analysis, do topological sort to compute levels
        dep_analysis = RDDLDependencyAnalysis(self._model)
        self.cpforder = dep_analysis.compute_levels()
        self._order_cpfs = list(sorted(self.cpforder.keys())) 
            
        self._action_fluents = set(self._model.actions.keys())
        self._init_actions = self._model.actions.copy()
        
        # non-fluent will never change
        self._subs = self._model.nonfluents.copy()
        
        # is a POMDP
        self._pomdp = bool(self._model.observ)
        
    @staticmethod
    def _print_stack_trace(expr):
        return '...\n' + str(expr) + '\n...'

    def check_state_invariants(self) -> None:
        '''Throws an exception if the state invariants are not satisfied.'''
        for idx, invariant in enumerate(self._model.invariants):
            sample = self._sample(invariant, self._subs)
            if not isinstance(sample, bool):
                raise RDDLTypeError(
                    'State invariant must evaluate to bool, got {}.'.format(sample) + 
                    '\n' + RDDLSimulator._print_stack_trace(invariant))
            elif not sample:
                raise RDDLStateInvariantNotSatisfiedError(
                    'State invariant {} is not satisfied.'.format(idx + 1) + 
                    '\n' + RDDLSimulator._print_stack_trace(invariant))
    
    def check_action_preconditions(self) -> None:
        '''Throws an exception if the action preconditions are not satisfied.'''
        for idx, precondition in enumerate(self._model.preconditions):
            sample = self._sample(precondition, self._subs)
            if not isinstance(sample, bool):
                raise RDDLTypeError(
                    'Action precondition must evaluate to bool, got {}.'.format(sample) + 
                    '\n' + RDDLSimulator._print_stack_trace(precondition))
            elif not sample:
                raise RDDLActionPreconditionNotSatisfiedError(
                    'Action precondition {} is not satisfied.'.format(idx + 1) + 
                    '\n' + RDDLSimulator._print_stack_trace(precondition))
    
    def check_terminal_states(self) -> bool:
        '''return True if a terminal state has been reached.'''
        for idx, terminal in enumerate(self._model.terminals):
            sample = self._sample(terminal, self._subs)
            if not isinstance(sample, bool):
                raise RDDLTypeError(
                    'Terminal state condition must evaluate to bool, got {}.'.format(sample) + 
                    '\n' + RDDLSimulator._print_stack_trace(terminal))
            elif sample:
                return True
        return False
    
    def sample_reward(self) -> float:
        '''Samples the current reward given the current state and action.'''
        reward = self._sample(self._model.reward, self._subs)
        return float(reward)
    
    def reset(self) -> Args:
        '''Resets the state variables to their initial values.'''
        
        # states and actions are set to their initial values
        self._model.states.update(self._model.init_state)
        self._model.actions.update(self._init_actions)
        
        # remaining fluents do not have default or initial values
        for var in self._model.derived.keys():
            self._model.derived[var] = None
        for var in self._model.interm.keys():
            self._model.interm[var] = None
        for var in self._model.observ.keys():
            self._model.observ[var] = None
            
        # prepare initial fluent sub table
        self._subs.update(self._model.states)
        self._subs.update(self._model.actions)  
        self._subs.update(self._model.derived)    
        self._subs.update(self._model.interm)    
        self._subs.update(self._model.observ)
        for var in self._model.prev_state.keys():
            self._subs[var] = None
        
        # check termination condition for the initial state
        done = self.check_terminal_states()
        
        if self._pomdp:
            return self._model.observ.copy(), done
        else:
            return self._model.states.copy(), done
    
    def step(self, actions: Args) -> Args:
        '''Samples and returns the next state from the cpfs.
        
        :param actions: a dict mapping current action fluents to their values
        '''
        
        # if actions are missing use their default values
        self._model.actions = {var: actions.get(var, default_value) 
                               for var, default_value in self._init_actions.items()}
        subs = self._subs 
        subs.update(self._model.actions)   
        
        # evaluate all CPFs, record next state fluents, update sub table 
        next_states, next_obs = {}, {}
        for order in self._order_cpfs:
            for cpf in self.cpforder[order]: 
                if cpf in self._model.next_state:
                    primed_cpf = self._model.next_state[cpf]
                    expr = self._model.cpfs[primed_cpf]
                    sample = self._sample(expr, subs)
                    if primed_cpf not in self._model.prev_state:
                        raise KeyError(
                            'Internal error: variable <{}> is not in prev_state.'.format(primed_cpf))
                    next_states[cpf] = sample   
                    subs[primed_cpf] = sample   
                elif cpf in self._model.derived:
                    expr = self._model.cpfs[cpf]
                    sample = self._sample(expr, subs)
                    self._model.derived[cpf] = sample
                    subs[cpf] = sample   
                elif cpf in self._model.interm:
                    expr = self._model.cpfs[cpf]
                    sample = self._sample(expr, subs)
                    self._model.interm[cpf] = sample
                    subs[cpf] = sample
                elif cpf in self._model.observ:
                    expr = self._model.cpfs[cpf]
                    sample = self._sample(expr, subs)
                    self._model.observ[cpf] = sample
                    subs[cpf] = sample
                    next_obs[cpf] = sample
                else:
                    raise RDDLUndefinedCPFError('CPF <{}> is not defined.'.format(cpf))     
        
        # evaluate the immediate reward
        reward = self.sample_reward()
        
        # update the internal model state and the sub table
        self._model.states = {}
        for cpf, value in next_states.items():
            self._model.states[cpf] = value
            subs[cpf] = value
            primed_cpf = self._model.next_state[cpf]
            subs[primed_cpf] = None
        
        # check the termination condition
        done = self.check_terminal_states()
        
        if self._pomdp:
            return next_obs, reward, done
        else:
            return next_states, reward, done
    
    # start of sampling subroutines
    # skipped: aggregations (sum, prod)
    #          existential (exists, forall)
    # TODO:    replace isinstance checks with something better (polymorphism)    
    def _sample(self, expr, subs):
        if isinstance(expr, Expression):
            etype, op = expr.etype
            if etype == 'constant':
                return self._sample_constant(expr, subs)
            elif etype == 'pvar':
                return self._sample_pvar(expr, op, subs)
            elif etype == 'relational':
                return self._sample_relational(expr, op, subs)
            elif etype == 'arithmetic':
                return self._sample_arithmetic(expr, op, subs)
            elif etype == 'aggregation':
                return self._sample_aggregation(expr, op, subs)
            elif etype == 'func':
                return self._sample_func(expr, op, subs)
            elif etype == 'boolean':
                return self._sample_logical(expr, op, subs)
            elif etype == 'control':
                return self._sample_control(expr, op, subs)
            elif etype == 'randomvar':
                return self._sample_random(expr, op, subs)
            else:
                raise RDDLNotImplementedError(
                    'Internal error: expression type {} is not supported.'.format(etype) + 
                    '\n' + RDDLSimulator._print_stack_trace(expr))
        else:
            raise RDDLNotImplementedError(
                'Internal error: type {} is not supported.'.format(expr) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))

    # simple expressions
    def _sample_constant(self, expr, subs):
        return expr.args
    
    def _sample_pvar(self, expr, name, subs):
        args = expr.args
        if len(args) != 2:
            raise RDDLInvalidNumberOfArgumentsError(
                'Internal error: pvar {} requires 2 args, got {}.'.format(expr, len(args)) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
        
        var = args[0]
        if var not in subs:
            raise RDDLUndefinedVariableError(
                'Variable <{}> is not defined in the instance.'.format(var) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
            
        val = subs[var]
        if val is None:
            raise RDDLUndefinedVariableError(
                'Variable <{}> is referenced before it has been assigned a value.'.format(var) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
            
        return val
    
    def _sample_relational(self, expr, op, subs):
        if op not in VALID_RELATIONAL_OPS:
            raise RDDLNotImplementedError(
                'Relational operator {} is not supported: must be one of {}.'.format(
                    op, VALID_RELATIONAL_OPS) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
            
        args = expr.args
        if not isinstance(args, Sized):
            raise RDDLTypeError(
                'Internal error: expected Sized, got {}.'.format(type(args)) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
        elif len(args) != 2:
            raise RDDLInvalidNumberOfArgumentsError(
                'Relational operator {} requires 2 args, got {}.'.format(op, len(args)) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
        
        arg1 = self._sample(args[0], subs)
        arg2 = self._sample(args[1], subs)
        arg1 = 1 * arg1  # bool -> int
        arg2 = 1 * arg2
        
        if op == '>=':
            return arg1 >= arg2
        elif op == '<=':
            return arg1 <= arg2
        elif op == '<':
            return arg1 < arg2
        elif op == '>':
            return arg1 > arg2
        elif op == '==':
            return arg1 == arg2
        else:  # '!='
            return arg1 != arg2
    
    def _sample_arithmetic(self, expr, op, subs):
        if op not in VALID_ARITHMETIC_OPS:
            raise RDDLNotImplementedError(
                'Arithmetic operator {} is not supported: must be one of {}.'.format(
                    op, VALID_ARITHMETIC_OPS) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
        
        args = expr.args        
          
        if len(args) == 1 and op == '-':
            arg = self._sample(args[0], subs)
            arg = -1 * arg  # bool -> int  
            return arg
        
        elif len(args) >= 1:
            if op == '*':
                return self._sample_short_circuit_product(expr, args, subs)
            elif op == '+':
                terms = (1 * self._sample(arg, subs) for arg in args)  # bool -> int
                return sum(terms)
            elif len(args) == 2:
                arg1 = self._sample(args[0], subs)
                arg2 = self._sample(args[1], subs)
                arg1 = 1 * arg1  # bool -> int
                arg2 = 1 * arg2
                if op == '-':
                    return arg1 - arg2
                elif op == '/':
                    return RDDLSimulator._safe_division(expr, arg1, arg2)
            
        raise RDDLInvalidNumberOfArgumentsError(
            'Arithmetic operator {} does not have the required number of args, got {}.'.format(
                op, len(args)) + 
            '\n' + RDDLSimulator._print_stack_trace(expr))
    
    @staticmethod
    def _safe_division(expr, arg1, arg2):
        if arg1 != arg1 or arg2 != arg2:
            raise ArithmeticError(
                'Invalid (NaN) values in quotient, got {} and {}.'.format(arg1, arg2) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
        elif arg2 == 0:
            raise ArithmeticError(
                'Division by zero.' + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
        
        return arg1 / arg2  # int -> float
        
    def _sample_short_circuit_product(self, expr, args, subs):
        product = 1
        
        # evaluate variables and constants first (these are fast)
        for arg in args:
            if arg.is_constant_expression() or arg.is_pvariable_expression():
                term = self._sample(arg, subs)
                product *= term  # bool -> int
                if product == 0:
                    return product
        
        # evaluate nested expressions (can't optimize the order in general)
        for arg in args:
            if not (arg.is_constant_expression() or arg.is_pvariable_expression()):
                term = self._sample(arg, subs)
                product *= term  # bool -> int
                if product == 0:
                    return product
                
        return product
    
    # aggregations
    def _sample_aggregation(self, expr, op, subs):
        if op not in VALID_AGGREGATE_OPS:
            raise RDDLNotImplementedError(
                'Aggregation operator {} is not supported: must be one of {}.'.format(
                    op, VALID_AGGREGATE_OPS) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
        
        args = expr.args
        if not args:
            raise RDDLInvalidNumberOfArgumentsError(
                'Aggregation operator {} requires at least 1 arg, got {}.'.format(op, len(args)) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
        
        terms = (1 * self._sample(arg, subs) for arg in args)  # bool -> int
        if op == 'minimum':
            return min(terms)
        else:  # 'maximum'
            return max(terms)
        
    # functions
    KNOWN_UNARY = {        
        'abs': abs,
        'sgn': lambda x: (-1 if x < 0 else (1 if x > 0 else 0)),
        'round': round,
        'floor': math.floor,
        'ceil': math.ceil,
        'cos': math.cos,
        'sin': math.sin,
        'tan': math.tan,
        'acos': math.acos,
        'asin': math.asin,
        'atan': math.atan,
        'cosh': math.cosh,
        'sinh': math.sinh,
        'tanh': math.tanh,
        'exp': math.exp,
        'ln': math.log,
        'sqrt': math.sqrt
    }
    
    KNOWN_BINARY = {
        'div': lambda x, y: int(x) // int(y),
        'mod': lambda x, y: int(x) % int(y),
        'min': min,
        'max': max,
        'pow': math.pow,
        'log': math.log
    }
    
    def _sample_func(self, expr, name, subs):
        args = expr.args
        if isinstance(args, Expression):
            args = (args,)
        elif not isinstance(args, Sized):
            raise RDDLTypeError(
                'Internal error: expected type Sized, got {}.'.format(type(args)) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
        
        if name in RDDLSimulator.KNOWN_UNARY:
            if len(args) != 1:
                raise RDDLInvalidNumberOfArgumentsError(
                    'Unary function {} requires 1 arg, got {}.'.format(name, len(args)) + 
                    '\n' + RDDLSimulator._print_stack_trace(expr))    
                            
            arg = self._sample(args[0], subs)
            arg = 1 * arg  # bool -> int
            try:
                return RDDLSimulator.KNOWN_UNARY[name](arg)
            except:
                raise RDDLValueOutOfRangeError(
                    'Unary function {} could not be evaluated with arg {}.'.format(name, arg) + 
                    '\n' + RDDLSimulator._print_stack_trace(expr))
        
        elif name in RDDLSimulator.KNOWN_BINARY:
            if len(args) != 2:
                raise RDDLInvalidNumberOfArgumentsError(
                    'Binary function {} requires 2 args, got {}.'.format(name, len(args)) + 
                    '\n' + RDDLSimulator._print_stack_trace(expr))      
                          
            arg1 = self._sample(args[0], subs)
            arg2 = self._sample(args[1], subs)
            arg1 = 1 * arg1  # bool -> int
            arg2 = 1 * arg2
            try:
                return RDDLSimulator.KNOWN_BINARY[name](arg1, arg2)
            except:
                raise RDDLValueOutOfRangeError(
                    'Binary function {} could not be evaluated with args {} and {}.'.format(
                        name, arg1, arg2) + 
                    '\n' + RDDLSimulator._print_stack_trace(expr))
        
        else:
            raise RDDLNotImplementedError(
                'Function {} is not supported.'.format(name) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))   
    
    # logical expressions
    def _sample_logical(self, expr, op, subs):
        if op not in VALID_LOGICAL_OPS:
            raise RDDLNotImplementedError(
                'Logical operator {} is not supported: must be one of {}.'.format(
                    op, VALID_LOGICAL_OPS) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
        
        args = expr.args
        
        if len(args) == 1 and op == '~':
            arg = self._sample(args[0], subs)
            if not isinstance(arg, bool):
                raise RDDLTypeError(
                    'Logical operator {} requires boolean arg, got {}.'.format(op, arg) + 
                    '\n' + RDDLSimulator._print_stack_trace(expr))
            return not arg
        
        elif len(args) >= 1:
            if op == '|' or op == '^':
                return self._sample_short_circuit_and_or(expr, args, op, subs)
            elif len(args) == 2:
                if op == '~':
                    return self._sample_xor(expr, *args, subs)
                elif op == '=>':
                    return self._sample_short_circuit_implies(expr, *args, subs)
                elif op == '<=>':
                    return self._sample_equivalent(expr, *args, subs)
                
        raise RDDLInvalidNumberOfArgumentsError(
            'Logical operator {} does not have the required number of args, got {}.'.format(
                op, len(args)) + 
            '\n' + RDDLSimulator._print_stack_trace(expr))
        
    def _sample_short_circuit_and_or(self, expr, args, op, subs):
        
        # evaluate variables and constants first (these are fast)
        for arg in args:
            if arg.is_constant_expression() or arg.is_pvariable_expression():
                term = self._sample(arg, subs)
                if not isinstance(term, bool):
                    raise RDDLTypeError(
                        'Logical operator {} requires boolean arg, got {}.'.format(op, term) + 
                        '\n' + RDDLSimulator._print_stack_trace(expr))
                if op == '|' and term:
                    return True
                elif op == '^' and not term:
                    return False
        
        # evaluate nested expressions (can't optimize the order in general)
        for arg in args:
            if not (arg.is_constant_expression() or arg.is_pvariable_expression()):
                term = self._sample(arg, subs)
                if not isinstance(term, bool):
                    raise RDDLTypeError(
                        'Logical operator {} requires boolean arg, got {}.'.format(op, term) + 
                        '\n' + RDDLSimulator._print_stack_trace(expr))
                if op == '|' and term:
                    return True
                elif op == '^' and not term:
                    return False
        
        return op != '|'
    
    def _sample_xor(self, expr, arg1, arg2, subs):
        arg1 = self._sample(arg1, subs)
        arg2 = self._sample(arg2, subs)
        if not (isinstance(arg1, bool) and isinstance(arg2, bool)):
            raise RDDLTypeError(
                'Logical operator ~ requires boolean args, got {} and {}.'.format(arg1, arg2) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
            
        return arg1 != arg2
    
    def _sample_short_circuit_implies(self, expr, arg1, arg2, subs):
        arg1 = self._sample(arg1, subs)
        if not isinstance(arg1, bool):
            raise RDDLTypeError(
                'Logical operator => requires boolean arg, got {}.'.format(arg1) + 
                '\n' + RDDLSimulator._print_stack_trace(expr)) 
        
        if not arg1:
            return True
        
        arg2 = self._sample(arg2, subs)
        if not isinstance(arg2, bool):
            raise RDDLTypeError(
                'Logical operator => requires boolean args, got {} and {}.'.format(arg1, arg2) + 
                '\n' + RDDLSimulator._print_stack_trace(expr)) 
            
        return arg2
    
    def _sample_equivalent(self, expr, arg1, arg2, subs):
        arg1 = self._sample(arg1, subs)
        arg2 = self._sample(arg2, subs)
        if not (isinstance(arg1, bool) and isinstance(arg2, bool)):
            raise RDDLTypeError(
                'Logical operator <=> requires boolean args, got {} and {}.'.format(arg1, arg2) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
            
        return arg1 == arg2
        
    # control
    def _sample_control(self, expr, op, subs):
        if op not in VALID_CONTROL_OPS:
            raise RDDLNotImplementedError(
                'Control structure {} is not supported, must be one of {}.'.format(
                    op, VALID_CONTROL_OPS) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
        
        args = expr.args
        if len(args) != 3:
            raise RDDLInvalidNumberOfArgumentsError(
                'If statement requires 3 statements, got {}.'.format(len(args)) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
        
        condition = self._sample(args[0], subs)
        if not isinstance(condition, bool):
            raise RDDLTypeError(
                'If condition must evaluate to bool, got {}.'.format(condition) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
        
        branch = args[1 if condition else 2]
        return self._sample(branch, subs)
        
    # random variables
    def _sample_random(self, expr, name, subs):
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
        else:  # no support for enum
            raise RDDLNotImplementedError(
                'Distribution {} is not supported.'.format(name) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))

    def _sample_kron_delta(self, expr, subs):
        args = expr.args
        if len(args) != 1:
            raise RDDLInvalidNumberOfArgumentsError(
                'KronDelta requires 1 parameter, got {}.'.format(len(args)) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
        
        arg = self._sample(args[0], subs)
        if not isinstance(arg, (int, bool)):
            raise RDDLTypeError(
                'KronDelta requires int or boolean parameter, got {}.'.format(arg) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
            
        return arg
    
    def _sample_dirac_delta(self, expr, subs):
        args = expr.args
        if len(args) != 1:
            raise RDDLInvalidNumberOfArgumentsError(
                'DiracDelta requires 1 parameter, got {}.'.format(len(args)) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
            
        arg = self._sample(args[0], subs)
        if not isinstance(arg, float):
            raise RDDLTypeError(
                'DiracDelta requires float parameter, got {}.'.format(arg) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
            
        return arg
    
    def _sample_uniform(self, expr, subs):
        args = expr.args
        if len(args) != 2:
            raise RDDLInvalidNumberOfArgumentsError(
                'Uniform requires 2 parameters, got {}.'.format(len(args)) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
            
        lb = float(self._sample(args[0], subs))
        ub = float(self._sample(args[1], subs))
        if not (lb <= ub):
            raise RDDLValueOutOfRangeError(
                'Uniform bounds do not satisfy {} <= {}.'.format(lb, ub) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
        
        return self._rng.uniform(lb, ub)      
    
    def _sample_bernoulli(self, expr, subs):
        args = expr.args
        if len(args) != 1:
            raise RDDLInvalidNumberOfArgumentsError(
                'Bernoulli requires 1 parameter, got {}.'.format(len(args)) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
            
        p = float(self._sample(args[0], subs))
        if not (0 <= p <= 1):
            raise RDDLValueOutOfRangeError(
                'Bernoulli parameter p should be in [0, 1], got {}.'.format(p) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
        
        return self._rng.uniform() <= p
    
    def _sample_normal(self, expr, subs):
        args = expr.args
        if len(args) != 2:
            raise RDDLInvalidNumberOfArgumentsError(
                'Normal requires 2 parameters, got {}.'.format(len(args)) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
            
        mean = float(self._sample(args[0], subs))
        var = float(self._sample(args[1], subs))
        if not (var >= 0.):
            raise RDDLValueOutOfRangeError(
                'Normal variance {} is not positive.'.format(var) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
            
        scale = math.sqrt(var)
        return self._rng.normal(loc=mean, scale=scale)
    
    def _sample_poisson(self, expr, subs):
        args = expr.args
        if len(args) != 1:
            raise RDDLInvalidNumberOfArgumentsError(
                'Poisson requires 1 parameter, got {}.'.format(len(args)) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
            
        rate = float(self._sample(args[0], subs))
        if not (rate >= 0.):
            raise RDDLValueOutOfRangeError(
                'Poisson rate {} is not positive.'.format(rate) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
        
        return self._rng.poisson(lam=rate)
    
    def _sample_exponential(self, expr, subs):
        args = expr.args
        if len(args) != 1:
            raise RDDLInvalidNumberOfArgumentsError(
                'Exponential requires 1 parameter, got {}.'.format(len(args)) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
            
        scale = float(self._sample(args[0], subs))
        if not (scale > 0.):
            raise RDDLValueOutOfRangeError(
                'Exponential rate {} is not positive.'.format(scale) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
        
        return self._rng.exponential(scale=scale)
    
    def _sample_weibull(self, expr, subs):
        args = expr.args
        if len(args) != 2:
            raise RDDLInvalidNumberOfArgumentsError(
                'Weibull requires 2 parameters, got {}.'.format(len(args)) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
            
        shape = float(self._sample(args[0], subs))
        if not (shape > 0.):
            raise RDDLValueOutOfRangeError(
                'Weibull shape {} is not positive.'.format(shape) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
        
        scale = float(self._sample(args[1], subs))
        if not (scale > 0.):
            raise RDDLValueOutOfRangeError(
                'Weibull scale {} is not positive.'.format(scale) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
        
        return scale * self._rng.weibull(shape)
    
    def _sample_gamma(self, expr, subs):
        args = expr.args
        if len(args) != 2:
            raise RDDLInvalidNumberOfArgumentsError(
                'Gamma requires 2 parameters, got {}.'.format(len(args)) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
            
        shape = float(self._sample(args[0], subs))
        if not (shape > 0.):
            raise RDDLValueOutOfRangeError(
                'Gamma shape {} is not positive.'.format(shape) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
            
        scale = float(self._sample(args[1], subs))
        if not (scale > 0.):
            raise RDDLValueOutOfRangeError(
                'Gamma scale {} is not positive.'.format(scale) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
        
        return self._rng.gamma(shape=shape, scale=scale)


class RDDLSimulatorWConstraints(RDDLSimulator):

    def __init__(self,
                 model: PlanningModel,
                 rng: np.random.Generator = np.random.default_rng(),
                 compute_levels: bool = True,
                 max_bound: float = np.inf) -> None:
        super().__init__(model, rng)#, compute_levels)

        self.epsilon = 0.001
        # self.BigM = float(max_bound)
        self.BigM = max_bound
        self._bounds = {}
        for state in model.states:
            self._bounds[state] = [-self.BigM, self.BigM]
        for derived in model.derived:
            self._bounds[derived] = [-self.BigM, self.BigM]
        for interm in model.interm:
            self._bounds[interm] = [-self.BigM, self.BigM]
        for action in model.actions:
            self._bounds[action] = [-self.BigM, self.BigM]
        for obs in model.observ:
            self._bounds[obs]  = [-self.BigM, self.BigM]

        # actions and states bounds extraction for gym's action and state spaces repots only!
        # currently supports only linear in\equality constraints
        for action_precond in model.preconditions:
            self._parse_bounds_rec(action_precond, self._model.actions)

        for state_inv in model.invariants:
            self._parse_bounds_rec(state_inv, self._model.states)

        for name in self._bounds:
            if self._bounds[name][0] > self._bounds[name][1]:
                raise RDDLValueOutOfRangeError(
                    'Variable <{}> bounds are invalid, max value cannot be lower than min.'.format(name) +
                    '\n' + RDDLSimulator._print_stack_trace(self._bounds[name]))

    def _parse_bounds_rec(self, cond, search_dict):
        if cond.etype[0] == "boolean" and cond.etype[1] == "^":
            for arg in cond.args:
                self._parse_bounds_rec(arg, search_dict)
        if cond.etype[0] == "relational":
            var, lim, loc = self.get_bounds(cond.args[0], cond.args[1], cond.etype[1], search_dict)
            if var is not None and loc is not None:
                if loc == 1:
                    if self._bounds[var][loc] > lim:
                        self._bounds[var][loc] = lim
                else:
                    if self._bounds[var][loc] < lim:
                        self._bounds[var][loc] = lim

    def get_bounds(self, left_arg, right_arg, op, search_dict=None) -> Tuple[str, float, int]:
        variable = None
        lim = 0
        loc = None
        if search_dict is None:
            return variable, float(lim), loc

        # make sure at least one of the args is a pvar of the correct type
        act_count = 0
        if left_arg.etype[0] == 'pvar':
            if left_arg.args[0] in search_dict:
                act_count = act_count + 1
        if right_arg.etype[0] == 'pvar':
            if right_arg.args[0] in search_dict:
                act_count = act_count + 1
        if act_count == 2:
            raise RDDLInvalidExpressionError(
                "Error in action-precondition block, " +
                "constraint {} {} {} does not have a structure of " +
                "action/state fluent <=/>= f(non-fluents, constants)".format(
                left_arg.args[0], right_arg.args[0]) +
                '\n' + RDDLSimulator._print_stack_trace(
                    left_arg.args[0] + op + right_arg.args[0]))
        if act_count == 0:
            return None, 0, None

        if left_arg.etype[0] == 'pvar':
            variable = left_arg.args[0]
            if variable in search_dict:
                if self.verify_tree_is_box(right_arg):
                    lim_temp = self._sample(right_arg, self._subs)
                    lim, loc = self.get_op_code(op, is_right=True)
                    return variable, float(lim + lim_temp), loc
                else:
                    raise RDDLInvalidExpressionError(
                        "Error in action-precondition block, bound {} must be a " + 
                        "determinisic function of non-fluents and constants only".format(
                            right_arg) + "\n" + RDDLSimulator._print_stack_trace(right_arg))

        elif right_arg.etype[0] == 'pvar':
            variable = right_arg.args[0]
            if variable in search_dict:
                if self.verify_tree_is_box(left_arg):
                    lim_temp = self._sample(left_arg, self._subs)
                    lim, loc = self.get_op_code(op, is_right=False)
                    return variable, float(lim + lim_temp), loc
                else:
                    raise RDDLInvalidExpressionError(
                        "Error in action-precondition block, bound {} must be a " + 
                        "determinisic function of non-fluents and constants only".format(
                            left_arg) + "\n" + RDDLSimulator._print_stack_trace(right_arg))
        else:
            raise RDDLInvalidExpressionError(
                "Error in action-precondition block, " +
                "constraint {} {} {} does not have a structure of " +
                "action/state fluent <=/>= f(non-fluents, constants)".format(
                        left_arg.args[0], right_arg.args[0]) +
                '\n' + RDDLSimulator._print_stack_trace(
                    left_arg.args[0] + op + right_arg.args[0]))

    def verify_tree_is_box(self, expr):
        if hasattr(expr, 'args') == False:
            return False
        if expr.etype[0] == "randomvar":
            return False
        if expr.etype[0] == 'pvar':
            if expr.args[0] in self._model.nonfluents:
                return True
            return False
        if expr.etype[0] == 'constant':
            return True
        else:
            result = True
            for elem in expr.args:
                if not self.verify_tree_is_box(elem):
                    return False
                # result = result and self.verify_tree_is_box(elem)
            return result

    def get_op_code(self, op, is_right: bool) -> (float, int):
        '''

        Args:
            op: inequality operator
            is_right: True if the evaluated element is to the left of the inequality, False otherwise
        Returns:
            (lim , loc)
            lim =  zero for soft inequality, minimum real number of strong inequality
        '''
        lim = 0
        if is_right:
            if op in ['<=', '<']:
                loc = 1
                if op == '<':
                    lim = -self.epsilon
            elif op in ['>=', '>']:
                loc = 0
                if op == '>':
                    lim = self.epsilon
        else:
            if op in ['<=', '<']:
                loc = 0
                if op == '<':
                    lim = self.epsilon
            elif op in ['>=', '>']:
                loc = 1
                if op == '>':
                    lim = -self.epsilon
        return (lim, loc)

    # def get_bounds(self, left_arg, right_arg, op, search_dict=None) -> Tuple[str, float, int]:
    #     variable = None
    #     lim = 0
    #     loc = None
    #     if search_dict is None:
    #         return variable, float(lim), loc
    #
    #     if left_arg.etype[0] == 'pvar':
    #         name = left_arg.args[0]
    #         if name in search_dict:
    #             variable = name
    #             if op in ['<=', '<']:
    #                 loc = 1
    #                 if op == '<':
    #                     lim = -self.epsilon
    #             elif op in ['>=', '>']:
    #                 loc = 0
    #                 if op == '>':
    #                     lim = self.epsilon
    #         elif name in self._model.nonfluents:
    #             lim = self._model.nonfluents[name]
    #     elif left_arg.etype[0] == 'constant':
    #         lim += left_arg.args
    #     else:
    #         return variable, float(lim), loc
    #
    #     if right_arg.etype[0] == 'pvar':
    #         name = right_arg.args[0]
    #         if name in search_dict:
    #             variable = name
    #             if op in ['<=', '<']:
    #                 loc = 0
    #                 if op == '<':
    #                     lim = self.epsilon
    #             elif op in ['>=', '>']:
    #                 loc = 1
    #                 if op == '>':
    #                     lim = -self.epsilon
    #         elif name in self._model.nonfluents:
    #             lim = self._model.nonfluents[name]
    #     elif right_arg.etype[0] == 'constant':
    #         lim += right_arg.args
    #     else:
    #         return variable, float(lim), loc
    #
    #     return variable, float(lim), loc

    @property
    def bounds(self):
        return self._bounds

    @bounds.setter
    def bounds(self, value):
        self._bounds = value

    @property
    def states(self):
        return self._model.states.copy()

    @property
    def isPOMDP(self):
        return self._pomdp

