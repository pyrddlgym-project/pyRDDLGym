from collections.abc import Sized
import math
import numpy as np
import random

from Grounder.RDDLModel import PlanningModel
from Parser.expr import Expression


class RDDLSimulator:
    
    def __init__(self, model: PlanningModel):
        '''Creates a new simulator for the given RDDL model.
        
        :param model: the RDDL model
        '''
        self._model = model        
        self._order_cpfs = sorted(self._model.cpforder.keys())
        
        self._subs = {}
        self._next_state = None        
        
    def reset_state(self):
        '''Resets the state variables to their initial values.'''
        self._model.states = self._model.init_state
        self._next_state = None
        return self._model.states
    
    def _update_subs(self):
        self._subs.clear()
        self._subs.update(self._model.nonfluents)
        self._subs.update(self._model.states)
        self._subs.update(self._model.actions)        
        return self._subs
        
    def check_state_invariants(self):
        '''Throws an exception if the state invariants for the current state variables are not satisfied.'''
        subs = self._update_subs()
        for idx, invariant in enumerate(self._model.invariants):
            sample = self._sample(invariant, subs)
            if not isinstance(sample, bool):
                raise Exception('State invariant must evaluate to bool, {} provided.'.format(sample))
            if not sample:
                raise Exception('State invariant #{} failed to be satisfied.'.format(idx + 1))
        
    def sample_next_state(self, actions):
        '''Samples and returns the next state from the cpfs.
        
        :param actions: a dict mapping current action fluents to their values
        '''
        if self._next_state is not None:
            raise Exception('A state has already been sampled: call update_state to update it.')
        self._model.actions = actions
        subs = self._update_subs()  
        next_state = {}
        for order in self._order_cpfs:
            for cpf in self._model.cpforder[order]: 
                if cpf in self._model.next_state:
                    primed_cpf = self._model.next_state[cpf]
                    expr = self._model.cpfs[primed_cpf].expr
                    sample = self._sample(expr, self._subs)
                    if primed_cpf not in self._model.prev_state:
                        raise Exception('Internal error: var {} not in prev_state.'.format(primed_cpf))
                    next_state[self._model.prev_state[primed_cpf]] = sample                    
                elif cpf in self._model.interm:
                    raise Exception('Interm vars not supported yet.')  # TODO: implement this
                else:
                    raise Exception('Undefined cpf {}.'.format(cpf))                
        self._next_state = next_state
        return next_state
    
    def sample_reward(self):
        '''Samples the current reward given the current state and action.'''
        expr = self._model.reward
        subs = self._update_subs()
        return float(self._sample(expr, subs))
    
    def update_state(self):
        '''Updates the state of the simulation to the sampled state.'''
        if self._next_state is None:
            raise Exception('Next state not sampled yet: call sample_next_state first.')
        self._model.states = self._next_state
        self._next_state = None
    
    # start of sampling subroutines
    # skipped: aggregations (sum, prod)
    #          existential (exists, forall)
    #          =>, <=>
    # TODO:    switch and enum
    #          replace isinstance checks with something better (polymorphism)
    @staticmethod
    def _print_stack_trace(expr):
        return '...\n' + str(expr) + '\n...'
    
    def _sample(self, expr, subs):
        if isinstance(expr, Expression):
            etype, eop = expr.etype
            if etype == 'constant':
                return self._sample_constant(expr, subs)
            elif etype == 'pvar':
                return self._sample_pvar(expr, eop, subs)
            elif etype == 'relational':
                return self._sample_relational(expr, eop, subs)
            elif etype == 'arithmetic':
                return self._sample_arithmetic(expr, eop, subs)
            elif etype == 'func':
                return self._sample_func(expr, eop, subs)
            elif etype == 'boolean':
                return self._sample_logical(expr, eop, subs)
            elif etype == 'control':
                return self._sample_control(expr, eop, subs)
            elif etype == 'randomvar':
                return self._sample_random(expr, eop, subs)
            else:
                raise Exception('Expression type {} is not supported.'.format(etype) + 
                                '\n' + RDDLSimulator._print_stack_trace(expr))
        else:
            raise Exception('Type {} is not an expression.'.format(expr) + 
                            '\n' + RDDLSimulator._print_stack_trace(expr))

    # simple expressions
    def _sample_constant(self, expr, subs):
        return expr.args
    
    def _sample_pvar(self, expr, name, subs):
        args = expr.args
        if len(args) != 2:
            raise Exception('Malformed pvar def {}.'.format(expr) + 
                            '\n' + RDDLSimulator._print_stack_trace(expr))
        
        var = args[0]
        if var not in subs:
            raise Exception('Variable {} is not defined.'.format(var) + 
                            '\n' + RDDLSimulator._print_stack_trace(expr))
        return subs[var]
        
    def _sample_relational(self, expr, eop, subs):
        args = expr.args
        if not isinstance(args, Sized):
            raise Exception('Expression args are invalid.' + 
                            '\n' + RDDLSimulator._print_stack_trace(expr))
        if len(args) != 2:
            raise Exception('Relational expression {} requires two args.'.format(eop) + 
                            '\n' + RDDLSimulator._print_stack_trace(expr))
        
        arg1 = self._sample(args[0], subs)
        arg2 = self._sample(args[1], subs)
        arg1 = 1 * arg1  # bool -> int
        arg2 = 1 * arg2
        
        if eop == '>=':
            return arg1 >= arg2
        elif eop == '<=':
            return arg1 <= arg2
        elif eop == '<':
            return arg1 < arg2
        elif eop == '>':
            return arg1 > arg2
        elif eop == '==':
            return arg1 == arg2
        elif eop == '~=':
            return arg1 != arg2
        else:
            raise Exception('Relational operator {} is not supported.'.format(eop) + 
                            '\n' + RDDLSimulator._print_stack_trace(expr))
    
    def _sample_arithmetic(self, expr, eop, subs):
        args = expr.args        
        if len(args) == 1 and eop == '-':
            arg = self._sample(args[0], subs)
            return -1 * arg  # bool -> int
        elif len(args) == 2:
            if eop == '*':
                return self._sample_short_circuit_product(expr, *args, subs)
            else:
                arg1 = self._sample(args[0], subs)
                arg2 = self._sample(args[1], subs)
                return RDDLSimulator._apply_arithmetic_rule(expr, arg1, arg2, eop)        
        else:
            raise Exception('Arithmetic expression {} cannot be evaluated.'.format(eop) + 
                            '\n' + RDDLSimulator._print_stack_trace(expr))
    
    @staticmethod
    def _apply_arithmetic_rule(expr, arg1, arg2, op):
        arg1 = 1 * arg1  # bool -> int
        arg2 = 1 * arg2
            
        if op == '+':
            return arg1 + arg2
        elif op == '-':
            return arg1 - arg2
        elif op == '*':
            return arg1 * arg2
        elif op == '/':
            return arg1 / arg2  # int -> float
        else:
            raise Exception('Invalid arithmetic operator {}.'.format(op) + 
                            '\n' + RDDLSimulator._print_stack_trace(expr))
    
    def _sample_short_circuit_product(self, expr, expr1, expr2, subs):
        if expr2.is_constant_expression() or expr2.is_pvariable_expression():  # TODO: is this ok?
            expr1, expr2 = expr2, expr1
        
        arg1 = self._sample(expr1, subs)
        arg1 = 1 * arg1  # bool -> int
        if arg1 == 0:
            return arg1
        
        arg2 = self._sample(expr2, subs)
        arg2 = 1 * arg2  # bool -> int
        return arg1 * arg2
    
    # functions
    KNOWN_UNARY = {        
        'abs': abs,
        'sgn': lambda x: math.copysign(1, x),
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
            raise Exception('Func args are invalid.' + 
                            '\n' + RDDLSimulator._print_stack_trace(expr))
        
        if name in RDDLSimulator.KNOWN_UNARY:
            if len(args) != 1:
                raise Exception('Func {} requires one arg.'.format(name) + 
                                '\n' + RDDLSimulator._print_stack_trace(expr))
            arg = self._sample(args[0], subs)
            arg = 1 * arg  # bool -> int
            return RDDLSimulator.KNOWN_UNARY[name](arg)
        
        elif name in RDDLSimulator.KNOWN_BINARY:
            if len(args) != 2:
                raise Exception('Func {} requires two args.'.format(name) + 
                                '\n' + RDDLSimulator._print_stack_trace(expr))
            arg1 = self._sample(args[0], subs)
            arg2 = self._sample(args[1], subs)
            arg1 = 1 * arg1  # bool -> int
            arg2 = 1 * arg2
            return RDDLSimulator.KNOWN_BINARY[name](arg1, arg2)
        
        else:
            raise Exception('Func {} is not supported.'.format(name) + 
                            '\n' + RDDLSimulator._print_stack_trace(expr))   
    
    # def _sample_aggregation(self, expr, eop, subs):
    #     if eop == 'sum': 
    #         eop = '+'
    #     elif eop == 'prod': 
    #         eop = '*'
    #     elif eop != 'min' and eop != 'max':
    #         raise Exception('Aggregation op {} is not supported.'.format(eop))
    #
    #     args = expr.args
    #     if len(args) <= 1:
    #         raise Exception('Aggregation must contain at least one var and one arg.')
    #
    #     typed_vars = args[:-1]
    #     subexpr = args[-1]
    #     names = [var for _, (var, obj) in typed_vars]
    #     subs = [self._ground.objects[obj] for _, (var, obj) in typed_vars]
    #     agg = None
    #     for i, sub in enumerate(itertools.product(*subs)):
    #         subs.update(zip(names, sub))
    #         term = self._sample(subexpr, subs)
    #         agg = RDDLSimulator._update_arithmetic(agg, term, eop) if i else term
    #
    #     for name in names:
    #         del subs[name]
    #     return agg
    
    # logical expressions
    def _sample_logical(self, expr, eop, subs):
        args = expr.args
        if len(args) == 1 and eop == '~':
            arg = self._sample(args[0], subs)
            if not isinstance(arg, bool):
                raise Exception('Binary negation requires boolean arg.' + 
                                '\n' + RDDLSimulator._print_stack_trace(expr))
            return not arg
        elif len(args) == 2:
            if eop == '|' or eop == '^':
                return self._sample_short_circuit(expr, *args, eop, subs)
            elif eop == '~':
                return self._sample_xor(expr, *args, subs)
            else:  # TODO: do we need =>, <=> for grounded?
                raise Exception('Binary operator {} is not supported.'.format(eop) + 
                                '\n' + RDDLSimulator._print_stack_trace(expr))
        else:
            raise Exception('Logical expression {} cannot be evaluated.'.format(eop) + 
                            '\n' + RDDLSimulator._print_stack_trace(expr))
        
    def _sample_short_circuit(self, expr, expr1, expr2, op, subs):
        if expr2.is_constant_expression() or expr2.is_pvariable_expression():  # TODO: is this ok?
            expr1, expr2 = expr2, expr1
        
        arg1 = self._sample(expr1, subs)
        if not isinstance(arg1, bool):
            raise Exception('Binary operator {} requires boolean arg, {} provided.'.format(op, arg1) + 
                            '\n' + RDDLSimulator._print_stack_trace(expr))
        if op == '|' and arg1:
            return True
        elif op == '^' and not arg1:
            return False
        
        arg2 = self._sample(expr2, subs)
        if not isinstance(arg1, bool):
            raise Exception('Binary operator {} requires boolean arg, {} provided.'.format(op, arg2) + 
                            '\n' + RDDLSimulator._print_stack_trace(expr))
        return arg2
        
    def _sample_xor(self, expr, arg1, arg2, subs):
        arg1 = self._sample(arg1, subs)
        arg2 = self._sample(arg2, subs)
        if not (isinstance(arg1, bool) and isinstance(arg2, bool)):
            raise Exception('Xor requires boolean args.' + 
                            '\n' + RDDLSimulator._print_stack_trace(expr))
        return arg1 != arg2
    
    # control
    def _sample_control(self, expr, eop, subs):
        args = expr.args
        if eop != 'if':
            raise Exception('Control type {} is not supported.'.format(eop) + 
                            '\n' + RDDLSimulator._print_stack_trace(expr))
        if len(args) != 3:
            raise Exception('If statement requires three args.' + 
                            '\n' + RDDLSimulator._print_stack_trace(expr))
        
        cond = self._sample(args[0], subs)
        if not isinstance(cond, bool):
            raise Exception('If condition must evaluate to bool, {} provided.'.format(cond) + 
                            '\n' + RDDLSimulator._print_stack_trace(expr))
        return self._sample(args[1 if cond else 2], subs)
        
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
        elif name == 'Discrete':  # TODO, implement Discrete
            raise Exception('Discrete not yet supported...' + 
                            '\n' + RDDLSimulator._print_stack_trace(expr))
        elif name == 'Multinomial':  # TODO, implement Multinomial
            raise Exception('Multinomial not yet supported...' + 
                            '\n' + RDDLSimulator._print_stack_trace(expr))            
        elif name == 'Dirichlet':  # TODO, implement Dirichlet
            raise Exception('Dirichlet not yet supported.' + 
                            '\n' + RDDLSimulator._print_stack_trace(expr))     
        else:
            raise Exception('Distribution {} is not supported.'.format(name) + 
                            '\n' + RDDLSimulator._print_stack_trace(expr))

    def _sample_kron_delta(self, expr, subs):
        args = expr.args
        if len(args) != 1:
            raise Exception('KronDelta requires one parameter.' + 
                            '\n' + RDDLSimulator._print_stack_trace(expr))
        
        arg = self._sample(args[0], subs)
        if not isinstance(arg, (int, bool)):
            raise Exception('KronDelta requires integer or boolean parameter.')
        return arg
    
    def _sample_dirac_delta(self, expr, subs):
        args = expr.args
        if len(args) != 1:
            raise Exception('DiracDelta requires one parameter.' + 
                            '\n' + RDDLSimulator._print_stack_trace(expr))
            
        arg = self._sample(args[0], subs)
        if not isinstance(arg, float):
            raise Exception('DiracDelta requires float parameter.')
        return arg
    
    def _sample_uniform(self, expr, subs):
        args = expr.args
        if len(args) != 2:
            raise Exception('Uniform requires two parameters.')
            
        lb = float(self._sample(args[0], subs))
        ub = float(self._sample(args[1], subs))
        if not (lb <= ub):
            raise Exception('Uniform bounds do not satisfy {} <= {}.'.format(lb, ub) + 
                            '\n' + RDDLSimulator._print_stack_trace(expr))
        
        return random.uniform(lb, ub)      
    
    def _sample_bernoulli(self, expr, subs):
        args = expr.args
        if len(args) != 1:
            raise Exception('Bernoulli distribution requires one parameter.' + 
                            '\n' + RDDLSimulator._print_stack_trace(expr))
            
        p = float(self._sample(args[0], subs))
        if not (0 <= p <= 1):
            raise Exception('Bernoulli parameter p should be in [0, 1], found = {}.'.format(p) + 
                            '\n' + RDDLSimulator._print_stack_trace(expr))
        
        return random.random() <= p
    
    def _sample_normal(self, expr, subs):
        args = expr.args
        if len(args) != 2:
            raise Exception('Normal requires two parameters.' + 
                            '\n' + RDDLSimulator._print_stack_trace(expr))
            
        mean = float(self._sample(args[0], subs))
        var = float(self._sample(args[1], subs))
        if not (var >= 0.):
            raise Exception('Variance of Normal {} is not positive.'.format(var) + 
                            '\n' + RDDLSimulator._print_stack_trace(expr))
        stdev = math.sqrt(var)
        
        return np.random.normal(loc=mean, scale=stdev)
    
    def _sample_poisson(self, expr, subs):
        args = expr.args
        if len(args) != 1:
            raise Exception('Poisson requires one parameter.' + 
                            '\n' + RDDLSimulator._print_stack_trace(expr))
            
        rate = float(self._sample(args[0], subs))
        if not (rate >= 0.):
            raise Exception('Rate of Poisson {} is not positive.'.format(rate) + 
                            '\n' + RDDLSimulator._print_stack_trace(expr))
        
        return np.random.poisson(lam=rate)
    
    def _sample_exponential(self, expr, subs):
        args = expr.args
        if len(args) != 1:
            raise Exception('Exponential requires one parameter.' + 
                            '\n' + RDDLSimulator._print_stack_trace(expr))
            
        scale = float(self._sample(args[0], subs))
        if not (scale > 0.):
            raise Exception('Rate of Exponential {} is not positive.'.format(scale) + 
                            '\n' + RDDLSimulator._print_stack_trace(expr))
        
        return np.random.exponential(scale=scale)
    
    def _sample_weibull(self, expr, subs):
        args = expr.args
        if len(args) != 2:
            raise Exception('Weibull requires two parameters.' + 
                            '\n' + RDDLSimulator._print_stack_trace(expr))
            
        shape = float(self._sample(args[0], subs))
        if not (shape > 0.):
            raise Exception('Shape of Weibull {} is not positive.'.format(shape) + 
                            '\n' + RDDLSimulator._print_stack_trace(expr))
        
        scale = float(self._sample(args[1], subs))
        if not (scale > 0.):
            raise Exception('Scale of Weibull {} is not positive.'.format(scale) + 
                            '\n' + RDDLSimulator._print_stack_trace(expr))
        
        return scale * np.random.weibull(shape)
    
    def _sample_gamma(self, expr, subs):
        args = expr.args
        if len(args) != 2:
            raise Exception('Gamma requires two parameters.' + 
                            '\n' + RDDLSimulator._print_stack_trace(expr))
            
        shape = float(self._sample(args[0], subs))
        if not (shape > 0.):
            raise Exception('Shape of Gamma {} is not positive.'.format(shape) + 
                            '\n' + RDDLSimulator._print_stack_trace(expr))
        
        scale = float(self._sample(args[1], subs))
        if not (scale > 0.):
            raise Exception('Scale of Gamma {} is not positive.'.format(scale) + 
                            '\n' + RDDLSimulator._print_stack_trace(expr))
        
        return np.random.gamma(shape=shape, scale=scale)
    
