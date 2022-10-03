from collections.abc import Sized
import math
import numpy as np
from typing import Dict

from Grounder.RDDLModel import PlanningModel
from Parser.expr import Expression, Value

Args = Dict[str, Value]


class RDDLRuntimeError(RuntimeError):
    pass


class RDDLSimulator:
    
    def __init__(self, model: PlanningModel, rng=np.random.default_rng()) -> None:
        '''Creates a new simulator for the given RDDL model.
        
        :param model: the RDDL model
        :param rng: the random number generator
        '''
        self._model = model        
        self._rng = rng
        
        self._order_cpfs = sorted(self._model.cpforder.keys())
        self._action_fluents = set(self._model.actions.keys())
        self._init_actions = self._model.actions.copy()
        
        self._subs = self._model.nonfluents.copy()  # these won't change
        self._next_state = None


    def reset_state(self) -> Args:
        '''Resets the state variables to their initial values.'''
        self._model.states = self._model.init_state
        self._model.actions = self._init_actions
        for var in self._model.interm.keys():  # TODO: unless can be default (parser throws error)
            self._model.interm[var] = None
        for var in self._model.derived.keys():
            self._model.derived[var] = None
        self._next_state = None
        return self._model.states
    
    def _update_subs(self) -> Args:
        self._subs.update(self._model.states)
        self._subs.update(self._model.actions)  
        self._subs.update(self._model.interm)    
        self._subs.update(self._model.derived)    
        return self._subs
        
    def check_state_invariants(self) -> None:
        '''Throws an exception if the state invariants are not satisfied.'''
        subs = self._update_subs()
        for idx, invariant in enumerate(self._model.invariants):
            sample = self._sample(invariant, subs)
            if not isinstance(sample, bool):
                raise TypeError(
                    'State invariant must evaluate to bool, got {}.'.format(sample) + 
                    '\n' + RDDLSimulator._print_stack_trace(invariant))
            if not sample:
                raise RDDLRuntimeError(
                    'State invariant {} is not satisfied.'.format(idx + 1) + 
                    '\n' + RDDLSimulator._print_stack_trace(invariant))
    
    def check_action_preconditions(self) -> None:
        '''Throws an exception if the action preconditions are not satisfied.'''
        subs = self._update_subs()
        for idx, precondition in enumerate(self._model.preconditions):
            sample = self._sample(precondition, subs)
            if not isinstance(sample, bool):
                raise TypeError(
                    'Action precondition must evaluate to bool, got {}.'.format(sample) + 
                    '\n' + RDDLSimulator._print_stack_trace(precondition))
            if not sample:
                raise RDDLRuntimeError(
                    'Action precondition {} is not satisfied.'.format(idx + 1) + 
                    '\n' + RDDLSimulator._print_stack_trace(precondition))
    
    def sample_next_state(self, actions: Args) -> Args:
        '''Samples and returns the next state from the cpfs.
        
        :param actions: a dict mapping current action fluents to their values
        '''
        if self._next_state is not None:
            raise RDDLRuntimeError(
                'A state has already been sampled: call update_state to update it.')

        actions_with_defaults = {}
        for k, default_action in self._init_actions.items():
            actions_with_defaults[k] = actions[k] if k in actions else default_action
        self._model.actions = actions_with_defaults
        subs = self._update_subs()  
        next_state = {}
        local_order = list(self._order_cpfs.copy())
        local_order.remove(0)
        local_order.append(0) 
        for order in local_order:
            for cpf in self._model.cpforder[order]: 
                if cpf in self._model.next_state:
                    primed_cpf = self._model.next_state[cpf]
                    expr = self._model.cpfs[primed_cpf]
                    sample = self._sample(expr, subs)
                    if primed_cpf not in self._model.prev_state:
                        raise KeyError(
                            'Internal error: variable {} is not in prev_state.'.format(primed_cpf))
                    next_state[self._model.prev_state[primed_cpf]] = sample   
                    subs[primed_cpf] = sample  # TODO: don't know if we need this             
                elif cpf in self._model.interm:
                    expr = self._model.cpfs[cpf]
                    sample = self._sample(expr, subs)
                    subs[cpf] = sample
                    self._model.interm[cpf] = sample
                elif cpf in self._model.derived:
                    expr = self._model.cpfs[cpf]
                    sample = self._sample(expr, subs)
                    subs[cpf] = sample
                    self._model.derived[cpf] = sample
                else:
                    raise SyntaxError('CPF {} is not defined in the instance.'.format(cpf))     
                           
        self._next_state = next_state
        return next_state
    
    def sample_reward(self) -> float:
        '''Samples the current reward given the current state and action.'''
        expr = self._model.reward
        subs = self._update_subs()
        return float(self._sample(expr, subs))
    
    def update_state(self, forced_state = None) -> None:
        '''Updates the state of the simulation to the sampled state.'''
        if forced_state is None:
            if self._next_state is None:
                raise RDDLRuntimeError(
                    'Next state has not been sampled: call sample_next_state.')
            self._model.states = self._next_state
        else:
            self._model.states = forced_state
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
                raise NotImplementedError(
                    'Internal error: expression type {} is not supported.'.format(etype) + 
                    '\n' + RDDLSimulator._print_stack_trace(expr))
        else:
            raise NotImplementedError(
                'Internal error: type {} is not supported.'.format(expr) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))

    # simple expressions
    def _sample_constant(self, expr, subs):
        return expr.args
    
    def _sample_pvar(self, expr, name, subs):
        args = expr.args
        if len(args) != 2:
            raise ValueError(
                'Internal error: pvar {} requires 2 args, got {}'.format(expr, len(args)) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
        
        var = args[0]
        if var not in subs:
            raise SyntaxError(
                'Variable {} is not defined in the instance.'.format(var) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
        val = subs[var]
        
        if val is None:
            raise ValueError(
                'The value of variable {} is not set.'.format(var) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
        return val
    
    def _sample_relational(self, expr, eop, subs):
        args = expr.args
        if not isinstance(args, Sized):
            raise TypeError(
                'Internal error: expected Sized, got {}.'.format(type(args)) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
        if len(args) != 2:
            raise ValueError(
                'Relational expression {} requires 2 args, got {}.'.format(eop, len(args)) + 
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
            raise NotImplementedError(
                'Relational operator {} is not supported.'.format(eop) + 
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
            raise ValueError(
                'Arithmetic operator {} requires 2 args, got {}'.format(eop, len(args)) + 
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
            raise NotImplementedError(
                'Arithmetic operator {} is not supported.'.format(op) + 
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
            raise TypeError(
                'Internal error: expected type Sized, got {}.'.format(type(args)) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
        
        if name in RDDLSimulator.KNOWN_UNARY:
            if len(args) != 1:
                raise ValueError(
                    'Function {} requires 1 arg, got {}.'.format(name, len(args)) + 
                    '\n' + RDDLSimulator._print_stack_trace(expr))    
                            
            arg = self._sample(args[0], subs)
            arg = 1 * arg  # bool -> int
            return RDDLSimulator.KNOWN_UNARY[name](arg)
        
        elif name in RDDLSimulator.KNOWN_BINARY:
            if len(args) != 2:
                raise ValueError(
                    'Function {} requires 2 args, got {}.'.format(name, len(args)) + 
                    '\n' + RDDLSimulator._print_stack_trace(expr))      
                          
            arg1 = self._sample(args[0], subs)
            arg2 = self._sample(args[1], subs)
            arg1 = 1 * arg1  # bool -> int
            arg2 = 1 * arg2
            return RDDLSimulator.KNOWN_BINARY[name](arg1, arg2)
        
        else:
            raise NotImplementedError(
                'Function {} is not supported.'.format(name) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))   
    
    # logical expressions
    def _sample_logical(self, expr, eop, subs):
        args = expr.args
        if len(args) == 1 and eop == '~':
            arg = self._sample(args[0], subs)
            if not isinstance(arg, bool):
                raise TypeError(
                    'Logical operator {} requires boolean arg, got {}.'.format(eop, arg) + 
                    '\n' + RDDLSimulator._print_stack_trace(expr))
            return not arg
        
        elif len(args) == 2:
            if eop == '|' or eop == '^':
                return self._sample_short_circuit(expr, *args, eop, subs)
            elif eop == '~':
                return self._sample_xor(expr, *args, subs)
            else:  # TODO: do we need =>, <=> for grounded?
                raise NotImplementedError(
                    'Logical operator {} is not supported.'.format(eop) + 
                    '\n' + RDDLSimulator._print_stack_trace(expr))
                
        else:
            raise ValueError(
                'Logical operator {} require 2 args, got {}'.format(eop, len(args)) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
        
    def _sample_short_circuit(self, expr, expr1, expr2, op, subs):
        if expr2.is_constant_expression() or expr2.is_pvariable_expression():  # TODO: is this ok?
            expr1, expr2 = expr2, expr1
        
        arg1 = self._sample(expr1, subs)
        if not isinstance(arg1, bool):
            raise TypeError(
                'Logical operator {} requires boolean arg, got {}.'.format(op, arg1) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
            
        if op == '|' and arg1:
            return True
        elif op == '^' and not arg1:
            return False
        
        arg2 = self._sample(expr2, subs)
        if not isinstance(arg1, bool):
            raise TypeError(
                'Logical operator {} requires boolean arg, got {}.'.format(op, arg2) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
        return arg2
        
    def _sample_xor(self, expr, arg1, arg2, subs):
        arg1 = self._sample(arg1, subs)
        arg2 = self._sample(arg2, subs)
        if not (isinstance(arg1, bool) and isinstance(arg2, bool)):
            raise TypeError(
                'Xor requires boolean args, got {} and {}.'.format(arg1, arg2) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
        return arg1 != arg2
    
    # control
    def _sample_control(self, expr, eop, subs):
        args = expr.args
        if eop != 'if':
            raise NotImplementedError(
                'Control type {} is not supported.'.format(eop) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
        if len(args) != 3:
            raise ValueError(
                'If statement requires 3 args, got {}.'.format(len(args)) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
        
        cond = self._sample(args[0], subs)
        if not isinstance(cond, bool):
            raise TypeError(
                'If condition must evaluate to bool, got {}.'.format(cond) + 
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
            raise NotImplementedError(
                'Internal error: Discrete not yet supported...' + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
        elif name == 'Multinomial':  # TODO, implement Multinomial
            raise NotImplementedError(
                'Internal error: Multinomial not yet supported...' + 
                '\n' + RDDLSimulator._print_stack_trace(expr))            
        elif name == 'Dirichlet':  # TODO, implement Dirichlet
            raise NotImplementedError(
                'Internal error: Dirichlet not yet supported...' + 
                '\n' + RDDLSimulator._print_stack_trace(expr))     
        else:
            raise NotImplementedError(
                'Distribution {} is not supported.'.format(name) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))

    def _sample_kron_delta(self, expr, subs):
        args = expr.args
        if len(args) != 1:
            raise ValueError(
                'KronDelta requires 1 parameter, got {}.'.format(len(args)) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
        
        arg = self._sample(args[0], subs)
        if not isinstance(arg, (int, bool)):
            raise TypeError(
                'KronDelta requires int or boolean parameter, got {}.'.format(arg) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
        return arg
    
    def _sample_dirac_delta(self, expr, subs):
        args = expr.args
        if len(args) != 1:
            raise ValueError(
                'DiracDelta requires 1 parameter, got {}.'.format(len(args)) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
            
        arg = self._sample(args[0], subs)
        if not isinstance(arg, float):
            raise TypeError(
                'DiracDelta requires float parameter, got {}.'.format(arg) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
        return arg
    
    def _sample_uniform(self, expr, subs):
        args = expr.args
        if len(args) != 2:
            raise ValueError(
                'Uniform requires 2 parameters, got {}.'.format(len(args)) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
            
        lb = float(self._sample(args[0], subs))
        ub = float(self._sample(args[1], subs))
        if not (lb <= ub):
            raise ValueError(
                'Uniform bounds do not satisfy {} <= {}.'.format(lb, ub) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
        
        return self._rng.uniform(lb, ub)      
    
    def _sample_bernoulli(self, expr, subs):
        args = expr.args
        if len(args) != 1:
            raise ValueError(
                'Bernoulli requires 1 parameter, got {}.'.format(len(args)) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
            
        p = float(self._sample(args[0], subs))
        if not (0 <= p <= 1):
            raise ValueError(
                'Bernoulli parameter p should be in [0, 1], got {}.'.format(p) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
        
        return self._rng.uniform() <= p
    
    def _sample_normal(self, expr, subs):
        args = expr.args
        if len(args) != 2:
            raise ValueError(
                'Normal requires 2 parameters, got {}.'.format(len(args)) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
            
        mean = float(self._sample(args[0], subs))
        var = float(self._sample(args[1], subs))
        if not (var >= 0.):
            raise ValueError(
                'Normal variance {} is not positive.'.format(var) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
        stdev = math.sqrt(var)
        
        return self._rng.normal(loc=mean, scale=stdev)
    
    def _sample_poisson(self, expr, subs):
        args = expr.args
        if len(args) != 1:
            raise ValueError(
                'Poisson requires 1 parameter, got {}.'.format(len(args)) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
            
        rate = float(self._sample(args[0], subs))
        if not (rate >= 0.):
            raise ValueError(
                'Poisson rate {} is not positive.'.format(rate) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
        
        return self._rng.poisson(lam=rate)
    
    def _sample_exponential(self, expr, subs):
        args = expr.args
        if len(args) != 1:
            raise ValueError(
                'Exponential requires 1 parameter, got {}.'.format(len(args)) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
            
        scale = float(self._sample(args[0], subs))
        if not (scale > 0.):
            raise ValueError(
                'Exponential rate {} is not positive.'.format(scale) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
        
        return self._rng.exponential(scale=scale)
    
    def _sample_weibull(self, expr, subs):
        args = expr.args
        if len(args) != 2:
            raise ValueError(
                'Weibull requires 2 parameters, got {}.'.format(len(args)) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
            
        shape = float(self._sample(args[0], subs))
        if not (shape > 0.):
            raise ValueError(
                'Weibull shape {} is not positive.'.format(shape) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
        
        scale = float(self._sample(args[1], subs))
        if not (scale > 0.):
            raise ValueError(
                'Weibull scale {} is not positive.'.format(scale) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
        
        return scale * self._rng.weibull(shape)
    
    def _sample_gamma(self, expr, subs):
        args = expr.args
        if len(args) != 2:
            raise ValueError(
                'Gamma requires 2 parameters, got {}.'.format(len(args)) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
            
        shape = float(self._sample(args[0], subs))
        if not (shape > 0.):
            raise ValueError(
                'Gamma shape {} is not positive.'.format(shape) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
        
        scale = float(self._sample(args[1], subs))
        if not (scale > 0.):
            raise ValueError(
                'Gamma scale {} is not positive.'.format(scale) + 
                '\n' + RDDLSimulator._print_stack_trace(expr))
        
        return self._rng.gamma(shape=shape, scale=scale)
    
class RDDLSimulatorWConstraints(RDDLSimulator):
    def __init__(self, model: PlanningModel, rng=np.random.default_rng(), max_bound=1000):
        super().__init__(model, rng)
        self.epsilon = 0.001
        self.BigM = float(max_bound)
        self._bounds = {}
        for state in model.states:
            self._bounds[state] = [-self.BigM, self.BigM]
        for derived in model.derived:
            self._bounds[derived] = [-self.BigM, self.BigM]
        for interm in model.interm:
            self._bounds[interm] = [-self.BigM, self.BigM]
        for action in model.actions:
            self._bounds[action] = [-self.BigM, self.BigM]

        # actions and states bounds extraction
        # currently supports only linear in\equality constraints
        for action_precond in model.preconditions:
            if action_precond.etype[0] != 'rational':
                pass
            var, lim, loc = self.get_bounds(action_precond.args[0], action_precond.args[1], action_precond.etype[1])
            if var is not None and loc is not None:
                self._bounds[var][loc] = lim

        for state_inv in model.invariants:
            if state_inv.etype[0] != 'rational':
                pass
            var, lim, loc = self.get_bounds(state_inv.args[0], state_inv.args[1], state_inv.etype[1], False)
            if var is not None and loc is not None:
                self._bounds[var][loc] = lim

    # def check_state_invariants(self) -> None:
    #     '''Throws an exception if the state invariants are not satisfied.'''
    #     # TODO: fix this method
    #     subs = self._update_subs()
    #     for idx, invariant in enumerate(self._model.invariants):
    #         sample = self._sample(invariant, subs)
    #         if not isinstance(sample, bool):
    #             raise TypeError(
    #                 'State invariant must evaluate to bool, got {}.'.format(sample) +
    #                 '\n' + RDDLSimulator._print_stack_trace(invariant))
    #         if not sample:
    #             raise RDDLRuntimeError(
    #                 'State invariant {} is not satisfied.'.format(idx + 1) +
    #                 '\n' + RDDLSimulator._print_stack_trace(invariant))

    def get_bounds(self, left_arg, right_arg, op, is_action=True):
        variable = None
        lim = 0
        loc = None
        if is_action:
            search_dict = self._model.actions
        else:
            search_dict = self._model.states

        if left_arg.etype[0] == 'pvar':
            name = left_arg.args[0]
            if name in search_dict:
                variable = name
                if op in ['<=', '<']:
                    loc = 1
                    if op == '<':
                        lim = -self.epsilon
                elif op in ['>=', '>']:
                    loc = 0
                    if op == '>':
                        lim = self.epsilon
            elif name in self._model.nonfluents:
                lim = self._model.nonfluents[name]
        elif left_arg.etype[0] == 'constant':
            lim += left_arg.args
        else:
            return variable, float(lim), loc

        if right_arg.etype[0] == 'pvar':
            name = right_arg.args[0]
            if name in search_dict:
                variable = name
                if op in ['<=', '<']:
                    loc = 0
                    if op == '<':
                        lim = self.epsilon
                elif op in ['>=', '>']:
                    loc = 1
                    if op == '>':
                        lim = -self.epsilon
            elif name in self._model.nonfluents:
                lim = self._model.nonfluents[name]
        elif right_arg.etype[0] == 'constant':
            lim += right_arg.args
        else:
            return variable, float(lim), loc

        return variable, float(lim), loc

    @property
    def bounds(self):
        return self._bounds

    @bounds.setter
    def bounds(self, value):
        self._bounds = value
