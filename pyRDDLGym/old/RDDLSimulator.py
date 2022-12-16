# import math
# import numpy as np
# from typing import Dict, Tuple
#
# from pyRDDLGym.Core.Debug.decompiler import RDDLDecompiler
# from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLActionPreconditionNotSatisfiedError
# from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidExpressionError
# from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidNumberOfArgumentsError
# from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLNotImplementedError
# from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLStateInvariantNotSatisfiedError
# from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLTypeError
# from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLUndefinedCPFError
# from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLUndefinedVariableError
# from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLValueOutOfRangeError
# from pyRDDLGym.Core.Grounder.RDDLModel import PlanningModel
# from pyRDDLGym.Core.Parser.expr import Expression, Value
# from pyRDDLGym.Core.Simulator.RDDLDependencyAnalysis import RDDLDependencyAnalysis
#
# Args = Dict[str, Value]
#
# VALID_ARITHMETIC_OPS = {'+', '-', '*', '/'}
# VALID_RELATIONAL_OPS = {'>=', '>', '<=', '<', '==', '~='}
# VALID_LOGICAL_OPS = {'|', '^', '~', '=>', '<=>'}
# VALID_AGGREGATE_OPS = {'minimum', 'maximum'}
# VALID_CONTROL_OPS = {'if'}
#
#
# class RDDLSimulator:
#
#     def __init__(self,
#                  model: PlanningModel,
#                  rng: np.random.Generator=np.random.default_rng()) -> None:
#         '''Creates a new simulator for the given RDDL model.
#
#         :param model: the RDDL model
#         :param rng: the random number generator
#         '''
#         self._model = model        
#         self._rng = rng
#
#         # perform a dependency analysis and topological sort to compute levels
#         dep_analysis = RDDLDependencyAnalysis(self._model)
#         self.cpforder = dep_analysis.compute_levels()
#         self._order_cpfs = list(sorted(self.cpforder.keys())) 
#
#         self._action_fluents = set(self._model.actions.keys())
#         self._init_actions = self._model.actions.copy()
#
#         # non-fluent will never change
#         self._subs = self._model.nonfluents.copy()
#
#         # is a POMDP
#         self._pomdp = bool(self._model.observ)
#
#     # error checks
#     @staticmethod
#     def _print_stack_trace(expr):
#         if isinstance(expr, Expression):
#             trace = RDDLDecompiler().decompile_expr(expr)
#         else:
#             trace = str(expr)
#         return '>> ' + trace
#
#     @staticmethod
#     def _check_type(value, valid_types, what, expr):
#         if not isinstance(value, valid_types):
#             raise RDDLTypeError(
#                 '{} must evaluate to {}, got {}.'.format(
#                     what, valid_types, value) + 
#                 '\n' + RDDLSimulator._print_stack_trace(expr))
#
#     @staticmethod
#     def _check_types(value1, value2, valid_types, what, expr):
#         if not (isinstance(value1, valid_types) and 
#                 isinstance(value2, valid_types)):
#             raise RDDLTypeError(
#                 '{} must evaluate to {}, got {} and {}.'.format(
#                     what, valid_types, value1, value2) + 
#                 '\n' + RDDLSimulator._print_stack_trace(expr))
#
#     @staticmethod
#     def _check_op(op, valid_ops, what, expr):
#         if op not in valid_ops:
#             raise RDDLNotImplementedError(
#                 '{} operator {} is not supported: must be one of {}.'.format(
#                     what, op, valid_ops) + 
#                 '\n' + RDDLSimulator._print_stack_trace(expr))
#
#     @staticmethod
#     def _check_arity(args, num_required, what, expr):
#         num_actual = len(args)
#         if num_actual != num_required:
#             raise RDDLInvalidNumberOfArgumentsError(
#                 '{} requires {} arguments, got {}.'.format(
#                     what, num_required, num_actual) + 
#                 '\n' + RDDLSimulator._print_stack_trace(expr))
#
#     @staticmethod
#     def _check_positive(value, strict, what, expr):
#         if strict:
#             failed = not (value > 0)
#             message = '{} must be strictly positive, got {}.'
#         else:
#             failed = not (value >= 0)
#             message = '{} must be non-negative, got {}.'
#         if failed:
#             raise RDDLValueOutOfRangeError(
#                 message.format(what, value) + 
#                 '\n' + RDDLSimulator._print_stack_trace(expr))
#
#     @staticmethod
#     def _check_bounds(lb, ub, what, expr):
#         if not (lb <= ub):
#             raise RDDLValueOutOfRangeError(
#                 'Bounds of {} are invalid:'.format(what) + 
#                 'max value {} must be >= min value {}.'.format(ub, lb) + 
#                 '\n' + RDDLSimulator._print_stack_trace(expr))
#
#     @staticmethod
#     def _check_range(value, lb, ub, what, expr):
#         if not (lb <= value <= ub):
#             raise RDDLValueOutOfRangeError(
#                 '{} must be in the range [{}, {}], got {}.'.format(
#                     what, lb, ub, value) + 
#                 '\n' + RDDLSimulator._print_stack_trace(expr))
#
#     @staticmethod
#     def _raise_unsupported(what, expr):
#         raise RDDLNotImplementedError(
#             '{} is not supported in the current RDDL version.'.format(what) + 
#             '\n' + RDDLSimulator._print_stack_trace(expr))
#
#     @staticmethod
#     def _raise_no_args(op, what, expr):
#         raise RDDLInvalidNumberOfArgumentsError(
#             '{} operator {} has no arguments.'.format(what, op) + 
#             '\n' + RDDLSimulator._print_stack_trace(expr))
#
#     # sampling for RDDL components
#     def check_state_invariants(self) -> None:
#         '''Throws an exception if the state invariants are not satisfied.'''
#         for i, invariant in enumerate(self._model.invariants):
#             sample = self._sample(invariant, self._subs)
#             RDDLSimulator._check_type(
#                 sample, bool, 'Invariant ' + str(i + 1), invariant)
#
#             if not sample:
#                 raise RDDLStateInvariantNotSatisfiedError(
#                     'Invariant {} is not satisfied.'.format(i + 1) + 
#                     '\n' + RDDLSimulator._print_stack_trace(invariant))
#
#     def check_action_preconditions(self, actions: Args) -> None:
#         '''Throws an exception if the action preconditions are not satisfied.'''
#
#         # if actions are missing use their default values
#         self._model.actions = {var: actions.get(var, default) 
#                                for var, default in self._init_actions.items()}
#         subs = self._subs
#         subs.update(self._model.actions)   
#
#         for i, precond in enumerate(self._model.preconditions):
#             sample = self._sample(precond, subs)
#             RDDLSimulator._check_type(
#                 sample, bool, 'Precondition ' + str(i + 1), precond)
#
#             if not sample:
#                 raise RDDLActionPreconditionNotSatisfiedError(
#                     'Precondition {} is not satisfied.'.format(i + 1) + 
#                     '\n' + RDDLSimulator._print_stack_trace(precond))
#
#     def check_terminal_states(self) -> bool:
#         '''return True if a terminal state has been reached.'''
#         for i, terminal in enumerate(self._model.terminals):
#             sample = self._sample(terminal, self._subs)
#             RDDLSimulator._check_type(
#                 sample, bool, 'Termination ' + str(i + 1), terminal)
#
#             if sample:
#                 return True
#         return False
#
#     def sample_reward(self) -> float:
#         '''Samples the current reward given the current state and action.'''
#         reward = self._sample(self._model.reward, self._subs)
#         return float(reward)
#
#     def reset(self) -> Args:
#         '''Resets the state variables to their initial values.'''
#
#         # states and actions are set to their initial values
#         self._model.states.update(self._model.init_state)
#         self._model.actions.update(self._init_actions)
#
#         # remaining fluents do not have default or initial values
#         for var in self._model.derived.keys():
#             self._model.derived[var] = None
#         for var in self._model.interm.keys():
#             self._model.interm[var] = None
#         for var in self._model.observ.keys():
#             self._model.observ[var] = None
#
#         # prepare initial fluent sub table
#         self._subs.update(self._model.states)
#         self._subs.update(self._model.actions)  
#         self._subs.update(self._model.derived)    
#         self._subs.update(self._model.interm)    
#         self._subs.update(self._model.observ)
#         for var in self._model.prev_state.keys():
#             self._subs[var] = None
#
#         # check termination condition for the initial state
#         done = self.check_terminal_states()
#
#         if self._pomdp:
#             return self._model.observ.copy(), done
#         else:
#             return self._model.states.copy(), done
#
#     def step(self, actions: Args) -> Args:
#         '''Samples and returns the next state from the cpfs.
#
#         :param actions: a dict mapping current action fluents to their values
#         '''
#
#         # if actions are missing use their default values
#         self._model.actions = {var: actions.get(var, default) 
#                                for var, default in self._init_actions.items()}
#         subs = self._subs 
#         subs.update(self._model.actions)   
#
#         # evaluate all CPFs, record next state fluents, update sub table 
#         next_states, next_obs = {}, {}
#         for order in self._order_cpfs:
#             for cpf in self.cpforder[order]: 
#                 if cpf in self._model.next_state:
#                     primed_cpf = self._model.next_state[cpf]
#                     expr = self._model.cpfs[primed_cpf]
#                     sample = self._sample(expr, subs)
#                     if primed_cpf not in self._model.prev_state:
#                         raise KeyError(
#                             'Internal error: variable <{}> not in prev_state.'.format(
#                                 primed_cpf))
#                     next_states[cpf] = sample   
#                     subs[primed_cpf] = sample   
#                 elif cpf in self._model.derived:
#                     expr = self._model.cpfs[cpf]
#                     sample = self._sample(expr, subs)
#                     self._model.derived[cpf] = sample
#                     subs[cpf] = sample   
#                 elif cpf in self._model.interm:
#                     expr = self._model.cpfs[cpf]
#                     sample = self._sample(expr, subs)
#                     self._model.interm[cpf] = sample
#                     subs[cpf] = sample
#                 elif cpf in self._model.observ:
#                     expr = self._model.cpfs[cpf]
#                     sample = self._sample(expr, subs)
#                     self._model.observ[cpf] = sample
#                     subs[cpf] = sample
#                     next_obs[cpf] = sample
#                 else:
#                     raise RDDLUndefinedCPFError('CPF <{}> is not defined.'.format(cpf))     
#
#         # evaluate the immediate reward
#         reward = self.sample_reward()
#
#         # update the internal model state and the sub table
#         self._model.states = {}
#         for cpf, value in next_states.items():
#             self._model.states[cpf] = value
#             subs[cpf] = value
#             primed_cpf = self._model.next_state[cpf]
#             subs[primed_cpf] = None
#
#         # check the termination condition
#         done = self.check_terminal_states()
#
#         if self._pomdp:
#             return next_obs, reward, done
#         else:
#             return next_states, reward, done
#
#     # start of sampling subroutines
#     def _sample(self, expr, subs):
#         if isinstance(expr, Expression):
#             etype, op = expr.etype
#             if etype == 'constant':
#                 return self._sample_constant(expr, subs)
#             elif etype == 'pvar':
#                 return self._sample_pvar(expr, op, subs)
#             elif etype == 'relational':
#                 return self._sample_relational(expr, op, subs)
#             elif etype == 'arithmetic':
#                 return self._sample_arithmetic(expr, op, subs)
#             elif etype == 'aggregation':
#                 return self._sample_aggregation(expr, op, subs)
#             elif etype == 'func':
#                 return self._sample_func(expr, op, subs)
#             elif etype == 'boolean':
#                 return self._sample_logical(expr, op, subs)
#             elif etype == 'control':
#                 return self._sample_control(expr, op, subs)
#             elif etype == 'randomvar':
#                 return self._sample_random(expr, op, subs)
#             else:
#                 raise RDDLNotImplementedError(
#                     'Internal error: expr {} is not recognized.'.format(expr))
#         else:
#             raise RDDLNotImplementedError(
#                 'Internal error: object {} is not recognized.'.format(expr))
#
#     # simple expressions
#     def _sample_constant(self, expr, subs):
#         return expr.args
#
#     def _sample_pvar(self, expr, name, subs):
#         args = expr.args
#         RDDLSimulator._check_arity(args, 2, 'Internal error: pvar ' + name, expr)
#
#         var, _ = args
#         if var not in subs:
#             raise RDDLUndefinedVariableError(
#                 'Variable <{}> is not defined in the instance.'.format(var) + 
#                 '\n' + RDDLSimulator._print_stack_trace(expr))
#
#         val = subs[var]
#         if val is None:
#             raise RDDLUndefinedVariableError(
#                 'Variable <{}> is referenced before assignment.'.format(var) + 
#                 '\n' + RDDLSimulator._print_stack_trace(expr))
#
#         return val
#
#     def _sample_relational(self, expr, op, subs):
#         RDDLSimulator._check_op(op, VALID_RELATIONAL_OPS, 'Relational', expr)
#
#         args = expr.args
#         RDDLSimulator._check_arity(args, 2, 'Relational operator ' + op, expr)
#
#         arg1 = self._sample(args[0], subs)
#         arg2 = self._sample(args[1], subs)
#         arg1 = 1 * arg1  # bool -> int
#         arg2 = 1 * arg2
#
#         if op == '>=':
#             return arg1 >= arg2
#         elif op == '<=':
#             return arg1 <= arg2
#         elif op == '<':
#             return arg1 < arg2
#         elif op == '>':
#             return arg1 > arg2
#         elif op == '==':
#             return arg1 == arg2
#         else:  # '!='
#             return arg1 != arg2
#
#     def _sample_arithmetic(self, expr, op, subs):
#         RDDLSimulator._check_op(op, VALID_ARITHMETIC_OPS, 'Arithmetic', expr)
#
#         args = expr.args        
#
#         if len(args) == 1 and op == '-':
#             arg = self._sample(args[0], subs)
#             arg = -1 * arg  # bool -> int  
#             return arg
#
#         elif len(args) >= 1:
#             if op == '*':
#                 return self._sample_short_circuit_product(expr, args, subs)
#             elif op == '+':
#                 terms = (1 * self._sample(arg, subs) for arg in args)  # bool -> int
#                 return sum(terms)
#             elif len(args) == 2:
#                 arg1 = self._sample(args[0], subs)
#                 arg2 = self._sample(args[1], subs)
#                 arg1 = 1 * arg1  # bool -> int
#                 arg2 = 1 * arg2
#                 if op == '-':
#                     return arg1 - arg2
#                 elif op == '/':
#                     return RDDLSimulator._safe_division(expr, arg1, arg2)
#
#         RDDLSimulator._raise_no_args(op, 'Arithmetic', expr)
#
#     @staticmethod
#     def _safe_division(expr, arg1, arg2):
#         if arg1 != arg1 or arg2 != arg2:
#             raise ArithmeticError(
#                 'Invalid values in quotient, got {} and {}.'.format(arg1, arg2) + 
#                 '\n' + RDDLSimulator._print_stack_trace(expr))
#         elif arg2 == 0:
#             raise ArithmeticError(
#                 'Division by zero.' + 
#                 '\n' + RDDLSimulator._print_stack_trace(expr))
#
#         return arg1 / arg2  # int -> float
#
#     def _sample_short_circuit_product(self, expr, args, subs):
#         product = 1
#
#         # evaluate variables and constants first (these are fast)
#         for arg in args:
#             if arg.is_constant_expression() or arg.is_pvariable_expression():
#                 term = self._sample(arg, subs)
#                 product *= term  # bool -> int
#                 if product == 0:
#                     return product
#
#         # evaluate nested expressions (can't optimize the order in general)
#         for arg in args:
#             if not (arg.is_constant_expression() or arg.is_pvariable_expression()):
#                 term = self._sample(arg, subs)
#                 product *= term  # bool -> int
#                 if product == 0:
#                     return product
#
#         return product
#
#     # aggregations
#     def _sample_aggregation(self, expr, op, subs):
#         RDDLSimulator._check_op(op, VALID_AGGREGATE_OPS, 'Aggregation', expr)
#
#         args = expr.args
#         if not args:
#             RDDLSimulator._raise_no_args(op, 'Aggregation', expr)
#
#         terms = (1 * self._sample(arg, subs) for arg in args)  # bool -> int
#         if op == 'minimum':
#             return min(terms)
#         else:  # 'maximum'
#             return max(terms)
#
#     # functions
#     KNOWN_UNARY = {        
#         'abs': abs,
#         'sgn': lambda x: (-1 if x < 0 else (1 if x > 0 else 0)),
#         'round': round,
#         'floor': math.floor,
#         'ceil': math.ceil,
#         'cos': math.cos,
#         'sin': math.sin,
#         'tan': math.tan,
#         'acos': math.acos,
#         'asin': math.asin,
#         'atan': math.atan,
#         'cosh': math.cosh,
#         'sinh': math.sinh,
#         'tanh': math.tanh,
#         'exp': math.exp,
#         'ln': math.log,
#         'sqrt': math.sqrt
#     }
#
#     KNOWN_BINARY = {
#         'div': lambda x, y: int(x) // int(y),
#         'mod': lambda x, y: int(x) % int(y),
#         'min': min,
#         'max': max,
#         'pow': math.pow,
#         'log': math.log
#     }
#
#     def _sample_func(self, expr, name, subs):
#         args = expr.args
#         if isinstance(args, Expression):
#             args = (args,)
#
#         if name in RDDLSimulator.KNOWN_UNARY:
#             RDDLSimulator._check_arity(args, 1, 'Unary function ' + name, expr)
#
#             arg = self._sample(args[0], subs)
#             arg = 1 * arg  # bool -> int
#
#             try:
#                 return RDDLSimulator.KNOWN_UNARY[name](arg)
#             except:
#                 raise RDDLValueOutOfRangeError(
#                     'Unary function {} could not be evaluated with {}.'.format(
#                         name, arg) + 
#                     '\n' + RDDLSimulator._print_stack_trace(expr))
#
#         elif name in RDDLSimulator.KNOWN_BINARY:
#             RDDLSimulator._check_arity(args, 2, 'Binary function ' + name, expr)
#
#             arg1 = self._sample(args[0], subs)
#             arg2 = self._sample(args[1], subs)
#             arg1 = 1 * arg1  # bool -> int
#             arg2 = 1 * arg2
#
#             try:
#                 return RDDLSimulator.KNOWN_BINARY[name](arg1, arg2)
#             except:
#                 raise RDDLValueOutOfRangeError(
#                     'Binary function {} could not be evaluated with {} and {}.'.format(
#                         name, arg1, arg2) + 
#                     '\n' + RDDLSimulator._print_stack_trace(expr))
#
#         RDDLSimulator._raise_unsupported('Function ' + name, expr)
#
#     # logical expressions
#     def _sample_logical(self, expr, op, subs):
#         RDDLSimulator._check_op(op, VALID_LOGICAL_OPS, 'Logical', expr)
#
#         args = expr.args
#
#         if len(args) == 1 and op == '~':
#             arg = self._sample(args[0], subs)
#             RDDLSimulator._check_type(
#                 arg, bool, 'Argument of logical operator ' + op, expr)
#             return not arg
#
#         elif len(args) >= 1:
#             if op == '|' or op == '^':
#                 return self._sample_short_circuit_and_or(expr, args, op, subs)
#             elif len(args) == 2:
#                 if op == '~':
#                     return self._sample_xor(expr, *args, subs)
#                 elif op == '=>':
#                     return self._sample_short_circuit_implies(expr, *args, subs)
#                 elif op == '<=>':
#                     return self._sample_equivalent(expr, *args, subs)
#
#         RDDLSimulator._raise_no_args(op, 'Logical', expr)
#
#     def _sample_short_circuit_and_or(self, expr, args, op, subs):
#
#         # evaluate variables and constants first (these are fast)
#         for arg in args:
#             if arg.is_constant_expression() or arg.is_pvariable_expression():
#                 term = self._sample(arg, subs)
#                 RDDLSimulator._check_type(
#                     term, bool, 'Argument of logical operator ' + op, expr)
#                 if op == '|' and term:
#                     return True
#                 elif op == '^' and not term:
#                     return False
#
#         # evaluate nested expressions (can't optimize the order in general)
#         for arg in args:
#             if not (arg.is_constant_expression() or arg.is_pvariable_expression()):
#                 term = self._sample(arg, subs)
#                 RDDLSimulator._check_type(
#                     term, bool, 'Argument of logical operator ' + op, expr)
#                 if op == '|' and term:
#                     return True
#                 elif op == '^' and not term:
#                     return False
#
#         return op != '|'
#
#     def _sample_xor(self, expr, arg1, arg2, subs):
#         arg1 = self._sample(arg1, subs)
#         arg2 = self._sample(arg2, subs)
#         RDDLSimulator._check_types(
#             arg1, arg2, bool, 'Arguments of logical operator ~', expr)
#
#         return arg1 != arg2
#
#     def _sample_short_circuit_implies(self, expr, arg1, arg2, subs):
#         arg1 = self._sample(arg1, subs)
#         RDDLSimulator._check_type(
#             arg1, bool, 'Argument of logical operator =>', expr)
#
#         if not arg1:
#             return True
#
#         arg2 = self._sample(arg2, subs)
#         RDDLSimulator._check_type(
#             arg2, bool, 'Argument of logical operator =>', expr)
#
#         return arg2
#
#     def _sample_equivalent(self, expr, arg1, arg2, subs):
#         arg1 = self._sample(arg1, subs)
#         arg2 = self._sample(arg2, subs)
#         RDDLSimulator._check_types(
#             arg1, arg2, bool, 'Arguments of logical operator <=>', expr)
#
#         return arg1 == arg2
#
#     # control
#     def _sample_control(self, expr, op, subs):
#         RDDLSimulator._check_op(op, VALID_CONTROL_OPS, 'Control', expr)
#
#         args = expr.args
#         RDDLSimulator._check_arity(args, 3, 'If then else', expr)
#
#         condition = self._sample(args[0], subs)
#         RDDLSimulator._check_type(condition, bool, 'If condition', expr)
#
#         branch = args[1 if condition else 2]
#         return self._sample(branch, subs)
#
#     # random variables
#     def _sample_random(self, expr, name, subs):
#         if name == 'KronDelta':
#             return self._sample_kron_delta(expr, subs)        
#         elif name == 'DiracDelta':
#             return self._sample_dirac_delta(expr, subs)
#         elif name == 'Uniform':
#             return self._sample_uniform(expr, subs)
#         elif name == 'Bernoulli':
#             return self._sample_bernoulli(expr, subs)
#         elif name == 'Normal':
#             return self._sample_normal(expr, subs)
#         elif name == 'Poisson':
#             return self._sample_poisson(expr, subs)
#         elif name == 'Exponential':
#             return self._sample_exponential(expr, subs)
#         elif name == 'Weibull':
#             return self._sample_weibull(expr, subs)        
#         elif name == 'Gamma':
#             return self._sample_gamma(expr, subs)
#         elif name == 'Binomial':
#             return self._sample_binomial(expr, subs)
#         elif name == 'NegativeBinomial':
#             return self._sample_negative_binomial(expr, subs)
#         elif name == 'Beta':
#             return self._sample_beta(expr, subs)
#         elif name == 'Geometric':
#             return self._sample_geometric(expr, subs)
#         elif name == 'Pareto':
#             return self._sample_pareto(expr, subs)
#         elif name == 'Student':
#             return self._sample_student(expr, subs)
#         elif name == 'Gumbel':
#             return self._sample_gumbel(expr, subs)
#         else:  # no support for enum
#             RDDLSimulator._raise_unsupported('Distribution ' + name, expr)
#
#     def _sample_kron_delta(self, expr, subs):
#         args = expr.args
#         RDDLSimulator._check_arity(args, 1, 'KronDelta', expr)
#
#         arg = self._sample(args[0], subs)
#         RDDLSimulator._check_type(arg, (int, bool), 'Argument of KronDelta', expr)
#
#         return arg
#
#     def _sample_dirac_delta(self, expr, subs):
#         args = expr.args
#         RDDLSimulator._check_arity(args, 1, 'DiracDelta', expr)
#
#         arg = self._sample(args[0], subs)
#         RDDLSimulator._check_type(arg, float, 'Argument of DiracDelta', expr)
#
#         return arg
#
#     def _sample_uniform(self, expr, subs):
#         args = expr.args
#         RDDLSimulator._check_arity(args, 2, 'Uniform', expr)
#
#         lb = float(self._sample(args[0], subs))
#         ub = float(self._sample(args[1], subs))
#         RDDLSimulator._check_bounds(lb, ub, 'Uniform', expr)
#
#         return self._rng.uniform(lb, ub)      
#
#     def _sample_bernoulli(self, expr, subs):
#         args = expr.args
#         RDDLSimulator._check_arity(args, 1, 'Bernoulli', expr)
#
#         p = float(self._sample(args[0], subs))
#         RDDLSimulator._check_range(p, 0, 1, 'Bernoulli probability', expr)
#
#         return self._rng.uniform() <= p
#
#     def _sample_normal(self, expr, subs):
#         args = expr.args
#         RDDLSimulator._check_arity(args, 2, 'Normal', expr)
#
#         mean = float(self._sample(args[0], subs))
#         var = float(self._sample(args[1], subs))
#         RDDLSimulator._check_positive(var, False, 'Normal variance', expr)
#
#         scale = math.sqrt(var)
#         return self._rng.normal(loc=mean, scale=scale)
#
#     def _sample_poisson(self, expr, subs):
#         args = expr.args
#         RDDLSimulator._check_arity(args, 1, 'Poisson', expr)
#
#         rate = float(self._sample(args[0], subs))
#         RDDLSimulator._check_positive(rate, False, 'Poisson rate', expr)
#
#         return self._rng.poisson(lam=rate)
#
#     def _sample_exponential(self, expr, subs):
#         args = expr.args
#         RDDLSimulator._check_arity(args, 1, 'Exponential', expr)
#
#         scale = float(self._sample(args[0], subs))
#         RDDLSimulator._check_positive(scale, True, 'Exponential rate', expr)
#
#         return self._rng.exponential(scale=scale)
#
#     def _sample_weibull(self, expr, subs):
#         args = expr.args
#         RDDLSimulator._check_arity(args, 2, 'Weibull', expr)
#
#         shape = float(self._sample(args[0], subs))
#         RDDLSimulator._check_positive(shape, True, 'Weibull shape', expr)
#
#         scale = float(self._sample(args[1], subs))
#         RDDLSimulator._check_positive(scale, True, 'Weibull scale', expr)
#
#         return scale * self._rng.weibull(shape)
#
#     def _sample_gamma(self, expr, subs):
#         args = expr.args
#         RDDLSimulator._check_arity(args, 2, 'Gamma', expr)
#
#         shape = float(self._sample(args[0], subs))
#         RDDLSimulator._check_positive(shape, True, 'Gamma shape', expr)
#
#         scale = float(self._sample(args[1], subs))
#         RDDLSimulator._check_positive(scale, True, 'Gamma scale', expr)
#
#         return self._rng.gamma(shape=shape, scale=scale)
#
#     def _sample_binomial(self, expr, subs):
#         args = expr.args
#         RDDLSimulator._check_arity(args, 2, 'Binomial', expr)
#
#         count = self._sample(args[0], subs)
#         RDDLSimulator._check_type(count, (int, bool), 'Binomial count', expr)
#
#         count = int(count)
#         RDDLSimulator._check_positive(count, False, 'Binomial count', expr)
#
#         p = float(self._sample(args[1], subs))
#         RDDLSimulator._check_range(p, 0, 1, 'Binomial probability', expr)
#
#         return self._rng.binomial(n=count, p=p)
#
#     def _sample_negative_binomial(self, expr, subs):
#         args = expr.args
#         RDDLSimulator._check_arity(args, 2, 'NegativeBinomial', expr)
#
#         threshold = self._sample(args[0], subs)
#         RDDLSimulator._check_positive(threshold, True, 'NegativeBinomial r', expr)
#
#         p = float(self._sample(args[1], subs))
#         RDDLSimulator._check_range(p, 0, 1, 'NegativeBinomial probability', expr)
#
#         return self._rng.negative_binomial(n=threshold, p=p)
#
#     def _sample_beta(self, expr, subs):
#         args = expr.args
#         RDDLSimulator._check_arity(args, 2, 'Beta', expr)
#
#         shape = float(self._sample(args[0], subs))
#         RDDLSimulator._check_positive(shape, True, 'Beta shape', expr)
#
#         rate = float(self._sample(args[1], subs))
#         RDDLSimulator._check_positive(rate, True, 'Beta rate', expr)
#
#         return self._rng.beta(a=shape, b=rate)
#
#     def _sample_geometric(self, expr, subs):
#         args = expr.args
#         RDDLSimulator._check_arity(args, 1, 'Geometric', expr)
#
#         p = float(self._sample(args[0], subs))
#         RDDLSimulator._check_range(p, 0, 1, 'Geometric probability', expr)
#
#         return self._rng.geometric(p=p)
#
#     def _sample_pareto(self, expr, subs):
#         args = expr.args
#         RDDLSimulator._check_arity(args, 2, 'Pareto', expr)
#
#         shape = float(self._sample(args[0], subs))
#         RDDLSimulator._check_positive(shape, True, 'Pareto shape', expr)
#
#         scale = float(self._sample(args[1], subs))
#         RDDLSimulator._check_positive(scale, True, 'Pareto scale', expr)
#
#         return scale * self._rng.pareto(a=shape)
#
#     def _sample_student(self, expr, subs):
#         args = expr.args
#         RDDLSimulator._check_arity(args, 1, 'Student', expr)
#
#         degrees = float(self._sample(args[0], subs))
#         RDDLSimulator._check_positive(degrees, True, 'Student df', expr)
#
#         return self._rng.standard_t(df=degrees)
#
#     def _sample_gumbel(self, expr, subs):
#         args = expr.args
#         RDDLSimulator._check_arity(args, 2, 'Gumbel', expr)
#
#         mean = float(self._sample(args[0], subs))
#         scale = float(self._sample(args[1], subs))
#         RDDLSimulator._check_positive(scale, True, 'Gumbel scale', expr)
#
#         return self._rng.gumbel(loc=mean, scale=scale)
#
#
# class RDDLSimulatorWConstraints(RDDLSimulator):
#
#     def __init__(self,
#                  model: PlanningModel,
#                  rng: np.random.Generator=np.random.default_rng(),
#                  compute_levels: bool=True,
#                  max_bound: float=np.inf) -> None:
#         super().__init__(model, rng)  # , compute_levels)
#
#         self.epsilon = 0.001
#         # self.BigM = float(max_bound)
#         self.BigM = max_bound
#         self._bounds = {}
#         for state in model.states:
#             self._bounds[state] = [-self.BigM, self.BigM]
#         for derived in model.derived:
#             self._bounds[derived] = [-self.BigM, self.BigM]
#         for interm in model.interm:
#             self._bounds[interm] = [-self.BigM, self.BigM]
#         for action in model.actions:
#             self._bounds[action] = [-self.BigM, self.BigM]
#         for obs in model.observ:
#             self._bounds[obs] = [-self.BigM, self.BigM]
#
#         # actions and states bounds extraction for gym's action and state spaces repots only!
#         # currently supports only linear in\equality constraints
#         for action_precond in model.preconditions:
#             self._parse_bounds_rec(action_precond, self._model.actions)
#
#         for state_inv in model.invariants:
#             self._parse_bounds_rec(state_inv, self._model.states)
#
#         for name in self._bounds:
#             lb, ub = self._bounds[name]
#             RDDLSimulator._check_bounds(
#                 lb, ub, 'Variable <{}>'.format(name), self._bounds[name])
#
#     def _parse_bounds_rec(self, cond, search_dict):
#         if cond.etype[0] == "boolean" and cond.etype[1] == "^":
#             for arg in cond.args:
#                 self._parse_bounds_rec(arg, search_dict)
#         if cond.etype[0] == "relational":
#             var, lim, loc = self.get_bounds(
#                 cond.args[0], cond.args[1], cond.etype[1], search_dict)
#             if var is not None and loc is not None:
#                 if loc == 1:
#                     if self._bounds[var][loc] > lim:
#                         self._bounds[var][loc] = lim
#                 else:
#                     if self._bounds[var][loc] < lim:
#                         self._bounds[var][loc] = lim
#
#     def get_bounds(self, left_arg, right_arg, op,
#                    search_dict=None) -> Tuple[str, float, int]:
#         variable = None
#         lim = 0
#         loc = None
#         if search_dict is None:
#             return variable, float(lim), loc
#
#         # make sure at least one of the args is a pvar of the correct type
#         act_count = 0
#         if left_arg.etype[0] == 'pvar':
#             if left_arg.args[0] in search_dict:
#                 act_count = act_count + 1
#         if right_arg.etype[0] == 'pvar':
#             if right_arg.args[0] in search_dict:
#                 act_count = act_count + 1
#         if act_count == 2:
#             raise RDDLInvalidExpressionError(
#                 "Error in action-precondition block, " + 
#                 "constraint {} {} {} does not have a structure of " + 
#                 "action/state fluent <=/>= f(non-fluents, constants)".format(
#                 left_arg.args[0], right_arg.args[0]) + 
#                 '\n' + RDDLSimulator._print_stack_trace(
#                     left_arg.args[0] + op + right_arg.args[0]))
#         if act_count == 0:
#             return None, 0, None
#
#         if left_arg.etype[0] == 'pvar':
#             variable = left_arg.args[0]
#             if variable in search_dict:
#                 if self.verify_tree_is_box(right_arg):
#                     lim_temp = self._sample(right_arg, self._subs)
#                     lim, loc = self.get_op_code(op, is_right=True)
#                     return variable, float(lim + lim_temp), loc
#                 else:
#                     raise RDDLInvalidExpressionError(
#                         "Error in action-precondition block, bound {} must be a " + 
#                         "determinisic function of non-fluents and constants only".format(
#                             right_arg) + "\n" + RDDLSimulator._print_stack_trace(right_arg))
#
#         elif right_arg.etype[0] == 'pvar':
#             variable = right_arg.args[0]
#             if variable in search_dict:
#                 if self.verify_tree_is_box(left_arg):
#                     lim_temp = self._sample(left_arg, self._subs)
#                     lim, loc = self.get_op_code(op, is_right=False)
#                     return variable, float(lim + lim_temp), loc
#                 else:
#                     raise RDDLInvalidExpressionError(
#                         "Error in action-precondition block, bound {} must be a " + 
#                         "determinisic function of non-fluents and constants only".format(
#                             left_arg) + "\n" + RDDLSimulator._print_stack_trace(right_arg))
#         else:
#             raise RDDLInvalidExpressionError(
#                 "Error in action-precondition block, " + 
#                 "constraint {} {} {} does not have a structure of " + 
#                 "action/state fluent <=/>= f(non-fluents, constants)".format(
#                         left_arg.args[0], right_arg.args[0]) + 
#                 '\n' + RDDLSimulator._print_stack_trace(
#                     left_arg.args[0] + op + right_arg.args[0]))
#
#     def verify_tree_is_box(self, expr):
#         if hasattr(expr, 'args') == False:
#             return False
#         if expr.etype[0] == "randomvar":
#             return False
#         if expr.etype[0] == 'pvar':
#             if expr.args[0] in self._model.nonfluents:
#                 return True
#             return False
#         if expr.etype[0] == 'constant':
#             return True
#         else:
#             result = True
#             for elem in expr.args:
#                 if not self.verify_tree_is_box(elem):
#                     return False
#                 # result = result and self.verify_tree_is_box(elem)
#             return result
#
#     def get_op_code(self, op, is_right: bool) -> (float, int):
#         '''
#
#         Args:
#             op: inequality operator
#             is_right: True if the evaluated element is to the left of the inequality, False otherwise
#         Returns:
#             (lim , loc)
#             lim =  zero for soft inequality, minimum real number of strong inequality
#         '''
#         lim = 0
#         if is_right:
#             if op in ['<=', '<']:
#                 loc = 1
#                 if op == '<':
#                     lim = -self.epsilon
#             elif op in ['>=', '>']:
#                 loc = 0
#                 if op == '>':
#                     lim = self.epsilon
#         else:
#             if op in ['<=', '<']:
#                 loc = 0
#                 if op == '<':
#                     lim = self.epsilon
#             elif op in ['>=', '>']:
#                 loc = 1
#                 if op == '>':
#                     lim = -self.epsilon
#         return (lim, loc)
#
#     # def get_bounds(self, left_arg, right_arg, op, search_dict=None) -> Tuple[str, float, int]:
#     #     variable = None
#     #     lim = 0
#     #     loc = None
#     #     if search_dict is None:
#     #         return variable, float(lim), loc
#     #
#     #     if left_arg.etype[0] == 'pvar':
#     #         name = left_arg.args[0]
#     #         if name in search_dict:
#     #             variable = name
#     #             if op in ['<=', '<']:
#     #                 loc = 1
#     #                 if op == '<':
#     #                     lim = -self.epsilon
#     #             elif op in ['>=', '>']:
#     #                 loc = 0
#     #                 if op == '>':
#     #                     lim = self.epsilon
#     #         elif name in self._model.nonfluents:
#     #             lim = self._model.nonfluents[name]
#     #     elif left_arg.etype[0] == 'constant':
#     #         lim += left_arg.args
#     #     else:
#     #         return variable, float(lim), loc
#     #
#     #     if right_arg.etype[0] == 'pvar':
#     #         name = right_arg.args[0]
#     #         if name in search_dict:
#     #             variable = name
#     #             if op in ['<=', '<']:
#     #                 loc = 0
#     #                 if op == '<':
#     #                     lim = self.epsilon
#     #             elif op in ['>=', '>']:
#     #                 loc = 1
#     #                 if op == '>':
#     #                     lim = -self.epsilon
#     #         elif name in self._model.nonfluents:
#     #             lim = self._model.nonfluents[name]
#     #     elif right_arg.etype[0] == 'constant':
#     #         lim += right_arg.args
#     #     else:
#     #         return variable, float(lim), loc
#     #
#     #     return variable, float(lim), loc
#
#     @property
#     def bounds(self):
#         return self._bounds
#
#     @bounds.setter
#     def bounds(self, value):
#         self._bounds = value
#
#     @property
#     def states(self):
#         return self._model.states.copy()
#
#     @property
#     def isPOMDP(self):
#         return self._pomdp

