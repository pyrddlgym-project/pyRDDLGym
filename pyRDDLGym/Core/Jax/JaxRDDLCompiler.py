import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy as scipy 
from typing import Dict

from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidNumberOfArgumentsError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLNotImplementedError

from pyRDDLGym.Core.Compiler.RDDLDecompiler import RDDLDecompiler
from pyRDDLGym.Core.Compiler.RDDLLevelAnalysis import RDDLLevelAnalysis
from pyRDDLGym.Core.Compiler.RDDLLiftedModel import RDDLLiftedModel
from pyRDDLGym.Core.Compiler.RDDLObjectsTracer import RDDLObjectsTracer
from pyRDDLGym.Core.Compiler.RDDLValueInitializer import RDDLValueInitializer
from pyRDDLGym.Core.Debug.Logger import Logger
from pyRDDLGym.Core.Parser.expr import Expression


class JaxRDDLCompiler:
    '''Compiles a RDDL AST representation into an equivalent JAX representation.
    All operations are identical to their numpy equivalents.
    '''
    
    INT = jnp.int32
    REAL = jnp.float32
    
    JAX_TYPES = {
        'int': INT,
        'real': REAL,
        'bool': bool
    }
    
    def __init__(self, rddl: RDDLLiftedModel,
                 allow_synchronous_state: bool=True,
                 logger: Logger=None) -> None:
        '''Creates a new RDDL to Jax compiler.
        
        :param rddl: the RDDL model to compile into Jax
        :param allow_synchronous_state: whether next-state components can depend
        on each other
        :param logger: to log information about compilation to file
        '''
        self.rddl = rddl
        self.logger = logger
        jax.config.update('jax_log_compiles', True)
        
        # compile initial values
        if self.logger is not None:
            self.logger.clear()
        initializer = RDDLValueInitializer(rddl, logger=self.logger)
        self.init_values = initializer.initialize()
        
        # compute dependency graph for CPFs and sort them by evaluation order
        sorter = RDDLLevelAnalysis(rddl, allow_synchronous_state, logger=self.logger)
        self.levels = sorter.compute_levels()        
        
        # trace expressions to cache information to be used later
        tracer = RDDLObjectsTracer(rddl, logger=self.logger)
        self.traced = tracer.trace()
        
        # basic operations        
        self.ARITHMETIC_OPS = {
            '+': jnp.add,
            '-': jnp.subtract,
            '*': jnp.multiply,
            '/': jnp.divide
        }    
        self.RELATIONAL_OPS = {
            '>=': jnp.greater_equal,
            '<=': jnp.less_equal,
            '<': jnp.less,
            '>': jnp.greater,
            '==': jnp.equal,
            '~=': jnp.not_equal
        }
        self.LOGICAL_NOT = jnp.logical_not
        self.LOGICAL_OPS = {
            '^': jnp.logical_and,
            '&': jnp.logical_and,
            '|': jnp.logical_or,
            '~': jnp.logical_xor,
            '=>': lambda e1, e2: jnp.logical_or(jnp.logical_not(e1), e2),
            '<=>': jnp.equal
        }
        self.AGGREGATION_OPS = {
            'sum': jnp.sum,
            'avg': jnp.mean,
            'prod': jnp.prod,
            'minimum': jnp.min,
            'maximum': jnp.max,
            'forall': jnp.all,
            'exists': jnp.any,
            'argmin': jnp.argmin,
            'argmax': jnp.argmax
        }
        self.KNOWN_UNARY = {        
            'abs': jnp.abs,
            'sgn': jnp.sign,
            'round': jnp.round,
            'floor': jnp.floor,
            'ceil': jnp.ceil,
            'cos': jnp.cos,
            'sin': jnp.sin,
            'tan': jnp.tan,
            'acos': jnp.arccos,
            'asin': jnp.arcsin,
            'atan': jnp.arctan,
            'cosh': jnp.cosh,
            'sinh': jnp.sinh,
            'tanh': jnp.tanh,
            'exp': jnp.exp,
            'ln': jnp.log,
            'sqrt': jnp.sqrt,
            'lngamma': scipy.special.gammaln,
            'gamma': lambda x: jnp.exp(scipy.special.gammaln(x))
        }        
        self.KNOWN_BINARY = {
            'div': jnp.floor_divide,
            'mod': jnp.mod,
            'min': jnp.minimum,
            'max': jnp.maximum,
            'pow': jnp.power,
            'log': lambda x, y: jnp.log(x) / jnp.log(y)
        }        
        self.CONTROL_OPS = {
            'if': jnp.where,
            'switch': self._jax_switch_helper()
        }
        
    # ===========================================================================
    # main compilation subroutines
    # ===========================================================================
     
    def compile(self, log_jax_expr: bool=False) -> None: 
        '''Compiles the current RDDL into Jax expressions.
        
        :param log_jax_expr: whether to pretty-print the compiled Jax functions
        to the log file
        '''
        self.invariants = self._compile_constraints(self.rddl.invariants)
        self.preconditions = self._compile_constraints(self.rddl.preconditions)
        self.termination = self._compile_constraints(self.rddl.terminals)
        self.cpfs = self._compile_cpfs()
        self.reward = self._compile_reward()
        
        if log_jax_expr and self.logger is not None:
            printed = self.print_jax()
            printed_cpfs = '\n\n'.join(f'{k}: {v}' 
                                       for (k, v) in printed['cpfs'].items())
            printed_reward = printed['reward']
            printed_invariants = '\n\n'.join(v for v in printed['invariants'])
            printed_preconds = '\n\n'.join(v for v in printed['preconditions'])
            printed_terminals = '\n\n'.join(v for v in printed['terminations'])
            message = (f'compiled JAX CPFs:\n\n'
                       f'{printed_cpfs}\n\n'
                       f'compiled JAX reward:\n\n'
                       f'{printed_reward}\n\n'
                       f'compiled JAX invariants:\n\n'
                       f'{printed_invariants}\n\n'
                       f'compiled JAX preconditions:\n\n'
                       f'{printed_preconds}\n\n'
                       f'compiled JAX terminations:\n\n'
                       f'{printed_terminals}\n')
            self.logger.log(message)
    
    def _compile_constraints(self, constraints):
        return [self._jax(expr, dtype=bool) for expr in constraints]
        
    def _compile_cpfs(self):
        jax_cpfs = {}
        for cpfs in self.levels.values():
            for cpf in cpfs:
                _, expr = self.rddl.cpfs[cpf]
                prange = self.rddl.variable_ranges[cpf]
                dtype = JaxRDDLCompiler.JAX_TYPES.get(prange, JaxRDDLCompiler.INT)
                jax_cpfs[cpf] = self._jax(expr, dtype=dtype)
        return jax_cpfs
    
    def _compile_reward(self):
        return self._jax(self.rddl.reward, dtype=JaxRDDLCompiler.REAL)
    
    def compile_rollouts(self, policy, n_steps: int, n_batch: int,
                         check_constraints: bool=False):
        '''Compiles the current RDDL into a wrapped function that samples multiple
        rollouts (state trajectories) in batched form for the given policy. The
        wrapped function takes the policy parameters and RNG key as input, and
        returns a dictionary of all logged information from the rollouts.
        
        :param policy: a Jax compiled function that takes the subs dict, step
        number, policy parameters and an RNG key and returns an action and the
        next RNG key in the sequence
        :param n_steps: the length of each rollout
        :param n_batch: how many rollouts each batch performs
        :param check_constraints: whether state, action and termination 
        conditions should be checked on each time step: this info is stored in the
        returned log and does not raise an exception
        '''
        NORMAL = JaxRDDLCompiler.ERROR_CODES['NORMAL']
            
        # compute a batched version of the initial values
        init_subs = {}
        for (name, value) in self.init_values.items():
            new_shape = (n_batch,) + np.shape(value)
            init_subs[name] = np.broadcast_to(value, shape=new_shape)            
        for (state, next_state) in self.rddl.next_state.items():
            init_subs[next_state] = init_subs[state]
        
        # this performs a single step update
        def _step(carried, step):
            subs, params, key = carried
            action, key = policy(subs, step, params, key)
            subs.update(action)
            error = NORMAL
            
            # check preconditions
            if check_constraints:
                preconds = jnp.zeros(shape=(len(self.preconditions),), dtype=bool)
                for (i, precond) in enumerate(self.preconditions):
                    sample, key, precond_err = precond(subs, key)
                    preconds = preconds.at[i].set(sample)
                    error |= precond_err
            
            # calculate CPFs in topological order and reward
            for (name, cpf) in self.cpfs.items():
                subs[name], key, cpf_err = cpf(subs, key)
                error |= cpf_err                
            reward, key, reward_err = self.reward(subs, key)
            error |= reward_err
            
            for (state, next_state) in self.rddl.next_state.items():
                subs[state] = subs[next_state]
            
            # check the invariants in the new state
            if check_constraints:
                invariants = jnp.zeros(shape=(len(self.invariants),), dtype=bool)
                for (i, invariant) in enumerate(self.invariants):
                    sample, key, invariant_err = invariant(subs, key)
                    invariants = invariants.at[i].set(sample)
                    error |= invariant_err
            
            # check the termination (TODO: zero out reward in s if terminated)
            if check_constraints:
                terminated = False
                for terminal in self.termination:
                    sample, key, terminal_err = terminal(subs, key)
                    terminated = jnp.logical_or(terminated, sample)
                    error |= terminal_err
            
            logged = {'fluent': subs,
                      'action': action,
                      'reward': reward,
                      'error': error}            
            if check_constraints:
                logged['preconditions'] = preconds
                logged['invariants'] = invariants
                logged['terminated'] = terminated
                
            carried = (subs, params, key)
            return carried, logged
        
        # this performs a single roll-out starting from subs
        def _rollout(subs, params, key):
            steps = jnp.arange(n_steps)
            carry = (subs, params, key)
            (* _, key), logged = jax.lax.scan(_step, carry, steps)
            return logged, key
        
        # this performs batched roll-outs
        def _rollouts(params, key): 
            subkeys = jax.random.split(key, num=n_batch)
            logged, keys = jax.vmap(_rollout, in_axes=(0, None, 0))(
                init_subs, params, subkeys)
            logged['keys'] = subkeys
            return logged, keys
        
        return _rollouts
    
    # ===========================================================================
    # error checks
    # ===========================================================================
    
    def print_jax(self) -> Dict:
        '''Returns a dictionary containing the string representations of all 
        Jax compiled expressions from the RDDL file.
        '''
        subs = self.init_values
        key = jax.random.PRNGKey(42)
        printed = {}
        printed['cpfs'] = {name: str(jax.make_jaxpr(expr)(subs, key))
                           for (name, expr) in self.cpfs.items()}
        printed['reward'] = str(jax.make_jaxpr(self.reward)(subs, key))
        printed['invariants'] = [str(jax.make_jaxpr(expr)(subs, key))
                                 for expr in self.invariants]
        printed['preconditions'] = [str(jax.make_jaxpr(expr)(subs, key))
                                    for expr in self.preconditions]
        printed['terminations'] = [str(jax.make_jaxpr(expr)(subs, key))
                                   for expr in self.termination]
        return printed
        
    @staticmethod
    def _print_stack_trace(expr):
        if isinstance(expr, Expression):
            trace = RDDLDecompiler().decompile_expr(expr)
        else:
            trace = str(expr)
        return '>> ' + trace
    
    @staticmethod
    def _check_valid_op(expr, valid_ops):
        etype, op = expr.etype
        if op not in valid_ops:
            valid_op_str = ','.join(valid_ops.keys())
            raise RDDLNotImplementedError(
                f'{etype} operator {op} is not supported: '
                f'must be in {valid_op_str}.\n' + 
                JaxRDDLCompiler._print_stack_trace(expr))
    
    @staticmethod
    def _check_num_args(expr, required_args):
        actual_args = len(expr.args)
        if actual_args != required_args:
            etype, op = expr.etype
            raise RDDLInvalidNumberOfArgumentsError(
                f'{etype} operator {op} requires {required_args} arguments, '
                f'got {actual_args}.\n' + 
                JaxRDDLCompiler._print_stack_trace(expr))
        
    ERROR_CODES = {
        'NORMAL': 0,
        'INVALID_CAST': 1,
        'INVALID_PARAM_UNIFORM': 2,
        'INVALID_PARAM_NORMAL': 4,
        'INVALID_PARAM_EXPONENTIAL': 8,
        'INVALID_PARAM_WEIBULL': 16,
        'INVALID_PARAM_BERNOULLI': 32,
        'INVALID_PARAM_POISSON': 64,
        'INVALID_PARAM_GAMMA': 128,
        'INVALID_PARAM_BETA': 256,
        'INVALID_PARAM_GEOMETRIC': 512,
        'INVALID_PARAM_PARETO': 1024,
        'INVALID_PARAM_STUDENT': 2048,
        'INVALID_PARAM_GUMBEL': 4096,
        'INVALID_PARAM_LAPLACE': 8192,
        'INVALID_PARAM_CAUCHY': 16384,
        'INVALID_PARAM_GOMPERTZ': 32768,
        'INVALID_PARAM_CHISQUARE': 65536,
        'INVALID_PARAM_DISCRETE': 131072,
        'INVALID_PARAM_KRON_DELTA': 262144,
        'INVALID_PARAM_DIRICHLET': 524288,
        'INVALID_PARAM_MULTIVARIATE_STUDENT': 1048576,
        'INVALID_PARAM_MULTINOMIAL': 2097152
    }
    
    INVERSE_ERROR_CODES = {
        0: 'Casting occurred that could result in loss of precision.',
        1: 'Found Uniform(a, b) distribution where a > b.',
        2: 'Found Normal(m, v^2) distribution where v < 0.',
        3: 'Found Exponential(s) distribution where s <= 0.',
        4: 'Found Weibull(k, l) distribution where either k <= 0 or l <= 0.',
        5: 'Found Bernoulli(p) distribution where either p < 0 or p > 1.',
        6: 'Found Poisson(l) distribution where l < 0.',
        7: 'Found Gamma(k, l) distribution where either k <= 0 or l <= 0.',
        8: 'Found Beta(a, b) distribution where either a <= 0 or b <= 0.',
        9: 'Found Geometric(p) distribution where either p < 0 or p > 1.',
        10: 'Found Pareto(k, l) distribution where either k <= 0 or l <= 0.',
        11: 'Found Student(df) distribution where df <= 0.',
        12: 'Found Gumbel(m, s) distribution where s <= 0.',
        13: 'Found Laplace(m, s) distribution where s <= 0.',
        14: 'Found Cauchy(m, s) distribution where s <= 0.',
        15: 'Found Gompertz(k, l) distribution where either k <= 0 or l <= 0.',
        16: 'Found ChiSquare(df) distribution where df <= 0.',
        17: 'Found Discrete(p) distribution where either p < 0 or p does not sum to 1.',
        18: 'Found KronDelta(x) distribution where x is not int nor bool.',
        19: 'Found Dirichlet(alpha) distribution where alpha < 0.',
        20: 'Found MultivariateStudent(mean, cov, df) distribution where df <= 0.',
        21: 'Found Multinomial(p, n) distribution where either p < 0, p does not sum to 1, or n <= 0.'
    }
    
    @staticmethod
    def get_error_codes(error):
        '''Given a compacted integer error flag from the execution of Jax, and 
        decomposes it into individual error codes.
        '''
        binary = reversed(bin(error)[2:])
        errors = [i for (i, c) in enumerate(binary) if c == '1']
        return errors
    
    @staticmethod
    def get_error_messages(error):
        '''Given a compacted integer error flag from the execution of Jax, and 
        decomposes it into error strings.
        '''
        codes = JaxRDDLCompiler.get_error_codes(error)
        messages = [JaxRDDLCompiler.INVERSE_ERROR_CODES[i] for i in codes]
        return messages
    
    # ===========================================================================
    # expression compilation
    # ===========================================================================
    
    def _jax(self, expr, dtype=None):
        etype, _ = expr.etype
        if etype == 'constant':
            jax_expr = self._jax_constant(expr)
        elif etype == 'pvar':
            jax_expr = self._jax_pvar(expr)
        elif etype == 'arithmetic':
            jax_expr = self._jax_arithmetic(expr)
        elif etype == 'relational':
            jax_expr = self._jax_relational(expr)
        elif etype == 'boolean':
            jax_expr = self._jax_logical(expr)
        elif etype == 'aggregation':
            jax_expr = self._jax_aggregation(expr)
        elif etype == 'func':
            jax_expr = self._jax_functional(expr)
        elif etype == 'control':
            jax_expr = self._jax_control(expr)
        elif etype == 'randomvar':
            jax_expr = self._jax_random(expr)
        elif etype == 'randomvector':
            jax_expr = self._jax_random_vector(expr)
        elif etype == 'matrix':
            jax_expr = self._jax_matrix(expr)
        else:
            raise RDDLNotImplementedError(
                f'Internal error: expression type {expr} is not supported.\n' + 
                JaxRDDLCompiler._print_stack_trace(expr))
                
        if dtype is not None:
            jax_expr = self._jax_cast(jax_expr, dtype)
        
        return jax_expr
            
    def _jax_cast(self, jax_expr, dtype):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_CAST']
        
        def _jax_wrapped_cast(x, key):
            val, key, err = jax_expr(x, key)
            sample = jnp.asarray(val, dtype=dtype)
            invalid = jnp.logical_and(jnp.logical_not(jnp.can_cast(val, dtype)),
                                      jnp.any(sample != val))
            err |= (invalid * ERR)
            return sample, key, err
        
        return _jax_wrapped_cast
   
    # ===========================================================================
    # leaves
    # ===========================================================================
    
    def _jax_constant(self, expr):
        NORMAL = JaxRDDLCompiler.ERROR_CODES['NORMAL']
        cached_value = self.traced.cached_sim_info(expr)
        
        def _jax_wrapped_constant(_, key):
            return cached_value, key, NORMAL

        return _jax_wrapped_constant
    
    def _jax_pvar(self, expr):
        NORMAL = JaxRDDLCompiler.ERROR_CODES['NORMAL']
        var, pvars = expr.args  
        cached_info = self.traced.cached_sim_info(expr)
        
        # boundary case: free variable is converted to array (0, 1, 2...)
        if self.rddl.is_free_variable(var):
            cached_value = cached_info

            def _jax_wrapped_pvar_free_variable(_, key):
                return cached_value, key, NORMAL
            
            return _jax_wrapped_pvar_free_variable
        
        # boundary case: enum literal is converted to canonical integer index
        elif not pvars and self.rddl.is_literal(var):
            cached_literal = cached_info
            
            def _jax_wrapped_pvar_literal(_, key):
                return cached_literal, key, NORMAL
            
            return _jax_wrapped_pvar_literal
        
        # boundary case: no shape information (e.g. scalar pvar)
        elif cached_info is None:
            
            def _jax_wrapped_pvar_scalar(x, key):
                sample = x[var]
                return sample, key, NORMAL
            
            return _jax_wrapped_pvar_scalar
        
        # must slice and/or reshape value tensor to match free variables
        else:
            slices, axis, shape, op_code, op_args = cached_info 
        
            # compile nested expressions
            if slices and op_code == -1:
                
                jax_nested_expr = [(self._jax(arg) if _slice is None 
                                    else (lambda _, key: (_slice, key, NORMAL)))
                                   for (arg, _slice) in zip(pvars, slices)]    
                
                def _jax_wrapped_pvar_tensor_nested(x, key):
                    error = NORMAL
                    sample = jnp.asarray(x[var])
                    new_slices = [None] * len(jax_nested_expr)
                    for (i, jax_expr) in enumerate(jax_nested_expr):
                        new_slices[i], key, err = jax_expr(x, key)
                        error |= err
                    sample = sample[tuple(new_slices)]
                    return sample, key, error
                
                return _jax_wrapped_pvar_tensor_nested
                
            # tensor variable but no nesting  
            else:
    
                def _jax_wrapped_pvar_tensor_non_nested(x, key):
                    sample = jnp.asarray(x[var])
                    if slices:
                        sample = sample[slices]
                    if axis:
                        sample = jnp.expand_dims(sample, axis=axis)
                        sample = jnp.broadcast_to(sample, shape=shape)
                    if op_code == 0:
                        sample = jnp.einsum(sample, *op_args)
                    elif op_code == 1:
                        sample = jnp.transpose(sample, axes=op_args)
                    return sample, key, NORMAL
                
                return _jax_wrapped_pvar_tensor_non_nested
    
    # ===========================================================================
    # mathematical
    # ===========================================================================
    
    @staticmethod
    def _jax_unary(jax_expr, jax_op, at_least_int=False, check_dtype=None):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_CAST']
        
        def _jax_wrapped_unary_op(x, key):
            sample, key, err = jax_expr(x, key)
            if at_least_int:
                sample = 1 * sample
            sample = jax_op(sample)
            if check_dtype is not None:
                invalid = jnp.logical_not(jnp.can_cast(sample, check_dtype))
                err |= (invalid * ERR)
            return sample, key, err
        
        return _jax_wrapped_unary_op
    
    @staticmethod
    def _jax_binary(jax_lhs, jax_rhs, jax_op, at_least_int=False, check_dtype=None):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_CAST']
        
        def _jax_wrapped_binary_op(x, key):
            sample1, key, err1 = jax_lhs(x, key)
            sample2, key, err2 = jax_rhs(x, key)
            if at_least_int:
                sample1 = 1 * sample1
                sample2 = 1 * sample2
            sample = jax_op(sample1, sample2)
            err = err1 | err2
            if check_dtype is not None:
                invalid = jnp.logical_not(jnp.logical_and(
                    jnp.can_cast(sample1, check_dtype),
                    jnp.can_cast(sample2, check_dtype)))
                err |= (invalid * ERR)
            return sample, key, err
        
        return _jax_wrapped_binary_op
        
    def _jax_arithmetic(self, expr):
        _, op = expr.etype
        valid_ops = self.ARITHMETIC_OPS
        JaxRDDLCompiler._check_valid_op(expr, valid_ops)
                    
        args = expr.args
        n = len(args)
        
        if n == 1 and op == '-':
            arg, = args
            jax_expr = self._jax(arg)
            return JaxRDDLCompiler._jax_unary(
                jax_expr, jnp.negative, at_least_int=True)
                    
        elif n == 2:
            lhs, rhs = args
            jax_lhs = self._jax(lhs)
            jax_rhs = self._jax(rhs)
            jax_op = valid_ops[op]
            return JaxRDDLCompiler._jax_binary(
                jax_lhs, jax_rhs, jax_op, at_least_int=True)
        
        JaxRDDLCompiler._check_num_args(expr, 2)
    
    def _jax_relational(self, expr):
        _, op = expr.etype
        valid_ops = self.RELATIONAL_OPS
        JaxRDDLCompiler._check_valid_op(expr, valid_ops)
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        lhs, rhs = expr.args
        jax_lhs = self._jax(lhs)
        jax_rhs = self._jax(rhs)
        jax_op = valid_ops[op]
        return JaxRDDLCompiler._jax_binary(
            jax_lhs, jax_rhs, jax_op, at_least_int=True)
           
    def _jax_logical(self, expr):
        _, op = expr.etype
        valid_ops = self.LOGICAL_OPS    
        JaxRDDLCompiler._check_valid_op(expr, valid_ops)
                
        args = expr.args
        n = len(args)
        
        if n == 1 and op == '~':
            arg, = args
            jax_expr = self._jax(arg)
            return JaxRDDLCompiler._jax_unary(
                jax_expr, self.LOGICAL_NOT, check_dtype=bool)
        
        elif n == 2:
            lhs, rhs = args
            jax_lhs = self._jax(lhs)
            jax_rhs = self._jax(rhs)
            jax_op = valid_ops[op]
            return JaxRDDLCompiler._jax_binary(
                jax_lhs, jax_rhs, jax_op, check_dtype=bool)
        
        JaxRDDLCompiler._check_num_args(expr, 2)
    
    def _jax_aggregation(self, expr):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_CAST']
        
        _, op = expr.etype
        valid_ops = self.AGGREGATION_OPS      
        JaxRDDLCompiler._check_valid_op(expr, valid_ops) 
        is_floating = op not in {'forall', 'exists'}
        
        * _, arg = expr.args  
        _, axes = self.traced.cached_sim_info(expr)
        
        jax_expr = self._jax(arg)
        jax_op = valid_ops[op]        
        
        def _jax_wrapped_aggregation(x, key):
            sample, key, err = jax_expr(x, key)
            if is_floating:
                sample = 1 * sample
            else:
                invalid = jnp.logical_not(jnp.can_cast(sample, bool))
                err |= (invalid * ERR)
            sample = jax_op(sample, axis=axes)
            return sample, key, err
        
        return _jax_wrapped_aggregation
               
    def _jax_functional(self, expr):
        _, op = expr.etype
        
        # unary function
        if op in self.KNOWN_UNARY:
            JaxRDDLCompiler._check_num_args(expr, 1)                            
            arg, = expr.args
            jax_expr = self._jax(arg)
            jax_op = self.KNOWN_UNARY[op]
            return JaxRDDLCompiler._jax_unary(
                jax_expr, jax_op, at_least_int=True)
            
        # binary function
        elif op in self.KNOWN_BINARY:
            JaxRDDLCompiler._check_num_args(expr, 2)                
            lhs, rhs = expr.args
            jax_lhs = self._jax(lhs)
            jax_rhs = self._jax(rhs)
            jax_op = self.KNOWN_BINARY[op]
            return JaxRDDLCompiler._jax_binary(
                jax_lhs, jax_rhs, jax_op, at_least_int=True)
        
        raise RDDLNotImplementedError(
                f'Function {op} is not supported.\n' + 
                JaxRDDLCompiler._print_stack_trace(expr))   
    
    # ===========================================================================
    # control flow
    # ===========================================================================
    
    def _jax_control(self, expr):
        _, op = expr.etype
        valid_ops = self.CONTROL_OPS
        JaxRDDLCompiler._check_valid_op(expr, valid_ops)
        
        if op == 'if':
            return self._jax_if(expr)
        else:
            return self._jax_switch(expr)
    
    def _jax_if(self, expr):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_CAST']
        JaxRDDLCompiler._check_num_args(expr, 3)
        jax_op = self.CONTROL_OPS['if']
        
        pred, if_true, if_false = expr.args        
        jax_pred = self._jax(pred)
        jax_true = self._jax(if_true)
        jax_false = self._jax(if_false)
        
        def _jax_wrapped_if_then_else(x, key):
            sample1, key, err1 = jax_pred(x, key)
            sample2, key, err2 = jax_true(x, key)
            sample3, key, err3 = jax_false(x, key)
            sample = jax_op(sample1, sample2, sample3)
            err = err1 | err2 | err3
            invalid = jnp.logical_not(jnp.can_cast(sample1, bool))
            err |= (invalid * ERR)
            return sample, key, err
            
        return _jax_wrapped_if_then_else
    
    def _jax_switch_helper(self):
        
        def _jax_switch_calc_exact(pred, cases):
            pred = pred[jnp.newaxis, ...]
            sample = jnp.take_along_axis(cases, pred, axis=0)
            assert sample.shape[0] == 1
            return sample[0, ...]

        return _jax_switch_calc_exact
        
    def _jax_switch(self, expr):
        pred, *_ = expr.args             
        jax_pred = self._jax(pred)
        jax_switch = self.CONTROL_OPS['switch']
        
        # wrap cases as JAX expressions
        cases, default = self.traced.cached_sim_info(expr) 
        jax_default = None if default is None else self._jax(default)
        jax_cases = [(jax_default if _case is None else self._jax(_case))
                     for _case in cases]
                    
        def _jax_wrapped_switch(x, key):
            
            # sample predicate
            sample_pred, key, err = jax_pred(x, key) 
            
            # sample cases
            sample_cases = [None] * len(jax_cases)
            for (i, jax_case) in enumerate(jax_cases):
                sample_cases[i], key, err_case = jax_case(x, key)
                err |= err_case                
            sample_cases = jnp.asarray(sample_cases)
            
            # predicate (enum) is an integer - use it to extract from case array
            sample = jax_switch(sample_pred, sample_cases)
            return sample, key, err    
        
        return _jax_wrapped_switch
    
    # ===========================================================================
    # random variables
    # ===========================================================================
    
    def _jax_random(self, expr):
        _, name = expr.etype
        if name == 'KronDelta':
            return self._jax_kron(expr)        
        elif name == 'DiracDelta':
            return self._jax_dirac(expr)
        elif name == 'Uniform':
            return self._jax_uniform(expr)
        elif name == 'Normal':
            return self._jax_normal(expr)
        elif name == 'Exponential':
            return self._jax_exponential(expr)
        elif name == 'Weibull':
            return self._jax_weibull(expr)   
        elif name == 'Bernoulli':
            return self._jax_bernoulli(expr)
        elif name == 'Poisson':
            return self._jax_poisson(expr)
        elif name == 'Gamma':
            return self._jax_gamma(expr)
        elif name == 'Beta':
            return self._jax_beta(expr)
        elif name == 'Geometric':
            return self._jax_geometric(expr)
        elif name == 'Pareto':
            return self._jax_pareto(expr)
        elif name == 'Student':
            return self._jax_student(expr)
        elif name == 'Gumbel':
            return self._jax_gumbel(expr)
        elif name == 'Laplace':
            return self._jax_laplace(expr)
        elif name == 'Cauchy':
            return self._jax_cauchy(expr)
        elif name == 'Gompertz':
            return self._jax_gompertz(expr)
        elif name == 'ChiSquare':
            return self._jax_chisquare(expr)
        elif name == 'Discrete':
            return self._jax_discrete(expr, unnorm=False)
        elif name == 'UnnormDiscrete':
            return self._jax_discrete(expr, unnorm=True)
        elif name == 'Discrete(p)':
            return self._jax_discrete_pvar(expr, unnorm=False)
        elif name == 'UnnormDiscrete(p)':
            return self._jax_discrete_pvar(expr, unnorm=True)
        else:
            raise RDDLNotImplementedError(
                f'Distribution {name} is not supported.\n' + 
                JaxRDDLCompiler._print_stack_trace(expr))
        
    def _jax_kron(self, expr):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_KRON_DELTA']
        JaxRDDLCompiler._check_num_args(expr, 1)
        arg, = expr.args
        arg = self._jax(arg)
        
        # just check that the sample can be cast to int
        def _jax_wrapped_distribution_kron(x, key):
            sample, key, err = arg(x, key)
            invalid = jnp.logical_not(jnp.can_cast(sample, JaxRDDLCompiler.INT))
            err |= (invalid * ERR)
            return sample, key, err
                        
        return _jax_wrapped_distribution_kron
    
    def _jax_dirac(self, expr):
        JaxRDDLCompiler._check_num_args(expr, 1)
        arg, = expr.args
        arg = self._jax(arg, dtype=JaxRDDLCompiler.REAL)
        return arg
    
    def _jax_uniform(self, expr):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_UNIFORM']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_lb, arg_ub = expr.args
        jax_lb = self._jax(arg_lb)
        jax_ub = self._jax(arg_ub)
        
        # reparameterization trick U(a, b) = a + (b - a) * U(0, 1)
        def _jax_wrapped_distribution_uniform(x, key):
            lb, key, err1 = jax_lb(x, key)
            ub, key, err2 = jax_ub(x, key)
            key, subkey = random.split(key)
            U = random.uniform(
                key=subkey, shape=jnp.shape(lb), dtype=JaxRDDLCompiler.REAL)
            sample = lb + (ub - lb) * U
            out_of_bounds = jnp.logical_not(jnp.all(lb <= ub))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_uniform
    
    def _jax_normal(self, expr):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_NORMAL']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_mean, arg_var = expr.args
        jax_mean = self._jax(arg_mean)
        jax_var = self._jax(arg_var)
        
        # reparameterization trick N(m, s^2) = m + s * N(0, 1)
        def _jax_wrapped_distribution_normal(x, key):
            mean, key, err1 = jax_mean(x, key)
            var, key, err2 = jax_var(x, key)
            std = jnp.sqrt(var)
            key, subkey = random.split(key)
            Z = random.normal(
                key=subkey, shape=jnp.shape(mean), dtype=JaxRDDLCompiler.REAL)
            sample = mean + std * Z
            out_of_bounds = jnp.logical_not(jnp.all(var >= 0))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_normal
    
    def _jax_exponential(self, expr):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_EXPONENTIAL']
        JaxRDDLCompiler._check_num_args(expr, 1)
        
        arg_scale, = expr.args
        jax_scale = self._jax(arg_scale)
        
        # reparameterization trick Exp(s) = s * Exp(1)
        def _jax_wrapped_distribution_exp(x, key):
            scale, key, err = jax_scale(x, key)
            key, subkey = random.split(key)
            Exp1 = random.exponential(
                key=subkey, shape=jnp.shape(scale), dtype=JaxRDDLCompiler.REAL)
            sample = scale * Exp1
            out_of_bounds = jnp.logical_not(jnp.all(scale > 0))
            err |= (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_exp
    
    def _jax_weibull(self, expr):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_WEIBULL']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_shape, arg_scale = expr.args
        jax_shape = self._jax(arg_shape)
        jax_scale = self._jax(arg_scale)
        
        # reparameterization trick W(s, r) = r * (-ln(1 - U(0, 1))) ** (1 / s)
        def _jax_wrapped_distribution_weibull(x, key):
            shape, key, err1 = jax_shape(x, key)
            scale, key, err2 = jax_scale(x, key)
            key, subkey = random.split(key)
            U = random.uniform(
                key=subkey, shape=jnp.shape(scale), dtype=JaxRDDLCompiler.REAL)
            sample = scale * jnp.power(-jnp.log1p(-U), 1.0 / shape)
            out_of_bounds = jnp.logical_not(jnp.all((shape > 0) & (scale > 0)))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_weibull
            
    def _jax_bernoulli(self, expr):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_BERNOULLI']
        JaxRDDLCompiler._check_num_args(expr, 1)
        
        arg_prob, = expr.args
        jax_prob = self._jax(arg_prob)
        
        # uses the implicit JAX subroutine
        def _jax_wrapped_distribution_bernoulli(x, key):
            prob, key, err = jax_prob(x, key)
            key, subkey = random.split(key)
            sample = random.bernoulli(key=subkey, p=prob)
            out_of_bounds = jnp.logical_not(jnp.all((prob >= 0) & (prob <= 1)))
            err |= (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_bernoulli
    
    def _jax_poisson(self, expr):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_POISSON']
        JaxRDDLCompiler._check_num_args(expr, 1)
        
        arg_rate, = expr.args
        jax_rate = self._jax(arg_rate)
        
        # uses the implicit JAX subroutine
        def _jax_wrapped_distribution_poisson(x, key):
            rate, key, err = jax_rate(x, key)
            key, subkey = random.split(key)
            sample = random.poisson(
                key=subkey, lam=rate, dtype=JaxRDDLCompiler.INT)
            out_of_bounds = jnp.logical_not(jnp.all(rate >= 0))
            err |= (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_poisson
    
    def _jax_gamma(self, expr):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_GAMMA']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_shape, arg_scale = expr.args
        jax_shape = self._jax(arg_shape)
        jax_scale = self._jax(arg_scale)
        
        # partial reparameterization trick Gamma(s, r) = r * Gamma(s, 1)
        # uses the implicit JAX subroutine for Gamma(s, 1) 
        def _jax_wrapped_distribution_gamma(x, key):
            shape, key, err1 = jax_shape(x, key)
            scale, key, err2 = jax_scale(x, key)
            key, subkey = random.split(key)
            Gamma = random.gamma(key=subkey, a=shape, dtype=JaxRDDLCompiler.REAL)
            sample = scale * Gamma
            out_of_bounds = jnp.logical_not(jnp.all((shape > 0) & (scale > 0)))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_gamma
    
    def _jax_beta(self, expr):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_BETA']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_shape, arg_rate = expr.args
        jax_shape = self._jax(arg_shape)
        jax_rate = self._jax(arg_rate)
        
        # uses the implicit JAX subroutine
        def _jax_wrapped_distribution_beta(x, key):
            shape, key, err1 = jax_shape(x, key)
            rate, key, err2 = jax_rate(x, key)
            key, subkey = random.split(key)
            sample = random.beta(key=subkey, a=shape, b=rate)
            out_of_bounds = jnp.logical_not(jnp.all((shape > 0) & (rate > 0)))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_beta
    
    def _jax_geometric(self, expr):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_GEOMETRIC']
        JaxRDDLCompiler._check_num_args(expr, 1)
        
        arg_prob, = expr.args
        jax_prob = self._jax(arg_prob)
        
        # reparameterization trick Geom(p) = floor(ln(U(0, 1)) / ln(p)) + 1
        def _jax_wrapped_distribution_geometric(x, key):
            prob, key, err = jax_prob(x, key)
            key, subkey = random.split(key)
            U = random.uniform(
                key=subkey, shape=jnp.shape(prob), dtype=JaxRDDLCompiler.REAL)
            sample = jnp.floor(jnp.log1p(-U) / jnp.log1p(-prob)) + 1
            out_of_bounds = jnp.logical_not(jnp.all((prob >= 0) & (prob <= 1)))
            err |= (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_geometric
    
    def _jax_pareto(self, expr):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_PARETO']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_shape, arg_scale = expr.args
        jax_shape = self._jax(arg_shape)
        jax_scale = self._jax(arg_scale)
        
        # partial reparameterization trick Pareto(s, r) = r * Pareto(s, 1)
        # uses the implicit JAX subroutine for Pareto(s, 1) 
        def _jax_wrapped_distribution_pareto(x, key):
            shape, key, err1 = jax_shape(x, key)
            scale, key, err2 = jax_scale(x, key)
            key, subkey = random.split(key)
            sample = scale * random.pareto(key=subkey, b=shape)
            out_of_bounds = jnp.logical_not(jnp.all((shape > 0) & (scale > 0)))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_pareto
    
    def _jax_student(self, expr):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_STUDENT']
        JaxRDDLCompiler._check_num_args(expr, 1)
        
        arg_df, = expr.args
        jax_df = self._jax(arg_df)
        
        # uses the implicit JAX subroutine for student(df)
        def _jax_wrapped_distribution_t(x, key):
            df, key, err = jax_df(x, key)
            key, subkey = random.split(key)
            sample = random.t(key=subkey, df=df)
            out_of_bounds = jnp.logical_not(jnp.all(df > 0))
            err |= (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_t
    
    def _jax_gumbel(self, expr):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_GUMBEL']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_mean, arg_scale = expr.args
        jax_mean = self._jax(arg_mean)
        jax_scale = self._jax(arg_scale)
        
        # reparameterization trick Gumbel(m, s) = m + s * Gumbel(0, 1)
        def _jax_wrapped_distribution_gumbel(x, key):
            mean, key, err1 = jax_mean(x, key)
            scale, key, err2 = jax_scale(x, key)
            key, subkey = random.split(key)
            Gumbel01 = random.gumbel(
                key=subkey, shape=jnp.shape(mean), dtype=JaxRDDLCompiler.REAL)
            sample = mean + scale * Gumbel01
            out_of_bounds = jnp.logical_not(jnp.all(scale > 0))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_gumbel
    
    def _jax_laplace(self, expr):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_LAPLACE']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_mean, arg_scale = expr.args
        jax_mean = self._jax(arg_mean)
        jax_scale = self._jax(arg_scale)
        
        # reparameterization trick Laplace(m, s) = m + s * Laplace(0, 1)
        def _jax_wrapped_distribution_laplace(x, key):
            mean, key, err1 = jax_mean(x, key)
            scale, key, err2 = jax_scale(x, key)
            key, subkey = random.split(key)
            Laplace01 = random.laplace(
                key=subkey, shape=jnp.shape(mean), dtype=JaxRDDLCompiler.REAL)
            sample = mean + scale * Laplace01
            out_of_bounds = jnp.logical_not(jnp.all(scale > 0))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_laplace
    
    def _jax_cauchy(self, expr):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_CAUCHY']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_mean, arg_scale = expr.args
        jax_mean = self._jax(arg_mean)
        jax_scale = self._jax(arg_scale)
        
        # reparameterization trick Cauchy(m, s) = m + s * Cauchy(0, 1)
        def _jax_wrapped_distribution_cauchy(x, key):
            mean, key, err1 = jax_mean(x, key)
            scale, key, err2 = jax_scale(x, key)
            key, subkey = random.split(key)
            Cauchy01 = random.cauchy(
                key=subkey, shape=jnp.shape(mean), dtype=JaxRDDLCompiler.REAL)
            sample = mean + scale * Cauchy01
            out_of_bounds = jnp.logical_not(jnp.all(scale > 0))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_cauchy
    
    def _jax_gompertz(self, expr):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_GOMPERTZ']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_shape, arg_scale = expr.args
        jax_shape = self._jax(arg_shape)
        jax_scale = self._jax(arg_scale)
        
        # reparameterization trick Gompertz(s, r) = ln(1 - log(U(0, 1)) / s) / r
        def _jax_wrapped_distribution_gompertz(x, key):
            shape, key, err1 = jax_shape(x, key)
            scale, key, err2 = jax_scale(x, key)
            key, subkey = random.split(key)
            U = random.uniform(
                key=subkey, shape=jnp.shape(scale), dtype=JaxRDDLCompiler.REAL)
            sample = jnp.log(1.0 - jnp.log1p(-U) / shape) / scale
            out_of_bounds = jnp.logical_not(jnp.all((shape > 0) & (scale > 0)))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_gompertz
    
    def _jax_chisquare(self, expr):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_CHISQUARE']
        JaxRDDLCompiler._check_num_args(expr, 1)
        
        arg_df, = expr.args
        jax_df = self._jax(arg_df)
        
        # use the fact that ChiSquare(df) = Gamma(df/2, 2)
        def _jax_wrapped_distribution_chisquare(x, key):
            df, key, err1 = jax_df(x, key)
            key, subkey = random.split(key)
            shape = df / 2.0
            Gamma = random.gamma(key=subkey, a=shape, dtype=JaxRDDLCompiler.REAL)
            sample = 2.0 * Gamma
            out_of_bounds = jnp.logical_not(jnp.all(df > 0))
            err = err1 | (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_chisquare
    
    # ===========================================================================
    # random variables with enum support
    # ===========================================================================
    
    def _jax_discrete_helper(self):
        
        def _jax_discrete_calc_exact(prob, subkey):
            out_of_bounds = jnp.logical_not(jnp.logical_and(
                jnp.all(prob >= 0),
                jnp.allclose(jnp.sum(prob, axis=-1), 1.0)))
            logits = jnp.log(prob)
            sample = random.categorical(key=subkey, logits=logits, axis=-1)
            return sample, out_of_bounds
        
        return _jax_discrete_calc_exact
            
    def _jax_discrete(self, expr, unnorm):
        NORMAL = JaxRDDLCompiler.ERROR_CODES['NORMAL']
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_DISCRETE']
        
        ordered_args = self.traced.cached_sim_info(expr)
        jax_probs = [self._jax(arg) for arg in ordered_args]
        jax_discrete = self._jax_discrete_helper()
        
        def _jax_wrapped_distribution_discrete(x, key):
            
            # sample case probabilities and normalize as needed
            error = NORMAL
            prob = [None] * len(jax_probs)
            for (i, jax_prob) in enumerate(jax_probs):
                prob[i], key, error_pdf = jax_prob(x, key)
                error |= error_pdf
            prob = jnp.stack(prob, axis=-1)
            if unnorm:
                normalizer = jnp.sum(prob, axis=-1, keepdims=True)
                prob = prob / normalizer
            
            # dispatch to sampling subroutine
            key, subkey = random.split(key)
            sample, out_of_bounds = jax_discrete(prob, subkey)
            error |= (out_of_bounds * ERR)
            return sample, key, error
        
        return _jax_wrapped_distribution_discrete
    
    def _jax_discrete_pvar(self, expr, unnorm):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_DISCRETE']
        JaxRDDLCompiler._check_num_args(expr, 1)
        
        _, args = expr.args
        arg, = args
        jax_probs = self._jax(arg)
        jax_discrete = self._jax_discrete_helper()

        def _jax_wrapped_distribution_discrete_pvar(x, key):
            
            # sample probabilities
            prob, key, error = jax_probs(x, key)
            if unnorm:
                normalizer = jnp.sum(prob, axis=-1, keepdims=True)
                prob = prob / normalizer
            
            # dispatch to sampling subroutine
            key, subkey = random.split(key)
            sample, out_of_bounds = jax_discrete(prob, subkey)
            error |= (out_of_bounds * ERR)
            return sample, key, error
        
        return _jax_wrapped_distribution_discrete_pvar

    # ===========================================================================
    # random vectors
    # ===========================================================================
    
    def _jax_random_vector(self, expr):
        _, name = expr.etype
        if name == 'MultivariateNormal':
            return self._jax_multivariate_normal(expr)   
        elif name == 'MultivariateStudent':
            return self._jax_multivariate_student(expr)  
        elif name == 'Dirichlet':
            return self._jax_dirichlet(expr)
        elif name == 'Multinomial':
            return self._jax_multinomial(expr)
        else:
            raise RDDLNotImplementedError(
                f'Distribution {name} is not supported.\n' + 
                JaxRDDLCompiler._print_stack_trace(expr))
    
    def _jax_multivariate_normal(self, expr): 
        _, args = expr.args
        mean, cov = args
        jax_mean = self._jax(mean)
        jax_cov = self._jax(cov)
        index, = self.traced.cached_sim_info(expr)
        
        # reparameterization trick MN(m, LL') = LZ + m, where Z ~ Normal(0, 1)
        def _jax_wrapped_distribution_multivariate_normal(x, key):
            
            # sample the mean and covariance
            sample_mean, key, err1 = jax_mean(x, key)
            sample_cov, key, err2 = jax_cov(x, key)
            
            # sample Normal(0, 1)
            key, subkey = random.split(key)
            Z = random.normal(
                key=subkey,
                shape=jnp.shape(sample_mean) + (1,),
                dtype=JaxRDDLCompiler.REAL)       
            
            # compute L s.t. cov = L * L' and reparameterize
            L = jnp.linalg.cholesky(sample_cov)
            sample = jnp.matmul(L, Z)[..., 0] + sample_mean
            sample = jnp.moveaxis(sample, source=-1, destination=index)
            err = err1 | err2
            return sample, key, err
        
        return _jax_wrapped_distribution_multivariate_normal
    
    def _jax_multivariate_student(self, expr):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_MULTIVARIATE_STUDENT']
        
        _, args = expr.args
        mean, cov, df = args
        jax_mean = self._jax(mean)
        jax_cov = self._jax(cov)
        jax_df = self._jax(df)
        index, = self.traced.cached_sim_info(expr)
        
        # reparameterization trick MN(m, LL') = LZ + m, where Z ~ StudentT(0, 1)
        def _jax_wrapped_distribution_multivariate_student(x, key):
            
            # sample the mean and covariance and degrees of freedom
            sample_mean, key, err1 = jax_mean(x, key)
            sample_cov, key, err2 = jax_cov(x, key)
            sample_df, key, err3 = jax_df(x, key)
            out_of_bounds = jnp.logical_not(jnp.all(sample_df > 0))
            
            # sample StudentT(0, 1, df) -- broadcast df to same shape as cov
            sample_df = sample_df[..., jnp.newaxis, jnp.newaxis]
            sample_df = jnp.broadcast_to(sample_df, shape=sample_mean.shape + (1,))
            key, subkey = random.split(key)
            Z = random.t(key=subkey, df=sample_df, shape=sample_df.shape,
                         dtype=JaxRDDLCompiler.REAL)   
            
            # compute L s.t. cov = L * L' and reparameterize
            L = jnp.linalg.cholesky(sample_cov)
            sample = jnp.matmul(L, Z)[..., 0] + sample_mean
            sample = jnp.moveaxis(sample, source=-1, destination=index)
            error = err1 | err2 | err3 | (out_of_bounds * ERR)
            return sample, key, error
        
        return _jax_wrapped_distribution_multivariate_student
    
    def _jax_dirichlet(self, expr):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_DIRICHLET']
        
        _, args = expr.args
        alpha, = args
        jax_alpha = self._jax(alpha)
        index, = self.traced.cached_sim_info(expr)
        
        # sample Gamma(alpha_i, 1) and normalize across i
        def _jax_wrapped_distribution_dirichlet(x, key):
            alpha, key, error = jax_alpha(x, key)
            out_of_bounds = jnp.logical_not(jnp.all(alpha > 0))
            error |= (out_of_bounds * ERR)
            key, subkey = random.split(key)
            Gamma = random.gamma(key=subkey, a=alpha)
            sample = Gamma / jnp.sum(Gamma, axis=-1, keepdims=True)
            sample = jnp.moveaxis(sample, source=-1, destination=index)
            return sample, key, error
        
        return _jax_wrapped_distribution_dirichlet
    
    def _jax_multinomial(self, expr):
        NORMAL = JaxRDDLCompiler.ERROR_CODES['NORMAL']
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_MULTINOMIAL']
        
        _, args = expr.args
        prob, trials = args
        jax_prob = self._jax(prob)
        jax_trials = self._jax(trials)
        jax_discrete = self._jax_discrete_helper()
        index, = self.traced.cached_sim_info(expr)
        
        # samples from a discrete(prob), computes a one-hot mask w.r.t categories
        def _jax_wrapped_multinomial_trial(_, values):
            samples, errors, key, prob, categories = values
            key, subkey = random.split(key)
            sample, error = jax_discrete(prob, subkey)            
            sample = sample[jnp.newaxis, ...]
            masked = (sample == categories)
            samples += masked
            errors |= error
            return (samples, errors, key, prob, categories)            
            
        def _jax_wrapped_distribution_multinomial(x, key):
            
            # sample multinomial parameters
            # note that # of trials must not be parameterized
            prob, key, err1 = jax_prob(x, key)
            trials, key, err2 = jax_trials(x, key)
            num_trials = jnp.ravel(trials)[0]
            out_of_bounds = jnp.logical_not(num_trials > 0)
            error = err1 | err2 | (out_of_bounds * ERR)
            
            # create a categories mask that spans support of discrete(prob)
            num_categories = prob.shape[-1]
            categories = np.arange(num_categories)
            categories = categories[(...,) + (jnp.newaxis,) * len(prob.shape[:-1])]
            
            # do a for loop over trials, accumulate category counts in samples
            counts = jnp.zeros(shape=(num_categories,) + prob.shape[:-1],
                               dtype=JaxRDDLCompiler.INT)
            sample, err3, key, *_ = jax.lax.fori_loop(
                lower=0,
                upper=num_trials,
                body_fun=_jax_wrapped_multinomial_trial,
                init_val=(counts, NORMAL, key, prob, categories))
            sample = jnp.moveaxis(sample, source=0, destination=index)
            error |= err3
            return sample, key, error
        
        return _jax_wrapped_distribution_multinomial
    
    # ===========================================================================
    # matrix algebra
    # ===========================================================================
    
    def _jax_matrix(self, expr):
        _, op = expr.etype
        if op == 'det':
            return self._jax_matrix_det(expr)
        elif op == 'inverse':
            return self._jax_matrix_inv(expr)
        else:
            raise RDDLNotImplementedError(
                f'Matrix operation {op} is not supported.\n' + 
                JaxRDDLCompiler._print_stack_trace(expr))
    
    def _jax_matrix_det(self, expr):
        * _, arg = expr.args
        jax_arg = self._jax(arg)
        
        def _jax_wrapped_matrix_operation_det(x, key):
            sample_arg, key, error = jax_arg(x, key)
            sample = jnp.linalg.det(sample_arg)
            return sample, key, error
        
        return _jax_wrapped_matrix_operation_det
    
    def _jax_matrix_inv(self, expr):
        _, arg = expr.args
        jax_arg = self._jax(arg)
        indices = self.traced.cached_sim_info(expr)
        
        def _jax_wrapped_matrix_operation_inv(x, key):
            sample_arg, key, error = jax_arg(x, key)
            sample = jnp.linalg.inv(sample_arg)
            sample = jnp.moveaxis(sample, source=(-2, -1), destination=indices)
            return sample, key, error
        
        return _jax_wrapped_matrix_operation_inv
            
