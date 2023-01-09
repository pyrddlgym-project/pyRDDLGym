import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy as scipy 

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
        self.rddl = rddl
        self.logger = logger
        jax.config.update('jax_log_compiles', True)
        
        # compile initial values
        if self.logger is not None:
            self.logger.clear()
        initializer = RDDLValueInitializer(rddl, logger=self.logger)
        self.init_values = initializer.initialize()
        
        # compute dependency graph for CPFs and sort them by evaluation order
        sorter = RDDLLevelAnalysis(rddl, allow_synchronous_state)
        self.levels = sorter.compute_levels()        
        
        # log dependency graph information to file
        if self.logger is not None: 
            levels_info = '\n\t'.join(f"{k}: {{{', '.join(v)}}}"
                                      for (k, v) in self.levels.items())
            message = (f'computed order of CPF evaluation:\n' 
                       f'\t{levels_info}\n')
            self.logger.log(message)
        
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
            'exists': jnp.any  
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
            'switch': jnp.select
        }
        
    # ===========================================================================
    # main compilation subroutines
    # ===========================================================================
     
    def compile(self, log_jax_expr: bool=False) -> None: 
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
        return [self._jax(c, dtype=bool) for c in constraints]
        
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
            return (carried, logged)
        
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
    
    def print_jax(self):
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
        'INVALID_PARAM_DISCRETE': 65536,
        'INVALID_PARAM_KRON_DELTA': 131072
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
        16: 'Found Discrete(p) distribution where either p < 0 or p does not sum to 1.',
        17: 'Found KronDelta(x) where x is not int nor bool.'
    }
    
    @staticmethod
    def get_error_codes(error):
        binary = reversed(bin(error)[2:])
        errors = [i for (i, c) in enumerate(binary) if c == '1']
        return errors
    
    @staticmethod
    def get_error_messages(error):
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
        else:
            raise RDDLNotImplementedError(
                f'Internal error: expression {expr} is not supported.\n' + 
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
        ERR = JaxRDDLCompiler.ERROR_CODES['NORMAL']
        cached_value = self.traced.cached_sim_info(expr)
        
        def _f_constant(_, key):
            return cached_value, key, ERR

        return _f_constant
    
    def _jax_pvar(self, expr):
        ERR = JaxRDDLCompiler.ERROR_CODES['NORMAL']
        var, pvars = expr.args  
        
        # boundary case: enum literal is converted to canonical integer index
        if not pvars and self.rddl.is_literal(var):
            cached_literal = self.traced.cached_sim_info(expr)
            
            def _jax_wrapped_pvar_literal(_, key):
                return cached_literal, key, ERR
            
            return _jax_wrapped_pvar_literal
        
        # boundary case: no shape information (e.g. scalar pvar)
        shape_info = self.traced.cached_sim_info(expr)
        if shape_info is None:
            
            def _jax_wrapped_pvar_scalar(x, key):
                sample = x[var]
                return sample, key, ERR
            
            return _jax_wrapped_pvar_scalar
        
        # must slice and/or reshape value tensor to match free variables
        slices, axis, shape, op_code, op_args = shape_info 
                
        def _jax_wrapped_pvar_tensor(x, key):
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
            return sample, key, ERR
        
        return _jax_wrapped_pvar_tensor
    
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
    
    def _jax_switch(self, expr):
        pred, *_ = expr.args             
        jax_pred = self._jax(pred)
        cases, default = self.traced.cached_sim_info(expr)  
        
        # wrap cases as JAX expressions
        jax_def = None if default is None else self._jax(default)
        jax_cases = [(jax_def if arg is None else self._jax(arg))
                     for arg in cases]
                    
        def _jax_wrapped_switch(x, key):
            
            # sample predicate
            sample_pred, key, err = jax_pred(x, key) 
            
            # sample cases
            sample_cases = []
            for jax_case in jax_cases:
                sample_case, key, err_case = jax_case(x, key)
                sample_cases.append(sample_case)
                err |= err_case                
            sample_cases = jnp.asarray(sample_cases)
            
            # predicate (enum) is an integer - use it to extract from case array
            sample_pred = jnp.expand_dims(sample_pred, axis=0)
            sample = jnp.take_along_axis(sample_cases, sample_pred, axis=0)
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
        elif name == 'Discrete':
            return self._jax_discrete(expr, unnorm=False)
        elif name == 'UnnormDiscrete':
            return self._jax_discrete(expr, unnorm=True)
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
        
        # reparameterization trick Bern(p) = U(0, 1) < p
        def _jax_wrapped_distribution_bernoulli(x, key):
            prob, key, err = jax_prob(x, key)
            key, subkey = random.split(key)
            U = random.uniform(
                key=subkey, shape=jnp.shape(prob), dtype=JaxRDDLCompiler.REAL)
            sample = jnp.less(U, prob)
            out_of_bounds = jnp.logical_not(jnp.all((prob >= 0) & (prob <= 1)))
            err |= (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_bernoulli
    
    def _jax_poisson(self, expr):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_POISSON']
        JaxRDDLCompiler._check_num_args(expr, 1)
        
        arg_rate, = expr.args
        jax_rate = self._jax(arg_rate)
        
        # uses the implicit JAX subroutine, which seems to be reparameterized
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
        # which seems to be reparameterized
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
        
        # uses the implicit JAX subroutine, which seems to be reparameterized
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
        # which seems to be reparameterized
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
        # which seems to be reparameterized
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
    
    def _jax_discrete(self, expr, unnorm):
        ERR0 = JaxRDDLCompiler.ERROR_CODES['NORMAL']
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_DISCRETE']
        
        jax_pdfs = [self._jax(arg) for arg in self.traced.cached_sim_info(expr)]
        
        def _jax_wrapped_distribution_discrete(x, key):
            
            # sample case probabilities
            err = ERR0
            sample_pdfs = []
            for jax_pdf in jax_pdfs:
                sample_pdf, key, err_pdf = jax_pdf(x, key)
                sample_pdfs.append(sample_pdf)
                err |= err_pdf
            sample_pdfs = jnp.asarray(sample_pdfs)
            
            # compute cumulative distribution function
            sample_cdf = jnp.cumsum(sample_pdfs, axis=0)
            
            # check this is a valid PDF
            if unnorm:
                out_of_bounds = jnp.logical_not(jnp.all(sample_pdfs >= 0))
                sample_cdf = sample_cdf / sample_cdf[-1:, ...]
            else:
                out_of_bounds = jnp.logical_not(jnp.logical_and(
                    jnp.all(sample_pdfs >= 0),
                    jnp.allclose(sample_cdf[-1, ...], 1.0)))
            err |= (out_of_bounds * ERR)
            
            # reparameterization trick using inverse CDF sampling
            key, subkey = random.split(key)
            shape = (1,) + jnp.shape(sample_cdf)[1:]
            U = random.uniform(key=subkey, shape=shape, dtype=JaxRDDLCompiler.REAL)
            sample = jnp.argmax(jnp.less(U, sample_cdf), axis=0)
            return sample, key, err
        
        return _jax_wrapped_distribution_discrete
