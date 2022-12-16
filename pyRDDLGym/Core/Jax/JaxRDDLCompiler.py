import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
import warnings

from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidNumberOfArgumentsError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidObjectError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLNotImplementedError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLTypeError

from pyRDDLGym.Core.Compiler.RDDLDecompiler import RDDLDecompiler
from pyRDDLGym.Core.Compiler.RDDLLiftedModel import RDDLLiftedModel
from pyRDDLGym.Core.Compiler.RDDLLevelAnalysis import RDDLLevelAnalysis
from pyRDDLGym.Core.Parser.expr import Expression
from pyRDDLGym.Core.Simulator.RDDLTensors import RDDLTensors


class JaxRDDLCompiler:
    
    INT = jnp.int32
    REAL = jnp.float32
    
    JAX_TYPES = {
        'int': INT,
        'real': REAL,
        'bool': bool
    }
    
    def __init__(self, rddl: RDDLLiftedModel,
                 allow_synchronous_state: bool=True,
                 debug: bool=True) -> None:
        self.rddl = rddl
        self.debug = debug
        jax.config.update('jax_log_compiles', self.debug)
        
        # static analysis and compilation
        self.static = RDDLLevelAnalysis(rddl, allow_synchronous_state)
        self.levels = self.static.compute_levels()
        self.tensors = RDDLTensors(rddl, debug)
        
        # initialize all fluent and non-fluent values
        self.init_values = self.tensors.init_values
        self.next_states = {var + '\'': var
                            for var, ftype in rddl.variable_types.items()
                            if ftype == 'state-fluent'}
        
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
            '|': jnp.logical_or,
            '~': jnp.logical_xor,
            '=>': lambda e1, e2: jnp.logical_or(jnp.logical_not(e1), e2),
            '<=>': jnp.equal
        }
        self.AGGREGATION_OPS = {
            'sum': jnp.sum,
            'avg': jnp.mean,
            'prod': jnp.prod,
            'min': jnp.min,
            'max': jnp.max,
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
            'sqrt': jnp.sqrt
        }        
        self.KNOWN_BINARY = {
            'min': jnp.minimum,
            'max': jnp.maximum,
            'pow': jnp.power
        }
        self.CONTROL_OPS = {
            'if': jnp.where
        }
        
    # ===========================================================================
    # main compilation subroutines
    # ===========================================================================
     
    def compile(self) -> None: 
        self.invariants = self._compile_constraints(self.rddl.invariants)
        self.preconditions = self._compile_constraints(self.rddl.preconditions)
        self.termination = self._compile_constraints(self.rddl.terminals)
        self.cpfs = self._compile_cpfs()
        self.reward = self._compile_reward()
    
    def _compile_constraints(self, constraints):
        return [self._jax(c, [], dtype=bool) for c in constraints]
        
    def _compile_cpfs(self):
        jax_cpfs = {}
        for _, cpfs in self.levels.items():
            for cpf in cpfs:
                objects, expr = self.rddl.cpfs[cpf]
                prange = self.rddl.variable_ranges[cpf]
                if prange not in JaxRDDLCompiler.JAX_TYPES:
                    raise RDDLTypeError(
                        f'Type <{prange}> of CPF <{cpf}> is not valid, '
                        f'must be one of {set(JaxRDDLCompiler.JAX_TYPES.keys())}.')
                dtype = JaxRDDLCompiler.JAX_TYPES[prange]
                jax_cpfs[cpf] = self._jax(expr, objects, dtype=dtype)
        return jax_cpfs
    
    def _compile_reward(self):
        return self._jax(self.rddl.reward, [], dtype=JaxRDDLCompiler.REAL)
    
    def compile_rollouts(self, policy, n_steps: int, n_batch: int,
                         check_constraints: bool=False):
        NORMAL = JaxRDDLCompiler.ERROR_CODES['NORMAL']
            
        # compute a batched version of the initial values
        init_subs = {}
        for name, value in self.init_values.items():
            init_subs[name] = np.broadcast_to(
                value, shape=(n_batch,) + np.shape(value))            
        for next_state, state in self.next_states.items():
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
                for i, precond in enumerate(self.preconditions):
                    sample, key, precond_err = precond(subs, key)
                    preconds = preconds.at[i].set(sample)
                    error |= precond_err
            
            # calculate CPFs in topological order and reward
            for name, cpf in self.cpfs.items():
                subs[name], key, cpf_err = cpf(subs, key)
                error |= cpf_err                
            reward, key, reward_err = self.reward(subs, key)
            error |= reward_err
            
            for next_state, state in self.next_states.items():
                subs[state] = subs[next_state]
            
            # check the invariants in the new state
            if check_constraints:
                invariants = jnp.zeros(shape=(len(self.invariants),), dtype=bool)
                for i, invariant in enumerate(self.invariants):
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
            
            carried = (subs, params, key)
            logged = {'fluent': subs,
                      'action': action,
                      'reward': reward,
                      'error': error}
            
            if check_constraints:
                logged['preconditions'] = preconds
                logged['invariants'] = invariants
                logged['terminated'] = terminated
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
        'INVALID_PARAM_GOMPERTZ': 32768      
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
        15: 'Found Gompertz(k, l) distribution where either k <= 0 or l <= 0.'
    }
    
    @staticmethod
    def get_error_codes(error):
        binary = reversed(bin(error)[2:])
        errors = [i for i, c in enumerate(binary) if c == '1']
        return errors
    
    @staticmethod
    def get_error_messages(error):
        codes = JaxRDDLCompiler.get_error_codes(error)
        messages = [JaxRDDLCompiler.INVERSE_ERROR_CODES[i] for i in codes]
        return messages
    
    # ===========================================================================
    # expression compilation
    # ===========================================================================
    
    def _jax(self, expr, objects, dtype=None):
        etype, _ = expr.etype
        if etype == 'constant':
            jax_expr = self._jax_constant(expr, objects)
        elif etype == 'pvar':
            jax_expr = self._jax_pvar(expr, objects)
        elif etype == 'arithmetic':
            jax_expr = self._jax_arithmetic(expr, objects)
        elif etype == 'relational':
            jax_expr = self._jax_relational(expr, objects)
        elif etype == 'boolean':
            jax_expr = self._jax_logical(expr, objects)
        elif etype == 'aggregation':
            jax_expr = self._jax_aggregation(expr, objects)
        elif etype == 'func':
            jax_expr = self._jax_functional(expr, objects)
        elif etype == 'control':
            jax_expr = self._jax_control(expr, objects)
        elif etype == 'randomvar':
            jax_expr = self._jax_random(expr, objects)
        else:
            raise RDDLNotImplementedError(
                f'Internal error: expression {expr} is not supported.\n' + 
                JaxRDDLCompiler._print_stack_trace(expr))
                
        if dtype is not None:
            jax_expr = self._jax_cast(jax_expr, dtype)
        
        return jax_expr
            
    def _jax_cast(self, jax_expr, dtype):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_CAST']
        
        def _f(x, key):
            val, key, err = jax_expr(x, key)
            sample = jnp.asarray(val, dtype=dtype)
            invalid = jnp.logical_not(jnp.can_cast(val, dtype, casting='safe'))
            err |= invalid * ERR
            return sample, key, err
        
        return _f
   
    # ===========================================================================
    # leaves
    # ===========================================================================
    
    def _jax_constant(self, expr, objects):
        ERR = JaxRDDLCompiler.ERROR_CODES['NORMAL']
        * _, shape = self.tensors.map(
            '', [], objects, 
            str(expr), JaxRDDLCompiler._print_stack_trace(expr))
        const = expr.args
        
        def _f(_, key):
            sample = jnp.full(shape=shape, fill_value=const)
            return sample, key, ERR

        return _f
    
    def _jax_pvar(self, expr, objects):
        ERR = JaxRDDLCompiler.ERROR_CODES['NORMAL']
        var, pvars = expr.args            
        permute, identity, new_dims = self.tensors.map(
            var, pvars, objects, 
            str(expr), JaxRDDLCompiler._print_stack_trace(expr))
        new_axes = (1,) * len(new_dims)
        
        def _f(x, key):
            val = jnp.asarray(x[var])
            sample = val
            if new_dims:
                sample = jnp.reshape(val, newshape=val.shape + new_axes) 
                sample = jnp.broadcast_to(sample, shape=val.shape + new_dims)
            if not identity:
                sample = jnp.einsum(permute, sample)
            return sample, key, ERR
        
        return _f
    
    # ===========================================================================
    # mathematical
    # ===========================================================================
    
    @staticmethod
    def _jax_unary(jax_expr, jax_op):
        
        def _f(x, key):
            val, key, err = jax_expr(x, key)
            sample = jax_op(val)
            return sample, key, err
        
        return _f
    
    @staticmethod
    def _jax_binary(jax_lhs, jax_rhs, jax_op):
        
        def _f(x, key):
            val1, key, err1 = jax_lhs(x, key)
            val2, key, err2 = jax_rhs(x, key)
            sample = jax_op(val1, val2)
            err = err1 | err2
            return sample, key, err
        
        return _f
        
    def _jax_arithmetic(self, expr, objects):
        _, op = expr.etype
        valid_ops = self.ARITHMETIC_OPS
        JaxRDDLCompiler._check_valid_op(expr, valid_ops)
                    
        args = expr.args
        n = len(args)
        
        if n == 1 and op == '-':
            arg, = args
            jax_expr = self._jax(arg, objects)
            return JaxRDDLCompiler._jax_unary(jax_expr, jnp.negative)
                    
        elif n == 2:
            lhs, rhs = args
            jax_lhs = self._jax(lhs, objects)
            jax_rhs = self._jax(rhs, objects)
            jax_op = valid_ops[op]
            return JaxRDDLCompiler._jax_binary(jax_lhs, jax_rhs, jax_op)
        
        JaxRDDLCompiler._check_num_args(expr, 2)
    
    def _jax_relational(self, expr, objects):
        _, op = expr.etype
        valid_ops = self.RELATIONAL_OPS
        JaxRDDLCompiler._check_valid_op(expr, valid_ops)
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        lhs, rhs = expr.args
        jax_lhs = self._jax(lhs, objects)
        jax_rhs = self._jax(rhs, objects)
        jax_op = valid_ops[op]
        return JaxRDDLCompiler._jax_binary(jax_lhs, jax_rhs, jax_op)
           
    def _jax_logical(self, expr, objects):
        _, op = expr.etype
        valid_ops = self.LOGICAL_OPS    
        JaxRDDLCompiler._check_valid_op(expr, valid_ops)
                
        args = expr.args
        n = len(args)
        
        if n == 1 and op == '~':
            arg, = args
            jax_expr = self._jax(arg, objects)
            return JaxRDDLCompiler._jax_unary(jax_expr, self.LOGICAL_NOT)
        
        elif n == 2:
            lhs, rhs = args
            jax_lhs = self._jax(lhs, objects)
            jax_rhs = self._jax(rhs, objects)
            jax_op = valid_ops[op]
            return JaxRDDLCompiler._jax_binary(jax_lhs, jax_rhs, jax_op)
        
        JaxRDDLCompiler._check_num_args(expr, 2)
    
    def _jax_aggregation(self, expr, objects):
        _, op = expr.etype
        valid_ops = self.AGGREGATION_OPS      
        JaxRDDLCompiler._check_valid_op(expr, valid_ops) 
        
        * pvars, arg = expr.args  
        new_objects = objects + [p[1] for p in pvars]
        axis = tuple(range(len(objects), len(new_objects)))
        
        fails = {p for _, p in new_objects if p not in self.rddl.objects}
        if fails:
            raise RDDLInvalidObjectError(
                f'Type(s) {fails} are not defined, '
                f'must be one of {set(self.rddl.objects.keys())}.\n' + 
                JaxRDDLCompiler._print_stack_trace(expr))
        
        jax_expr = self._jax(arg, new_objects)
        jax_op = valid_ops[op]        
        
        def _f(x, key):
            val, key, err = jax_expr(x, key)
            sample = jax_op(val, axis=axis)
            return sample, key, err
        
        if self.debug:
            warnings.warn(
                f'compiling static graph for aggregation:'
                f'\n\toperator       ={op} {pvars}'
                f'\n\tinput objects  ={new_objects}'
                f'\n\toutput objects ={objects}'
                f'\n\treduction axis ={axis}'
                f'\n\treduction op   ={valid_ops[op]}\n'
            )
                
        return _f
               
    def _jax_functional(self, expr, objects):
        _, op = expr.etype
        
        if op in self.KNOWN_UNARY:
            JaxRDDLCompiler._check_num_args(expr, 1)                            
            arg, = expr.args
            jax_expr = self._jax(arg, objects)
            jax_op = self.KNOWN_UNARY[op]
            return JaxRDDLCompiler._jax_unary(jax_expr, jax_op)
            
        elif op in self.KNOWN_BINARY:
            JaxRDDLCompiler._check_num_args(expr, 2)                
            lhs, rhs = expr.args
            jax_lhs = self._jax(lhs, objects)
            jax_rhs = self._jax(rhs, objects)
            jax_op = self.KNOWN_BINARY[op]
            return JaxRDDLCompiler._jax_binary(jax_lhs, jax_rhs, jax_op)
        
        raise RDDLNotImplementedError(
                f'Function {op} is not supported.\n' + 
                JaxRDDLCompiler._print_stack_trace(expr))   
    
    # ===========================================================================
    # control flow
    # ===========================================================================
    
    def _jax_control(self, expr, objects):
        _, op = expr.etype
        valid_ops = self.CONTROL_OPS
        JaxRDDLCompiler._check_valid_op(expr, valid_ops)
        JaxRDDLCompiler._check_num_args(expr, 3)
        
        pred, if_true, if_false = expr.args        
        jax_pred = self._jax(pred, objects)
        jax_true = self._jax(if_true, objects)
        jax_false = self._jax(if_false, objects)        
        jax_op = self.CONTROL_OPS[op]
        
        def _f(x, key):
            val1, key, err1 = jax_pred(x, key)
            val2, key, err2 = jax_true(x, key)
            val3, key, err3 = jax_false(x, key)
            sample = jax_op(val1, val2, val3)
            err = err1 | err2 | err3
            return sample, key, err
            
        return _f
    
    # ===========================================================================
    # random variables
    # ===========================================================================
    
    def _jax_random(self, expr, objects):
        _, name = expr.etype
        if name == 'KronDelta':
            return self._jax_kron(expr, objects)        
        elif name == 'DiracDelta':
            return self._jax_dirac(expr, objects)
        elif name == 'Uniform':
            return self._jax_uniform(expr, objects)
        elif name == 'Normal':
            return self._jax_normal(expr, objects)
        elif name == 'Exponential':
            return self._jax_exponential(expr, objects)
        elif name == 'Weibull':
            return self._jax_weibull(expr, objects)   
        elif name == 'Bernoulli':
            return self._jax_bernoulli(expr, objects)
        elif name == 'Poisson':
            return self._jax_poisson(expr, objects)
        elif name == 'Gamma':
            return self._jax_gamma(expr, objects)
        elif name == 'Beta':
            return self._jax_beta(expr, objects)
        elif name == 'Geometric':
            return self._jax_geometric(expr, objects)
        elif name == 'Pareto':
            return self._jax_pareto(expr, objects)
        elif name == 'Student':
            return self._jax_student(expr, objects)
        elif name == 'Gumbel':
            return self._jax_gumbel(expr, objects)
        elif name == 'Laplace':
            return self._jax_laplace(expr, objects)
        elif name == 'Cauchy':
            return self._jax_cauchy(expr, objects)
        elif name == 'Gompertz':
            return self._jax_gompertz(expr, objects)
        else:
            raise RDDLNotImplementedError(
                f'Distribution {name} is not supported.\n' + 
                JaxRDDLCompiler._print_stack_trace(expr))
        
    def _jax_kron(self, expr, objects):
        JaxRDDLCompiler._check_num_args(expr, 1)
        arg, = expr.args
        arg = self._jax(arg, objects, dtype=bool)
        return arg
    
    def _jax_dirac(self, expr, objects):
        JaxRDDLCompiler._check_num_args(expr, 1)
        arg, = expr.args
        arg = self._jax(arg, objects, dtype=JaxRDDLCompiler.REAL)
        return arg
    
    def _jax_uniform(self, expr, objects):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_UNIFORM']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_lb, arg_ub = expr.args
        jax_lb = self._jax(arg_lb, objects)
        jax_ub = self._jax(arg_ub, objects)
        
        def _f(x, key):
            lb, key, err1 = jax_lb(x, key)
            ub, key, err2 = jax_ub(x, key)
            key, subkey = random.split(key)
            U = random.uniform(
                key=subkey, shape=lb.shape, dtype=JaxRDDLCompiler.REAL)
            sample = lb + (ub - lb) * U
            out_of_bounds = jnp.any(lb > ub)
            err = err1 | err2 | out_of_bounds * ERR
            return sample, key, err
        
        return _f
    
    def _jax_normal(self, expr, objects):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_NORMAL']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_mean, arg_var = expr.args
        jax_mean = self._jax(arg_mean, objects)
        jax_var = self._jax(arg_var, objects)
        
        def _f(x, key):
            mean, key, err1 = jax_mean(x, key)
            var, key, err2 = jax_var(x, key)
            std = jnp.sqrt(var)
            key, subkey = random.split(key)
            Z = random.normal(
                key=subkey, shape=mean.shape, dtype=JaxRDDLCompiler.REAL)
            sample = mean + std * Z
            out_of_bounds = jnp.any(var < 0)
            err = err1 | err2 | out_of_bounds * ERR
            return sample, key, err
        
        return _f
    
    def _jax_exponential(self, expr, objects):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_EXPONENTIAL']
        JaxRDDLCompiler._check_num_args(expr, 1)
        
        arg_scale, = expr.args
        jax_scale = self._jax(arg_scale, objects)
                
        def _f(x, key):
            scale, key, err = jax_scale(x, key)
            key, subkey = random.split(key)
            Exp = random.exponential(
                key=subkey, shape=scale.shape, dtype=JaxRDDLCompiler.REAL)
            sample = scale * Exp
            out_of_bounds = jnp.any(scale <= 0)
            err |= out_of_bounds * ERR
            return sample, key, err
        
        return _f
    
    def _jax_weibull(self, expr, objects):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_WEIBULL']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_shape, arg_scale = expr.args
        jax_shape = self._jax(arg_shape, objects)
        jax_scale = self._jax(arg_scale, objects)
        
        def _f(x, key):
            shape, key, err1 = jax_shape(x, key)
            scale, key, err2 = jax_scale(x, key)
            key, subkey = random.split(key)
            U = random.uniform(
                key=subkey, shape=shape.shape, dtype=JaxRDDLCompiler.REAL)
            sample = scale * jnp.power(-jnp.log1p(-U), 1.0 / shape)
            out_of_bounds = jnp.any((shape <= 0) | (scale <= 0))
            err = err1 | err2 | out_of_bounds * ERR
            return sample, key, err
        
        return _f
            
    def _jax_bernoulli(self, expr, objects):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_BERNOULLI']
        JaxRDDLCompiler._check_num_args(expr, 1)
        
        arg_prob, = expr.args
        jax_prob = self._jax(arg_prob, objects)
        
        def _f(x, key):
            prob, key, err = jax_prob(x, key)
            key, subkey = random.split(key)
            U = random.uniform(
                key=subkey, shape=prob.shape, dtype=JaxRDDLCompiler.REAL)
            sample = U < prob
            out_of_bounds = jnp.any((prob < 0) | (prob > 1))
            err |= out_of_bounds * ERR
            return sample, key, err
        
        return _f
    
    def _jax_poisson(self, expr, objects):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_POISSON']
        JaxRDDLCompiler._check_num_args(expr, 1)
        
        arg_rate, = expr.args
        jax_rate = self._jax(arg_rate, objects)
        
        def _f(x, key):
            rate, key, err = jax_rate(x, key)
            key, subkey = random.split(key)
            sample = random.poisson(
                key=subkey, lam=rate, dtype=JaxRDDLCompiler.INT)
            out_of_bounds = jnp.any(rate < 0)
            err |= out_of_bounds * ERR
            return sample, key, err
        
        return _f
    
    def _jax_gamma(self, expr, objects):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_GAMMA']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_shape, arg_scale = expr.args
        jax_shape = self._jax(arg_shape, objects)
        jax_scale = self._jax(arg_scale, objects)
        
        def _f(x, key):
            shape, key, err1 = jax_shape(x, key)
            scale, key, err2 = jax_scale(x, key)
            key, subkey = random.split(key)
            Gamma = random.gamma(key=subkey, a=shape, dtype=JaxRDDLCompiler.REAL)
            sample = scale * Gamma
            out_of_bounds = jnp.any((shape <= 0) | (scale <= 0))
            err = err1 | err2 | out_of_bounds * ERR
            return sample, key, err
        
        return _f
    
    def _jax_beta(self, expr, objects):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_BETA']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_shape, arg_rate = expr.args
        jax_shape = self._jax(arg_shape, objects)
        jax_rate = self._jax(arg_rate, objects)
        
        def _f(x, key):
            shape, key, err1 = jax_shape(x, key)
            rate, key, err2 = jax_rate(x, key)
            key, subkey = random.split(key)
            sample = random.beta(key=subkey, a=shape, b=rate)
            out_of_bounds = jnp.any((shape <= 0) | (rate <= 0))
            err = err1 | err2 | out_of_bounds * ERR
            return sample, key, err
        
        return _f
    
    def _jax_geometric(self, expr, objects):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_GEOMETRIC']
        JaxRDDLCompiler._check_num_args(expr, 1)
        
        arg_prob, = expr.args
        jax_prob = self._jax(arg_prob, objects)
        
        def _f(x, key):
            prob, key, err = jax_prob(x, key)
            key, subkey = random.split(key)
            U = random.uniform(
                key=subkey, shape=prob.shape, dtype=JaxRDDLCompiler.REAL)
            sample = jnp.floor(jnp.log1p(-U) / jnp.log1p(-prob)) + 1
            out_of_bounds = jnp.any((prob < 0) | (prob > 1))
            err |= out_of_bounds * ERR
            return sample, key, err
        
        return _f
    
    def _jax_pareto(self, expr, objects):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_PARETO']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_shape, arg_scale = expr.args
        jax_shape = self._jax(arg_shape, objects)
        jax_scale = self._jax(arg_scale, objects)
        
        def _f(x, key):
            shape, key, err1 = jax_shape(x, key)
            scale, key, err2 = jax_scale(x, key)
            key, subkey = random.split(key)
            sample = scale * random.pareto(key=subkey, b=shape)
            out_of_bounds = jnp.any((shape <= 0) | (scale <= 0))
            err = err1 | err2 | out_of_bounds * ERR
            return sample, key, err
        
        return _f
    
    def _jax_student(self, expr, objects):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_STUDENT']
        JaxRDDLCompiler._check_num_args(expr, 1)
        
        arg_df, = expr.args
        jax_df = self._jax(arg_df, objects)
        
        def _f(x, key):
            df, key, err = jax_df(x, key)
            key, subkey = random.split(key)
            sample = random.t(key=subkey, df=df)
            out_of_bounds = jnp.any(df <= 0)
            err |= out_of_bounds * ERR
            return sample, key, err
        
        return _f
    
    def _jax_gumbel(self, expr, objects):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_GUMBEL']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_mean, arg_scale = expr.args
        jax_mean = self._jax(arg_mean, objects)
        jax_scale = self._jax(arg_scale, objects)
        
        def _f(x, key):
            mean, key, err1 = jax_mean(x, key)
            scale, key, err2 = jax_scale(x, key)
            key, subkey = random.split(key)
            Gumbel01 = random.gumbel(
                key=subkey, shape=mean.shape, dtype=JaxRDDLCompiler.REAL)
            sample = mean + scale * Gumbel01
            out_of_bounds = jnp.any(scale <= 0)
            err = err1 | err2 | out_of_bounds * ERR
            return sample, key, err
        
        return _f
    
    def _jax_laplace(self, expr, objects):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_LAPLACE']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_mean, arg_scale = expr.args
        jax_mean = self._jax(arg_mean, objects)
        jax_scale = self._jax(arg_scale, objects)
        
        def _f(x, key):
            mean, key, err1 = jax_mean(x, key)
            scale, key, err2 = jax_scale(x, key)
            key, subkey = random.split(key)
            Laplace01 = random.laplace(
                key=subkey, shape=mean.shape, dtype=JaxRDDLCompiler.REAL)
            sample = mean + scale * Laplace01
            out_of_bounds = jnp.any(scale <= 0)
            err = err1 | err2 | out_of_bounds * ERR
            return sample, key, err
        
        return _f
    
    def _jax_cauchy(self, expr, objects):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_CAUCHY']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_mean, arg_scale = expr.args
        jax_mean = self._jax(arg_mean, objects)
        jax_scale = self._jax(arg_scale, objects)
        
        def _f(x, key):
            mean, key, err1 = jax_mean(x, key)
            scale, key, err2 = jax_scale(x, key)
            key, subkey = random.split(key)
            Cauchy01 = random.cauchy(
                key=subkey, shape=mean.shape, dtype=JaxRDDLCompiler.REAL)
            sample = mean + scale * Cauchy01
            out_of_bounds = jnp.any(scale <= 0)
            err = err1 | err2 | out_of_bounds * ERR
            return sample, key, err
        
        return _f
    
    def _jax_gompertz(self, expr, objects):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_GOMPERTZ']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_shape, arg_scale = expr.args
        jax_shape = self._jax(arg_shape, objects)
        jax_scale = self._jax(arg_scale, objects)
        
        def _f(x, key):
            shape, key, err1 = jax_shape(x, key)
            scale, key, err2 = jax_scale(x, key)
            key, subkey = random.split(key)
            U = random.uniform(
                key=subkey, shape=shape.shape, dtype=JaxRDDLCompiler.REAL)
            sample = jnp.log(1.0 - jnp.log1p(-U) / shape) / scale
            out_of_bounds = jnp.any((shape <= 0) | (scale <= 0))
            err = err1 | err2 | out_of_bounds * ERR
            return sample, key, err
        
        return _f
    