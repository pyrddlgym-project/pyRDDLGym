import jax
import jax.numpy as jnp
import jax.random as random
import warnings

from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidNumberOfArgumentsError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLNotImplementedError
from pyRDDLGym.Core.Parser.expr import Expression
from pyRDDLGym.Core.Parser.rddl import RDDL


class JaxRDDLCompiler:
    
    INT = jnp.int32
    REAL = jnp.float32
    
    RDDL_TO_JAX_TYPE = {
        'int': INT,
        'real': REAL,
        'bool': bool
    }
    
    def __init__(self, rddl: RDDL, enforce_diff: bool=False) -> None:
        self.rddl = rddl
        self.domain = rddl.domain
        self.enforce_diff = enforce_diff
        
        # extract object types and their objects
        self.objects = {}
        for obj, values in rddl.non_fluents.objects:
            self.objects[obj] = dict(zip(values, range(len(values))))
        
        # extract the parameters required for each pvariable
        self.pvars, self.states, self.pvar_ranges = {}, {}, {}
        for pvar in rddl.domain.pvariables:
            name = pvar.name
            if pvar.fluent_type == 'state-fluent':
                name = name + '\''
                self.states[name] = pvar.name
            self.pvars[name] = [] if pvar.param_types is None else pvar.param_types
            
            if self.enforce_diff:
                self.pvar_ranges[name] = JaxRDDLCompiler.REAL
                if pvar.range != 'real':
                    warnings.warn(
                        'Pvariable <{}> of type {} will be promoted to real.'.format(
                            name, pvar.range), FutureWarning, stacklevel=2)
            else:
                self.pvar_ranges[name] = JaxRDDLCompiler.RDDL_TO_JAX_TYPE[pvar.range]
            
    # start of compilation subroutines of RDDL programs 
    def compile(self) -> None:
        result = {'invariants': self._compile_invariants(),
                  'preconds': self._compile_preconditions(),
                  'terminals': self._compile_termination(),
                  'reward': self._compile_reward(),
                  'cpfs': self._compile_cpfs()}
        return result
        
    def _compile_invariants(self):
        jax_invariants = [JaxRDDLCompiler._jax_cast(self._jax(expr, []), bool)
                          for expr in self.domain.invariants]
        return jax.tree_map(jax.jit, jax_invariants)
        
    def _compile_preconditions(self):
        jax_preconds = [JaxRDDLCompiler._jax_cast(self._jax(expr, []), bool)
                        for expr in self.domain.preconds]
        return jax.tree_map(jax.jit, jax_preconds)
    
    def _compile_termination(self):
        jax_terminals = [JaxRDDLCompiler._jax_cast(self._jax(expr, []), bool)
                         for expr in self.domain.terminals]
        return jax.tree_map(jax.jit, jax_terminals)
    
    def _compile_cpfs(self):
        jax_cpfs = {}  
        for cpf in self.domain.cpfs[1]:
            expr = cpf.expr
            _, (name, params) = cpf.pvar
            if params is None:
                params = [] 
            pvar_inputs = self.pvars[name]
            params = [(p, pvar_inputs[i]) for i, p in enumerate(params)]
            jax_cpfs[name] = JaxRDDLCompiler._jax_cast(
                self._jax(expr, params), self.pvar_ranges[name])
        jit_cpfs = jax.tree_map(jax.jit, jax_cpfs)        
        return jax_cpfs, jit_cpfs
    
    def _compile_reward(self):
        jax_reward = JaxRDDLCompiler._jax_cast(
            self._jax(self.domain.reward, []), JaxRDDLCompiler.REAL)
        jit_reward = jax.jit(jax_reward)
        return jax_reward, jit_reward
    
    # start of compilation subroutines for expressions
    @staticmethod
    def _print_stack_trace(expr):
        return '...\n' + str(expr) + '\n...'

    ERROR_CODES = {
        'NORMAL': 0,
        'INVALID_CAST': 1,
        'INVALID_PARAM_UNIFORM': 2,
        'INVALID_PARAM_NORMAL': 4,
        'INVALID_PARAM_EXPONENTIAL': 8,
        'INVALID_PARAM_WEIBULL': 16,
        'INVALID_PARAM_BERNOULLI': 32,
        'INVALID_PARAM_POISSON': 64,
        'INVALID_PARAM_GAMMA': 128
    }
    
    INVERSE_ERROR_CODES = {
        0: 'Possible loss of precision during casting to a weaker type (e.g., real to int).',
        1: 'Found Uniform(a, b) distribution where a > b.',
        2: 'Found Normal(m, v^2) distribution where v < 0.',
        3: 'Found Exponential(s) distribution where s < 0.',
        4: 'Found Weibull(k, l) distribution where either k < 0 or l < 0.',
        5: 'Found Bernoulli(p) distribution where p < 0.',
        6: 'Found Poisson(l) distribution where l < 0.',
        7: 'Found Gamma(k, l) distribution where either k < 0 or l < 0.'
    }
    
    @staticmethod
    def get_error_codes(error):
        binary = reversed(bin(error)[2:])
        errors = [i for i, c in enumerate(binary) if c == '1']
        return errors
        
    def _jax(self, expr, params):
        if isinstance(expr, Expression):
            etype, op = expr.etype
            if etype == 'constant':
                return self._jax_constant(expr, params)
            elif etype == 'pvar':
                return self._jax_pvar(expr, params)
            elif etype == 'arithmetic':
                return self._jax_arithmetic(expr, op, params)
            elif etype == 'relational':
                return self._jax_relational(expr, op, params)
            elif etype == 'boolean':
                return self._jax_logical(expr, op, params)
            elif etype == 'aggregation':
                return self._jax_aggregation(expr, op, params)
            elif etype == 'func':
                return self._jax_functional(expr, op, params)
            elif etype == 'control':
                return self._jax_control(expr, op, params)
            elif etype == 'randomvar':
                return self._jax_random(expr, op, params)
            else:
                raise RDDLNotImplementedError(
                    'Internal error: expression type {} is not supported.'.format(etype) + 
                    '\n' + JaxRDDLCompiler._print_stack_trace(expr))
        else:
            raise RDDLNotImplementedError(
                'Internal error: type {} is not supported.'.format(expr) + 
                '\n' + JaxRDDLCompiler._print_stack_trace(expr))
            
    @staticmethod
    def _jax_cast(jax_expr, dtype):
        ERR_CODE = JaxRDDLCompiler.ERROR_CODES['INVALID_CAST']
        
        def _f(x, key):
            val, key, err = jax_expr(x, key)
            sample = jnp.asarray(val, dtype=dtype)
            invalid_cast = jnp.logical_not(jnp.can_cast(val, dtype, casting='safe'))
            err |= invalid_cast * ERR_CODE
            return sample, key, err
        
        return _f
   
    # leaves
    def _map_pvar_subs_to_subscript(self, given_params, desired_params, expr):
        symbols = 'abcdefghijklmnopqrstuvwxyz'   
        symbols = symbols + symbols.upper()   
        
        # check that number of parameters is valid
        if len(desired_params) > len(symbols):
            raise RDDLNotImplementedError(
                'Variable <{}> is {}-D, but current version supports up to {}.'.format(
                    expr.args[0], len(desired_params), len(symbols)) + 
                '\n' + JaxRDDLCompiler._print_stack_trace(expr))
        
        # compute a mapping permutation(a,b,c...) -> (a,b,c...) that performs the
        # correct variable substitution
        lhs = [None] * len(given_params)
        new_dims = []
        for i_desired, (param_desired, obj_desired) in enumerate(desired_params):
            add_new_param = True
            for i_given, param_given in enumerate(given_params):
                if param_given == param_desired:
                    lhs[i_given] = symbols[i_desired]
                    add_new_param = False
            if add_new_param:
                lhs.append(symbols[i_desired])
                new_dim = len(self.objects[obj_desired])
                new_dims.append(new_dim)
        rhs = symbols[:len(desired_params)]
        
        # safeguard against any remaining free variables
        free_vars = [given_params[i] for i, p in enumerate(lhs) if p is None]
        if free_vars:
            raise RDDLInvalidNumberOfArgumentsError(
                'Variable <{}> contains free parameter(s) {}.'.format(
                    expr.args[0], free_vars) + 
                '\n' + JaxRDDLCompiler._print_stack_trace(expr))
        
        lhs = ''.join(lhs)
        subscripts = lhs + ' -> ' + rhs
        id_map = lhs == rhs
        new_dims = tuple(new_dims)
        
        return subscripts, id_map, new_dims
    
    def _jax_constant(self, expr, params):
        const = expr.args
        
        subscripts, id_map, new_dims = self._map_pvar_subs_to_subscript([], params, expr)
        new_axes = (1,) * len(new_dims)
        
        ERR_CODE = JaxRDDLCompiler.ERROR_CODES['NORMAL']
        
        def _f(_, key):
            val = jnp.asarray(const)
            sample = val
            if new_dims:
                sample = jnp.reshape(val, newshape=val.shape + new_axes) 
                sample = jnp.broadcast_to(sample, shape=val.shape + new_dims)
            if not id_map:
                sample = jnp.einsum(subscripts, sample)
            return sample, key, ERR_CODE

        return _f
    
    def _jax_pvar(self, expr, params):
        var, pvars = expr.args
        pvars = [] if pvars is None else pvars
        
        len_pvars = len(pvars)
        len_req = len(self.pvars.get(var, []))
        if len_pvars < len_req:
            raise RDDLInvalidNumberOfArgumentsError(
                'Variable <{}> requires at least {} parameter arguments, got {}.'.format(
                    var, len_req, len_pvars) + 
                '\n' + JaxRDDLCompiler._print_stack_trace(expr))
            
        subscripts, id_map, new_dims = self._map_pvar_subs_to_subscript(pvars, params, expr)
        new_axes = (1,) * len(new_dims)
        
        ERR_CODE = JaxRDDLCompiler.ERROR_CODES['NORMAL']
        
        def _f(x, key):
            val = jnp.asarray(x[var])
            sample = val
            if new_dims:
                sample = jnp.reshape(val, newshape=val.shape + new_axes) 
                sample = jnp.broadcast_to(sample, shape=val.shape + new_dims)
            if not id_map:
                sample = jnp.einsum(subscripts, sample)
            return sample, key, ERR_CODE
        
        return _f
    
    # arithmetic expressions
    ARITHMETIC_OPS = {
        '+': jnp.add,
        '-': jnp.subtract,
        '*': jnp.multiply,
        '/': jnp.divide
    }
    
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
        
    def _jax_arithmetic(self, expr, op, params):
        if op not in JaxRDDLCompiler.ARITHMETIC_OPS:
            raise RDDLNotImplementedError(
                'Arithmetic operator {} is not supported: must be one of {}.'.format(
                    op, JaxRDDLCompiler.ARITHMETIC_OPS.keys()) + 
                '\n' + JaxRDDLCompiler._print_stack_trace(expr))
            
        args = expr.args
        n = len(args)
        
        if n == 1 and op == '-':
            arg, = args
            jax_expr = self._jax(arg, params)
            return JaxRDDLCompiler._jax_unary(jax_expr, jnp.negative)
                    
        elif n == 2:
            lhs, rhs = args
            jax_lhs = self._jax(lhs, params)
            jax_rhs = self._jax(rhs, params)
            jax_op = JaxRDDLCompiler.ARITHMETIC_OPS[op]
            return JaxRDDLCompiler._jax_binary(jax_lhs, jax_rhs, jax_op)
        
        raise RDDLInvalidNumberOfArgumentsError(
            'Arithmetic operator {} does not have the required number of args, got {}.'.format(
                op, n) + 
            '\n' + JaxRDDLCompiler._print_stack_trace(expr))
    
    # relational expressions
    RELATIONAL_OPS = {
        '>=': jnp.greater_equal,
        '<=': jnp.less_equal,
        '<': jnp.less,
        '>': jnp.greater,
        '==': jnp.equal,
        '~=': jnp.not_equal
    }
    
    def _jax_relational(self, expr, op, params):
        if op not in JaxRDDLCompiler.RELATIONAL_OPS:
            raise RDDLNotImplementedError(
                'Relational operator {} is not supported: must be one of {}.'.format(
                    op, JaxRDDLCompiler.RELATIONAL_OPS.keys()) + 
                '\n' + JaxRDDLCompiler._print_stack_trace(expr))
            
        args = expr.args
        n = len(args)
        
        if n != 2:
            raise RDDLInvalidNumberOfArgumentsError(
                'Relational operator {} requires 2 args, got {}.'.format(op, n) + 
                '\n' + JaxRDDLCompiler._print_stack_trace(expr))
        
        lhs, rhs = args
        jax_lhs = self._jax(lhs, params)
        jax_rhs = self._jax(rhs, params)
        jax_op = JaxRDDLCompiler.RELATIONAL_OPS[op]
        return JaxRDDLCompiler._jax_binary(jax_lhs, jax_rhs, jax_op)
        
    # logical expressions    
    LOGICAL_OPS = {
        False: {
            '^': jnp.logical_and,
            '|': jnp.logical_or,
            '~': jnp.logical_xor,
            '=>': lambda e1, e2: jnp.logical_or(jnp.logical_not(e1), e2),
            '<=>': jnp.equal
        },
        True: {
            '^': jnp.minimum,
            '|': jnp.maximum
        }
    }
        
    def _jax_logical(self, expr, op, params):
        OPS = JaxRDDLCompiler.LOGICAL_OPS[self.enforce_diff]        
        if op not in OPS:
            raise RDDLNotImplementedError(
                'Logical operator {} is not supported: must be one of {}.'.format(op, OPS) + 
                '\n' + JaxRDDLCompiler._print_stack_trace(expr))       
        
        args = expr.args
        n = len(args)
        
        if n == 1 and op == '~':
            arg, = args
            jax_expr = self._jax(arg, params)
            return JaxRDDLCompiler._jax_unary(jax_expr, jnp.logical_not)
        
        elif n == 2:
            lhs, rhs = args
            jax_lhs = self._jax(lhs, params)
            jax_rhs = self._jax(rhs, params)
            jax_op = OPS[op]
            return JaxRDDLCompiler._jax_binary(jax_lhs, jax_rhs, jax_op)
        
        raise RDDLInvalidNumberOfArgumentsError(
            'Logical operator {} does not have the required number of args, got {}.'.format(
                op, n) + 
            '\n' + JaxRDDLCompiler._print_stack_trace(expr))
    
    # aggregations
    AGGREGATION_OPS = {
        False: {
            'sum': jnp.sum,
            'avg': jnp.mean,
            'prod': jnp.prod,
            'min': jnp.min,
            'max': jnp.max,
            'forall': jnp.all,
            'exists': jnp.any
        },
        True: {
            'sum': jnp.sum,
            'avg': jnp.mean,
            'prod': jnp.prod,
            'min': jnp.min,
            'max': jnp.max,
            'forall': jnp.min,
            'exists': jnp.max
        }
    }
    
    def _jax_aggregation(self, expr, op, params):
        OPS = JaxRDDLCompiler.AGGREGATION_OPS[self.enforce_diff]            
        if op not in OPS:
            raise RDDLNotImplementedError(
                'Aggregation operator {} is not supported: must be one of {}.'.format(
                    op, OPS.keys()) + 
                '\n' + JaxRDDLCompiler._print_stack_trace(expr))
            
        * pvars, arg = expr.args
        pvars = list(map(lambda p: p[1], pvars))        
        new_params = params + pvars
        reduced_axes = tuple(range(len(params), len(new_params)))
        
        jax_expr = self._jax(arg, new_params)
        jax_op = OPS[op]        
        
        def _f(x, key):
            val, key, err = jax_expr(x, key)
            sample = jax_op(val, axis=reduced_axes)
            return sample, key, err
        
        return _f
                 
    # functions
    KNOWN_UNARY = {        
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
    
    KNOWN_BINARY = {
        'min': jnp.minimum,
        'max': jnp.maximum,
        'pow': jnp.power
    }
    
    def _jax_functional(self, expr, op, params):
        n = len(expr.args)
        
        if op in JaxRDDLCompiler.KNOWN_UNARY:
            if n != 1:
                raise RDDLInvalidNumberOfArgumentsError(
                    'Unary function {} requires 1 arg, got {}.'.format(op, n) + 
                    '\n' + JaxRDDLCompiler._print_stack_trace(expr))    
                            
            arg, = expr.args
            jax_expr = self._jax(arg, params)
            jax_op = JaxRDDLCompiler.KNOWN_UNARY[op]
            return JaxRDDLCompiler._jax_unary(jax_expr, jax_op)
            
        elif op in JaxRDDLCompiler.KNOWN_BINARY:
            if n != 2:
                raise RDDLInvalidNumberOfArgumentsError(
                    'Binary function {} requires 2 args, got {}.'.format(op, n) + 
                    '\n' + JaxRDDLCompiler._print_stack_trace(expr))  
                
            lhs, rhs = expr.args
            jax_lhs = self._jax(lhs, params)
            jax_rhs = self._jax(rhs, params)
            jax_op = JaxRDDLCompiler.KNOWN_BINARY[op]
            return JaxRDDLCompiler._jax_binary(jax_lhs, jax_rhs, jax_op)
        
        raise RDDLNotImplementedError(
                'Function {} is not supported.'.format(op) + 
                '\n' + JaxRDDLCompiler._print_stack_trace(expr))   
    
    # control flow
    def _jax_control(self, expr, op, params):
        if op not in {'if'}:
            raise RDDLNotImplementedError(
                'Control flow type <{}> is not supported: must be one of {}.'.format(
                    op, {'if'}) + 
                '\n' + JaxRDDLCompiler._print_stack_trace(expr))
            
        pred, if_true, if_false = expr.args
        
        jax_pred = self._jax(pred, params)
        jax_true = self._jax(if_true, params)
        jax_false = self._jax(if_false, params)
        
        def _f(x, key):
            val1, key, err1 = jax_pred(x, key)
            val2, key, err2 = jax_true(x, key)
            val3, key, err3 = jax_false(x, key)
            sample = jnp.where(val1, val2, val3)
            err = err1 | err2 | err3
            return sample, key, err
            
        return _f
    
    # random variables   
    # TODO: try these for Poisson and Gamma
    # https://arxiv.org/pdf/1611.01144.pdf?
    # https://arxiv.org/pdf/2003.01847.pdf?     
    def _jax_random(self, expr, name, params):
        if name == 'KronDelta':
            return self._jax_kron(expr, params)        
        elif name == 'DiracDelta':
            return self._jax_dirac(expr, params)
        elif name == 'Uniform':
            return self._jax_uniform(expr, params)
        elif name == 'Normal':
            return self._jax_normal(expr, params)
        elif name == 'Exponential':
            return self._jax_exponential(expr, params)
        elif name == 'Weibull':
            return self._jax_weibull(expr, params)   
        elif name == 'Bernoulli':
            return self._jax_bernoulli(expr, params)
        elif name == 'Poisson':
            return self._jax_poisson(expr, params)
        elif name == 'Gamma':
            return self._jax_gamma(expr, params)
        else:
            raise RDDLNotImplementedError(
                'Distribution {} is not supported.'.format(name) + 
                '\n' + JaxRDDLCompiler._print_stack_trace(expr))
        
    def _jax_kron(self, expr, params):
        arg, = expr.args
        arg = self._jax(arg, params)
        if self.enforce_diff:
            warnings.warn('KronDelta will be ignored.')
        else:
            arg = JaxRDDLCompiler._jax_cast(arg, bool)
        return arg
    
    def _jax_dirac(self, expr, params):
        arg, = expr.args
        arg = self._jax(arg, params)
        arg = JaxRDDLCompiler._jax_cast(arg, JaxRDDLCompiler.REAL)
        return arg
    
    def _jax_uniform(self, expr, params):
        expr_lb, expr_ub = expr.args
        jax_lb = self._jax(expr_lb, params)
        jax_ub = self._jax(expr_ub, params)
        
        ERR_CODE = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_UNIFORM']
        
        # U(a, b) = a + (b - a) * xi, where xi ~ U(0, 1)
        def _f(x, key):
            lb, key, err1 = jax_lb(x, key)
            ub, key, err2 = jax_ub(x, key)
            key, subkey = random.split(key)
            U = random.uniform(key=subkey, shape=lb.shape, dtype=JaxRDDLCompiler.REAL)
            sample = lb + (ub - lb) * U
            out_of_bounds = jnp.any(lb > ub)
            err = err1 | err2 | out_of_bounds * ERR_CODE
            return sample, key, err
        
        return _f
    
    def _jax_normal(self, expr, params):
        expr_mean, expr_var = expr.args
        jax_mean = self._jax(expr_mean, params)
        jax_var = self._jax(expr_var, params)
        
        ERR_CODE = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_NORMAL']
        
        # N(mu, s^2) = mu + s * N(0, 1)
        def _f(x, key):
            mean, key, err1 = jax_mean(x, key)
            var, key, err2 = jax_var(x, key)
            std = jnp.sqrt(var)
            key, subkey = random.split(key)
            Z = random.normal(key=subkey, shape=mean.shape, dtype=JaxRDDLCompiler.REAL)
            sample = mean + std * Z
            out_of_bounds = jnp.any(var < 0)
            err = err1 | err2 | out_of_bounds * ERR_CODE
            return sample, key, err
        
        return _f
    
    def _jax_exponential(self, expr, params):
        expr_scale, = expr.args
        jax_scale = self._jax(expr_scale, params)
        
        ERR_CODE = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_EXPONENTIAL']
        
        # Exp(scale) = scale * Exp(1)
        def _f(x, key):
            scale, key, err = jax_scale(x, key)
            key, subkey = random.split(key)
            Exp1 = random.exponential(
                key=subkey, shape=scale.shape, dtype=JaxRDDLCompiler.REAL)
            sample = scale * Exp1
            out_of_bounds = jnp.any(scale < 0)
            err |= out_of_bounds * ERR_CODE
            return sample, key, err
        
        return _f
    
    def _jax_weibull(self, expr, params):
        expr_shape, expr_scale = expr.args
        jax_shape = self._jax(expr_shape, params)
        jax_scale = self._jax(expr_scale, params)
        
        ERR_CODE = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_WEIBULL']
        
        # W(shape, scale) = scale * (-log(1 - U(0, 1)))^{1 / shape}
        # TODO: make this numerically stable
        def _f(x, key):
            shape, key, err1 = jax_shape(x, key)
            scale, key, err2 = jax_scale(x, key)
            key, subkey = random.split(key)
            U = random.uniform(key=subkey, shape=shape.shape, dtype=JaxRDDLCompiler.REAL)
            sample = scale * jnp.power(-jnp.log(1.0 - U), 1.0 / shape)
            out_of_bounds = jnp.any((shape < 0) | (scale < 0))
            err = err1 | err2 | out_of_bounds * ERR_CODE
            return sample, key, err
        
        return _f
            
    def _jax_bernoulli(self, expr, params):
        expr_prob, = expr.args
        jax_prob = self._jax(expr_prob, params)
        
        ERR_CODE = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_BERNOULLI']
        
        # Bernoulli(p) = 1[U(0, 1) < p]
        def _f(x, key):
            prob, key, err = jax_prob(x, key)
            key, subkey = random.split(key)
            U = random.uniform(key=subkey, shape=prob.shape, dtype=JaxRDDLCompiler.REAL)
            sample = jnp.less(U, prob)
            out_of_bounds = jnp.any((prob < 0) | (prob > 1))
            err |= out_of_bounds * ERR_CODE
            return sample, key, err
        
        return _f
    
    def _jax_poisson(self, expr, params):
        
        # no reparameterization so far
        if self.enforce_diff:
            raise RDDLNotImplementedError(
                'No reparameterization is implemented for Poisson.' + 
                '\n' + JaxRDDLCompiler._print_stack_trace(expr))
            
        expr_rate, = expr.args
        jax_rate = self._jax(expr_rate, params)
        
        ERR_CODE = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_POISSON']
        
        def _f(x, key):
            rate, key, err = jax_rate(x, key)
            key, subkey = random.split(key)
            sample = random.poisson(key=subkey, lam=rate, dtype=JaxRDDLCompiler.INT)
            out_of_bounds = jnp.any(rate < 0)
            err |= out_of_bounds * ERR_CODE
            return sample, key, err
        
        return _f
    
    def _jax_gamma(self, expr, params):
        
        # no reparameterization so far
        if self.enforce_diff:
            raise RDDLNotImplementedError(
                'No reparameterization is implemented for Gamma.' + 
                '\n' + JaxRDDLCompiler._print_stack_trace(expr))
            
        expr_shape, expr_scale = expr.args
        jax_shape = self._jax(expr_shape, params)
        jax_scale = self._jax(expr_scale, params)
        
        ERR_CODE = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_GAMMA']
        
        def _f(x, key):
            shape, key, err1 = jax_shape(x, key)
            scale, key, err2 = jax_scale(x, key)
            key, subkey = random.split(key)
            Gamma1 = random.gamma(key=subkey, a=shape, dtype=JaxRDDLCompiler.REAL)
            sample = scale * Gamma1
            out_of_bounds = jnp.any((shape < 0) | (scale < 0))
            err = err1 | err2 | out_of_bounds * ERR_CODE
            return sample, key, err
        
        return _f
            
