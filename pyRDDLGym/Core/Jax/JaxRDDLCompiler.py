import jax
import jax.numpy as jnp
import jax.random as random

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
    
    def __init__(self, rddl: RDDL) -> None:
        self.rddl = rddl
        self.domain = rddl.domain
        
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
        jax_invariants = [self._jax(expr, [], bool) for expr in self.domain.invariants]
        return jax.tree_map(jax.jit, jax_invariants)
        
    def _compile_preconditions(self):
        jax_preconds = [self._jax(expr, [], bool) for expr in self.domain.preconds]
        return jax.tree_map(jax.jit, jax_preconds)
    
    def _compile_termination(self):
        jax_terminals = [self._jax(expr, [], bool) for expr in self.domain.terminals]
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
            jax_cpfs[name] = self._jax(expr, params, self.pvar_ranges[name])
        jit_cpfs = jax.tree_map(jax.jit, jax_cpfs)        
        return jax_cpfs, jit_cpfs
    
    def _compile_reward(self):
        jax_reward = self._jax(self.domain.reward, [], JaxRDDLCompiler.REAL)
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
        'INVALID_PARAM_POISSON' : 64,
        'INVALID_PARAM_GAMMA' : 128
    }
    
    INVERSE_ERROR_CODES = {
        0: 'Type coercion failed: result in intermediate calculations has been truncated.',
        1: 'Found Uniform(a, b) distribution where a > b.',
        2: 'Found Normal(m, v^2) distribution where v < 0.',
        3: 'Found Exponential(s) distribution where s < 0.',
        4: 'Found Weibull(k, l) distribution where either k < 0 or l < 0.',
        5: 'Found Bernoulli(p) distribution where p < 0.',
        6: 'Found Poisson(l) distribution where l < 0.',
        7: 'Found Gamma(k, l) distribution where either k < 0 or l < 0.'
    }
    
    def _jax(self, expr, params, dtype):
        if isinstance(expr, Expression):
            etype, op = expr.etype
            if etype == 'constant':
                return self._jax_constant(expr, params, dtype)
            elif etype == 'pvar':
                return self._jax_pvar(expr, params, dtype)
            elif etype == 'arithmetic':
                return self._jax_arithmetic(expr, op, params, dtype)
            elif etype == 'relational':
                return self._jax_relational(expr, op, params, dtype)
            elif etype == 'boolean':
                return self._jax_logical(expr, op, params, dtype)
            elif etype == 'aggregation':
                return self._jax_aggregation(expr, op, params, dtype)
            elif etype == 'func':
                return self._jax_functional(expr, op, params, dtype)
            elif etype == 'control':
                return self._jax_control(expr, op, params, dtype)
            elif etype == 'randomvar':
                return self._jax_random(expr, op, params, dtype)
            else:
                raise RDDLNotImplementedError(
                    'Internal error: expression type {} is not supported.'.format(etype) + 
                    '\n' + JaxRDDLCompiler._print_stack_trace(expr))
        else:
            raise RDDLNotImplementedError(
                'Internal error: type {} is not supported.'.format(expr) + 
                '\n' + JaxRDDLCompiler._print_stack_trace(expr))
            
    # leaves
    def _map_pvar_subs_to_subscript(self, given_params, desired_params, expr):
        symbols = 'abcdefghijklmnopqrstuvwxyz'   
        symbols = symbols + symbols.upper()   
             
        if len(desired_params) > len(symbols):
            raise RDDLNotImplementedError(
                'Variable <{}> is {}-D, but current version supports up to {}.'.format(
                    expr.args[0], len(desired_params), len(symbols)) + 
                '\n' + JaxRDDLCompiler._print_stack_trace(expr))
            
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
    
    def _jax_constant(self, expr, params, dtype):
        const = expr.args
        subscripts, id_map, new_dims = self._map_pvar_subs_to_subscript([], params, expr)
        new_axes = (1,) * len(new_dims)
        ERR_CODE = JaxRDDLCompiler.ERROR_CODES['INVALID_CAST']
        
        if id_map:  # just cast - no shaping required
            
            def _f(x, key):
                err = (not jnp.can_cast(const, dtype, casting='safe')) * ERR_CODE
                sample = jnp.asarray(const, dtype=dtype)
                return sample, key, err
            
        else:  # must cast and shape array
                
            def _f(x, key):
                err = (not jnp.can_cast(const, dtype, casting='safe')) * ERR_CODE
                const_x = jnp.asarray(const, dtype=dtype)
                sample = jnp.reshape(const_x, newshape=const_x.shape + new_axes) 
                sample = jnp.broadcast_to(sample, shape=const_x.shape + new_dims)
                sample = jnp.einsum(subscripts, sample)
                return sample, key, err

        return _f
    
    def _jax_pvar(self, expr, params, dtype):
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
        ERR_CODE = JaxRDDLCompiler.ERROR_CODES['INVALID_CAST']
        
        if id_map:  # just cast - no shaping required
            
            def _f(x, key):
                var_x = x[var]
                err = (not jnp.can_cast(var_x, dtype, casting='safe')) * ERR_CODE
                sample = jnp.asarray(var_x, dtype=dtype)
                return sample, key, err
                
        else:  # must cast and shape array
            
            def _f(x, key):
                var_x = x[var]
                err = (not jnp.can_cast(var_x, dtype, casting='safe')) * ERR_CODE
                var_x = jnp.asarray(var_x, dtype=dtype)
                sample = jnp.reshape(var_x, newshape=var_x.shape + new_axes) 
                sample = jnp.broadcast_to(sample, shape=var_x.shape + new_dims)
                sample = jnp.einsum(subscripts, sample)
                return sample, key, err
        
        return _f

    # arithmetic expressions
    ARITHMETIC_OPS = {
        '+': jnp.add,
        '-': jnp.subtract,
        '*': jnp.multiply,
        '/': jnp.divide
    }
    
    @staticmethod
    def _jax_unary(jax_expr, jax_op, dtype):
        ERR_CODE = JaxRDDLCompiler.ERROR_CODES['INVALID_CAST']
        
        def _f(x, key):
            val_x, key, err_x = jax_expr(x, key)
            sample = jax_op(val_x)
            err = err_x | (not jnp.can_cast(sample, dtype, casting='safe')) * ERR_CODE
            sample = jnp.asarray(sample, dtype=dtype)
            return sample, key, err
        
        return _f
    
    @staticmethod
    def _jax_binary(jax_lhs, jax_rhs, jax_op, dtype):
        ERR_CODE = JaxRDDLCompiler.ERROR_CODES['INVALID_CAST']
        
        def _f(x, key):
            val1, key, err1_x = jax_lhs(x, key)
            val2, key, err2_x = jax_rhs(x, key)
            sample = jax_op(val1, val2)
            err = err1_x | err2_x | (not jnp.can_cast(sample, dtype, casting='safe')) * ERR_CODE
            sample = jnp.asarray(sample, dtype=dtype)
            return sample, key, err
        
        return _f
        
    def _jax_arithmetic(self, expr, op, params, dtype):
        if op not in JaxRDDLCompiler.ARITHMETIC_OPS:
            raise RDDLNotImplementedError(
                'Arithmetic operator {} is not supported: must be one of {}.'.format(
                    op, JaxRDDLCompiler.ARITHMETIC_OPS.keys()) + 
                '\n' + JaxRDDLCompiler._print_stack_trace(expr))
            
        args = expr.args
        n = len(args)
        
        if n == 1 and op == '-':
            arg, = args
            jax_expr = self._jax(arg, params, dtype)
            return JaxRDDLCompiler._jax_unary(jax_expr, jnp.negative, dtype)
                    
        elif n == 2:
            lhs, rhs = args
            jax_lhs = self._jax(lhs, params, dtype)
            jax_rhs = self._jax(rhs, params, dtype)
            jax_op = JaxRDDLCompiler.ARITHMETIC_OPS[op]
            return JaxRDDLCompiler._jax_binary(jax_lhs, jax_rhs, jax_op, dtype)
        
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
    
    def _jax_relational(self, expr, op, params, dtype):
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
        jax_lhs = self._jax(lhs, params, JaxRDDLCompiler.REAL)
        jax_rhs = self._jax(rhs, params, JaxRDDLCompiler.REAL)
        jax_op = JaxRDDLCompiler.RELATIONAL_OPS[op]
        return JaxRDDLCompiler._jax_binary(jax_lhs, jax_rhs, jax_op, dtype)
        
    # logical expressions    
    LOGICAL_OPS = {
        '^': jnp.logical_and,
        '|': jnp.logical_or,
        '~': jnp.logical_xor,
        '=>': lambda e1, e2: jnp.logical_or(jnp.logical_not(e1), e2),
        '<=>': jnp.equal
    }
    
    def _jax_logical(self, expr, op, params, dtype):
        if op not in JaxRDDLCompiler.LOGICAL_OPS:
            raise RDDLNotImplementedError(
                'Logical operator {} is not supported: must be one of {}.'.format(
                    op, JaxRDDLCompiler.LOGICAL_OPS) + 
                '\n' + JaxRDDLCompiler._print_stack_trace(expr))       
        
        args = expr.args
        n = len(args)
        
        if n == 1 and op == '~':
            arg, = args
            jax_expr = self._jax(arg, params, bool)
            return JaxRDDLCompiler._jax_unary(jax_expr, jnp.logical_not, dtype)
        
        elif n == 2:
            lhs, rhs = args
            jax_lhs = self._jax(lhs, params, bool)
            jax_rhs = self._jax(rhs, params, bool)
            jax_op = JaxRDDLCompiler.LOGICAL_OPS[op]
            return JaxRDDLCompiler._jax_binary(jax_lhs, jax_rhs, jax_op, dtype)
        
        raise RDDLInvalidNumberOfArgumentsError(
            'Logical operator {} does not have the required number of args, got {}.'.format(
                op, n) + 
            '\n' + JaxRDDLCompiler._print_stack_trace(expr))
    
    # aggregations
    AGGREGATION_OPS = {
        'sum': jnp.sum,
        'avg': jnp.mean,
        'prod': jnp.prod,
        'min': jnp.min,
        'max': jnp.max,
        'forall': jnp.all,
        'exists': jnp.any
    }
    
    def _jax_aggregation(self, expr, op, params, dtype):
        if op not in JaxRDDLCompiler.AGGREGATION_OPS:
            raise RDDLNotImplementedError(
                'Aggregation operator {} is not supported: must be one of {}.'.format(
                    op, JaxRDDLCompiler.AGGREGATION_OPS.keys()) + 
                '\n' + JaxRDDLCompiler._print_stack_trace(expr))
            
        * pvars, arg = expr.args
        pvars = list(map(lambda p: p[1], pvars))        
        new_params = params + pvars
        reduced_axes = tuple(range(len(params), len(new_params)))
        
        new_dtype = bool if op in {'forall', 'exists'} else dtype
        jax_expr = self._jax(arg, new_params, new_dtype)
        jax_op = JaxRDDLCompiler.AGGREGATION_OPS[op]        
        ERR_CODE = JaxRDDLCompiler.ERROR_CODES['INVALID_CAST']
        
        def _f(x, key):
            sample, key, err_x = jax_expr(x, key)
            sample = jax_op(sample, axis=reduced_axes)
            err = err_x | (not jnp.can_cast(sample, dtype, casting='safe')) * ERR_CODE
            sample = jnp.asarray(sample, dtype=dtype)
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
    
    def _jax_functional(self, expr, op, params, dtype):
        n = len(expr.args)
        
        if op in JaxRDDLCompiler.KNOWN_UNARY:
            if n != 1:
                raise RDDLInvalidNumberOfArgumentsError(
                    'Unary function {} requires 1 arg, got {}.'.format(op, n) + 
                    '\n' + JaxRDDLCompiler._print_stack_trace(expr))    
                            
            arg, = expr.args
            jax_expr = self._jax(arg, params, dtype)
            jax_op = JaxRDDLCompiler.KNOWN_UNARY[op]
            return JaxRDDLCompiler._jax_unary(jax_expr, jax_op, dtype)
            
        elif op in JaxRDDLCompiler.KNOWN_BINARY:
            if n != 2:
                raise RDDLInvalidNumberOfArgumentsError(
                    'Binary function {} requires 2 args, got {}.'.format(op, n) + 
                    '\n' + JaxRDDLCompiler._print_stack_trace(expr))  
                
            lhs, rhs = expr.args
            jax_lhs = self._jax(lhs, params, dtype)
            jax_rhs = self._jax(rhs, params, dtype)
            jax_op = JaxRDDLCompiler.KNOWN_BINARY[op]
            return JaxRDDLCompiler._jax_binary(jax_lhs, jax_rhs, jax_op, dtype)
        
        raise RDDLNotImplementedError(
                'Function {} is not supported.'.format(op) + 
                '\n' + JaxRDDLCompiler._print_stack_trace(expr))   
    
    # control flow
    def _jax_control(self, expr, op, params, dtype):
        if op not in {'if'}:
            raise RDDLNotImplementedError(
                'Control flow type <{}> is not supported: must be one of {}.'.format(
                    op, {'if'}) + 
                '\n' + JaxRDDLCompiler._print_stack_trace(expr))
            
        pred, if_true, if_false = expr.args
        
        jax_pred = self._jax(pred, params, bool)
        jax_true = self._jax(if_true, params, dtype)
        jax_false = self._jax(if_false, params, dtype)
        
        def _f(x, key):
            pred_x, key, err1_x = jax_pred(x, key)
            true_x, key, err2_x = jax_true(x, key)
            false_x, key, err3_x = jax_false(x, key)
            sample = jnp.where(pred_x, true_x, false_x)
            err = err1_x | err2_x | err3_x
            return sample, key, err
            
        return _f
    
    # random variables   
    # TODO: try these for Poisson and Gamma
    # https://arxiv.org/pdf/1611.01144.pdf?
    # https://arxiv.org/pdf/2003.01847.pdf?     
    def _jax_random(self, expr, name, params, dtype):
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
            # raise RDDLInvalidExpressionError(
            #     'Distribution {} is not reparameterizable.'.format(name) + 
            #     '\n' + JaxRDDLCompiler._print_stack_trace(expr))
        else:
            raise RDDLNotImplementedError(
                'Distribution {} is not supported.'.format(name) + 
                '\n' + JaxRDDLCompiler._print_stack_trace(expr))
        
    def _jax_kron(self, expr, params):
        arg, = expr.args
        return self._jax(arg, params, JaxRDDLCompiler.INT)
    
    def _jax_dirac(self, expr, params):
        arg, = expr.args
        return self._jax(arg, params, JaxRDDLCompiler.REAL)
    
    def _jax_uniform(self, expr, params):
        expr_lb, expr_ub = expr.args
        jax_lb = self._jax(expr_lb, params, JaxRDDLCompiler.REAL)
        jax_ub = self._jax(expr_ub, params, JaxRDDLCompiler.REAL)
        ERR_CODE = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_UNIFORM']
        
        # U(a, b) = a + (b - a) * xi, where xi ~ U(0, 1)
        def _f(x, key):
            lb_x, key, err1_x = jax_lb(x, key)
            ub_x, key, err2_x = jax_ub(x, key)
            err = err1_x | err2_x | jnp.any(lb_x > ub_x) * ERR_CODE
            key, subkey = random.split(key)
            U = random.uniform(
                key=subkey, shape=lb_x.shape, dtype=JaxRDDLCompiler.REAL)
            sample = lb_x + (ub_x - lb_x) * U
            return sample, key, err
        
        return _f
    
    def _jax_normal(self, expr, params):
        expr_mean, expr_var = expr.args
        jax_mean = self._jax(expr_mean, params, JaxRDDLCompiler.REAL)
        jax_var = self._jax(expr_var, params, JaxRDDLCompiler.REAL)
        ERR_CODE = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_NORMAL']
        
        # N(mu, s^2) = mu + s * N(0, 1)
        def _f(x, key):
            mean_x, key, err1_x = jax_mean(x, key)
            var_x, key, err2_x = jax_var(x, key)
            err = err1_x | err2_x | jnp.any(var_x < 0) * ERR_CODE
            std_x = jnp.sqrt(var_x)
            key, subkey = random.split(key)
            Z = random.normal(
                key=subkey, shape=mean_x.shape, dtype=JaxRDDLCompiler.REAL)
            sample = mean_x + std_x * Z
            return sample, key, err
        
        return _f
    
    def _jax_exponential(self, expr, params):
        expr_scale, = expr.args
        jax_scale = self._jax(expr_scale, params, JaxRDDLCompiler.REAL)
        ERR_CODE = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_EXPONENTIAL']
        
        # Exp(scale) = scale * Exp(1)
        def _f(x, key):
            scale_x, key, err_x = jax_scale(x, key)
            err = err_x | jnp.any(scale_x < 0) * ERR_CODE
            key, subkey = random.split(key)
            Exp1 = random.exponential(
                key=subkey, shape=scale_x.shape, dtype=JaxRDDLCompiler.REAL)
            sample = scale_x * Exp1
            return sample, key, err
        
        return _f
    
    def _jax_weibull(self, expr, params):
        expr_shape, expr_scale = expr.args
        jax_shape = self._jax(expr_shape, params, JaxRDDLCompiler.REAL)
        jax_scale = self._jax(expr_scale, params, JaxRDDLCompiler.REAL)
        ERR_CODE = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_WEIBULL']
        
        # W(shape, scale) = scale * (-log(1 - U(0, 1)))^{1 / shape}
        # TODO: make this numerically stable
        def _f(x, key):
            shape_x, key, err1_x = jax_shape(x, key)
            scale_x, key, err2_x = jax_scale(x, key)
            err = err1_x | err2_x | jnp.any((shape_x < 0) | (scale_x < 0)) * ERR_CODE
            key, subkey = random.split(key)
            U = random.uniform(
                key=subkey, shape=shape_x.shape, dtype=JaxRDDLCompiler.REAL)
            sample = scale_x * jnp.power(-jnp.log(1.0 - U), 1.0 / shape_x)
            return sample, key, err
        
        return _f
            
    def _jax_bernoulli(self, expr, params):
        expr_prob, = expr.args
        jax_prob = self._jax(expr_prob, params, JaxRDDLCompiler.REAL)
        ERR_CODE = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_BERNOULLI']
        
        # Bernoulli(p) = 1[U(0, 1) < p]
        def _f(x, key):
            prob_x, key, err_x = jax_prob(x, key)
            err = err_x | jnp.any((prob_x < 0) | (prob_x > 1)) * ERR_CODE
            key, subkey = random.split(key)
            U = random.uniform(
                key=subkey, shape=prob_x.shape, dtype=JaxRDDLCompiler.REAL)
            sample = jnp.less(U, prob_x)
            return sample, key, err
        
        return _f
    
    def _jax_poisson(self, expr, params):
        expr_rate, = expr.args
        jax_rate = self._jax(expr_rate, params, JaxRDDLCompiler.REAL)
        ERR_CODE = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_POISSON']
        
        # no reparameterization so far
        def _f(x, key):
            rate_x, key, err_x = jax_rate(x, key)
            err = err_x | jnp.any(rate_x < 0) * ERR_CODE
            key, subkey = random.split(key)
            sample = random.poisson(key=subkey, lam=rate_x, dtype=JaxRDDLCompiler.INT)
            return sample, key, err
        
        return _f
    
    def _jax_gamma(self, expr, params):
        expr_shape, expr_scale = expr.args
        jax_shape = self._jax(expr_shape, params, JaxRDDLCompiler.REAL)
        jax_scale = self._jax(expr_scale, params, JaxRDDLCompiler.REAL)
        ERR_CODE = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_GAMMA']
        
        # only partially reparameterizable
        def _f(x, key):
            shape_x, key, err1_x = jax_shape(x, key)
            scale_x, key, err2_x = jax_scale(x, key)
            err = err1_x | err2_x | jnp.any((shape_x < 0) | (scale_x < 0)) * ERR_CODE
            key, subkey = random.split(key)
            Gamma1 = random.gamma(key=subkey, a=shape_x, dtype=JaxRDDLCompiler.REAL)
            sample = scale_x * Gamma1
            return sample, key, err
        
        return _f
            
            