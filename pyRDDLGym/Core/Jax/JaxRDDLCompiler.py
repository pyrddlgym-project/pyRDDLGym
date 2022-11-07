import jax
import jax.numpy as jnp
import jax.random as random

from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLNotImplementedError
from pyRDDLGym.Core.Parser.expr import Expression
from pyRDDLGym.Core.Parser.rddl import RDDL


class JaxRDDLCompiler:
    
    INT = jnp.int32
    REAL = jnp.float64
    
    def __init__(self, rddl: RDDL) -> None:
        self.rddl = rddl
        self.domain = rddl.domain
        
        # extract objects and free parameters for each variable
        self.objects, self.pvars, self.states = {}, {}, {}
        for obj, values in rddl.non_fluents.objects:
            self.objects[obj] = dict(zip(values, range(len(values))))
        for pvar in rddl.domain.pvariables:
            name = pvar.name
            if pvar.fluent_type == 'state-fluent':
                name = name + '\''
                self.states[name] = pvar.name
            self.pvars[name] = pvar.param_types
            
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
            jax_cpfs[name] = self._jax(expr, params, JaxRDDLCompiler.REAL)
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
    def _map_pvar_subs_to_subscript(self, given_params, desired_params):
        symbols = 'abcdefghijklmnopqrstuvwxyz'   
        
        if given_params is None:
            given_params = []
            
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
        
        subscripts = ''.join(lhs) + ' -> ' + rhs
        return subscripts, new_dims
    
    def _jax_constant(self, expr, params, dtype):
        const = expr.args
        subscripts, new_dims = self._map_pvar_subs_to_subscript([], params)
        new_axes = (1,) * len(new_dims)
        
        def _f(x, key):
            constx = jnp.array(const, dtype)
            sample = jnp.reshape(constx, constx.shape + new_axes) 
            for i, dim in enumerate(new_dims):
                sample = jnp.repeat(sample, repeats=dim, axis=len(constx.shape) + i)
            sample = jnp.einsum(subscripts, sample)
            return sample, key

        return _f
    
    def _jax_pvar(self, expr, params, dtype):
        var, pvars = expr.args
        subscripts, new_dims = self._map_pvar_subs_to_subscript(pvars, params)
        new_axes = (1,) * len(new_dims)
        
        def _f(x, key):
            varx = jnp.array(x[var], dtype)
            sample = jnp.reshape(varx, varx.shape + new_axes) 
            for i, dim in enumerate(new_dims):
                sample = jnp.repeat(sample, repeats=dim, axis=len(varx.shape) + i)
            sample = jnp.einsum(subscripts, sample)
            return sample, key
        
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
        
        def _f(x, key):
            val, key = jax_expr(x, key)
            sample = jax_op(val)
            sample = sample.astype(dtype)
            return sample, key
        
        return _f
    
    @staticmethod
    def _jax_binary(jax_lhs, jax_rhs, jax_op, dtype):
        
        def _f(x, key):
            val1, key = jax_lhs(x, key)
            val2, key = jax_rhs(x, key)
            sample = jax_op(val1, val2)
            sample = sample.astype(dtype)
            return sample, key
        
        return _f
        
    def _jax_arithmetic(self, expr, op, params, dtype):
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
        
        raise Exception('e')
    
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
        args = expr.args
        n = len(args)
        
        if n == 2:
            lhs, rhs = args
            jax_lhs = self._jax(lhs, params, JaxRDDLCompiler.REAL)
            jax_rhs = self._jax(rhs, params, JaxRDDLCompiler.REAL)
            jax_op = JaxRDDLCompiler.RELATIONAL_OPS[op]
            return JaxRDDLCompiler._jax_binary(jax_lhs, jax_rhs, jax_op, dtype)
        
        raise Exception('e')
    
    # logical expressions    
    LOGICAL_OPS = {
        '^': jnp.logical_and,
        '|': jnp.logical_or,
        '~': jnp.logical_xor,
        '=>': lambda e1, e2: jnp.logical_or(jnp.logical_not(e1), e2),
        '<=>': jnp.equal
    }
    
    def _jax_logical(self, expr, op, params, dtype):
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
        
        raise Exception('e')
    
    # aggregations
    AGGREGATION_OPS = {
        'sum': jnp.sum,
        'prod': jnp.prod,
        'min': jnp.min,
        'max': jnp.max,
        'forall': jnp.all,
        'exists': jnp.any
    }
    
    def _jax_aggregation(self, expr, op, params, dtype):
        *pvars, arg = expr.args
        pvars = list(map(lambda p: p[1], pvars))        
        new_params = params + pvars
        reduced_axes = tuple(range(len(params), len(new_params)))
        
        new_dtype = bool if op in {'forall', 'exists'} else dtype
        jax_expr = self._jax(arg, new_params, new_dtype)
        jax_op = JaxRDDLCompiler.AGGREGATION_OPS[op]
        
        def _f(x, key):
            sample, key = jax_expr(x, key)
            sample = jax_op(sample, axis=reduced_axes)
            sample = sample.astype(dtype)
            return sample, key
        
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
            if n == 1:
                arg, = expr.args
                jax_expr = self._jax(arg, params, JaxRDDLCompiler.REAL)
                jax_op = JaxRDDLCompiler.KNOWN_UNARY[op]
                return JaxRDDLCompiler._jax_unary(jax_expr, jax_op, dtype)
        
        if op in JaxRDDLCompiler.KNOWN_BINARY:
            if n == 2:
                lhs, rhs = expr.args
                jax_lhs = self._jax(lhs, params, JaxRDDLCompiler.REAL)
                jax_rhs = self._jax(rhs, params, JaxRDDLCompiler.REAL)
                jax_op = JaxRDDLCompiler.KNOWN_BINARY[op]
                return JaxRDDLCompiler._jax_binary(jax_lhs, jax_rhs, jax_op, dtype)
        
        raise Exception('e')
    
    # control flow
    def _jax_control(self, expr, op, params, dtype):
        pred, if_true, if_false = expr.args
        
        jax_pred = self._jax(pred, params, bool)
        jax_true = self._jax(if_true, params, dtype)
        jax_false = self._jax(if_false, params, dtype)
        
        def _f(x, key):
            val, key = jax_pred(x, key)
            val_true, key = jax_true(x, key)
            val_false, key = jax_false(x, key)
            sample = jnp.where(val, val_true, val_false)
            sample = sample.astype(dtype)
            return sample, key
            
        return _f
    
    # random variables        
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
        elif name in {'Poisson', 'Gamma'}:
            raise Exception('e')
        else:
            raise Exception('e')
        
    def _jax_kron(self, expr, params):
        return self._jax(expr, params, JaxRDDLCompiler.INT)
    
    def _jax_dirac(self, expr, params):
        return self._jax(expr, params, JaxRDDLCompiler.REAL)
    
    def _jax_uniform(self, expr, params):
        expr_lb, expr_ub = expr.args
        jax_lb = self._jax(expr_lb, params, JaxRDDLCompiler.REAL)
        jax_ub = self._jax(expr_ub, params, JaxRDDLCompiler.REAL)
        
        def _f(x, key):
            lb, key = jax_lb(x, key)
            ub, key = jax_ub(x, key)
            key, subkey = random.split(key)
            U = random.uniform(key=subkey, shape=lb.shape)
            sample = lb + (ub - lb) * U
            return sample, key
        
        return _f
    
    def _jax_normal(self, expr, params):
        expr_mean, expr_var = expr.args
        jax_mean = self._jax(expr_mean, params, JaxRDDLCompiler.REAL)
        jax_var = self._jax(expr_var, params, JaxRDDLCompiler.REAL)
            
        def _f(x, key):
            mean, key = jax_mean(x, key)
            var, key = jax_var(x, key)
            std = jnp.sqrt(var)
            key, subkey = random.split(key)
            Z = random.normal(key=subkey, shape=mean.shape)
            sample = mean + std * Z
            return sample, key
        
        return _f
    
    def _jax_exponential(self, expr, params):
        expr_scale, = expr.args
        jax_scale = self._jax(expr_scale, params, JaxRDDLCompiler.REAL)
        
        def _f(x, key):
            scale, key = jax_scale(x, key)
            key, subkey = random.split(key)
            Exp1 = random.exponential(key=subkey)
            sample = scale * Exp1
            return sample, key
        
        return _f
    
    def _jax_weibull(self, expr, params):
        expr_shape, expr_scale = expr.args
        jax_shape = self._jax(expr_shape, params, JaxRDDLCompiler.REAL)
        jax_scale = self._jax(expr_scale, params, JaxRDDLCompiler.REAL)
        
        def _f(x, key):
            shape, key = jax_shape(x, key)
            scale, key = jax_scale(x, key)
            key, subkey = random.split(key)
            U = random.uniform(key=subkey)
            sample = scale * jnp.power(-jnp.log(1.0 - U), 1.0 / shape)
            return sample, key
        
        return _f
            
    def _jax_bernoulli(self, expr, params):
        expr_prob, = expr.args
        jax_prob = self._jax(expr_prob, params, JaxRDDLCompiler.REAL)
        
        def _f(x, key):
            prob, key = jax_prob(x, key)
            key, subkey = random.split(key)
            U = random.uniform(key=subkey, shape=prob.shape)
            sample = jnp.less(U, prob)
            return sample, key
        
        return _f
    
