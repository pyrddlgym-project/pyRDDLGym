import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
import warnings

from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidNumberOfArgumentsError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLNotImplementedError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLValueOutOfRangeError
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
    
    DEFAULT_VALUES = {
        'int': 0,
        'real': 0.0,
        'bool': False
    }
    
    def __init__(self, rddl: RDDL, allow_discrete: bool=True) -> None:
        self.rddl = rddl
        self.domain = rddl.domain
        self.instance = rddl.instance
        self.non_fluents = rddl.non_fluents
        self.allow_discrete = allow_discrete
        
        # basic Jax operations
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
    
    def compile(self) -> None:
        self.objects, self.states, self.pvars, self.pvar_types = self._compile_objects()        
        self.init_values, self.horizon, self.discount = self._compile_instance()
        
        self.invariants = self._compile_invariants()
        self.preconditions = self._compile_preconditions()
        self.termination = self._compile_termination()
        self.reward = self._compile_reward()
        self.cpfs = self._compile_cpfs()
        
        self.state_unprimed = {cpf: self.states.get(cpf, cpf) for cpf in self.cpfs[0].keys()}
        
        return self
    
    # compile RDDL files
    def _compile_objects(self):
        objects = {}
        for obj, values in self.non_fluents.objects:
            objects[obj] = dict(zip(values, range(len(values))))
        
        states, pvars, pvar_types = {}, {}, {}
        for pvar in self.domain.pvariables:
            name = pvar.name
            ptype = pvar.range
            params = pvar.param_types
            
            if pvar.fluent_type == 'state-fluent':
                name = name + '\''
                states[name] = pvar.name
                
            pvars[name] = [] if params is None else params
            
            if self.allow_discrete:
                pvar_types[name] = JaxRDDLCompiler.RDDL_TO_JAX_TYPE[ptype]
            else:
                pvar_types[name] = JaxRDDLCompiler.REAL
                if ptype != 'real':
                    warnings.warn('Variable <{}> of type {} will be cast to real.'.format(
                        name, ptype), FutureWarning, stacklevel=2)
                    
        return objects, states, pvars, pvar_types
    
    def _compile_instance(self):
        object_lookup = {}        
        for obj, values in self.non_fluents.objects:
            object_lookup.update(zip(values, [obj] * len(values)))  
        
        # initialize values with default
        init_values = {}
        for pvar in self.domain.pvariables:
            name = pvar.name
            params = pvar.param_types
            ptype = pvar.range
            
            value = pvar.default
            if value is None:
                value = JaxRDDLCompiler.DEFAULT_VALUES[ptype]
            if not self.allow_discrete:
                value = float(value)
                ptype = 'real'
            
            if params is None:
                init_values[name] = value              
            else: 
                init_values[name] = np.full(
                    shape=tuple(len(self.objects[p]) for p in params),
                    fill_value=value,
                    dtype=JaxRDDLCompiler.RDDL_TO_JAX_TYPE[ptype])
        
        # override values with instance
        if hasattr(self.instance, 'init_state'):
            for (name, params), value in self.instance.init_state:
                if params is not None:
                    coords = tuple(self.objects[object_lookup[p]][p] for p in params)
                    init_values[name][coords] = value   
        
        if hasattr(self.non_fluents, 'init_non_fluent'):
            for (name, params), value in self.non_fluents.init_non_fluent:
                if params is not None:
                    coords = tuple(self.objects[object_lookup[p]][p] for p in params)
                    init_values[name][coords] = value   
        
        # other constants from instance      
        horizon = self.instance.horizon
        if not (horizon > 0):
            raise RDDLValueOutOfRangeError(
                'Horizon {} in the instance is not > 0.'.format(horizon))
            
        gamma = self.instance.discount
        if not (0 <= gamma <= 1):
            raise RDDLValueOutOfRangeError(
                'Discount {} in the instance is not in [0, 1].'.format(gamma))
        
        return init_values, horizon, gamma
        
    # compile Jax expressions
    def _compile_invariants(self):
        jax_invariants = [self._jax_cast(self._jax(expr, []), bool)
                          for expr in self.domain.invariants]
        return jax.tree_map(jax.jit, jax_invariants)
        
    def _compile_preconditions(self):
        jax_preconds = [self._jax_cast(self._jax(expr, []), bool)
                        for expr in self.domain.preconds]
        return jax.tree_map(jax.jit, jax_preconds)
    
    def _compile_termination(self):
        jax_terminals = [self._jax_cast(self._jax(expr, []), bool)
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
            jax_cpfs[name] = self._jax_cast(
                self._jax(expr, params), self.pvar_types[name])
        
        # TODO: implement topological sort
        cpf_order = self.domain.derived_cpfs + \
                    self.domain.intermediate_cpfs + \
                    self.domain.state_cpfs + \
                    self.domain.observation_cpfs 
        cpf_order = [cpf.pvar[1][0] for cpf in cpf_order]
        jax_cpfs = {name: jax_cpfs[name] for name in cpf_order}
        
        jit_cpfs = jax.tree_map(jax.jit, jax_cpfs)        
        return jax_cpfs, jit_cpfs
    
    def _compile_reward(self):
        jax_reward = self._jax_cast(
            self._jax(self.domain.reward, []), JaxRDDLCompiler.REAL)
        jit_reward = jax.jit(jax_reward)
        return jax_reward, jit_reward
    
    # error handling
    @staticmethod
    def _print_stack_trace(expr):
        return '...\n' + str(expr) + '\n...'
    
    @staticmethod
    def _check_valid_op(expr, valid_ops):
        etype, op = expr.etype
        if op not in valid_ops:
            message = '{} operator {} is not supported: must be one of {}.'.format(
                etype, op, valid_ops.keys())
            trace = JaxRDDLCompiler._print_stack_trace(expr)
            raise RDDLNotImplementedError(message + '\n' + trace)
    
    @staticmethod
    def _check_num_args(expr, required_args):
        etype, op = expr.etype
        actual_args = len(expr.args)
        if actual_args != required_args:
            raise RDDLInvalidNumberOfArgumentsError(
                '{} operator {} requires {} arguments, got {}.'.format(
                    etype, op, required_args, actual_args) + 
                '\n' + JaxRDDLCompiler._print_stack_trace(expr))
        
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
        0: 'Possible loss of precision in cast to a weaker type (e.g., real to int).',
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
    
    @staticmethod
    def get_error_messages(error):
        codes = JaxRDDLCompiler.get_error_codes(error)
        messages = [JaxRDDLCompiler.INVERSE_ERROR_CODES[i] for i in codes]
        return messages
    
    # start of compilation subroutines for expressions
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
            
    def _jax_cast(self, jax_expr, dtype):
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
        symbols = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        
        # check that number of parameters is valid
        if len(desired_params) > len(symbols):
            raise RDDLNotImplementedError(
                'Variable <{}> is {}-D, but current version supports up to {}-D.'.format(
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
        valid_ops = self.ARITHMETIC_OPS
        JaxRDDLCompiler._check_valid_op(expr, valid_ops)
                    
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
            jax_op = valid_ops[op]
            return JaxRDDLCompiler._jax_binary(jax_lhs, jax_rhs, jax_op)
        
        JaxRDDLCompiler._check_num_args(expr, 2)
    
    # relational expressions
    def _jax_relational(self, expr, op, params):
        valid_ops = self.RELATIONAL_OPS
        JaxRDDLCompiler._check_valid_op(expr, valid_ops)
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        lhs, rhs = expr.args
        jax_lhs = self._jax(lhs, params)
        jax_rhs = self._jax(rhs, params)
        jax_op = valid_ops[op]
        return JaxRDDLCompiler._jax_binary(jax_lhs, jax_rhs, jax_op)
        
    # logical expressions         
    def _jax_logical(self, expr, op, params):
        valid_ops = self.LOGICAL_OPS    
        JaxRDDLCompiler._check_valid_op(expr, valid_ops)
                
        args = expr.args
        n = len(args)
        
        if n == 1 and op == '~':
            arg, = args
            jax_expr = self._jax(arg, params)
            return JaxRDDLCompiler._jax_unary(jax_expr, self.LOGICAL_NOT)
        
        elif n == 2:
            lhs, rhs = args
            jax_lhs = self._jax(lhs, params)
            jax_rhs = self._jax(rhs, params)
            jax_op = valid_ops[op]
            return JaxRDDLCompiler._jax_binary(jax_lhs, jax_rhs, jax_op)
        
        JaxRDDLCompiler._check_num_args(expr, 2)
    
    # aggregations    
    def _jax_aggregation(self, expr, op, params):
        valid_ops = self.AGGREGATION_OPS      
        JaxRDDLCompiler._check_valid_op(expr, valid_ops) 
        
        * pvars, arg = expr.args
        pvars = list(map(lambda p: p[1], pvars))        
        new_params = params + pvars
        reduced_axes = tuple(range(len(params), len(new_params)))
        
        jax_expr = self._jax(arg, new_params)
        jax_op = valid_ops[op]        
        
        def _f(x, key):
            val, key, err = jax_expr(x, key)
            sample = jax_op(val, axis=reduced_axes)
            return sample, key, err
        
        return _f
                 
    # functions
    def _jax_functional(self, expr, op, params): 
        if op in self.KNOWN_UNARY:
            JaxRDDLCompiler._check_num_args(expr, 1)                            
            arg, = expr.args
            jax_expr = self._jax(arg, params)
            jax_op = self.KNOWN_UNARY[op]
            return JaxRDDLCompiler._jax_unary(jax_expr, jax_op)
            
        elif op in self.KNOWN_BINARY:
            JaxRDDLCompiler._check_num_args(expr, 2)                
            lhs, rhs = expr.args
            jax_lhs = self._jax(lhs, params)
            jax_rhs = self._jax(rhs, params)
            jax_op = self.KNOWN_BINARY[op]
            return JaxRDDLCompiler._jax_binary(jax_lhs, jax_rhs, jax_op)
        
        raise RDDLNotImplementedError(
                'Function {} is not supported.'.format(op) + 
                '\n' + JaxRDDLCompiler._print_stack_trace(expr))   
    
    # control flow
    def _jax_control(self, expr, op, params):
        valid_ops = self.CONTROL_OPS
        JaxRDDLCompiler._check_valid_op(expr, valid_ops)
        JaxRDDLCompiler._check_num_args(expr, 3)
        
        pred, if_true, if_false = expr.args        
        jax_pred = self._jax(pred, params)
        jax_true = self._jax(if_true, params)
        jax_false = self._jax(if_false, params)
        
        jax_op = self.CONTROL_OPS[op]
        
        def _f(x, key):
            val1, key, err1 = jax_pred(x, key)
            val2, key, err2 = jax_true(x, key)
            val3, key, err3 = jax_false(x, key)
            sample = jax_op(val1, val2, val3)
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
        JaxRDDLCompiler._check_num_args(expr, 1)
        arg, = expr.args
        arg = self._jax(arg, params)
        arg = self._jax_cast(arg, bool)
        return arg
    
    def _jax_dirac(self, expr, params):
        JaxRDDLCompiler._check_num_args(expr, 1)
        arg, = expr.args
        arg = self._jax(arg, params)
        arg = self._jax_cast(arg, JaxRDDLCompiler.REAL)
        return arg
    
    def _jax_uniform(self, expr, params):
        JaxRDDLCompiler._check_num_args(expr, 2)
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
        JaxRDDLCompiler._check_num_args(expr, 2)
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
        JaxRDDLCompiler._check_num_args(expr, 1)
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
        JaxRDDLCompiler._check_num_args(expr, 2)
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
        JaxRDDLCompiler._check_num_args(expr, 1)
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
        JaxRDDLCompiler._check_num_args(expr, 1)
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
        JaxRDDLCompiler._check_num_args(expr, 2)
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
            
    
