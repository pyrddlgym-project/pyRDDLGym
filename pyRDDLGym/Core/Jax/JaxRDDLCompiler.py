import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
import warnings

from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidNumberOfArgumentsError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLNotImplementedError
from pyRDDLGym.Core.Parser.rddl import RDDL


class JaxRDDLCompiler:
    
    INT = jnp.int32
    REAL = jnp.float32
    
    JAX_TYPES = {
        'int': INT,
        'real': REAL,
        'bool': bool
    }
    
    DEFAULT_VALUES = {
        'int': 0,
        'real': 0.0,
        'bool': False
    }
    
    VALID_SYMBOLS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

    def __init__(self, 
                 rddl: RDDL,
                 force_continuous: bool=False, 
                 debug: bool=False) -> None:
        self.rddl = rddl
        self.force_continuous = force_continuous
        self.debug = debug
        jax.config.update('jax_log_compiles', self.debug)
        
        self.domain = rddl.domain
        self.instance = rddl.instance
        self.non_fluents = rddl.non_fluents
        
        # TODO: implement topological sort
        cpf_order = self.domain.derived_cpfs + \
                    self.domain.intermediate_cpfs + \
                    self.domain.state_cpfs + \
                    self.domain.observation_cpfs 
        self.cpf_order = [cpf.pvar[1][0] for cpf in cpf_order]
        
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
    # compile type and object information
    # ===========================================================================
    
    def _compile_types(self):
        self.pvar_types = {}
        for pvar in self.domain.pvariables:
            name = pvar.name
            ptypes = pvar.param_types
            if ptypes is None: 
                ptypes = []
            self.pvar_types[name] = ptypes
            if pvar.is_state_fluent():
                self.pvar_types[name + '\''] = ptypes
            
        self.cpf_types = {}
        for cpf in self.domain.cpfs[1]:
            _, (name, objects) = cpf.pvar
            if objects is None:
                objects = [] 
            types = self.pvar_types[name]
            self.cpf_types[name] = [(o, types[i]) for i, o in enumerate(objects)]
        
        if self.debug:
            pvar = ''.join(f'\n\t\t{k}: {v}' for k, v in self.pvar_types.items())
            cpf = ''.join(f'\n\t\t{k}: {v}' for k, v in self.cpf_types.items())
            warnings.warn(
                f'compiling type information:'
                f'\n\tpvar types ={pvar}'
                f'\n\tcpf types  ={cpf}\n'
            )
    
    def _compile_objects(self): 
        self.objects, self.objects_to_index = {}, {}     
        for name, ptype in self.non_fluents.objects:
            indices = {obj: index for index, obj in enumerate(ptype)}
            self.objects[name] = indices
            overlap = indices.keys() & self.objects_to_index.keys()
            if overlap:
                raise Exception(
                    f'Multiple types share the same object <{overlap}>.')
            self.objects_to_index.update(indices)
        
        if self.debug:
            obj = ''.join(f'\n\t\t{k}: {v}' for k, v in self.objects.items())
            warnings.warn(
                f'compiling object information:'
                f'\n\tobjects ={obj}\n'
            )
    
    def _objects_to_coordinates(self, objects, msg):
        try:
            return tuple(self.objects_to_index[obj] for obj in objects)
        except:
            for obj in objects:
                if obj not in self.objects_to_index:
                    raise Exception(
                        f'Object <{obj}> declared in {msg} is not valid.')
    
    def _compile_initial_values(self):
        
        # get default values from domain
        self.dtypes = {}
        self.init_values = {}
        self.noop_actions = {}
        for pvar in self.domain.pvariables:
            name = pvar.name                     
            prange = pvar.range
            if self.force_continuous:
                prange = 'real'
            dtype = JaxRDDLCompiler.JAX_TYPES[prange]
            self.dtypes[name] = dtype
            if pvar.is_state_fluent():
                self.dtypes[name + '\''] = dtype
                             
            ptypes = pvar.param_types
            value = pvar.default
            if value is None:
                value = JaxRDDLCompiler.DEFAULT_VALUES[prange]
            if ptypes is None:
                self.init_values[name] = value              
            else: 
                self.init_values[name] = np.full(
                    shape=tuple(len(self.objects[obj]) for obj in ptypes),
                    fill_value=value,
                    dtype=dtype)
            
            if pvar.is_action_fluent():
                self.noop_actions[name] = self.init_values[name]
        
        # override default values with instance
        if hasattr(self.instance, 'init_state'):
            for (name, objects), value in self.instance.init_state:
                if objects is not None:
                    coords = self._objects_to_coordinates(objects, 'init-state')
                    self.init_values[name][coords] = value   
        
        if hasattr(self.non_fluents, 'init_non_fluent'):
            for (name, objects), value in self.non_fluents.init_non_fluent:
                if objects is not None:
                    coords = self._objects_to_coordinates(objects, 'non-fluents')
                    self.init_values[name][coords] = value
        
        if self.debug:
            val = ''.join(f'\n\t\t{k}: {v}' for k, v in self.init_values.items())
            warnings.warn(
                f'compiling initial value information:'
                f'\n\tvalues ={val}\n'
            )
        
        # useful to have state lookup
        self.states = {}
        for pvar in self.domain.pvariables:
            if pvar.is_state_fluent():
                self.states[pvar.name + '\''] = pvar.name
                
    # ===========================================================================
    # main compilation subroutines
    # ===========================================================================
    
    def compile(self) -> None:
        self._compile_types()
        self._compile_objects()
        self._compile_initial_values()      
        
        self.invariants = self._compile_constraints(self.domain.invariants)
        self.preconditions = self._compile_constraints(self.domain.preconds)
        self.termination = self._compile_constraints(self.domain.terminals)
        self.cpfs = self._compile_cpfs()
        self.reward = self._compile_reward()
    
    def _compile_constraints(self, constraints):
        to_jax = lambda e: self._jax(e, [], dtype=bool)
        return list(map(to_jax, constraints))
        
    def _compile_cpfs(self):
        jax_cpfs = {}
        for cpf in self.domain.cpfs[1]:
            _, (name, _) = cpf.pvar
            expr = cpf.expr
            ptypes = self.cpf_types[name]
            dtype = self.dtypes[name]
            jax_cpfs[name] = self._jax(expr, ptypes, dtype=dtype)          
        jax_cpfs = {name: jax_cpfs[name] for name in self.cpf_order}
        return jax_cpfs
    
    def _compile_reward(self):
        return self._jax(self.domain.reward, [], dtype=JaxRDDLCompiler.REAL)
    
    # ===========================================================================
    # error checks
    # ===========================================================================
    
    @staticmethod
    def _print_stack_trace(expr):
        return '...\n' + str(expr) + '\n...'
    
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
        'INVALID_PARAM_GAMMA': 128
    }
    
    INVERSE_ERROR_CODES = {
        0: 'Casting occurred that could result in loss of precision.',
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
    
    def _get_subs_map(self, objects_has, types_has, objects_req, expr):
        
        # reached limit on number of valid dimensions (52)
        valid_symbols = JaxRDDLCompiler.VALID_SYMBOLS
        n_valid = len(valid_symbols)
        n_req = len(objects_req)
        if n_req > n_valid:
            raise RDDLNotImplementedError(
                f'Up to {n_valid}-D are supported, '
                f'but variable <{expr.args[0]}> is {n_req}-D.\n' + 
                JaxRDDLCompiler._print_stack_trace(expr))
                
        # find a map permutation(a,b,c...) -> (a,b,c...) for the correct einsum
        objects_has = tuple(zip(objects_has, types_has))
        lhs = [None] * len(objects_has)
        new_dims = []
        for i_req, (obj_req, type_req) in enumerate(objects_req):
            new_dim = True
            for i_has, (obj_has, type_has) in enumerate(objects_has):
                if obj_has == obj_req:
                    lhs[i_has] = valid_symbols[i_req]
                    new_dim = False
                    
                    # check evaluation matches the definition in pvariables {...}
                    if type_req != type_has: 
                        raise Exception(
                            f'Argument <{obj_req}> of variable <{expr.args[0]}> '
                            f'expects type <{type_has}>, got <{type_req}>.\n' + 
                            JaxRDDLCompiler._print_stack_trace(expr))
            
            # need to expand the shape of the value array
            if new_dim:
                lhs.append(valid_symbols[i_req])
                new_dims.append(len(self.objects[type_req]))
                
        # safeguard against any free types
        free = [objects_has[1][i] for i, p in enumerate(lhs) if p is None]
        if free:
            raise RDDLInvalidNumberOfArgumentsError(
                f'Variable <{expr.args[0]}> has free parameter(s) {free}.\n' + 
                JaxRDDLCompiler._print_stack_trace(expr))
            
        lhs = ''.join(lhs)
        rhs = valid_symbols[:n_req]
        permute = lhs + ' -> ' + rhs
        identity = lhs == rhs
        new_dims = tuple(new_dims)
        
        if self.debug:
            warnings.warn(
                f'caching static info for pvar transform:' 
                f'\n\texpr     ={expr}'
                f'\n\tinputs   ={objects_has}'
                f'\n\ttargets  ={objects_req}'
                f'\n\tnew axes ={new_dims}'
                f'\n\teinsum   ={permute}\n'
            )
            
        return (permute, identity, new_dims)
    
    def _jax_constant(self, expr, objects):        
        ERR = JaxRDDLCompiler.ERROR_CODES['NORMAL']
        *_, shape = self._get_subs_map([], [], objects, expr)
        const = expr.args
        
        def _f(_, key):
            sample = jnp.full(shape=shape, fill_value=const)
            return sample, key, ERR

        return _f
    
    def _jax_pvar(self, expr, objects):
        ERR = JaxRDDLCompiler.ERROR_CODES['NORMAL']
        args = expr.args        
        var, pvars = args        
        if pvars is None:
            pvars = []
        types = self.pvar_types[var]
        n_has = len(pvars)
        n_req = len(types)
        if n_has != n_req:
            raise RDDLInvalidNumberOfArgumentsError(
                f'Variable <{var}> requires {n_req} parameters, got {n_has}.\n' + 
                JaxRDDLCompiler._print_stack_trace(expr))
        
        permute, identity, new_dims = self._get_subs_map(
            pvars, types, objects, expr)
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
                f'\n\toutput objects ={objects}'
                f'\n\tinput objects  ={new_objects}'
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
                f'Function {op} is not supported.\n'
                + JaxRDDLCompiler._print_stack_trace(expr))   
    
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
        
        # U(a, b) = a + (b - a) * xi, where xi ~ U(0, 1)
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
        
        # N(m, s ^ 2) = m + s * N(0, 1)
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
                
        # Exp(scale) = scale * Exp(1)
        def _f(x, key):
            scale, key, err = jax_scale(x, key)
            key, subkey = random.split(key)
            Exp = random.exponential(
                key=subkey, shape=scale.shape, dtype=JaxRDDLCompiler.REAL)
            sample = scale * Exp
            out_of_bounds = jnp.any(scale < 0)
            err |= out_of_bounds * ERR
            return sample, key, err
        
        return _f
    
    def _jax_weibull(self, expr, objects):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_WEIBULL']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_shape, arg_scale = expr.args
        jax_shape = self._jax(arg_shape, objects)
        jax_scale = self._jax(arg_scale, objects)
        
        # W(shape, scale) = scale * (-log(1 - U(0, 1))) ^ {1 / shape}
        def _f(x, key):
            shape, key, err1 = jax_shape(x, key)
            scale, key, err2 = jax_scale(x, key)
            key, subkey = random.split(key)
            U = random.uniform(
                key=subkey, shape=shape.shape, dtype=JaxRDDLCompiler.REAL)
            sample = scale * jnp.power(-jnp.log1p(-U), 1.0 / shape)
            out_of_bounds = jnp.any((shape < 0) | (scale < 0))
            err = err1 | err2 | out_of_bounds * ERR
            return sample, key, err
        
        return _f
            
    def _jax_bernoulli(self, expr, objects):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_BERNOULLI']
        JaxRDDLCompiler._check_num_args(expr, 1)
        
        arg_prob, = expr.args
        jax_prob = self._jax(arg_prob, objects)
        
        # Bernoulli(p) = U(0, 1) < p
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
            out_of_bounds = jnp.any((shape < 0) | (scale < 0))
            err = err1 | err2 | out_of_bounds * ERR
            return sample, key, err
        
        return _f

