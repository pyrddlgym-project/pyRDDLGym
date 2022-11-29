import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
import warnings

from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidNumberOfArgumentsError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLNotImplementedError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLUndefinedVariableError
from pyRDDLGym.Core.Parser.rddl import RDDL
from pyRDDLGym.Core.Simulator.LiftedRDDLLevelAnalysis import LiftedRDDLLevelAnalysis
from pyRDDLGym.Core.Simulator.LiftedRDDLTypeAnalysis import LiftedRDDLTypeAnalysis


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
                 allow_synchronous_state: bool=True,
                 debug: bool=True) -> None:
        self.rddl = rddl
        self.force_continuous = force_continuous
        self.debug = debug
        jax.config.update('jax_log_compiles', self.debug)
        
        self.domain = rddl.domain
        self.instance = rddl.instance
        self.non_fluents = rddl.non_fluents
        
        # perform a dependency analysis for fluent variables
        self.static = LiftedRDDLLevelAnalysis(rddl, allow_synchronous_state)
        self.levels = self.static.compute_levels()
        self.next_states = self.static.next_states
        
        # extract information about types and their objects in the domain
        self.types = LiftedRDDLTypeAnalysis(rddl, debug=debug)
        self._compile_initial_values()
                
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
                    shape=self.types.shape(ptypes),
                    fill_value=value,
                    dtype=dtype)
            
            if pvar.is_action_fluent():
                self.noop_actions[name] = self.init_values[name]
        
        # override default values with instance
        blocks = {}
        if hasattr(self.instance, 'init_state'):
            blocks['init-state'] = self.instance.init_state
        if hasattr(self.non_fluents, 'init_non_fluent'):
            blocks['non-fluents'] = self.non_fluents.init_non_fluent
        
        for block_name, block_content in blocks.items():
            for (name, objects), value in block_content:
                init_values = self.init_values.get(name, None)
                if init_values is None:
                    raise RDDLUndefinedVariableError(
                        f'Variable <{name}> in {block_name} is not valid.')
                    
                ptypes = self.types.pvar_types[name]
                if not self.types.is_compatible(name, objects):
                    raise RDDLInvalidNumberOfArgumentsError(
                        f'Type arguments {objects} for variable <{name}> '
                        f'do not match definition {ptypes}.')
                        
                if ptypes:
                    coords = self.types.coordinates(objects, block_name)                                
                    self.init_values[name][coords] = value
                else:
                    self.init_values[name] = value
        
        if self.debug:
            val = ''.join(f'\n\t\t{k}: {v}' for k, v in self.init_values.items())
            warnings.warn(
                f'compiling initial value info:'
                f'\n\tvalues ={val}\n'
            )
            
    def compile(self) -> None:        
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
        for _, cpfs in self.levels.items():
            for cpf in cpfs:
                ptypes = self.types.cpf_types[cpf]
                expr = self.static.cpfs[cpf]
                dtype = self.dtypes[cpf]
                jax_cpfs[cpf] = self._jax(expr, ptypes, dtype=dtype)
        return jax_cpfs
    
    def _compile_reward(self):
        return self._jax(self.domain.reward, [], dtype=JaxRDDLCompiler.REAL)
    
    def compile_rollouts(self, policy, n_steps: int, n_batch: int):
        NORMAL = JaxRDDLCompiler.ERROR_CODES['NORMAL']
            
        # compute a batched version of the initial values
        init_subs = {}
        for name, value in self.init_values.items():
            shape = (n_batch,) + np.shape(value)
            init_subs[name] = np.broadcast_to(value, shape=shape)            
        for next_state, state in self.next_states.items():
            init_subs[next_state] = init_subs[state]
        
        # this performs a single step update
        def _step(carried, step):
            subs, params, key = carried
            action, key = policy(subs, step, params, key)
            subs.update(action)
            
            # check preconditions
            error = NORMAL
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
            
            # check the invariant in the new state
            invariants = jnp.zeros(shape=(len(self.invariants),), dtype=bool)
            for i, invariant in enumerate(self.invariants):
                sample, key, invariant_err = invariant(subs, key)
                invariants = invariants.at[i].set(sample)
                error |= invariant_err
            
            carried = (subs, params, key)
            logged = {'fluent': subs,
                      'action': action,
                      'reward': reward,
                      'preconditions': preconds,
                      'invariants': invariants,
                      'error': error}
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
    
    def _jax_constant(self, expr, objects):
        ERR = JaxRDDLCompiler.ERROR_CODES['NORMAL']
        * _, shape = self.types.map('', [], objects, expr)
        const = expr.args
        
        def _f(_, key):
            sample = jnp.full(shape=shape, fill_value=const)
            return sample, key, ERR

        return _f
    
    def _jax_pvar(self, expr, objects):
        ERR = JaxRDDLCompiler.ERROR_CODES['NORMAL']
        var, pvars = expr.args            
        permute, identity, new_dims = self.types.map(var, pvars, objects, expr)
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
        fails = self.types.validate_types(new_objects)
        if fails:
            raise RDDLUndefinedVariableError(
            f'Type(s) {fails} in aggregation {op} are not valid.\n' + 
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

