from functools import partial
import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy as scipy 
import traceback
from typing import Callable, Dict, List
import warnings

# more robust approach - if user does not have this or broken try to continue
try:
    from tensorflow_probability.substrates import jax as tfp
except Exception:
    warnings.warn('Failed to import tensorflow-probability: '
                  'compilation of some complex distributions will not work.',
                  stacklevel=2)
    traceback.print_exc()
    tfp = None
    
from pyRDDLGym.Core.ErrorHandling.RDDLException import print_stack_trace
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidNumberOfArgumentsError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLNotImplementedError

from pyRDDLGym.Core.Compiler.RDDLLevelAnalysis import RDDLLevelAnalysis
from pyRDDLGym.Core.Compiler.RDDLLiftedModel import RDDLLiftedModel
from pyRDDLGym.Core.Compiler.RDDLObjectsTracer import RDDLObjectsTracer
from pyRDDLGym.Core.Compiler.RDDLValueInitializer import RDDLValueInitializer
from pyRDDLGym.Core.Debug.Logger import Logger
from pyRDDLGym.Core.Env.RDDLConstraints import RDDLConstraints
from pyRDDLGym.Core.Simulator.RDDLSimulator import RDDLSimulatorPrecompiled


class JaxRDDLCompiler:
    '''Compiles a RDDL AST representation into an equivalent JAX representation.
    All operations are identical to their numpy equivalents.
    '''
    
    MODEL_PARAM_TAG_SEPARATOR = '___'
    
    def __init__(self, rddl: RDDLLiftedModel,
                 allow_synchronous_state: bool=True,
                 logger: Logger=None,
                 use64bit: bool=False) -> None:
        '''Creates a new RDDL to Jax compiler.
        
        :param rddl: the RDDL model to compile into Jax
        :param allow_synchronous_state: whether next-state components can depend
        on each other
        :param logger: to log information about compilation to file
        :param use64bit: whether to use 64 bit arithmetic
        '''
        self.rddl = rddl
        self.logger = logger
        # jax.config.update('jax_log_compiles', True) # for testing ONLY
        
        if use64bit:
            self.INT = jnp.int64
            self.REAL = jnp.float64
            jax.config.update('jax_enable_x64', True)
        else:
            self.INT = jnp.int32
            self.REAL = jnp.float32
        self.ONE = jnp.asarray(1, dtype=self.INT)
        self.JAX_TYPES = {
            'int': self.INT,
            'real': self.REAL,
            'bool': bool
        }
        
        # compile initial values
        if self.logger is not None:
            self.logger.clear()
        initializer = RDDLValueInitializer(rddl, logger=self.logger)
        self.init_values = initializer.initialize()
        
        # compute dependency graph for CPFs and sort them by evaluation order
        sorter = RDDLLevelAnalysis(rddl, allow_synchronous_state, logger=self.logger)
        self.levels = sorter.compute_levels()        
        
        # trace expressions to cache information to be used later
        tracer = RDDLObjectsTracer(rddl, logger=self.logger, cpf_levels=self.levels)
        self.traced = tracer.trace()
        
        # extract the box constraints on actions
        simulator = RDDLSimulatorPrecompiled(
            rddl=self.rddl,
            init_values=self.init_values,
            levels=self.levels,
            trace_info=self.traced)  
        constraints = RDDLConstraints(simulator, vectorized=True)
        self.constraints = constraints
        
        # basic operations
        self.NEGATIVE = lambda x, param: jnp.negative(x)  
        self.ARITHMETIC_OPS = {
            '+': lambda x, y, param: jnp.add(x, y),
            '-': lambda x, y, param: jnp.subtract(x, y),
            '*': lambda x, y, param: jnp.multiply(x, y),
            '/': lambda x, y, param: jnp.divide(x, y)
        }    
        self.RELATIONAL_OPS = {
            '>=': lambda x, y, param: jnp.greater_equal(x, y),
            '<=': lambda x, y, param: jnp.less_equal(x, y),
            '<': lambda x, y, param: jnp.less(x, y),
            '>': lambda x, y, param: jnp.greater(x, y),
            '==': lambda x, y, param: jnp.equal(x, y),
            '~=': lambda x, y, param: jnp.not_equal(x, y)
        }
        self.LOGICAL_NOT = lambda x, param: jnp.logical_not(x)
        self.LOGICAL_OPS = {
            '^': lambda x, y, param: jnp.logical_and(x, y),
            '&': lambda x, y, param: jnp.logical_and(x, y),
            '|': lambda x, y, param: jnp.logical_or(x, y),
            '~': lambda x, y, param: jnp.logical_xor(x, y),
            '=>': lambda x, y, param: jnp.logical_or(jnp.logical_not(x), y),
            '<=>': lambda x, y, param: jnp.equal(x, y)
        }
        self.AGGREGATION_OPS = {
            'sum': lambda x, axis, param: jnp.sum(x, axis=axis),
            'avg': lambda x, axis, param: jnp.mean(x, axis=axis),
            'prod': lambda x, axis, param: jnp.prod(x, axis=axis),
            'minimum': lambda x, axis, param: jnp.min(x, axis=axis),
            'maximum': lambda x, axis, param: jnp.max(x, axis=axis),
            'forall': lambda x, axis, param: jnp.all(x, axis=axis),
            'exists': lambda x, axis, param: jnp.any(x, axis=axis),
            'argmin': lambda x, axis, param: jnp.argmin(x, axis=axis),
            'argmax': lambda x, axis, param: jnp.argmax(x, axis=axis)
        }
        self.AGGREGATION_BOOL = {'forall', 'exists'}
        self.KNOWN_UNARY = {        
            'abs': lambda x, param: jnp.abs(x),
            'sgn': lambda x, param: jnp.sign(x),
            'round': lambda x, param: jnp.round(x),
            'floor': lambda x, param: jnp.floor(x),
            'ceil': lambda x, param: jnp.ceil(x),
            'cos': lambda x, param: jnp.cos(x),
            'sin': lambda x, param: jnp.sin(x),
            'tan': lambda x, param: jnp.tan(x),
            'acos': lambda x, param: jnp.arccos(x),
            'asin': lambda x, param: jnp.arcsin(x),
            'atan': lambda x, param: jnp.arctan(x),
            'cosh': lambda x, param: jnp.cosh(x),
            'sinh': lambda x, param: jnp.sinh(x),
            'tanh': lambda x, param: jnp.tanh(x),
            'exp': lambda x, param: jnp.exp(x),
            'ln': lambda x, param: jnp.log(x),
            'sqrt': lambda x, param: jnp.sqrt(x),
            'lngamma': lambda x, param: scipy.special.gammaln(x),
            'gamma': lambda x, param: jnp.exp(scipy.special.gammaln(x))
        }        
        self.KNOWN_BINARY = {
            'div': lambda x, y, param: jnp.floor_divide(x, y),
            'mod': lambda x, y, param: jnp.mod(x, y),
            'fmod': lambda x, y, param: jnp.mod(x, y),
            'min': lambda x, y, param: jnp.minimum(x, y),
            'max': lambda x, y, param: jnp.maximum(x, y),
            'pow': lambda x, y, param: jnp.power(x, y),
            'log': lambda x, y, param: jnp.log(x) / jnp.log(y),
            'hypot': lambda x, y, param: jnp.hypot(x, y)
        }
        
    # ===========================================================================
    # main compilation subroutines
    # ===========================================================================
     
    def compile(self, log_jax_expr: bool=False) -> None: 
        '''Compiles the current RDDL into Jax expressions.
        
        :param log_jax_expr: whether to pretty-print the compiled Jax functions
        to the log file
        '''
        info = {}
        self.invariants = self._compile_constraints(self.rddl.invariants, info)
        self.preconditions = self._compile_constraints(self.rddl.preconditions, info)
        self.termination = self._compile_constraints(self.rddl.terminals, info)
        self.cpfs = self._compile_cpfs(info)
        self.reward = self._compile_reward(info)
        self.model_params = info
        
        if log_jax_expr and self.logger is not None:
            printed = self.print_jax()
            printed_cpfs = '\n\n'.join(f'{k}: {v}' 
                                       for (k, v) in printed['cpfs'].items())
            printed_reward = printed['reward']
            printed_invariants = '\n\n'.join(v for v in printed['invariants'])
            printed_preconds = '\n\n'.join(v for v in printed['preconditions'])
            printed_terminals = '\n\n'.join(v for v in printed['terminations'])
            printed_params = '\n'.join(f'{k}: {v}' for (k, v) in info.items())
            message = (
                f'[info] compiled JAX CPFs:\n\n'
                f'{printed_cpfs}\n\n'
                f'[info] compiled JAX reward:\n\n'
                f'{printed_reward}\n\n'
                f'[info] compiled JAX invariants:\n\n'
                f'{printed_invariants}\n\n'
                f'[info] compiled JAX preconditions:\n\n'
                f'{printed_preconds}\n\n'
                f'[info] compiled JAX terminations:\n\n'
                f'{printed_terminals}\n'
                f'[info] model parameters:\n'
                f'{printed_params}\n'
            )
            self.logger.log(message)
    
    def _compile_constraints(self, constraints, info):
        return [self._jax(expr, info, dtype=bool) for expr in constraints]
        
    def _compile_cpfs(self, info):
        jax_cpfs = {}
        for cpfs in self.levels.values():
            for cpf in cpfs:
                _, expr = self.rddl.cpfs[cpf]
                prange = self.rddl.variable_ranges[cpf]
                dtype = self.JAX_TYPES.get(prange, self.INT)
                jax_cpfs[cpf] = self._jax(expr, info, dtype=dtype)
        return jax_cpfs
    
    def _compile_reward(self, info):
        return self._jax(self.rddl.reward, info, dtype=self.REAL)
    
    def _extract_inequality_constraint(self, expr):
        result = []
        etype, op = expr.etype
        if etype == 'relational':
            left, right = expr.args
            if op == '<' or op == '<=':
                result.append((left, right))
            elif op == '>' or op == '>=':
                result.append((right, left))
        elif etype == 'boolean' and op == '^':
            for arg in expr.args:
                result.extend(self._extract_inequality_constraint(arg))
        return result
    
    def _extract_equality_constraint(self, expr):
        result = []
        etype, op = expr.etype
        if etype == 'relational':
            left, right = expr.args
            if op == '==':
                result.append((left, right))
        elif etype == 'boolean' and op == '^':
            for arg in expr.args:
                result.extend(self._extract_equality_constraint(arg))
        return result
            
    def _jax_nonlinear_constraints(self): 
        rddl = self.rddl
        
        # extract the non-box inequality constraints on actions
        inequalities = [constr 
                        for (i, expr) in enumerate(rddl.preconditions)
                        for constr in self._extract_inequality_constraint(expr)
                        if not self.constraints.is_box_preconditions[i]]
        
        # compile them to JAX and write as h(s, a) <= 0
        op = self.ARITHMETIC_OPS['-']        
        jax_inequalities = []
        for (left, right) in inequalities:
            jax_lhs = self._jax(left, {})
            jax_rhs = self._jax(right, {})
            jax_constr = self._jax_binary(jax_lhs, jax_rhs, op, '', at_least_int=True)
            jax_inequalities.append(jax_constr)
        
        # extract the non-box equality constraints on actions
        equalities = [constr 
                      for (i, expr) in enumerate(rddl.preconditions)
                      for constr in self._extract_equality_constraint(expr)
                      if not self.constraints.is_box_preconditions[i]]
        
        # compile them to JAX and write as g(s, a) == 0
        jax_equalities = []
        for (left, right) in equalities:
            jax_lhs = self._jax(left, {})
            jax_rhs = self._jax(right, {})
            jax_constr = self._jax_binary(jax_lhs, jax_rhs, op, '', at_least_int=True)
            jax_equalities.append(jax_constr)
            
        return jax_inequalities, jax_equalities
    
    def compile_rollouts(self, policy: Callable,
                         n_steps: int, 
                         n_batch: int,
                         check_constraints: bool=False,
                         constraint_func: bool=False):
        '''Compiles the current RDDL into a wrapped function that samples multiple
        rollouts (state trajectories) in batched form for the given policy. The
        wrapped function takes the policy parameters and RNG key as input, and
        returns a dictionary of all logged information from the rollouts.
        
        constraint_func provides the option to compile nonlinear constraints:
        
            1. f(s, a) ?? g(s, a)
            2. f1(s, a) ^ f2(s, a) ^ ... ?? g(s, a)
            3. forall_{?p1, ...} f(s, a, ?p1, ...) ?? g(s, a) where f is of the
               form 1 or 2 above.
        
        and where ?? is <, <=, > or >= into JAX expressions h(s, a) representing 
        the constraints of the form: 
            
            h(s, a) <= 0
            g(s, a) == 0
                
        for which a penalty or barrier-type method can be used to enforce 
        constraint satisfaction. A list is returned containing values for all
        non-box inequality constraints.
        
        :param policy: a Jax compiled function that takes the policy parameters, 
        decision epoch, state dict, and an RNG key and returns an action dict
        :param n_steps: the length of each rollout
        :param n_batch: how many rollouts each batch performs
        :param check_constraints: whether state, action and termination 
        conditions should be checked on each time step: this info is stored in the
        returned log and does not raise an exception
        :param constraint_func: produces the h(s, a) function described above
        in addition to the usual outputs
        '''
        NORMAL = JaxRDDLCompiler.ERROR_CODES['NORMAL']
        
        rddl = self.rddl
        reward_fn, cpfs = self.reward, self.cpfs
        preconds, invariants, terminals = \
            self.preconditions, self.invariants, self.termination
            
        if constraint_func:
            inequality_fns, equality_fns = self._jax_nonlinear_constraints()
        else:
            inequality_fns, equality_fns = None, None
        
        # do a single step update from the RDDL model
        def _jax_wrapped_single_step(key, policy_params, hyperparams, 
                                     step, subs, model_params):
            errors = NORMAL
            
            # compute action
            key, subkey = random.split(key)
            states = {var: values 
                      for (var, values) in subs.items()
                      if rddl.variable_types[var] == 'state-fluent'}
            actions = policy(subkey, policy_params, hyperparams, step, states)
            subs.update(actions)
            
            # check action preconditions
            precond_check = True
            if check_constraints:
                for precond in preconds:
                    sample, key, err = precond(subs, model_params, key)
                    precond_check = jnp.logical_and(precond_check, sample)
                    errors |= err
            
            # compute h(s, a) <= 0 and g(s, a) == 0 constraint functions
            inequalities, equalities = [], []
            if constraint_func:
                for constraint in inequality_fns:
                    sample, key, err = constraint(subs, model_params, key)
                    inequalities.append(sample)
                    errors |= err
                for constraint in equality_fns:
                    sample, key, err = constraint(subs, model_params, key)
                    equalities.append(sample)
                    errors |= err
                
            # calculate CPFs in topological order
            for (name, cpf) in cpfs.items():
                subs[name], key, err = cpf(subs, model_params, key)
                errors |= err                
                
            # calculate the immediate reward
            reward, key, err = reward_fn(subs, model_params, key)
            errors |= err
            
            # set the next state to the current state
            for (state, next_state) in rddl.next_state.items():
                subs[state] = subs[next_state]
            
            # check the state invariants
            invariant_check = True
            if check_constraints:
                for invariant in invariants:
                    sample, key, err = invariant(subs, model_params, key)
                    invariant_check = jnp.logical_and(invariant_check, sample)
                    errors |= err
            
            # check the termination (TODO: zero out reward in s if terminated)
            terminated_check = False
            if check_constraints:
                for terminal in terminals:
                    sample, key, err = terminal(subs, model_params, key)
                    terminated_check = jnp.logical_or(terminated_check, sample)
                    errors |= err
            
            # prepare the return value
            log = {
                'pvar': subs,
                'action': actions,
                'reward': reward,
                'error': errors,
                'precondition': precond_check,
                'invariant': invariant_check,
                'termination': terminated_check
            }            
            if constraint_func:
                log['inequalities'] = inequalities
                log['equalities'] = equalities
                
            return log, subs
        
        # do a batched step update from the RDDL model
        def _jax_wrapped_batched_step(carry, step):
            key, policy_params, hyperparams, subs, model_params = carry  
            key, *subkeys = random.split(key, num=1 + n_batch)
            keys = jnp.asarray(subkeys)
            batched_step = jax.vmap(
                _jax_wrapped_single_step,
                in_axes=(0, None, None, None, 0, None)
            ) 
            log, subs = batched_step(
                keys, policy_params, hyperparams, step, subs,  model_params)            
            carry = (key, policy_params, hyperparams, subs, model_params)
            return carry, log            
            
        # do a batched roll-out from the RDDL model
        def _jax_wrapped_batched_rollout(key, policy_params, hyperparams, 
                                         subs, model_params):
            start = (key, policy_params, hyperparams, subs, model_params)
            steps = jnp.arange(n_steps)
            _, log = jax.lax.scan(_jax_wrapped_batched_step, start, steps)
            log = jax.tree_map(partial(jnp.swapaxes, axis1=0, axis2=1), log)
            return log
        
        return _jax_wrapped_batched_rollout
    
    # ===========================================================================
    # error checks
    # ===========================================================================
    
    def print_jax(self) -> Dict[str, object]:
        '''Returns a dictionary containing the string representations of all 
        Jax compiled expressions from the RDDL file.
        '''
        subs = self.init_values
        params = self.model_params
        key = jax.random.PRNGKey(42)
        printed = {}
        printed['cpfs'] = {name: str(jax.make_jaxpr(expr)(subs, params, key))
                           for (name, expr) in self.cpfs.items()}
        printed['reward'] = str(jax.make_jaxpr(self.reward)(subs, params, key))
        printed['invariants'] = [str(jax.make_jaxpr(expr)(subs, params, key))
                                 for expr in self.invariants]
        printed['preconditions'] = [str(jax.make_jaxpr(expr)(subs, params, key))
                                    for expr in self.preconditions]
        printed['terminations'] = [str(jax.make_jaxpr(expr)(subs, params, key))
                                   for expr in self.termination]
        return printed
        
    @staticmethod
    def _check_valid_op(expr, valid_ops):
        etype, op = expr.etype
        if op not in valid_ops:
            valid_op_str = ','.join(valid_ops.keys())
            raise RDDLNotImplementedError(
                f'{etype} operator {op} is not supported: '
                f'must be in {valid_op_str}.\n' + 
                print_stack_trace(expr))
    
    @staticmethod
    def _check_num_args(expr, required_args):
        actual_args = len(expr.args)
        if actual_args != required_args:
            etype, op = expr.etype
            raise RDDLInvalidNumberOfArgumentsError(
                f'{etype} operator {op} requires {required_args} arguments, '
                f'got {actual_args}.\n' + 
                print_stack_trace(expr))
        
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
        'INVALID_PARAM_KUMARASWAMY': 131072,
        'INVALID_PARAM_DISCRETE': 262144,
        'INVALID_PARAM_KRON_DELTA': 524288,
        'INVALID_PARAM_DIRICHLET': 1048576,
        'INVALID_PARAM_MULTIVARIATE_STUDENT': 2097152,
        'INVALID_PARAM_MULTINOMIAL': 4194304,
        'INVALID_PARAM_BINOMIAL': 8388608,
        'INVALID_PARAM_NEGATIVE_BINOMIAL': 16777216        
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
        17: 'Found Kumaraswamy(a, b) distribution where either a <= 0 or b <= 0.',
        18: 'Found Discrete(p) distribution where either p < 0 or p does not sum to 1.',
        19: 'Found KronDelta(x) distribution where x is not int nor bool.',
        20: 'Found Dirichlet(alpha) distribution where alpha < 0.',
        21: 'Found MultivariateStudent(mean, cov, df) distribution where df <= 0.',
        22: 'Found Multinomial(n, p) distribution where either p < 0, p does not sum to 1, or n <= 0.',
        23: 'Found Binomial(n, p) distribution where either p < 0, p > 1, or n <= 0.',
        24: 'Found NegativeBinomial(n, p) distribution where either p < 0, p > 1, or n <= 0.'        
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
    # handling of auxiliary data (e.g. model tuning parameters)
    # ===========================================================================
    
    def _unwrap(self, op, expr_id, info):
        sep = JaxRDDLCompiler.MODEL_PARAM_TAG_SEPARATOR
        jax_op, name = op, None
        if isinstance(op, tuple):
            jax_op, param = op
            if param is not None:
                tags, values = param
                if isinstance(tags, tuple):
                    name = sep.join(tags)
                else:
                    name = str(tags)
                name = f'{name}{sep}{expr_id}'
                if name in info:
                    raise Exception(f'Model parameter {name} is already defined.')
                info[name] = values
        return jax_op, name
    
    def get_ids_of_parameterized_expressions(self) -> List[int]:
        '''Returns a list of expression IDs that have tuning parameters.'''
        sep = JaxRDDLCompiler.MODEL_PARAM_TAG_SEPARATOR
        ids = [int(key.split(sep)[-1]) for key in self.model_params] 
        return ids
    
    # ===========================================================================
    # expression compilation
    # ===========================================================================
    
    def _jax(self, expr, info, dtype=None):
        etype, _ = expr.etype
        if etype == 'constant':
            jax_expr = self._jax_constant(expr, info)
        elif etype == 'pvar':
            jax_expr = self._jax_pvar(expr, info)
        elif etype == 'arithmetic':
            jax_expr = self._jax_arithmetic(expr, info)
        elif etype == 'relational':
            jax_expr = self._jax_relational(expr, info)
        elif etype == 'boolean':
            jax_expr = self._jax_logical(expr, info)
        elif etype == 'aggregation':
            jax_expr = self._jax_aggregation(expr, info)
        elif etype == 'func':
            jax_expr = self._jax_functional(expr, info)
        elif etype == 'control':
            jax_expr = self._jax_control(expr, info)
        elif etype == 'randomvar':
            jax_expr = self._jax_random(expr, info)
        elif etype == 'randomvector':
            jax_expr = self._jax_random_vector(expr, info)
        elif etype == 'matrix':
            jax_expr = self._jax_matrix(expr, info)
        else:
            raise RDDLNotImplementedError(
                f'Internal error: expression type {expr} is not supported.\n' + 
                print_stack_trace(expr))
                
        if dtype is not None:
            jax_expr = self._jax_cast(jax_expr, dtype)
        
        return jax_expr
            
    def _jax_cast(self, jax_expr, dtype):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_CAST']
        
        def _jax_wrapped_cast(x, params, key):
            val, key, err = jax_expr(x, params, key)
            sample = jnp.asarray(val, dtype=dtype)
            invalid_cast = jnp.logical_and(
                jnp.logical_not(jnp.can_cast(val, dtype)),
                jnp.any(sample != val))
            err |= (invalid_cast * ERR)
            return sample, key, err
        
        return _jax_wrapped_cast
   
    # ===========================================================================
    # leaves
    # ===========================================================================
    
    def _jax_constant(self, expr, info):
        NORMAL = JaxRDDLCompiler.ERROR_CODES['NORMAL']
        cached_value = self.traced.cached_sim_info(expr)
        
        def _jax_wrapped_constant(x, params, key):
            sample = jnp.asarray(cached_value)
            return sample, key, NORMAL

        return _jax_wrapped_constant
    
    def _jax_pvar_slice(self, _slice):
        NORMAL = JaxRDDLCompiler.ERROR_CODES['NORMAL']
        
        def _jax_wrapped_pvar_slice(x, params, key):
            return _slice, key, NORMAL
        
        return _jax_wrapped_pvar_slice
            
    def _jax_pvar(self, expr, info):
        NORMAL = JaxRDDLCompiler.ERROR_CODES['NORMAL']
        var, pvars = expr.args  
        is_value, cached_info = self.traced.cached_sim_info(expr)
        
        # boundary case: free variable is converted to array (0, 1, 2...)
        # boundary case: domain object is converted to canonical integer index
        if is_value:
            cached_value = cached_info

            def _jax_wrapped_object(x, params, key):
                sample = jnp.asarray(cached_value)
                return sample, key, NORMAL
            
            return _jax_wrapped_object
        
        # boundary case: no shape information (e.g. scalar pvar)
        elif cached_info is None:
            
            def _jax_wrapped_pvar_scalar(x, params, key):
                sample = jnp.asarray(x[var])
                return sample, key, NORMAL
            
            return _jax_wrapped_pvar_scalar
        
        # must slice and/or reshape value tensor to match free variables
        else:
            slices, axis, shape, op_code, op_args = cached_info 
        
            # compile nested expressions
            if slices and op_code == RDDLObjectsTracer.NUMPY_OP_CODE.NESTED_SLICE:
                
                jax_nested_expr = [(self._jax(arg, info) 
                                    if _slice is None 
                                    else self._jax_pvar_slice(_slice))
                                   for (arg, _slice) in zip(pvars, slices)]    
                
                def _jax_wrapped_pvar_tensor_nested(x, params, key):
                    error = NORMAL
                    sample = jnp.asarray(x[var])
                    new_slices = [None] * len(jax_nested_expr)
                    for (i, jax_expr) in enumerate(jax_nested_expr):
                        new_slices[i], key, err = jax_expr(x, params, key)
                        error |= err
                    new_slices = tuple(new_slices)
                    sample = sample[new_slices]
                    return sample, key, error
                
                return _jax_wrapped_pvar_tensor_nested
                
            # tensor variable but no nesting  
            else:
    
                def _jax_wrapped_pvar_tensor_non_nested(x, params, key):
                    sample = jnp.asarray(x[var])
                    if slices:
                        sample = sample[slices]
                    if axis:
                        sample = jnp.expand_dims(sample, axis=axis)
                        sample = jnp.broadcast_to(sample, shape=shape)
                    if op_code == RDDLObjectsTracer.NUMPY_OP_CODE.EINSUM:
                        sample = jnp.einsum(sample, *op_args)
                    elif op_code == RDDLObjectsTracer.NUMPY_OP_CODE.TRANSPOSE:
                        sample = jnp.transpose(sample, axes=op_args)
                    return sample, key, NORMAL
                
                return _jax_wrapped_pvar_tensor_non_nested
    
    # ===========================================================================
    # mathematical
    # ===========================================================================
    
    def _jax_unary(self, jax_expr, jax_op, jax_param,
                   at_least_int=False, check_dtype=None):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_CAST']

        def _jax_wrapped_unary_op(x, params, key):
            sample, key, err = jax_expr(x, params, key)
            if at_least_int:
                sample = self.ONE * sample
            param = params.get(jax_param, None)
            sample = jax_op(sample, param)
            if check_dtype is not None:
                invalid_cast = jnp.logical_not(jnp.can_cast(sample, check_dtype))
                err |= (invalid_cast * ERR)
            return sample, key, err
        
        return _jax_wrapped_unary_op
    
    def _jax_binary(self, jax_lhs, jax_rhs, jax_op, jax_param,
                    at_least_int=False, check_dtype=None):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_CAST']
        
        def _jax_wrapped_binary_op(x, params, key):
            sample1, key, err1 = jax_lhs(x, params, key)
            sample2, key, err2 = jax_rhs(x, params, key)
            if at_least_int:
                sample1 = self.ONE * sample1
                sample2 = self.ONE * sample2
            param = params.get(jax_param, None)
            sample = jax_op(sample1, sample2, param)
            err = err1 | err2
            if check_dtype is not None:
                invalid_cast = jnp.logical_not(jnp.logical_and(
                    jnp.can_cast(sample1, check_dtype),
                    jnp.can_cast(sample2, check_dtype)))
                err |= (invalid_cast * ERR)
            return sample, key, err
        
        return _jax_wrapped_binary_op
    
    def _jax_arithmetic(self, expr, info):
        _, op = expr.etype
        valid_ops = self.ARITHMETIC_OPS
        JaxRDDLCompiler._check_valid_op(expr, valid_ops)
                    
        args = expr.args
        n = len(args)
        
        if n == 1 and op == '-':
            arg, = args
            jax_expr = self._jax(arg, info)
            jax_op, jax_param = self._unwrap(self.NEGATIVE, expr.id, info)
            return self._jax_unary(jax_expr, jax_op, jax_param, at_least_int=True)
                    
        elif n == 2:
            lhs, rhs = args
            jax_lhs = self._jax(lhs, info)
            jax_rhs = self._jax(rhs, info)
            jax_op, jax_param = self._unwrap(valid_ops[op], expr.id, info)
            return self._jax_binary(
                jax_lhs, jax_rhs, jax_op, jax_param, at_least_int=True)
        
        JaxRDDLCompiler._check_num_args(expr, 2)
    
    def _jax_relational(self, expr, info):
        _, op = expr.etype
        valid_ops = self.RELATIONAL_OPS
        JaxRDDLCompiler._check_valid_op(expr, valid_ops)
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        lhs, rhs = expr.args
        jax_lhs = self._jax(lhs, info)
        jax_rhs = self._jax(rhs, info)
        jax_op, jax_param = self._unwrap(valid_ops[op], expr.id, info)
        return self._jax_binary(
            jax_lhs, jax_rhs, jax_op, jax_param, at_least_int=True)
           
    def _jax_logical(self, expr, info):
        _, op = expr.etype
        valid_ops = self.LOGICAL_OPS    
        JaxRDDLCompiler._check_valid_op(expr, valid_ops)
                
        args = expr.args
        n = len(args)
        
        if n == 1 and op == '~':
            arg, = args
            jax_expr = self._jax(arg, info)
            jax_op, jax_param = self._unwrap(self.LOGICAL_NOT, expr.id, info)
            return self._jax_unary(jax_expr, jax_op, jax_param, check_dtype=bool)
        
        elif n == 2:
            lhs, rhs = args
            jax_lhs = self._jax(lhs, info)
            jax_rhs = self._jax(rhs, info)
            jax_op, jax_param = self._unwrap(valid_ops[op], expr.id, info)
            return self._jax_binary(
                jax_lhs, jax_rhs, jax_op, jax_param, check_dtype=bool)
        
        JaxRDDLCompiler._check_num_args(expr, 2)
    
    def _jax_aggregation(self, expr, info):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_CAST']
        
        _, op = expr.etype
        valid_ops = self.AGGREGATION_OPS      
        JaxRDDLCompiler._check_valid_op(expr, valid_ops) 
        is_floating = op not in self.AGGREGATION_BOOL
        
        * _, arg = expr.args  
        _, axes = self.traced.cached_sim_info(expr)
        
        jax_expr = self._jax(arg, info)
        jax_op, jax_param = self._unwrap(valid_ops[op], expr.id, info)
        
        def _jax_wrapped_aggregation(x, params, key):
            sample, key, err = jax_expr(x, params, key)
            if is_floating:
                sample = self.ONE * sample
            else:
                invalid_cast = jnp.logical_not(jnp.can_cast(sample, bool))
                err |= (invalid_cast * ERR)
            param = params.get(jax_param, None)
            sample = jax_op(sample, axis=axes, param=param)
            return sample, key, err
        
        return _jax_wrapped_aggregation
               
    def _jax_functional(self, expr, info):
        _, op = expr.etype
        
        # unary function
        if op in self.KNOWN_UNARY:
            JaxRDDLCompiler._check_num_args(expr, 1)                            
            arg, = expr.args
            jax_expr = self._jax(arg, info)
            jax_op, jax_param = self._unwrap(self.KNOWN_UNARY[op], expr.id, info)
            return self._jax_unary(jax_expr, jax_op, jax_param, at_least_int=True)
            
        # binary function
        elif op in self.KNOWN_BINARY:
            JaxRDDLCompiler._check_num_args(expr, 2)                
            lhs, rhs = expr.args
            jax_lhs = self._jax(lhs, info)
            jax_rhs = self._jax(rhs, info)
            jax_op, jax_param = self._unwrap(self.KNOWN_BINARY[op], expr.id, info)
            return self._jax_binary(
                jax_lhs, jax_rhs, jax_op, jax_param, at_least_int=True)
        
        raise RDDLNotImplementedError(
            f'Function {op} is not supported.\n' + 
            print_stack_trace(expr))   
    
    # ===========================================================================
    # control flow
    # ===========================================================================
    
    def _jax_control(self, expr, info):
        _, op = expr.etype        
        if op == 'if':
            return self._jax_if(expr, info)
        elif op == 'switch':
            return self._jax_switch(expr, info)
        
        raise RDDLNotImplementedError(
            f'Control operator {op} is not supported.\n' + 
            print_stack_trace(expr))   
    
    def _jax_if_helper(self):
        
        def _jax_wrapped_if_calc_exact(c, a, b, param):
            return jnp.where(c, a, b)
        
        return _jax_wrapped_if_calc_exact
    
    def _jax_if(self, expr, info):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_CAST']
        JaxRDDLCompiler._check_num_args(expr, 3)
        jax_if, jax_param = self._unwrap(self._jax_if_helper(), expr.id, info)
        
        pred, if_true, if_false = expr.args        
        jax_pred = self._jax(pred, info)
        jax_true = self._jax(if_true, info)
        jax_false = self._jax(if_false, info)
        
        def _jax_wrapped_if_then_else(x, params, key):
            sample1, key, err1 = jax_pred(x, params, key)
            sample2, key, err2 = jax_true(x, params, key)
            sample3, key, err3 = jax_false(x, params, key)
            param = params.get(jax_param, None)
            sample = jax_if(sample1, sample2, sample3, param)
            err = err1 | err2 | err3
            invalid_cast = jnp.logical_not(jnp.can_cast(sample1, bool))
            err |= (invalid_cast * ERR)
            return sample, key, err
            
        return _jax_wrapped_if_then_else
    
    def _jax_switch_helper(self):
        
        def _jax_wrapped_switch_calc_exact(pred, cases, param):
            pred = pred[jnp.newaxis, ...]
            sample = jnp.take_along_axis(cases, pred, axis=0)
            assert sample.shape[0] == 1
            return sample[0, ...]

        return _jax_wrapped_switch_calc_exact
        
    def _jax_switch(self, expr, info):
        pred, *_ = expr.args             
        jax_pred = self._jax(pred, info)
        jax_switch, jax_param = self._unwrap(
            self._jax_switch_helper(), expr.id, info)
        
        # wrap cases as JAX expressions
        cases, default = self.traced.cached_sim_info(expr) 
        jax_default = None if default is None else self._jax(default, info)
        jax_cases = [(jax_default if _case is None else self._jax(_case, info))
                     for _case in cases]
                    
        def _jax_wrapped_switch(x, params, key):
            
            # sample predicate
            sample_pred, key, err = jax_pred(x, params, key) 
            
            # sample cases
            sample_cases = [None] * len(jax_cases)
            for (i, jax_case) in enumerate(jax_cases):
                sample_cases[i], key, err_case = jax_case(x, params, key)
                err |= err_case                
            sample_cases = jnp.asarray(sample_cases)
            
            # predicate (enum) is an integer - use it to extract from case array
            param = params.get(jax_param, None)
            sample = jax_switch(sample_pred, sample_cases, param)
            return sample, key, err    
        
        return _jax_wrapped_switch
    
    # ===========================================================================
    # random variables
    # ===========================================================================
    
    # distributions with complete reparameterization support:
    # KronDelta: complete
    # DiracDelta: complete
    # Uniform: complete
    # Bernoulli: complete (subclass uses Gumbel-softmax)
    # Normal: complete
    # Exponential: complete
    # Weibull: complete
    # Pareto: complete
    # Gumbel: complete
    # Laplace: complete
    # Cauchy: complete
    # Gompertz: complete
    # Kumaraswamy: complete
    # Discrete: complete (subclass uses Gumbel-softmax)
    # UnnormDiscrete: complete (subclass uses Gumbel-softmax)
    # Discrete(p): complete (subclass uses Gumbel-softmax)
    # UnnormDiscrete(p): complete (subclass uses Gumbel-softmax)
    
    # distributions with incomplete reparameterization support (TODO):
    # Binomial: (use truncation and Gumbel-softmax)
    # NegativeBinomial: (no reparameterization)
    # Poisson: (use truncation and Gumbel-softmax)
    # Gamma, ChiSquare: (no shape reparameterization)
    # Beta: (no reparameterization)
    # Geometric: (implement safe floor)
    # Student: (no reparameterization)
    
    def _jax_random(self, expr, info):
        _, name = expr.etype
        if name == 'KronDelta':
            return self._jax_kron(expr, info)        
        elif name == 'DiracDelta':
            return self._jax_dirac(expr, info)
        elif name == 'Uniform':
            return self._jax_uniform(expr, info)
        elif name == 'Bernoulli':
            return self._jax_bernoulli(expr, info)
        elif name == 'Normal':
            return self._jax_normal(expr, info)  
        elif name == 'Poisson':
            return self._jax_poisson(expr, info)
        elif name == 'Exponential':
            return self._jax_exponential(expr, info)
        elif name == 'Weibull':
            return self._jax_weibull(expr, info) 
        elif name == 'Gamma':
            return self._jax_gamma(expr, info)
        elif name == 'Binomial':
            return self._jax_binomial(expr, info)
        elif name == 'NegativeBinomial':
            return self._jax_negative_binomial(expr, info)
        elif name == 'Beta':
            return self._jax_beta(expr, info)
        elif name == 'Geometric':
            return self._jax_geometric(expr, info)
        elif name == 'Pareto':
            return self._jax_pareto(expr, info)
        elif name == 'Student':
            return self._jax_student(expr, info)
        elif name == 'Gumbel':
            return self._jax_gumbel(expr, info)
        elif name == 'Laplace':
            return self._jax_laplace(expr, info)
        elif name == 'Cauchy':
            return self._jax_cauchy(expr, info)
        elif name == 'Gompertz':
            return self._jax_gompertz(expr, info)
        elif name == 'ChiSquare':
            return self._jax_chisquare(expr, info)
        elif name == 'Kumaraswamy':
            return self._jax_kumaraswamy(expr, info)
        elif name == 'Discrete':
            return self._jax_discrete(expr, info, unnorm=False)
        elif name == 'UnnormDiscrete':
            return self._jax_discrete(expr, info, unnorm=True)
        elif name == 'Discrete(p)':
            return self._jax_discrete_pvar(expr, info, unnorm=False)
        elif name == 'UnnormDiscrete(p)':
            return self._jax_discrete_pvar(expr, info, unnorm=True)
        else:
            raise RDDLNotImplementedError(
                f'Distribution {name} is not supported.\n' + 
                print_stack_trace(expr))
        
    def _jax_kron(self, expr, info):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_KRON_DELTA']
        JaxRDDLCompiler._check_num_args(expr, 1)
        arg, = expr.args
        arg = self._jax(arg, info)
        
        # just check that the sample can be cast to int
        def _jax_wrapped_distribution_kron(x, params, key):
            sample, key, err = arg(x, params, key)
            invalid_cast = jnp.logical_not(jnp.can_cast(sample, self.INT))
            err |= (invalid_cast * ERR)
            return sample, key, err
                        
        return _jax_wrapped_distribution_kron
    
    def _jax_dirac(self, expr, info):
        JaxRDDLCompiler._check_num_args(expr, 1)
        arg, = expr.args
        arg = self._jax(arg, info, dtype=self.REAL)
        return arg
    
    def _jax_uniform(self, expr, info):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_UNIFORM']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_lb, arg_ub = expr.args
        jax_lb = self._jax(arg_lb, info)
        jax_ub = self._jax(arg_ub, info)
        
        # reparameterization trick U(a, b) = a + (b - a) * U(0, 1)
        def _jax_wrapped_distribution_uniform(x, params, key):
            lb, key, err1 = jax_lb(x, params, key)
            ub, key, err2 = jax_ub(x, params, key)
            key, subkey = random.split(key)
            U = random.uniform(key=subkey, shape=jnp.shape(lb), dtype=self.REAL)
            sample = lb + (ub - lb) * U
            out_of_bounds = jnp.logical_not(jnp.all(lb <= ub))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_uniform
    
    def _jax_normal(self, expr, info):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_NORMAL']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_mean, arg_var = expr.args
        jax_mean = self._jax(arg_mean, info)
        jax_var = self._jax(arg_var, info)
        
        # reparameterization trick N(m, s^2) = m + s * N(0, 1)
        def _jax_wrapped_distribution_normal(x, params, key):
            mean, key, err1 = jax_mean(x, params, key)
            var, key, err2 = jax_var(x, params, key)
            std = jnp.sqrt(var)
            key, subkey = random.split(key)
            Z = random.normal(key=subkey, shape=jnp.shape(mean), dtype=self.REAL)
            sample = mean + std * Z
            out_of_bounds = jnp.logical_not(jnp.all(var >= 0))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_normal
    
    def _jax_exponential(self, expr, info):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_EXPONENTIAL']
        JaxRDDLCompiler._check_num_args(expr, 1)
        
        arg_scale, = expr.args
        jax_scale = self._jax(arg_scale, info)
        
        # reparameterization trick Exp(s) = s * Exp(1)
        def _jax_wrapped_distribution_exp(x, params, key):
            scale, key, err = jax_scale(x, params, key)
            key, subkey = random.split(key)
            Exp1 = random.exponential(
                key=subkey, shape=jnp.shape(scale), dtype=self.REAL)
            sample = scale * Exp1
            out_of_bounds = jnp.logical_not(jnp.all(scale > 0))
            err |= (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_exp
    
    def _jax_weibull(self, expr, info):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_WEIBULL']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_shape, arg_scale = expr.args
        jax_shape = self._jax(arg_shape, info)
        jax_scale = self._jax(arg_scale, info)
        
        # reparameterization trick W(s, r) = r * (-ln(1 - U(0, 1))) ** (1 / s)
        def _jax_wrapped_distribution_weibull(x, params, key):
            shape, key, err1 = jax_shape(x, params, key)
            scale, key, err2 = jax_scale(x, params, key)
            key, subkey = random.split(key)
            U = random.uniform(key=subkey, shape=jnp.shape(scale), dtype=self.REAL)
            sample = scale * jnp.power(-jnp.log1p(-U), 1.0 / shape)
            out_of_bounds = jnp.logical_not(jnp.all((shape > 0) & (scale > 0)))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_weibull
    
    def _jax_bernoulli_helper(self):
        
        def _jax_wrapped_calc_bernoulli_exact(key, prob, param):
            return random.bernoulli(key, prob)
        
        return _jax_wrapped_calc_bernoulli_exact
        
    def _jax_bernoulli(self, expr, info):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_BERNOULLI']
        JaxRDDLCompiler._check_num_args(expr, 1)
        jax_bern, jax_param = self._unwrap(
            self._jax_bernoulli_helper(), expr.id, info)
        
        arg_prob, = expr.args
        jax_prob = self._jax(arg_prob, info)
        
        # uses the implicit JAX subroutine
        def _jax_wrapped_distribution_bernoulli(x, params, key):
            prob, key, err = jax_prob(x, params, key)
            key, subkey = random.split(key)
            param = params.get(jax_param, None)
            sample = jax_bern(subkey, prob, param)
            out_of_bounds = jnp.logical_not(jnp.all((prob >= 0) & (prob <= 1)))
            err |= (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_bernoulli
    
    def _jax_poisson(self, expr, info):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_POISSON']
        JaxRDDLCompiler._check_num_args(expr, 1)
        
        arg_rate, = expr.args
        jax_rate = self._jax(arg_rate, info)
        
        # uses the implicit JAX subroutine
        def _jax_wrapped_distribution_poisson(x, params, key):
            rate, key, err = jax_rate(x, params, key)
            key, subkey = random.split(key)
            sample = random.poisson(key=subkey, lam=rate, dtype=self.INT)
            out_of_bounds = jnp.logical_not(jnp.all(rate >= 0))
            err |= (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_poisson
    
    def _jax_gamma(self, expr, info):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_GAMMA']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_shape, arg_scale = expr.args
        jax_shape = self._jax(arg_shape, info)
        jax_scale = self._jax(arg_scale, info)
        
        # partial reparameterization trick Gamma(s, r) = r * Gamma(s, 1)
        # uses the implicit JAX subroutine for Gamma(s, 1) 
        def _jax_wrapped_distribution_gamma(x, params, key):
            shape, key, err1 = jax_shape(x, params, key)
            scale, key, err2 = jax_scale(x, params, key)
            key, subkey = random.split(key)
            Gamma = random.gamma(key=subkey, a=shape, dtype=self.REAL)
            sample = scale * Gamma
            out_of_bounds = jnp.logical_not(jnp.all((shape > 0) & (scale > 0)))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_gamma
    
    def _jax_binomial(self, expr, info):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_BINOMIAL']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_trials, arg_prob = expr.args
        jax_trials = self._jax(arg_trials, info)
        jax_prob = self._jax(arg_prob, info)
        
        # uses the JAX substrate of tensorflow-probability
        def _jax_wrapped_distribution_binomial(x, params, key):
            trials, key, err2 = jax_trials(x, params, key)       
            prob, key, err1 = jax_prob(x, params, key)
            trials = jnp.asarray(trials, self.REAL)
            prob = jnp.asarray(prob, self.REAL)
            key, subkey = random.split(key)
            dist = tfp.distributions.Binomial(total_count=trials, probs=prob)
            sample = dist.sample(seed=subkey).astype(self.INT)
            out_of_bounds = jnp.logical_not(jnp.all(
                (prob >= 0) & (prob <= 1) & (trials >= 0)))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_binomial
    
    def _jax_negative_binomial(self, expr, info):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_NEGATIVE_BINOMIAL']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_trials, arg_prob = expr.args
        jax_trials = self._jax(arg_trials, info)
        jax_prob = self._jax(arg_prob, info)
        
        # uses the JAX substrate of tensorflow-probability
        def _jax_wrapped_distribution_negative_binomial(x, params, key):
            trials, key, err2 = jax_trials(x, params, key)       
            prob, key, err1 = jax_prob(x, params, key)
            trials = jnp.asarray(trials, self.REAL)
            prob = jnp.asarray(prob, self.REAL)
            key, subkey = random.split(key)
            dist = tfp.distributions.NegativeBinomial(
                total_count=trials, probs=prob)
            sample = dist.sample(seed=subkey).astype(self.INT)
            out_of_bounds = jnp.logical_not(jnp.all(
                (prob >= 0) & (prob <= 1) & (trials > 0)))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_negative_binomial    
        
    def _jax_beta(self, expr, info):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_BETA']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_shape, arg_rate = expr.args
        jax_shape = self._jax(arg_shape, info)
        jax_rate = self._jax(arg_rate, info)
        
        # uses the implicit JAX subroutine
        def _jax_wrapped_distribution_beta(x, params, key):
            shape, key, err1 = jax_shape(x, params, key)
            rate, key, err2 = jax_rate(x, params, key)
            key, subkey = random.split(key)
            sample = random.beta(key=subkey, a=shape, b=rate)
            out_of_bounds = jnp.logical_not(jnp.all((shape > 0) & (rate > 0)))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_beta
    
    def _jax_geometric(self, expr, info):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_GEOMETRIC']
        JaxRDDLCompiler._check_num_args(expr, 1)
        
        arg_prob, = expr.args
        jax_prob = self._jax(arg_prob, info)
        floor_op, jax_param = self._unwrap(
            self.KNOWN_UNARY['floor'], expr.id, info)
        
        # reparameterization trick Geom(p) = floor(ln(U(0, 1)) / ln(p)) + 1
        def _jax_wrapped_distribution_geometric(x, params, key):
            prob, key, err = jax_prob(x, params, key)
            key, subkey = random.split(key)
            U = random.uniform(key=subkey, shape=jnp.shape(prob), dtype=self.REAL)
            param = params.get(jax_param, None)
            sample = floor_op(jnp.log1p(-U) / jnp.log1p(-prob), param) + 1
            out_of_bounds = jnp.logical_not(jnp.all((prob >= 0) & (prob <= 1)))
            err |= (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_geometric
    
    def _jax_pareto(self, expr, info):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_PARETO']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_shape, arg_scale = expr.args
        jax_shape = self._jax(arg_shape, info)
        jax_scale = self._jax(arg_scale, info)
        
        # partial reparameterization trick Pareto(s, r) = r * Pareto(s, 1)
        # uses the implicit JAX subroutine for Pareto(s, 1) 
        def _jax_wrapped_distribution_pareto(x, params, key):
            shape, key, err1 = jax_shape(x, params, key)
            scale, key, err2 = jax_scale(x, params, key)
            key, subkey = random.split(key)
            sample = scale * random.pareto(key=subkey, b=shape)
            out_of_bounds = jnp.logical_not(jnp.all((shape > 0) & (scale > 0)))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_pareto
    
    def _jax_student(self, expr, info):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_STUDENT']
        JaxRDDLCompiler._check_num_args(expr, 1)
        
        arg_df, = expr.args
        jax_df = self._jax(arg_df, info)
        
        # uses the implicit JAX subroutine for student(df)
        def _jax_wrapped_distribution_t(x, params, key):
            df, key, err = jax_df(x, params, key)
            key, subkey = random.split(key)
            sample = random.t(key=subkey, df=df, shape=jnp.shape(df))
            out_of_bounds = jnp.logical_not(jnp.all(df > 0))
            err |= (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_t
    
    def _jax_gumbel(self, expr, info):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_GUMBEL']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_mean, arg_scale = expr.args
        jax_mean = self._jax(arg_mean, info)
        jax_scale = self._jax(arg_scale, info)
        
        # reparameterization trick Gumbel(m, s) = m + s * Gumbel(0, 1)
        def _jax_wrapped_distribution_gumbel(x, params, key):
            mean, key, err1 = jax_mean(x, params, key)
            scale, key, err2 = jax_scale(x, params, key)
            key, subkey = random.split(key)
            Gumbel01 = random.gumbel(
                key=subkey, shape=jnp.shape(mean), dtype=self.REAL)
            sample = mean + scale * Gumbel01
            out_of_bounds = jnp.logical_not(jnp.all(scale > 0))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_gumbel
    
    def _jax_laplace(self, expr, info):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_LAPLACE']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_mean, arg_scale = expr.args
        jax_mean = self._jax(arg_mean, info)
        jax_scale = self._jax(arg_scale, info)
        
        # reparameterization trick Laplace(m, s) = m + s * Laplace(0, 1)
        def _jax_wrapped_distribution_laplace(x, params, key):
            mean, key, err1 = jax_mean(x, params, key)
            scale, key, err2 = jax_scale(x, params, key)
            key, subkey = random.split(key)
            Laplace01 = random.laplace(
                key=subkey, shape=jnp.shape(mean), dtype=self.REAL)
            sample = mean + scale * Laplace01
            out_of_bounds = jnp.logical_not(jnp.all(scale > 0))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_laplace
    
    def _jax_cauchy(self, expr, info):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_CAUCHY']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_mean, arg_scale = expr.args
        jax_mean = self._jax(arg_mean, info)
        jax_scale = self._jax(arg_scale, info)
        
        # reparameterization trick Cauchy(m, s) = m + s * Cauchy(0, 1)
        def _jax_wrapped_distribution_cauchy(x, params, key):
            mean, key, err1 = jax_mean(x, params, key)
            scale, key, err2 = jax_scale(x, params, key)
            key, subkey = random.split(key)
            Cauchy01 = random.cauchy(
                key=subkey, shape=jnp.shape(mean), dtype=self.REAL)
            sample = mean + scale * Cauchy01
            out_of_bounds = jnp.logical_not(jnp.all(scale > 0))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_cauchy
    
    def _jax_gompertz(self, expr, info):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_GOMPERTZ']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_shape, arg_scale = expr.args
        jax_shape = self._jax(arg_shape, info)
        jax_scale = self._jax(arg_scale, info)
        
        # reparameterization trick Gompertz(s, r) = ln(1 - log(U(0, 1)) / s) / r
        def _jax_wrapped_distribution_gompertz(x, params, key):
            shape, key, err1 = jax_shape(x, params, key)
            scale, key, err2 = jax_scale(x, params, key)
            key, subkey = random.split(key)
            U = random.uniform(key=subkey, shape=jnp.shape(scale), dtype=self.REAL)
            sample = jnp.log(1.0 - jnp.log1p(-U) / shape) / scale
            out_of_bounds = jnp.logical_not(jnp.all((shape > 0) & (scale > 0)))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_gompertz
    
    def _jax_chisquare(self, expr, info):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_CHISQUARE']
        JaxRDDLCompiler._check_num_args(expr, 1)
        
        arg_df, = expr.args
        jax_df = self._jax(arg_df, info)
        
        # use the fact that ChiSquare(df) = Gamma(df/2, 2)
        def _jax_wrapped_distribution_chisquare(x, params, key):
            df, key, err1 = jax_df(x, params, key)
            key, subkey = random.split(key)
            shape = df / 2.0
            Gamma = random.gamma(key=subkey, a=shape, dtype=self.REAL)
            sample = 2.0 * Gamma
            out_of_bounds = jnp.logical_not(jnp.all(df > 0))
            err = err1 | (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_chisquare
    
    def _jax_kumaraswamy(self, expr, info):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_KUMARASWAMY']
        JaxRDDLCompiler._check_num_args(expr, 2)
        
        arg_a, arg_b = expr.args
        jax_a = self._jax(arg_a, info)
        jax_b = self._jax(arg_b, info)
        
        # uses the reparameterization K(a, b) = (1 - (1 - U(0, 1))^{1/b})^{1/a}
        def _jax_wrapped_distribution_kumaraswamy(x, params, key):
            a, key, err1 = jax_a(x, params, key)
            b, key, err2 = jax_b(x, params, key)
            key, subkey = random.split(key)
            U = random.uniform(key=subkey, shape=jnp.shape(a), dtype=self.REAL)            
            sample = jnp.power(1.0 - jnp.power(U, 1.0 / b), 1.0 / a)
            out_of_bounds = jnp.logical_not(jnp.all((a > 0) & (b > 0)))
            err = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_kumaraswamy
    
    # ===========================================================================
    # random variables with enum support
    # ===========================================================================
    
    def _jax_discrete_helper(self):
        
        def _jax_wrapped_discrete_calc_exact(key, prob, param):
            logits = jnp.log(prob)
            sample = random.categorical(key=key, logits=logits, axis=-1)
            out_of_bounds = jnp.logical_not(jnp.logical_and(
                jnp.all(prob >= 0),
                jnp.allclose(jnp.sum(prob, axis=-1), 1.0)))
            return sample, out_of_bounds
        
        return _jax_wrapped_discrete_calc_exact
            
    def _jax_discrete(self, expr, info, unnorm):
        NORMAL = JaxRDDLCompiler.ERROR_CODES['NORMAL']
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_DISCRETE']
        jax_discrete, jax_param = self._unwrap(
            self._jax_discrete_helper(), expr.id, info)
        
        ordered_args = self.traced.cached_sim_info(expr)
        jax_probs = [self._jax(arg, info) for arg in ordered_args]
        
        def _jax_wrapped_distribution_discrete(x, params, key):
            
            # sample case probabilities and normalize as needed
            error = NORMAL
            prob = [None] * len(jax_probs)
            for (i, jax_prob) in enumerate(jax_probs):
                prob[i], key, error_pdf = jax_prob(x, params, key)
                error |= error_pdf
            prob = jnp.stack(prob, axis=-1)
            if unnorm:
                normalizer = jnp.sum(prob, axis=-1, keepdims=True)
                prob = prob / normalizer
            
            # dispatch to sampling subroutine
            key, subkey = random.split(key)
            param = params.get(jax_param, None)
            sample, out_of_bounds = jax_discrete(subkey, prob, param)
            error |= (out_of_bounds * ERR)
            return sample, key, error
        
        return _jax_wrapped_distribution_discrete
    
    def _jax_discrete_pvar(self, expr, info, unnorm):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_DISCRETE']
        JaxRDDLCompiler._check_num_args(expr, 1)
        jax_discrete, jax_param = self._unwrap(
            self._jax_discrete_helper(), expr.id, info)
        
        _, args = expr.args
        arg, = args
        jax_probs = self._jax(arg, info)

        def _jax_wrapped_distribution_discrete_pvar(x, params, key):
            
            # sample probabilities
            prob, key, error = jax_probs(x, params, key)
            if unnorm:
                normalizer = jnp.sum(prob, axis=-1, keepdims=True)
                prob = prob / normalizer
            
            # dispatch to sampling subroutine
            key, subkey = random.split(key)
            param = params.get(jax_param, None)
            sample, out_of_bounds = jax_discrete(subkey, prob, param)
            error |= (out_of_bounds * ERR)
            return sample, key, error
        
        return _jax_wrapped_distribution_discrete_pvar

    # ===========================================================================
    # random vectors
    # ===========================================================================
    
    def _jax_random_vector(self, expr, info):
        _, name = expr.etype
        if name == 'MultivariateNormal':
            return self._jax_multivariate_normal(expr, info)   
        elif name == 'MultivariateStudent':
            return self._jax_multivariate_student(expr, info)  
        elif name == 'Dirichlet':
            return self._jax_dirichlet(expr, info)
        elif name == 'Multinomial':
            return self._jax_multinomial(expr, info)
        else:
            raise RDDLNotImplementedError(
                f'Distribution {name} is not supported.\n' + 
                print_stack_trace(expr))
    
    def _jax_multivariate_normal(self, expr, info): 
        _, args = expr.args
        mean, cov = args
        jax_mean = self._jax(mean, info)
        jax_cov = self._jax(cov, info)
        index, = self.traced.cached_sim_info(expr)
        
        # reparameterization trick MN(m, LL') = LZ + m, where Z ~ Normal(0, 1)
        def _jax_wrapped_distribution_multivariate_normal(x, params, key):
            
            # sample the mean and covariance
            sample_mean, key, err1 = jax_mean(x, params, key)
            sample_cov, key, err2 = jax_cov(x, params, key)
            
            # sample Normal(0, 1)
            key, subkey = random.split(key)
            Z = random.normal(
                key=subkey,
                shape=jnp.shape(sample_mean) + (1,),
                dtype=self.REAL)       
            
            # compute L s.t. cov = L * L' and reparameterize
            L = jnp.linalg.cholesky(sample_cov)
            sample = jnp.matmul(L, Z)[..., 0] + sample_mean
            sample = jnp.moveaxis(sample, source=-1, destination=index)
            err = err1 | err2
            return sample, key, err
        
        return _jax_wrapped_distribution_multivariate_normal
    
    def _jax_multivariate_student(self, expr, info):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_MULTIVARIATE_STUDENT']
        
        _, args = expr.args
        mean, cov, df = args
        jax_mean = self._jax(mean, info)
        jax_cov = self._jax(cov, info)
        jax_df = self._jax(df, info)
        index, = self.traced.cached_sim_info(expr)
        
        # reparameterization trick MN(m, LL') = LZ + m, where Z ~ StudentT(0, 1)
        def _jax_wrapped_distribution_multivariate_student(x, params, key):
            
            # sample the mean and covariance and degrees of freedom
            sample_mean, key, err1 = jax_mean(x, params, key)
            sample_cov, key, err2 = jax_cov(x, params, key)
            sample_df, key, err3 = jax_df(x, params, key)
            out_of_bounds = jnp.logical_not(jnp.all(sample_df > 0))
            
            # sample StudentT(0, 1, df) -- broadcast df to same shape as cov
            sample_df = sample_df[..., jnp.newaxis, jnp.newaxis]
            sample_df = jnp.broadcast_to(sample_df, shape=sample_mean.shape + (1,))
            key, subkey = random.split(key)
            Z = random.t(
                key=subkey, 
                df=sample_df, 
                shape=jnp.shape(sample_df),
                dtype=self.REAL)   
            
            # compute L s.t. cov = L * L' and reparameterize
            L = jnp.linalg.cholesky(sample_cov)
            sample = jnp.matmul(L, Z)[..., 0] + sample_mean
            sample = jnp.moveaxis(sample, source=-1, destination=index)
            error = err1 | err2 | err3 | (out_of_bounds * ERR)
            return sample, key, error
        
        return _jax_wrapped_distribution_multivariate_student
    
    def _jax_dirichlet(self, expr, info):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_DIRICHLET']
        
        _, args = expr.args
        alpha, = args
        jax_alpha = self._jax(alpha, info)
        index, = self.traced.cached_sim_info(expr)
        
        # sample Gamma(alpha_i, 1) and normalize across i
        def _jax_wrapped_distribution_dirichlet(x, params, key):
            alpha, key, error = jax_alpha(x, params, key)
            out_of_bounds = jnp.logical_not(jnp.all(alpha > 0))
            error |= (out_of_bounds * ERR)
            key, subkey = random.split(key)
            Gamma = random.gamma(key=subkey, a=alpha)
            sample = Gamma / jnp.sum(Gamma, axis=-1, keepdims=True)
            sample = jnp.moveaxis(sample, source=-1, destination=index)
            return sample, key, error
        
        return _jax_wrapped_distribution_dirichlet
    
    def _jax_multinomial(self, expr, info):
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_MULTINOMIAL']
        
        _, args = expr.args
        trials, prob = args
        jax_trials = self._jax(trials, info)
        jax_prob = self._jax(prob, info)
        index, = self.traced.cached_sim_info(expr)
        
        def _jax_wrapped_distribution_multinomial(x, params, key):
            trials, key, err1 = jax_trials(x, params, key)
            prob, key, err2 = jax_prob(x, params, key)
            trials = jnp.asarray(trials, self.REAL)
            prob = jnp.asarray(prob, self.REAL)
            key, subkey = random.split(key)
            dist = tfp.distributions.Multinomial(total_count=trials, probs=prob)
            sample = dist.sample(seed=subkey).astype(self.INT)
            sample = jnp.moveaxis(sample, source=-1, destination=index)
            out_of_bounds = jnp.logical_not(jnp.all(
                (prob >= 0)
                & jnp.allclose(jnp.sum(prob, axis=-1), 1.0)
                & (trials >= 0)))
            error = err1 | err2 | (out_of_bounds * ERR)
            return sample, key, error            
        
        return _jax_wrapped_distribution_multinomial
    
    # ===========================================================================
    # matrix algebra
    # ===========================================================================
    
    def _jax_matrix(self, expr, info):
        _, op = expr.etype
        if op == 'det':
            return self._jax_matrix_det(expr, info)
        elif op == 'inverse':
            return self._jax_matrix_inv(expr, info, pseudo=False)
        elif op == 'pinverse':
            return self._jax_matrix_inv(expr, info, pseudo=True)
        elif op == 'cholesky':
            return self._jax_matrix_cholesky(expr, info)
        else:
            raise RDDLNotImplementedError(
                f'Matrix operation {op} is not supported.\n' + 
                print_stack_trace(expr))
    
    def _jax_matrix_det(self, expr, info):
        * _, arg = expr.args
        jax_arg = self._jax(arg, info)
        
        def _jax_wrapped_matrix_operation_det(x, params, key):
            sample_arg, key, error = jax_arg(x, params, key)
            sample = jnp.linalg.det(sample_arg)
            return sample, key, error
        
        return _jax_wrapped_matrix_operation_det
    
    def _jax_matrix_inv(self, expr, info, pseudo):
        _, arg = expr.args
        jax_arg = self._jax(arg, info)
        indices = self.traced.cached_sim_info(expr)
        op = jnp.linalg.pinv if pseudo else jnp.linalg.inv
        
        def _jax_wrapped_matrix_operation_inv(x, params, key):
            sample_arg, key, error = jax_arg(x, params, key)
            sample = op(sample_arg)
            sample = jnp.moveaxis(sample, source=(-2, -1), destination=indices)
            return sample, key, error
        
        return _jax_wrapped_matrix_operation_inv
    
    def _jax_matrix_cholesky(self, expr, info):
        _, arg = expr.args
        jax_arg = self._jax(arg, info)
        indices = self.traced.cached_sim_info(expr)
        op = jnp.linalg.cholesky
        
        def _jax_wrapped_matrix_operation_cholesky(x, params, key):
            sample_arg, key, error = jax_arg(x, params, key)
            sample = op(sample_arg)
            sample = jnp.moveaxis(sample, source=(-2, -1), destination=indices)
            return sample, key, error
        
        return _jax_wrapped_matrix_operation_cholesky
            
