import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import warnings

from pyRDDLGym.Core.Jax.JaxRDDLCompiler import JaxRDDLCompiler
from pyRDDLGym.Core.Jax.JaxRDDLLogic import FuzzyLogic, ProductLogic


class JaxRDDLCompilerWithGrad(JaxRDDLCompiler):
    '''Compiles a RDDL AST representation to an equivalent JAX representation. 
    Unlike its parent class, this class treats all fluents as real-valued, and
    replaces all mathematical operations by equivalent ones with a well defined 
    (e.g. non-zero) gradient where appropriate. 
    '''
    
    def __init__(self, *args,
                 logic: FuzzyLogic=ProductLogic(),
                 temp: float=0.1,
                 eps: float=1e-12,
                 **kwargs) -> None:
        '''Creates a new RDDL to Jax compiler, where operations that are not
        differentiable are converted to approximate forms that have defined 
        gradients.
        
        :param *args: arguments to pass to base compiler
        :param logic: Fuzzy logic object that specifies how exact operations
        are converted to their approximate forms: this class may be subclassed
        to customize these operations
        :param temp: temperature parameter for Gumbel-softmax applied to Bernoulli
        and discrete distributions
        :param eps: minimum value of a probability before mapping through log: 
        used to avoid underflow in log calculation
        :param *kwargs: keyword arguments to pass to base compiler
        '''
        super(JaxRDDLCompilerWithGrad, self).__init__(*args, **kwargs)
        self.temp = temp
        self.eps = eps
        
        # actions and CPFs must be continuous
        warnings.warn(f'Initial values of CPFs and action-fluents '
                      f'will be cast to real.', stacklevel=2)   
        for (var, values) in self.init_values.items():
            if self.rddl.variable_types[var] != 'non-fluent':
                self.init_values[var] = np.asarray(values, dtype=JaxRDDLCompiler.REAL) 
        
        # overwrite basic operations with fuzzy ones
        self.RELATIONAL_OPS = {
            '>=': logic.greaterEqual,
            '<=': logic.lessEqual,
            '<': logic.less,
            '>': logic.greater,
            '==': logic.equal,
            '~=': logic.notEqual
        }
        self.LOGICAL_NOT = logic.Not  
        self.LOGICAL_OPS = {
            '^': logic.And,
            '&': logic.And,
            '|': logic.Or,
            '~': logic.xor,
            '=>': logic.implies,
            '<=>': logic.equiv
        }
        self.AGGREGATION_OPS = {
            'sum': jnp.sum,
            'avg': jnp.mean,
            'prod': jnp.prod,
            'minimum': jnp.min,
            'maximum': jnp.max,
            'forall': logic.forall,
            'exists': logic.exists,
            'argmin': logic.argmin,
            'argmax': logic.argmax
        }
        self.KNOWN_UNARY['sgn'] = logic.signum        
        self.CONTROL_OPS = {
            'if': logic.If,
            'switch': logic.Switch
        }
            
    def _compile_cpfs(self):
        warnings.warn('CPFs outputs will be cast to real.', stacklevel=2)      
        jax_cpfs = {}
        for (_, cpfs) in self.levels.items():
            for cpf in cpfs:
                _, expr = self.rddl.cpfs[cpf]
                jax_cpfs[cpf] = self._jax(expr, dtype=JaxRDDLCompiler.REAL)
        return jax_cpfs
    
    def _jax_relational(self, expr):
        _, op = expr.etype
        warnings.warn(f'Relational operator {op} uses sigmoid approximation.',
                      stacklevel=2)        
        return super(JaxRDDLCompilerWithGrad, self)._jax_relational(expr)
    
    def _jax_logical(self, expr):
        _, op = expr.etype
        warnings.warn(f'Logical operator {op} uses fuzzy logic.', stacklevel=2)        
        return super(JaxRDDLCompilerWithGrad, self)._jax_logical(expr)
    
    def _jax_aggregation(self, expr):
        _, op = expr.etype
        if op in ['forall', 'exists', 'argmin', 'argmax']:
            warnings.warn(f'Aggregation operator <{op}> uses fuzzy logic.',
                          stacklevel=2)        
        return super(JaxRDDLCompilerWithGrad, self)._jax_aggregation(expr)
    
    def _jax_functional(self, expr):
        _, op = expr.etype
        if op == 'sgn':
            warnings.warn(f'Function {op} uses tanh approximation.',
                          stacklevel=2) 
        return super(JaxRDDLCompilerWithGrad, self)._jax_functional(expr)
    
    def _jax_control(self, expr):
        _, op = expr.etype
        warnings.warn(f'Control operator <{op}> uses linear approximation.',
                      stacklevel=2)    
        return super(JaxRDDLCompilerWithGrad, self)._jax_control(expr)
    
    def _jax_kron(self, expr):
        warnings.warn('KronDelta will be ignored.', stacklevel=2)                       
        arg, = expr.args
        arg = self._jax(arg)
        return arg
    
    def _jax_bernoulli(self, expr):
        warnings.warn(f'Bernoulli uses Gumbel-softmax reparameterization.',
                      stacklevel=2) 
         
        ERR = JaxRDDLCompiler.ERROR_CODES['INVALID_PARAM_BERNOULLI']
        JaxRDDLCompiler._check_num_args(expr, 1)
        
        arg_prob, = expr.args
        jax_prob = self._jax(arg_prob)
        tau, eps = self.temp, self.eps
        
        # use the Gumbel-softmax trick to make this differentiable
        def _jax_wrapped_distribution_bernoulli_gumbel_softmax(x, key):
            prob, key, err = jax_prob(x, key)
            key, subkey = random.split(key)
            dist = jnp.asarray([prob, 1.0 - prob])
            Gumbel01 = random.gumbel(key=subkey, shape=dist.shape)
            clipped_dist = jnp.maximum(dist, eps)
            sample = (Gumbel01 + jnp.log(clipped_dist)) / tau
            sample = jax.nn.softmax(sample, axis=0)[0, ...]      
            out_of_bounds = jnp.logical_not(jnp.all((prob >= 0) & (prob <= 1)))
            err |= (out_of_bounds * ERR)
            return sample, key, err
        
        return _jax_wrapped_distribution_bernoulli_gumbel_softmax
    
    def _jax_discrete_helper(self):
        warnings.warn(f'Discrete uses Gumbel-softmax reparameterization.',
                      stacklevel=2) 
        tau, eps = self.temp, self.eps
        
        def _jax_discrete_calc_gumbel_softmax(prob, subkey):
            Gumbel01 = random.gumbel(key=subkey, shape=prob.shape)
            clipped_prob = jnp.maximum(prob, eps)
            sample = (Gumbel01 + jnp.log(clipped_prob)) / tau
            sample = jax.nn.softmax(sample, axis=0)
            indices = jnp.arange(prob.shape[0])
            indices = jnp.expand_dims(indices, axis=tuple(range(1, len(prob.shape))))
            return jnp.sum(sample * indices, axis=0)
        
        return _jax_discrete_calc_gumbel_softmax
