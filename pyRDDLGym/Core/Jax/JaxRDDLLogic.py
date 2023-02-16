import jax
import jax.numpy as jnp
import jax.random as random
import math
import warnings


class FuzzyLogic:
    '''A class representing fuzzy logic in JAX using the product t-norm. 
    Override this class directly to override its functionality.
    '''
    
    def __init__(self, sigmoid_weight: float=10.0):
        '''Creates a new fuzzy logic in Jax.
        
        :param sigmoid_weight: how concentrated the sigmoids should be;
        this is currently used to convert relational (e.g. <=, ==) operations
        to probabilities; higher values mean sharper values that fall off faster
        '''
        self._w = sigmoid_weight
        self._normalizer = math.tanh(0.25 * self._w)
        
    # ===========================================================================
    # logical operators
    # ===========================================================================
     
    def And(self, a, b):
        warnings.warn('Using the replacement rule: a ^ b --> a * b.', stacklevel=2)
        return a * b
    
    def Not(self, x):
        warnings.warn('Using the replacement rule: ~a --> 1 - a', stacklevel=2)
        return 1.0 - x
    
    def Or(self, a, b):
        return self.Not(self.And(self.Not(a), self.Not(b)))
    
    def xor(self, a, b):
        return self.And(self.Or(a, b), self.Not(self.And(a, b)))
    
    def implies(self, a, b):
        return self.Or(self.Not(a), b)
    
    def equiv(self, a, b):
        return self.And(self.implies(a, b), self.implies(b, a))
    
    def forall(self, x, axis=None):
        warnings.warn('Using the replacement rule: forall(a) --> prod(a)',
                      stacklevel=2)
        return jnp.prod(x, axis=axis)
    
    def exists(self, x, axis=None):
        return self.Not(self.forall(self.Not(x), axis=axis))
    
    # ===========================================================================
    # comparison operators
    # ===========================================================================
     
    def equal(self, a, b):
        warnings.warn('Using the replacement rule: '
                      'a == b --> sigmoid(a - b + 0.5) - sigmoid(a - b - 0.5)',
                      stacklevel=2)
        expr = a - b
        p1 = jax.nn.sigmoid(self._w * (expr + 0.5))
        p2 = jax.nn.sigmoid(self._w * (expr - 0.5))
        return (p1 - p2) / self._normalizer
    
    def notEqual(self, a, b):
        return self.Not(self.equal(a, b))
    
    def greaterEqual(self, a, b):
        warnings.warn('Using the replacement rule: a >= b --> sigmoid(a - b)',
                      stacklevel=2)
        return jax.nn.sigmoid(self._w * (a - b))
    
    def greater(self, a, b):
        warnings.warn('Using the replacement rule: a > b --> sigmoid(a - b)',
                      stacklevel=2)
        return self.greaterEqual(a, b)
    
    def less(self, a, b):
        return self.Not(self.greaterEqual(a, b))
    
    def lessEqual(self, a, b):
        return self.Not(self.greater(a, b))
    
    # ===========================================================================
    # aggregations
    # ===========================================================================
     
    def _literal_array(self, shape):
        num_values, *shape_batch = shape
        literals = jnp.arange(num_values)
        literals = literals[(...,) + (jnp.newaxis,) * len(shape_batch)]
        literals = jnp.broadcast_to(literals, shape=shape)
        return literals
        
    def argmax(self, x, axis):
        warnings.warn('Using the replacement rule: '
                      f'argmax(x) --> sum(i * softmax(x[i]))',
                      stacklevel=2)
        prob = jax.nn.softmax(self._w * x, axis=axis)
        prob = jnp.moveaxis(prob, source=axis, destination=0)
        literals = self._literal_array(prob.shape)
        return jnp.sum(literals * prob, axis=0)
    
    def argmin(self, x, axis):
        return self.argmax(-x, axis)
    
    # ===========================================================================
    # control flow
    # ===========================================================================
     
    def If(self, c, a, b):
        warnings.warn('Using the replacement rule: '
                      'if c then a else b --> c * a + (1 - c) * b', stacklevel=2)
        return c * a + (1 - c) * b
    
    def Switch(self, pred, cases):
        warnings.warn('Using the replacement rule: '
                      'switch(pred) { cond... } --> sum((pred == i) * cond[i])', 
                      stacklevel=2)
        pred = pred[jnp.newaxis, ...]
        pred = jnp.broadcast_to(pred, shape=cases.shape)    
        literals = self._literal_array(cases.shape)
        prob = self.equal(pred, literals)
        return jnp.sum(cases * prob, axis=0)
    
    def signum(self, x):
        warnings.warn('Using the replacement rule: signum(x) --> tanh(x)',
                      stacklevel=2)
        return jnp.tanh(self._w * x)
    
    def sqrt(self, x):
        warnings.warn('Using the replacement rule: sqrt(x) --> sqrt(x + eps)',
                      stacklevel=2)
        return jnp.sqrt(x + 1e-12)
    
    # ===========================================================================
    # random variables
    # ===========================================================================
     
    def _gumbel_softmax(self, key, prob):
        Gumbel01 = random.gumbel(key=key, shape=prob.shape)
        clipped_dist = jnp.maximum(prob, 1e-12)
        sample = self._w * (Gumbel01 + jnp.log(clipped_dist))
        return sample
        
    def bernoulli(self, key, prob):
        warnings.warn('Using the replacement rule: '
                      'Bernoulli(p) --> Gumbel-softmax', stacklevel=2)
        prob = jnp.stack([prob, 1.0 - prob], axis=-1)
        sample = self._gumbel_softmax(key, prob)
        sample = jax.nn.softmax(sample, axis=-1)[..., 0]     
        return sample
    
    def discrete(self, key, prob):
        warnings.warn('Using the replacement rule: '
                      'Discrete(p) --> Gumbel-softmax', stacklevel=2)
        sample = self._gumbel_softmax(key, prob)
        sample = jax.nn.softmax(sample, axis=-1)
        *shape_batch, num_values = prob.shape
        indices = jnp.arange(num_values)
        indices = indices[(jnp.newaxis,) * len(shape_batch) + (...,)]
        sample = jnp.sum(sample * indices, axis=-1)
        return sample
        
    
if __name__ == '__main__':
    logic = FuzzyLogic(10)
    
    # https://towardsdatascience.com/emulating-logical-gates-with-a-neural-network-75c229ec4cc9
    def test_logic(x1, x2):
        q1 = logic.And(logic.greater(x1, 0), logic.greater(x2, 0))
        q2 = logic.And(logic.Not(logic.greater(x1, 0)), logic.Not(logic.greater(x2, 0)))
        cond = logic.Or(q1, q2)
        pred = logic.If(cond, +1, -1)
        return pred
    
    x1 = jnp.asarray([1, 1, -1, -1, 0.1, 15, -0.5]).astype(float)
    x2 = jnp.asarray([1, -1, 1, -1, 10, -30, 6]).astype(float)
    print(test_logic(x1, x2))
    
    def switch(pred, cases):
        return logic.Switch(pred, cases)
    
    pred = jnp.asarray(jnp.linspace(0, 2, 10))
    case1 = jnp.asarray([-10.] * 10)
    case2 = jnp.asarray([1.5] * 10)
    case3 = jnp.asarray([10.] * 10)
    cases = jnp.asarray([case1, case2, case3])
    print(switch(pred, cases))
    
    def argmaxmin(x):
        amax = logic.argmax(x, axis=0)
        amin = logic.argmin(x, axis=0)
        return amax, amin
        
    values = jnp.asarray([2., 3., 5., 4.9, 4., 1., -1., -2.])
    amax, amin = argmaxmin(values)
    print(amax)
    print(amin)
