import jax
import jax.numpy as jnp
import jax.random as random
import warnings

DEFAULT_WEIGHT = 10.0


class TNorm:
    
    def norm(self, x, y):
        raise NotImplementedError
    
    def norms(self, x, axis):
        raise NotImplementedError
        

class ProductTNorm(TNorm):
    
    def norm(self, x, y):
        return x * y
    
    def norms(self, x, axis):
        return jnp.prod(x, axis=axis)

    
class FuzzyLogic:
    '''A class representing fuzzy logic in JAX.
    
    Functionality can be customized by either providing a tnorm as parameters, 
    or by overriding its methods.
    '''
    
    def __init__(self, tnorm: TNorm=ProductTNorm(),
                 weight: float=DEFAULT_WEIGHT,
                 eps: float=1e-12):
        '''Creates a new fuzzy logic in Jax.
        
        :param tnorm: t-norm for doing fuzzy boolean logic
        :param weight: a concentration parameter (larger means better accuracy)
        :param eps: small positive float to mitigate underflow
        '''
        self.tnorm = tnorm
        self.weight = weight
        self.eps = eps
        
    # ===========================================================================
    # logical operators
    # ===========================================================================
     
    def And(self, a, b):
        warnings.warn('Using the replacement rule: '
                      'a ^ b --> tnorm(a, b).', stacklevel=2)
        return self.tnorm.norm(a, b)
    
    def Not(self, x):
        warnings.warn('Using the replacement rule: '
                      '~a --> 1 - a', stacklevel=2)
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
        warnings.warn('Using the replacement rule: '
                      'forall(a) --> tnorm(a[1], tnorm(a[2], ...))', stacklevel=2)
        return self.tnorm.norms(x, axis=axis)
    
    def exists(self, x, axis=None):
        return self.Not(self.forall(self.Not(x), axis=axis))
    
    # ===========================================================================
    # comparison operators
    # ===========================================================================
     
    def greaterEqual(self, a, b):
        warnings.warn('Using the replacement rule: '
                      'a >= b --> sigmoid(a - b)', stacklevel=2)
        return jax.nn.sigmoid(self.weight * (a - b))
    
    def greater(self, a, b):
        warnings.warn('Using the replacement rule: '
                      'a > b --> sigmoid(a - b)', stacklevel=2)
        return jax.nn.sigmoid(self.weight * (a - b))
    
    def lessEqual(self, a, b):
        return self.greaterEqual(-a, -b)
    
    def less(self, a, b):
        return self.greater(-a, -b)

    def equal(self, a, b):
        warnings.warn('Using the replacement rule: '
                      'a == b --> sech^2(b - a)', stacklevel=2)
        return 1.0 - jnp.square(jnp.tanh(self.weight * (b - a)))
    
    def notEqual(self, a, b):
        return self.Not(self.equal(a, b))
    
    # ===========================================================================
    # special functions
    # ===========================================================================
     
    def signum(self, x):
        warnings.warn('Using the replacement rule: '
                      'signum(x) --> tanh(x)', stacklevel=2)
        return jnp.tanh(self.weight * x)
    
    def floor(self, x):
        warnings.warn('floor() will have zero gradient', stacklevel=2)
        return jnp.floor(x)
    
    def ceil(self, x):
        warnings.warn('ceil() will have zero gradient', stacklevel=2)
        return jnp.ceil(x)
    
    def round(self, x):
        warnings.warn('round() will have zero gradient', stacklevel=2)
        return jnp.round(x)
    
    def sqrt(self, x):
        warnings.warn('Using the replacement rule: '
                      'sqrt(x) --> sqrt(x + eps)', stacklevel=2)
        return jnp.sqrt(x + self.eps)
    
    # ===========================================================================
    # indexing
    # ===========================================================================
     
    @staticmethod
    def _literals(shape, axis):
        literals = jnp.arange(shape[axis])
        literals = literals[(...,) + (jnp.newaxis,) * (len(shape) - 1)]
        literals = jnp.moveaxis(literals, source=0, destination=axis)
        literals = jnp.broadcast_to(literals, shape=shape)
        return literals
    
    def argmax(self, x, axis):
        warnings.warn('Using the replacement rule: '
                      f'argmax(x) --> sum(i * softmax(x[i]))', stacklevel=2)
        prob_max = jax.nn.softmax(self.weight * x, axis=axis)
        literals = FuzzyLogic._literals(prob_max.shape, axis=axis)
        softargmax = jnp.sum(literals * prob_max, axis=axis)
        trueargmax = jnp.argmax(x, axis=axis)
        sample = softargmax + jax.lax.stop_gradient(trueargmax - softargmax)
        return sample
    
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
                      'switch(pred) { cases } --> sum(cases[i] * (pred == i))',
                      stacklevel=2)    
        pred = jnp.broadcast_to(pred[jnp.newaxis, ...], shape=cases.shape)
        literals = FuzzyLogic._literals(cases.shape, axis=0)
        proximity = -jnp.abs(pred - literals)
        softcase = jax.nn.softmax(self.weight * proximity, axis=0)
        softswitch = jnp.sum(cases * softcase, axis=0)
        hardcase = jnp.argmax(proximity, axis=0)[jnp.newaxis, ...]        
        hardswitch = jnp.take_along_axis(cases, hardcase, axis=0)[0, ...]
        sample = softswitch + jax.lax.stop_gradient(hardswitch - softswitch)
        return sample
    
    # ===========================================================================
    # random variables
    # ===========================================================================
     
    def _gumbel_softmax(self, key, prob):
        Gumbel01 = random.gumbel(key=key, shape=prob.shape)
        sample = Gumbel01 + jnp.log(prob + self.eps)
        return sample
        
    def bernoulli(self, key, prob):
        warnings.warn('Using the replacement rule: '
                      'Bernoulli(p) --> Gumbel-softmax(p)', stacklevel=2)
        prob = jnp.stack([1.0 - prob, prob], axis=-1)
        sample = self._gumbel_softmax(key, prob)
        sample = self.argmax(sample, axis=-1)
        return sample
    
    def discrete(self, key, prob):
        warnings.warn('Using the replacement rule: '
                      'Discrete(p) --> Gumbel-softmax(p)', stacklevel=2)
        sample = self._gumbel_softmax(key, prob) 
        sample = self.argmax(sample, axis=-1)
        return sample
        

# UNIT TESTS
logic = FuzzyLogic()


def _test_logical():
    
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


def _test_indexing():

    def argmaxmin(x):
        amax = logic.argmax(x, axis=0)
        amin = logic.argmin(x, axis=0)
        return amax, amin
        
    values = jnp.asarray([2., 3., 5., 4.9, 4., 1., -1., -2.])
    amax, amin = argmaxmin(values)
    print(amax)
    print(amin)


def _test_control():
    
    def switch(pred, cases):
        return logic.Switch(pred, cases)

    pred = jnp.asarray(jnp.linspace(0, 2, 10))
    case1 = jnp.asarray([-10.] * 10)
    case2 = jnp.asarray([1.5] * 10)
    case3 = jnp.asarray([10.] * 10)
    cases = jnp.asarray([case1, case2, case3])
    print(switch(pred, cases))


def _test_random():
    key = random.PRNGKey(42)

    def bern(n):
        prob = jnp.asarray([0.3] * n)
        sample = logic.bernoulli(key, prob)
        return sample
    
    samples = bern(5000)
    print(jnp.mean(samples))


if __name__ == '__main__':
    _test_logical()
    _test_indexing()
    _test_control()
    _test_random()
    
