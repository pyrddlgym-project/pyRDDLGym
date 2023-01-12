import jax
import jax.numpy as jnp
import math


class FuzzyLogic:
    '''A general class representing fuzzy logic in Jax.
    '''
    
    def And(self, a, b):
        raise NotImplementedError
    
    def Not(self, x):
        raise NotImplementedError
    
    def Or(self, a, b):
        return self.Not(self.And(self.Not(a), self.Not(b)))
    
    def xor(self, a, b):
        return self.And(self.Or(a, b), self.Not(self.And(a, b)))
    
    def implies(self, a, b):
        return self.Or(self.Not(a), b)
    
    def equiv(self, a, b):
        return self.And(self.implies(a, b), self.implies(b, a))
    
    def forall(self, x, axis=None):
        raise NotImplementedError
    
    def exists(self, x, axis=None):
        return self.Not(self.forall(self.Not(x), axis=axis))
    
    def If(self, c, a, b):
        raise NotImplementedError
    
    def Switch(self, pred, cases):
        raise NotImplementedError
        
    def equal(self, a, b):
        raise NotImplementedError
    
    def notEqual(self, a, b):
        return self.Not(self.equal(a, b))
    
    def greater(self, a, b):
        raise NotImplementedError
    
    def greaterEqual(self, a, b):
        raise NotImplementedError
    
    def less(self, a, b):
        return self.Not(self.greaterEqual(a, b))
    
    def lessEqual(self, a, b):
        return self.Not(self.greater(a, b))
    
    def signum(self, x):
        raise NotImplementedError
    

class ProductLogic(FuzzyLogic):
    '''A class representing the Product t-norm fuzzy logic in Jax.
    '''
    
    def __init__(self, sigmoid_weight: float=10.0):
        '''Creates a new product fuzzy logic in Jax.
        
        :param sigmoid_weight: how concentrated the sigmoids should be;
        this is currently used to convert relational (e.g. <=, ==) operations
        to probabilities; higher values mean sharper probabilities that fall off
        faster
        '''
        self._w = sigmoid_weight
        self._normalizer = math.tanh(0.25 * self._w)
        
    def And(self, a, b):
        return a * b
    
    def Not(self, x):
        return 1.0 - x
    
    def forall(self, x, axis=None):
        return jnp.prod(x, axis=axis)
    
    def If(self, c, a, b):
        return c * a + (1 - c) * b
    
    def Switch(self, pred, cases):
        shape = cases.shape  
        pred = jnp.expand_dims(pred, axis=0)
        pred = jnp.broadcast_to(pred, shape=shape)    
        num_cases = shape[0]
        literals = jnp.arange(num_cases)
        literals = jnp.expand_dims(literals, axis=tuple(range(1, len(shape))))
        literals = jnp.broadcast_to(literals, shape=shape)
        weight = self.equal(pred, literals)
        prob = jax.nn.softmax(self._w * weight, axis=0)
        sample = jnp.sum(cases * prob, axis=0)
        return sample
    
    def equal(self, a, b):
        expr = a - b
        p1 = jax.nn.sigmoid(self._w * (expr + 0.5))
        p2 = jax.nn.sigmoid(self._w * (expr - 0.5))
        return (p1 - p2) / self._normalizer
    
    def greaterEqual(self, a, b):
        expr = a - b
        return jax.nn.sigmoid(self._w * expr)
    
    def greater(self, a, b):
        return self.greaterEqual(a, b)
    
    def signum(self, x):
        return jax.nn.tanh(self._w * x)

    
if __name__ == '__main__':
    logic = ProductLogic(10)
    
    # https://towardsdatascience.com/emulating-logical-gates-with-a-neural-network-75c229ec4cc9
    def test_logic(x1, x2):
        q1 = logic.And(logic.greater(x1, 0), logic.greater(x2, 0))
        q2 = logic.And(logic.Not(logic.greater(x1, 0)), logic.Not(logic.greater(x2, 0)))
        cond = logic.Or(q1, q2)
        pred = logic.If(cond, +1, -1)
        return pred
    
    x1 = jnp.asarray([1, 1, -1, -1, 0.1, 15, -0.5])
    x2 = jnp.asarray([1, -1, 1, -1, 10, -30, 6])
    print(test_logic(x1, x2))
    
    def switch(pred, cases):
        return logic.Switch(pred, cases)
    
    pred = jnp.asarray(jnp.linspace(0, 2, 10))
    case1 = jnp.asarray([-10.] * 10)
    case2 = jnp.asarray([1.5] * 10)
    case3 = jnp.asarray([10.] * 10)
    cases = jnp.asarray([case1, case2, case3])
    print(switch(pred, cases))
