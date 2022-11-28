import jax.numpy as jnp
import warnings

from pyRDDLGym.Core.Jax.JaxRDDLCompiler import JaxRDDLCompiler


class FuzzyLogic:
    
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
        raise NotImplemented


class ProductLogic(FuzzyLogic):
    
    def And(self, a, b):
        return a * b
    
    def Not(self, x):
        return 1.0 - x
    
    def implies(self, a, b):
        return 1.0 - a * (1.0 - b)

    def forall(self, x, axis=None):
        return jnp.prod(x, axis=axis)
    
    def If(self, c, a, b):
        return c * a + (1 - c) * b


class JaxRDDLCompilerWithGrad(JaxRDDLCompiler):
    
    def __init__(self, *args, logic: FuzzyLogic=ProductLogic(), **kwargs) -> None:
        super(JaxRDDLCompilerWithGrad, self).__init__(
            *args, force_continuous=True, **kwargs)
        
        # overwrite basic operations with fuzzy ones
        self.LOGICAL_OPS = {
            '^': logic.And,
            '|': logic.Or,
            '~': logic.xor,
            '=>': logic.implies,
            '<=>': logic.equiv
        }
        self.LOGICAL_NOT = logic.Not  
        self.AGGREGATION_OPS['exists'] = logic.exists
        self.AGGREGATION_OPS['forall'] = logic.forall
        self.CONTROL_OPS['if'] = logic.If
    
    def _jax_logical(self, expr, objects):
        _, op = expr.etype
        warnings.warn(f'Logical operator {op} uses fuzzy logic.',
                      FutureWarning, stacklevel=2)        
        return super(JaxRDDLCompilerWithGrad, self)._jax_logical(expr, objects)
    
    def _jax_aggregation(self, expr, objects):
        _, op = expr.etype
        warnings.warn(f'Aggregation operator {op} uses fuzzy logic.',
                      FutureWarning, stacklevel=2)        
        return super(JaxRDDLCompilerWithGrad, self)._jax_aggregation(expr, objects)
        
    def _jax_control(self, expr, objects):
        _, op = expr.etype
        warnings.warn(f'Control operator {op} uses fuzzy logic.',
                      FutureWarning, stacklevel=2)        
        return super(JaxRDDLCompilerWithGrad, self)._jax_control(expr, objects)

    def _jax_kron(self, expr, objects):
        warnings.warn('KronDelta will be ignored.', FutureWarning, stacklevel=2)                       
        arg, = expr.args
        arg = self._jax(arg, objects)
        return arg
