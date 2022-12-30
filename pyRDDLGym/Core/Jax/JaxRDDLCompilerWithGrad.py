import numpy as np
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
        raise NotImplementedError


class BooleanLogic(FuzzyLogic):
    
    def And(self, a, b):
        return jnp.logical_and(a, b)
    
    def Not(self, x):
        return jnp.logical_not(x)

    def forall(self, x, axis=None):
        return jnp.all(x, axis=axis)
    
    def If(self, c, a, b):
        return jnp.where(c, a, b)


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
        super(JaxRDDLCompilerWithGrad, self).__init__(*args, **kwargs)
        
        # actions and CPFs must be continuous
        warnings.warn(f'Initial values of CPFs and action-fluents '
                      f'will be cast to real.', stacklevel=2)   
        for var, values in self.init_values.items():
            if self.rddl.variable_types[var] != 'non-fluent':
                self.init_values[var] = np.asarray(
                    values, dtype=JaxRDDLCompiler.REAL)        
        
        # overwrite basic operations with fuzzy ones
        self.LOGICAL_OPS = {
            '^': logic.And,
            '&': logic.And,
            '|': logic.Or,
            '~': logic.xor,
            '=>': logic.implies,
            '<=>': logic.equiv
        }
        self.LOGICAL_NOT = logic.Not  
        self.AGGREGATION_OPS['exists'] = logic.exists
        self.AGGREGATION_OPS['forall'] = logic.forall
        self.CONTROL_OPS['if'] = logic.If
    
    def _compile_cpfs(self):
        warnings.warn('CPFs outputs will be cast to real.', stacklevel=2)      
        jax_cpfs = {}
        for _, cpfs in self.levels.items():
            for cpf in cpfs:
                objects, expr = self.rddl.cpfs[cpf]
                dtype = JaxRDDLCompiler.JAX_TYPES['real']
                jax_cpfs[cpf] = self._jax(expr, objects, dtype=dtype)
        return jax_cpfs
    
    def _jax_logical(self, expr, objects):
        _, op = expr.etype
        warnings.warn(f'Logical operator {op} uses fuzzy logic.', stacklevel=2)        
        return super(JaxRDDLCompilerWithGrad, self)._jax_logical(expr, objects)
    
    def _jax_aggregation(self, expr, objects):
        _, op = expr.etype
        warnings.warn(f'Aggregation operator {op} uses fuzzy logic.', stacklevel=2)        
        return super(JaxRDDLCompilerWithGrad, self)._jax_aggregation(expr, objects)
        
    def _jax_control(self, expr, objects):
        _, op = expr.etype
        warnings.warn(f'Control operator {op} uses fuzzy logic.', stacklevel=2)        
        return super(JaxRDDLCompilerWithGrad, self)._jax_control(expr, objects)

    def _jax_kron(self, expr, objects):
        warnings.warn('KronDelta will be ignored.', stacklevel=2)                       
        arg, = expr.args
        arg = self._jax(arg, objects)
        return arg
