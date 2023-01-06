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
    
    def __init__(self, *args, logic: FuzzyLogic=ProductLogic(), **kwargs) -> None:
        super(JaxRDDLCompilerWithGrad, self).__init__(*args, **kwargs)
        
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
        self.AGGREGATION_OPS['exists'] = logic.exists
        self.AGGREGATION_OPS['forall'] = logic.forall
        self.CONTROL_OPS['if'] = logic.If
    
    def _compile_cpfs(self):
        warnings.warn('CPFs outputs will be cast to real.', stacklevel=2)      
        jax_cpfs = {}
        for (_, cpfs) in self.levels.items():
            for cpf in cpfs:
                _, expr = self.rddl.cpfs[cpf]
                jax_cpfs[cpf] = self._jax(expr, dtype=JaxRDDLCompiler.REAL)
        return jax_cpfs
    
    def _jax_logical(self, expr):
        _, op = expr.etype
        warnings.warn(f'Logical operator {op} uses fuzzy logic.', stacklevel=2)        
        return super(JaxRDDLCompilerWithGrad, self)._jax_logical(expr)
    
    def _jax_aggregation(self, expr):
        _, op = expr.etype
        warnings.warn(f'Aggregation operator {op} uses fuzzy logic.', stacklevel=2)        
        return super(JaxRDDLCompilerWithGrad, self)._jax_aggregation(expr)
        
    def _jax_control(self, expr):
        _, op = expr.etype
        warnings.warn(f'Control operator {op} uses fuzzy logic.', stacklevel=2)        
        return super(JaxRDDLCompilerWithGrad, self)._jax_control(expr)

    def _jax_kron(self, expr):
        warnings.warn('KronDelta will be ignored.', stacklevel=2)                       
        arg, = expr.args
        arg = self._jax(arg)
        return arg

