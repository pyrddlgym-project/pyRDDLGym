import jax
import jax.numpy as jnp
import jax.random as jrng

from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidExpressionError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidNumberOfArgumentsError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLNotImplementedError
from pyRDDLGym.Core.Grounder.RDDLModel import PlanningModel
from pyRDDLGym.Core.Parser.expr import Expression
from pyRDDLGym.Core.Simulator.RDDLDependencyAnalysis import RDDLDependencyAnalysis

VALID_RESTRICTED_ARITHMETIC_OPS = {'-', '/'}
VALID_RESTRICTED_LOGICAL_OPS = {'~', '=>', '<=>'}
VALID_CONTROL_OPS = {'if'}


class JaxRDDLCompiler:
    
    def __init__(self, model: PlanningModel) -> None:
        self._model = model
        
        # perform a dependency analysis, do topological sort to compute levels
        dep_analysis = RDDLDependencyAnalysis(self._model)
        self.cpforder = dep_analysis.compute_levels()
        self.order_cpfs = list(sorted(self.cpforder.keys())) 
        
        # compiled expressions
        self.invariants = []
        self.preconditions = []
        self.termination = []
        self.reward, self.jit_reward = None, None
        self.cpfs, self.jit_cpfs = {}, {}
        
    # start of compilation subroutines of RDDL programs
    def compile(self) -> None:
        self.invariants = self._compile_invariants()
        self.preconditions = self._compile_preconditions()
        self.termination = self._compile_termination()
        self.reward, self.jit_reward = self._compile_reward()
        self.cpfs, self.jit_cpfs = self._compile_cpfs()
        return self
        
    def _compile_invariants(self):
        tree = list(map(self._jax, self._model.invariants))
        tree_jit = jax.tree_map(jax.jit, tree)
        return tree_jit
        
    def _compile_preconditions(self):
        tree = list(map(self._jax, self._model.preconditions))
        tree_jit = jax.tree_map(jax.jit, tree)
        return tree_jit
    
    def _compile_termination(self):
        tree = list(map(self._jax, self._model.terminals))
        tree_jit = jax.tree_map(jax.jit, tree)
        return tree_jit

    def _compile_cpfs(self):
        tree = {}  
        for order in self.order_cpfs:
            for cpf in self.cpforder[order]: 
                if cpf in self._model.next_state:
                    primed_cpf = self._model.next_state[cpf]
                    expr = self._model.cpfs[primed_cpf]
                    tree[primed_cpf] = self._jax(expr)
                else:
                    expr = self._model.cpfs[cpf]
                    tree[cpf] = self._jax(expr)
        tree_jit = jax.tree_map(jax.jit, tree)        
        return tree, tree_jit
    
    def _compile_reward(self):
        reward = self._jax(self._model.reward)
        reward_jit = jax.jit(reward)
        return reward, reward_jit
        
    # start of compilation subroutines for expressions
    # skipped: aggregations (sum, prod) for now
    @staticmethod
    def _print_stack_trace(expr):
        return '...\n' + str(expr) + '\n...'

    def _jax(self, expr):
        if isinstance(expr, Expression):
            etype, op = expr.etype
            if etype == 'constant':
                return self._jax_constant(expr)
            elif etype == 'pvar':
                return self._jax_pvar(expr)
            elif etype == 'arithmetic':
                return self._jax_arithmetic(expr, op)
            elif etype == 'aggregation':
                return self._jax_aggregation(expr, op)
            elif etype == 'relational':
                return self._jax_relational(expr, op)
            elif etype == 'func':
                return self._jax_func(expr, op)
            elif etype == 'boolean':
                return self._jax_logical(expr, op)
            elif etype == 'control':
                return self._jax_control(expr, op)
            elif etype == 'randomvar':
                return self._jax_random(expr, op)
            else:
                raise RDDLNotImplementedError(
                    'Internal error: expression type {} is not supported.'.format(etype) + 
                    '\n' + JaxRDDLCompiler._print_stack_trace(expr))
        else:
            raise RDDLNotImplementedError(
                'Internal error: type {} is not supported.'.format(expr) + 
                '\n' + JaxRDDLCompiler._print_stack_trace(expr))
            
    # simple expressions
    def _jax_constant(self, expr):
        sample = expr.args
        
        def _f(x, key):
            return sample, key

        return _f
    
    def _jax_pvar(self, expr):
        var, *_ = expr.args
        
        def _f(x, key):
            sample = x[var]
            return sample, key

        return _f
    
    def _jax_forward(self, jaxop, *args):
        args = tuple(args)
        
        if len(args) == 1:
            arg, = args
            
            def _f(x, key):
                val, key = arg(x, key)
                sample = jaxop(val)
                return sample, key
        
        elif len(args) == 2:
            arg1, arg2 = args
            
            def _f(x, key):
                val1, key = arg1(x, key)
                val2, key = arg2(x, key)
                sample = jaxop(val1, val2)
                return sample, key
        
        else:
            arg1, *arg2etc = args
            
            def _f(x, key):
                sample, key = arg1(x, key)
                for arg in arg2etc:
                    val, key = arg(x, key)
                    sample = jaxop(sample, val)
                return sample, key
        
        return _f
            
    AGGREGATION_OPS = {
        '+': jnp.add,
        '*': jnp.multiply,
        '^': jnp.logical_and,
        '|': jnp.logical_or,
        'minimum': jnp.minimum,
        'maximum': jnp.maximum
    }
    
    def _jax_aggregation(self, expr, op):
        if op not in JaxRDDLCompiler.AGGREGATION_OPS:
            raise RDDLNotImplementedError(
                'Aggregation operator {} is not supported: must be one of {}.'.format(
                    op, JaxRDDLCompiler.AGGREGATION_OPS.keys()) + 
                '\n' + JaxRDDLCompiler._print_stack_trace(expr))
            
        jaxop = JaxRDDLCompiler.AGGREGATION_OPS[op]  
        args = map(self._jax, expr.args)
        return self._jax_forward(jaxop, *args)
    
    def _jax_arithmetic(self, expr, op):
        
        # can have arbitrary number of args
        if op in JaxRDDLCompiler.AGGREGATION_OPS:
            return self._jax_aggregation(expr, op)
        
        # can only have one or two args
        if op not in VALID_RESTRICTED_ARITHMETIC_OPS:
            raise RDDLNotImplementedError(
                'Arithmetic operator {} is not supported: must be one of {}.'.format(
                    op, VALID_RESTRICTED_ARITHMETIC_OPS) + 
                '\n' + JaxRDDLCompiler._print_stack_trace(expr))
            
        n = len(expr.args)
        args = map(self._jax, expr.args)
                
        if op == '-':
            if n == 1:
                return self._jax_forward(jnp.negative, *args)
            elif n == 2:
                return self._jax_forward(jnp.subtract, *args)
                
        elif op == '/':
            if n == 2:
                return self._jax_forward(jnp.divide, *args)
            
        raise RDDLInvalidNumberOfArgumentsError(
            'Arithmetic operator {} does not have the required number of args, got {}.'.format(op, n) + 
            '\n' + JaxRDDLCompiler._print_stack_trace(expr))

    RELATIONAL_OPS = {
        '>=': jnp.greater_equal,
        '<=': jnp.less_equal,
        '<': jnp.less,
        '>': jnp.greater,
        '==': jnp.equal,
        '~=': jnp.not_equal
    }
    
    def _jax_relational(self, expr, op):
        if op not in JaxRDDLCompiler.RELATIONAL_OPS:
            raise RDDLNotImplementedError(
                'Relational operator {} is not supported: must be one of {}.'.format(
                    op, JaxRDDLCompiler.RELATIONAL_OPS.keys()) + 
                '\n' + JaxRDDLCompiler._print_stack_trace(expr))
        
        if len(expr.args) != 2:
            raise RDDLInvalidNumberOfArgumentsError(
                'Relational operator {} requires 2 args, got {}.'.format(op, len(expr.args)) + 
                '\n' + JaxRDDLCompiler._print_stack_trace(expr))

        jaxop = JaxRDDLCompiler.RELATIONAL_OPS[op]            
        args = map(self._jax, expr.args)
        return self._jax_forward(jaxop, *args)
        
    # functions
    KNOWN_UNARY = {        
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
    
    KNOWN_BINARY = {
        'min': jnp.minimum,
        'max': jnp.maximum,
        'pow': jnp.power
    }
    
    def _jax_func(self, expr, op):
        n = len(expr.args)
        
        if op in JaxRDDLCompiler.KNOWN_UNARY:
            if n != 1:
                raise RDDLInvalidNumberOfArgumentsError(
                    'Unary function {} requires 1 arg, got {}.'.format(op, n) + 
                    '\n' + JaxRDDLCompiler._print_stack_trace(expr))    
                            
            jaxop = JaxRDDLCompiler.KNOWN_UNARY[op]            
            args = map(self._jax, expr.args)
            return self._jax_forward(jaxop, *args)

        elif op in JaxRDDLCompiler.KNOWN_BINARY:
            if n != 2:
                raise RDDLInvalidNumberOfArgumentsError(
                    'Binary function {} requires 2 args, got {}.'.format(op, n) + 
                    '\n' + JaxRDDLCompiler._print_stack_trace(expr))  
                
            jaxop = JaxRDDLCompiler.KNOWN_BINARY[op]            
            args = map(self._jax, expr.args)
            return self._jax_forward(jaxop, *args)
        
        raise RDDLNotImplementedError(
                'Function {} is not supported.'.format(op) + 
                '\n' + JaxRDDLCompiler._print_stack_trace(expr))   
        
    # logical expressions    
    def _jax_logical(self, expr, op):
        
        # can have arbitrary number of args
        if op in JaxRDDLCompiler.AGGREGATION_OPS:
            return self._jax_aggregation(expr, op)
        
        # can only have one or two args (TODO: implement => and <=>)
        if op not in VALID_RESTRICTED_LOGICAL_OPS:
            raise RDDLNotImplementedError(
                'Logical operator {} is not supported: must be one of {}.'.format(
                    op, VALID_RESTRICTED_LOGICAL_OPS) + 
                '\n' + JaxRDDLCompiler._print_stack_trace(expr))
            
        n = len(expr.args)
        args = map(self._jax, expr.args)
        
        if op == '~':
            if n == 1:
                return self._jax_forward(jnp.logical_not, *args)
            elif n == 2:
                return self._jax_forward(jnp.logical_or, *args)
        
        elif op == '=>':
            if n == 2:
                arg1, arg2 = args
                
                def _f(x, key):
                    val1, key = arg1(x, key)
                    val2, key = arg2(x, key)
                    return jnp.logical_or(jnp.logical_not(val1), val2)
                 
                return _f
        
        elif op == '<=>':
            if n == 2:
                arg1, arg2 = args
                
                def _f(x, key):
                    val1, key = arg1(x, key)
                    val2, key = arg2(x, key)
                    return jnp.equal(val1, val2)
                 
                return _f
            
        raise RDDLInvalidNumberOfArgumentsError(
            'Logical operator {} does not have the required number of args, got {}.'.format(op, n) + 
            '\n' + JaxRDDLCompiler._print_stack_trace(expr))
    
    # control flow
    def _jax_control(self, expr, op):
        if op not in VALID_CONTROL_OPS:
            raise RDDLNotImplementedError(
                'Control flow type {} is not supported: must be one of {}.'.format(
                    op, VALID_CONTROL_OPS) + 
                '\n' + JaxRDDLCompiler._print_stack_trace(expr))
        
        # must be an if then else statement
        pred, true_f, false_f = map(self._jax, expr.args)
        
        def _f(x, key):
            predx, key = pred(x, key)
            return jax.lax.cond(predx, true_f, false_f, x, key)

        return _f
        
    # random variables        
    def _jax_random(self, expr, name):
        if name == 'KronDelta':
            return self._jax_kron(expr)        
        elif name == 'DiracDelta':
            return self._jax_dirac(expr)
        elif name == 'Uniform':
            return self._jax_uniform(expr)
        elif name == 'Normal':
            return self._jax_normal(expr)
        elif name == 'Exponential':
            return self._jax_exponential(expr)
        elif name == 'Weibull':
            return self._jax_weibull(expr)   
        elif name in {'Bernoulli', 'Poisson', 'Gamma'}:
            
            # TODO: try these
            # https://arxiv.org/pdf/1611.01144.pdf?
            # https://arxiv.org/pdf/2003.01847.pdf?
            raise RDDLInvalidExpressionError(
                'Distribution {} is not reparameterizable.'.format(name) + 
                '\n' + JaxRDDLCompiler._print_stack_trace(expr))
        else:
            raise RDDLNotImplementedError(
                'Distribution {} is not supported.'.format(name) + 
                '\n' + JaxRDDLCompiler._print_stack_trace(expr))
    
    def _jax_kron(self, expr):
        arg, = map(self._jax, expr.args)
        return arg
    
    def _jax_dirac(self, expr):
        arg, = map(self._jax, expr.args)
        return arg
    
    def _jax_uniform(self, expr):
        a, b = map(self._jax, expr.args)
        
        # U(a, b) = a + (b - a) * xi, where xi ~ U(0, 1)
        def _f(x, key):
            ax, key = a(x, key)
            bx, key = b(x, key)
            key, subkey = jrng.split(key)
            U = jrng.uniform(key=subkey)
            sample = ax + (bx - ax) * U
            return sample, key

        return _f

    def _jax_normal(self, expr):
        mean, s2 = map(self._jax, expr.args)

        # N(mu, s^2) = mu + s * N(0, 1)         
        def _f(x, key):
            mx, key = mean(x, key)
            s2x, key = s2(x, key)
            s1x = jnp.sqrt(s2x)
            key, subkey = jrng.split(key)
            Z = jrng.normal(key=subkey)
            sample = mx + s1x * Z
            return sample, key
            
        return _f
    
    def _jax_exponential(self, expr):
        scale, = map(self._jax, expr.args)
        
        # Exp(scale) = scale * Exp(1)
        def _f(x, key):
            scalex, key = scale(x, key)
            key, subkey = jrng.split(key)
            Exp1 = jrng.exponential(key=subkey)
            sample = scalex * Exp1
            return sample, key

        return _f
    
    def _jax_weibull(self, expr):
        shape, scale = map(self._jax, expr.args)
        
        # W(shape, scale) = scale * (-log(1 - U(0, 1)))^{1 / shape}
        # TODO: make this numerically stable
        def _f(x, key):
            kx, key = shape(x, key)
            lx, key = scale(x, key)
            key, subkey = jrng.split(key)
            U = jrng.uniform(key=subkey)
            sample = lx * jnp.power(-jnp.log(1 - U), 1 / kx)
            return sample, key
        
        return _f
