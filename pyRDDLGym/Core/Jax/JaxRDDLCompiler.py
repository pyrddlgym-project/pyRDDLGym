import jax
import jax.numpy as jnp
import jax.random as jrng

from pyRDDLGym.Core.Grounder.RDDLModel import PlanningModel
from pyRDDLGym.Core.Parser.expr import Expression
from pyRDDLGym.Core.Simulator.RDDLDependencyAnalysis import RDDLDependencyAnalysis


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
    
    def _jax_agg_1(self, arg, jaxop):
        
        def _f(x, key):
            val, key = arg(x, key)
            sample = jaxop(val)
            return sample, key
        
        return _f
    
    def _jax_agg_2(self, arg1, arg2, jaxop):
        
        def _f(x, key):
            val1, key = arg1(x, key)
            val2, key = arg2(x, key)
            sample = jaxop(val1, val2)
            return sample, key
        
        return _f

    AGGREGATION_OPS = {
        '+': (jnp.add, 0),
        '*': (jnp.multiply, 1),
        '^': (jnp.logical_and, True),
        '|': (jnp.logical_or, False),
        'minimum': (jnp.minimum, None),
        'maximum': (jnp.maximum, None)
    }
    
    def _jax_aggregation(self, expr, op):
        n = len(expr.args)
        args = map(self._jax, expr.args)
        jaxop, _ = JaxRDDLCompiler.AGGREGATION_OPS[op]
            
        # simple case of one or two arguments; can produce a compact jaxpr
        if n == 1:
            _f, = args
        elif n == 2:
            _f = self._jax_agg_2(*args, jaxop)
        
        # TODO: n-ary; this produces a large jaxpr!
        
        return _f
    
    def _jax_arithmetic(self, expr, op):
        
        # can have arbitrary number of args
        if op in JaxRDDLCompiler.AGGREGATION_OPS:
            return self._jax_aggregation(expr, op)
        
        # can only have one or two args
        n = len(expr.args)
        args = map(self._jax, expr.args)
        
        if op == '-':
            if n == 1:
                _f = self._jax_agg_1(*args, jnp.negative)
            elif n == 2:
                _f = self._jax_agg_2(*args, jnp.subtract)
                
        elif op == '/':
            _f = self._jax_agg_2(*args, jnp.divide)
            
        return _f 
        
    RELATIONAL_OPS = {
        '>=': jnp.greater_equal,
        '<=': jnp.less_equal,
        '<': jnp.less,
        '>': jnp.greater,
        '==': jnp.equal,
        '~=': jnp.not_equal
    }
    
    def _jax_relational(self, expr, op):
        args = map(self._jax, expr.args)
        jaxop = JaxRDDLCompiler.RELATIONAL_OPS[op]
        _f = self._jax_agg_2(*args, jaxop)
        return _f
        
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
        args = map(self._jax, expr.args)
        
        if op in JaxRDDLCompiler.KNOWN_BINARY:
            jaxop = JaxRDDLCompiler.KNOWN_BINARY[op]
            _f = self._jax_agg_2(*args, jaxop)

        elif op in JaxRDDLCompiler.KNOWN_UNARY:
            jaxop = JaxRDDLCompiler.KNOWN_UNARY[op]
            _f = self._jax_agg_1(*args, jaxop)
            
        return _f
    
    # logical expressions    
    def _jax_logical(self, expr, op):
        
        # can have arbitrary number of args
        if op in JaxRDDLCompiler.AGGREGATION_OPS:
            return self._jax_aggregation(expr, op)
        
        # can only have one or two args (TODO: implement => and <=>)
        n = len(expr.args)
        args = map(self._jax, expr.args)
        
        if op == '~':
            if n == 1:
                _f = self._jax_agg_1(*args, jnp.logical_not)
            elif n == 2:
                _f = self._jax_agg_2(*args, jnp.logical_or)
                
        return _f 
    
    # control flow
    def _jax_control(self, expr, _):
        pred, true_f, false_f = map(self._jax, expr.args)
        
        def _f(x, key):
            predx, key = pred(x, key)
            sample, key = jax.lax.cond(predx, true_f, false_f, x, key)
            return sample, key

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
        else:
            # TODO: try these
            # https://arxiv.org/pdf/1611.01144.pdf?
            # https://arxiv.org/pdf/2003.01847.pdf?
            pass
    
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
