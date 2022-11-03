import jax
import jax.numpy as jnp

from pyRDDLGym.Core.Grounder.RDDLModel import PlanningModel
from pyRDDLGym.Core.Parser.expr import Expression
from pyRDDLGym.Core.Simulator.RDDLDependencyAnalysis import RDDLDependencyAnalysis


class JaxCompiler:
    
    def __init__(self,
                 model: PlanningModel) -> None:
        self._model = model
        
        # perform a dependency analysis, do topological sort to compute levels
        dep_analysis = RDDLDependencyAnalysis(self._model)
        self.cpforder = dep_analysis.compute_levels()
        self._order_cpfs = list(sorted(self.cpforder.keys())) 
            
        self._action_fluents = set(self._model.actions.keys())
        self._init_actions = self._model.actions.copy()
        
        # non-fluent will never change
        self._subs = self._model.nonfluents.copy()
        
        # is a POMDP
        self._pomdp = bool(self._model.observ)
    
    def compile(self):
        jax_cpfs = {}
        self._reparam = {}
        
        for order in self._order_cpfs:
            for cpf in self.cpforder[order]: 
                if cpf in self._model.next_state:
                    primed_cpf = self._model.next_state[cpf]
                    expr = self._model.cpfs[primed_cpf]
                    jax_cpfs[primed_cpf] = jax.jit(self._jax(expr))
                else:
                    expr = self._model.cpfs[cpf]
                    jax_cpfs[cpf] = jax.jit(self._jax(expr))
        jax_reward = jax.jit(self._jax(self._model.reward))
        
        return jax_cpfs, jax_reward, self._reparam
        
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
        arg = expr.args
        _f = lambda _: arg
        return _f
    
    def _jax_pvar(self, expr):
        var, *_ = expr.args
        _f = lambda x: x[var]
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
        agg, init_val = JaxCompiler.AGGREGATION_OPS[op]
            
        # simple case of one or two arguments; can produce a compact jaxpr
        if n == 1:
            _f, = args
        elif n == 2:
            arg1, arg2 = args
            _f = lambda x: agg(arg1(x), arg2(x))
        
        # n-ary; this produces a large jaxpr!
        else:
            # seed with first arg value or provided arg
            if init_val is None:
                init_f = lambda x: args[0](x)  
            else:
                init_f = lambda _: init_val
            
            # accumulate expression
            _f = lambda x: jax.lax.fori_loop(
                0, n,
                lambda i, accum: agg(accum, jax.lax.switch(i, args, x)),
                init_f(x))
            
        return _f
    
    def _jax_arithmetic(self, expr, op):
        
        # can have arbitrary number of args
        if op in JaxCompiler.AGGREGATION_OPS:
            return self._jax_aggregation(expr, op)
        
        # can only have one or two args
        n = len(expr.args)
        args = map(self._jax, expr.args)
        
        if op == '-':
            if n == 1:
                arg, = args
                _f = lambda x: jnp.negative(arg(x))
            elif n == 2:
                arg1, arg2 = args
                _f = lambda x: jnp.subtract(arg1(x), arg2(x))
                
        elif op == '/':
            arg1, arg2 = args
            _f = lambda x: jnp.divide(arg1(x), arg2(x))
        
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
        arg1, arg2 = args
        agg = JaxCompiler.RELATIONAL_OPS[op]
        _f = lambda x: agg(arg1(x), arg2(x))
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
        
        if op in JaxCompiler.KNOWN_BINARY:
            arg1, arg2 = args
            agg = JaxCompiler.KNOWN_BINARY[op]
            _f = lambda x: agg(arg1(x), arg2(x))
            
        elif op in JaxCompiler.KNOWN_UNARY:
            arg, = args
            agg = JaxCompiler.KNOWN_UNARY[op]
            _f = lambda x: agg(arg(x))
            
        return _f
    
    # logical expressions    
    def _jax_logical(self, expr, op):
        
        # can have arbitrary number of args
        if op in JaxCompiler.AGGREGATION_OPS:
            return self._jax_aggregation(expr, op)
        
        # can only have one or two args (TODO: implement => and <=>)
        n = len(expr.args)
        args = map(self._jax, expr.args)
        
        if op == '~':
            if n == 1:
                arg, = args
                _f = lambda x: jnp.logical_not(arg(x))
            elif n == 2:
                arg1, arg2 = args
                _f = lambda x: jnp.logical_xor(arg1(x), arg2(x))
             
        return _f 
    
    # control flow
    def _jax_control(self, expr, _):
        pred, true_f, false_f = map(self._jax, expr.args)
        _f = lambda x: jax.lax.cond(pred(x), true_f(x), false_f(x))
        return _f
        
    # random variables
    def _register_reparameterization(self, sampler):
        nparam = len(self._reparam)
        var_name = 'jax:xi-' + str(nparam + 1)
        self._reparam[var_name] = sampler
        return var_name
        
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
        xi = self._register_reparameterization(jax.random.uniform)
        _f = lambda x: a(x) * (1 - x[xi]) + b(x) * x[xi]
        return _f

    def _jax_normal(self, expr):
        mean, var = map(self._jax, expr.args)

        # N(mu, s^2) = mu + s * N(0, 1) 
        xi = self._register_reparameterization(jax.random.normal)
        _f = lambda x: mean(x) + jnp.sqrt(var(x)) * x[xi]        
        return _f
    
    def _jax_exponential(self, expr):
        scale, = map(self._jax, expr.args)
        
        # Exp(lam) = lam * Exp(1)
        xi = self._register_reparameterization(jax.random.exponential)
        _f = lambda x: scale(x) * x[xi]
        return _f
    
    def _jax_weibull(self, expr):
        shape, scale = map(self._jax, expr.args)
        
        # W(k, lam) = lam * (-ln(1 - U(0, 1)))^{1 / k}
        xi = self._register_reparameterization(jax.random.uniform)
        
        # TODO: make this numerically stable
        def _f(x):
            kx, lx = shape(x), scale(x)
            f = lx * jnp.power(-jnp.log(1 - x[xi]), 1 / kx)
            return f
        
        return _f
