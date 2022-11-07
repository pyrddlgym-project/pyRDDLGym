import jax
import jax.numpy as jnp
import jax.random as jrng

from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidExpressionError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidNumberOfArgumentsError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLNotImplementedError
from pyRDDLGym.Core.Grounder.RDDLModel import PlanningModel
from pyRDDLGym.Core.Parser.expr import Expression
from pyRDDLGym.Core.Simulator.RDDLDependencyAnalysis import RDDLDependencyAnalysis


class JaxRDDLCompiler:
    
    def __init__(self, model: PlanningModel, tau_reparam: float=1e-4) -> None:
        self._model = model
        self._tau = tau_reparam
        
        # perform a dependency analysis, do topological sort to compute levels
        dep_analysis = RDDLDependencyAnalysis(self._model)
        self.cpforder = dep_analysis.compute_levels()
        self.order_cpfs = list(sorted(self.cpforder.keys()))         
        
        self.cpf_types = {**model.intermranges, **model.derivedranges, **model.statesranges}
        print('dict is ' + str(self.cpf_types))
        
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
        tree = [self._jax(expr, bool) for expr in self._model.invariants]
        tree_jit = jax.tree_map(jax.jit, tree)
        return tree_jit
        
    def _compile_preconditions(self):
        tree = [self._jax(expr, bool) for expr in self._model.preconditions]
        tree_jit = jax.tree_map(jax.jit, tree)
        return tree_jit
    
    def _compile_termination(self):
        tree = [self._jax(expr, bool) for expr in self._model.terminals]
        tree_jit = jax.tree_map(jax.jit, tree)
        return tree_jit
    
    CPF_TYPES = {
        'int' : jnp.int32,
        'real' : jnp.float32,
        'bool' : bool
    }
    
    def _compile_cpfs(self):
        tree = {}  
        for order in self.order_cpfs:
            for cpf in self.cpforder[order]: 
                dtype = JaxRDDLCompiler.CPF_TYPES[self.cpf_types[cpf]]
                if cpf in self._model.next_state:
                    primed_cpf = self._model.next_state[cpf]
                    expr = self._model.cpfs[primed_cpf]
                    tree[primed_cpf] = self._jax(expr, dtype)
                else:
                    expr = self._model.cpfs[cpf]
                    tree[cpf] = self._jax(expr, dtype)
        tree_jit = jax.tree_map(jax.jit, tree)        
        return tree, tree_jit
    
    def _compile_reward(self):
        reward = self._jax(self._model.reward, jnp.float32)
        reward_jit = jax.jit(reward)
        return reward, reward_jit
        
    # start of compilation subroutines for expressions
    # skipped: aggregations (sum, prod) for now
    @staticmethod
    def _print_stack_trace(expr):
        return '...\n' + str(expr) + '\n...'

    def _jax(self, expr, dtype):
        if isinstance(expr, Expression):
            etype, op = expr.etype
            if etype == 'constant':
                return self._jax_constant(expr, dtype)
            elif etype == 'pvar':
                return self._jax_pvar(expr, dtype)
            elif etype == 'arithmetic':
                return self._jax_arithmetic(expr, op, dtype)
            elif etype == 'aggregation':
                return self._jax_aggregation(expr, op, dtype)
            elif etype == 'relational':
                return self._jax_relational(expr, op, dtype)
            elif etype == 'func':
                return self._jax_func(expr, op, dtype)
            elif etype == 'boolean':
                return self._jax_logical(expr, op, dtype)
            elif etype == 'control':
                return self._jax_control(expr, op, dtype)
            elif etype == 'randomvar':
                return self._jax_random(expr, op, dtype)
            else:
                raise RDDLNotImplementedError(
                    'Internal error: expression type {} is not supported.'.format(etype) + 
                    '\n' + JaxRDDLCompiler._print_stack_trace(expr))
        else:
            raise RDDLNotImplementedError(
                'Internal error: type {} is not supported.'.format(expr) + 
                '\n' + JaxRDDLCompiler._print_stack_trace(expr))
            
    # simple expressions
    def _jax_constant(self, expr, dtype):
        const = expr.args
        
        def _f(x, key):
            sample = jnp.array(const, dtype)
            return sample, key

        return _f
    
    def _jax_pvar(self, expr, dtype):
        var, *_ = expr.args
        
        def _f(x, key):
            sample = jnp.array(x[var], dtype)
            return sample, key

        return _f
    
    def _jax_eval_unary_op(self, jaxop, dtype, jaxexpr):
        
        def _f(x, key):
            val, key = jaxexpr(x, key)
            sample = jaxop(val).astype(dtype)
            return sample, key
        
        return _f
        
    def _jax_eval_binary_op(self, jaxop, dtype, jaxexpr1, jaxexpr2):
        
        def _f(x, key):
            val1, key = jaxexpr1(x, key)
            val2, key = jaxexpr2(x, key)
            sample = jaxop(val1, val2).astype(dtype)
            return sample, key
        
        return _f
    
    RELATIONAL_OPS = {
        '>=': jnp.greater_equal,
        '<=': jnp.less_equal,
        '<': jnp.less,
        '>': jnp.greater,
        '==': jnp.equal,
        '~=': jnp.not_equal
    }
    
    def _jax_relational(self, expr, op, dtype):
        if op not in JaxRDDLCompiler.RELATIONAL_OPS:
            raise RDDLNotImplementedError(
                'Relational operator {} is not supported: must be one of {}.'.format(
                    op, JaxRDDLCompiler.RELATIONAL_OPS.keys()) + 
                '\n' + JaxRDDLCompiler._print_stack_trace(expr))
        
        jaxexprs = [self._jax(e, jnp.float32) for e in expr.args]
        n = len(jaxexprs)        
        
        if n != 2:
            raise RDDLInvalidNumberOfArgumentsError(
                'Relational operator {} requires 2 args, got {}.'.format(op, n) + 
                '\n' + JaxRDDLCompiler._print_stack_trace(expr))

        jaxop = JaxRDDLCompiler.RELATIONAL_OPS[op]  
        return self._jax_eval_binary_op(jaxop, dtype, *jaxexprs)
        
    ARITHMETIC_OPS = {
        '+': jnp.add,
        '-': jnp.subtract,
        '*': jnp.multiply,
        '/': jnp.divide
    }
    
    def _jax_arithmetic(self, expr, op, dtype):
        n = len(expr.args)
        if n != 2 and op in JaxRDDLCompiler.AGGREGATION_OPS:
            return self._jax_aggregation(expr, op, dtype)
        
        if op not in JaxRDDLCompiler.ARITHMETIC_OPS:
            raise RDDLNotImplementedError(
                'Arithmetic operator {} is not supported: must be one of {}.'.format(
                    op, JaxRDDLCompiler.ARITHMETIC_OPS.keys()) + 
                '\n' + JaxRDDLCompiler._print_stack_trace(expr))
        
        jaxexprs = [self._jax(e, dtype) for e in expr.args]
                
        if n == 1 and op == '-':
            return self._jax_eval_unary_op(jnp.negative, dtype, *jaxexprs)
        elif n == 2:
            jaxop = JaxRDDLCompiler.ARITHMETIC_OPS[op]
            return self._jax_eval_binary_op(jaxop, dtype, *jaxexprs)
            
        raise RDDLInvalidNumberOfArgumentsError(
            'Arithmetic operator {} does not have the required number of args, got {}.'.format(
                op, n) + 
            '\n' + JaxRDDLCompiler._print_stack_trace(expr))
        
    # aggregations
    def _jax_eval_nary_op(self, jaxop, dtype, *jaxexprs):
        n = len(jaxexprs)
        
        def _eval_args(carry, i):
            x, key = carry
            val, key = jax.lax.switch(i, jaxexprs, x, key)
            carry = (x, key)
            return carry, val
                
        def _f(x, key):
            ids = jnp.arange(n)
            seed = (x, key)
            (_, key), vals = jax.lax.scan(_eval_args, seed, ids)
            sample = jaxop(vals).astype(dtype)
            return sample, key
        
        return _f
        
    AGGREGATION_OPS = {
        '+': jnp.sum,
        '*': jnp.prod,
        '^': jnp.all,
        '|': jnp.any,
        'minimum': jnp.min,
        'maximum': jnp.max
    }
    
    def _jax_aggregation(self, expr, op, dtype):
        if op not in JaxRDDLCompiler.AGGREGATION_OPS:
            raise RDDLNotImplementedError(
                'Aggregation operator {} is not supported: must be one of {}.'.format(
                    op, JaxRDDLCompiler.AGGREGATION_OPS.keys()) + 
                '\n' + JaxRDDLCompiler._print_stack_trace(expr))
        
        if op in {'^', '|'}:
            jaxexprs = [self._jax(e, bool) for e in expr.args]
        else:
            jaxexprs = [self._jax(e, dtype) for e in expr.args]
        n = len(jaxexprs)
        
        if n == 1:
            jaxexpr, = jaxexprs
            return jaxexpr
        elif n == 2:
            jaxop = JaxRDDLCompiler.AGGREGATION_OPS[op]  
            return self._jax_eval_binary_op(jaxop, dtype, *jaxexprs)
        else:
            jaxop = JaxRDDLCompiler.AGGREGATION_OPS[op]  
            return self._jax_eval_nary_op(jaxop, dtype, *jaxexprs)
            
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
    
    def _jax_func(self, expr, op, dtype):
        n = len(expr.args)
        
        if op in JaxRDDLCompiler.KNOWN_UNARY:
            if n != 1:
                raise RDDLInvalidNumberOfArgumentsError(
                    'Unary function {} requires 1 arg, got {}.'.format(op, n) + 
                    '\n' + JaxRDDLCompiler._print_stack_trace(expr))    
                            
            jaxop = JaxRDDLCompiler.KNOWN_UNARY[op] 
            jaxexprs = [self._jax(e, jnp.float32) for e in expr.args]
            return self._jax_eval_unary_op(jaxop, dtype, *jaxexprs)

        elif op in JaxRDDLCompiler.KNOWN_BINARY:
            if n != 2:
                raise RDDLInvalidNumberOfArgumentsError(
                    'Binary function {} requires 2 args, got {}.'.format(op, n) + 
                    '\n' + JaxRDDLCompiler._print_stack_trace(expr))  
                
            jaxop = JaxRDDLCompiler.KNOWN_BINARY[op]      
            jaxexprs = [self._jax(e, jnp.float32) for e in expr.args]
            return self._jax_eval_binary_op(jaxop, dtype, *jaxexprs)
        
        raise RDDLNotImplementedError(
                'Function {} is not supported.'.format(op) + 
                '\n' + JaxRDDLCompiler._print_stack_trace(expr))   
        
    # logical expressions    
    LOGICAL_OPS = {
        '^': jnp.logical_and,
        '|': jnp.logical_or,
        '~': jnp.logical_xor,
        '=>': lambda e1, e2: jnp.logical_or(jnp.logical_not(e1), e2),
        '<=>': jnp.equal
    }
    
    def _jax_logical(self, expr, op, dtype):
        n = len(expr.args)
        if n != 2 and op in JaxRDDLCompiler.AGGREGATION_OPS:
            return self._jax_aggregation(expr, op, dtype)
        
        if op not in JaxRDDLCompiler.LOGICAL_OPS:
            raise RDDLNotImplementedError(
                'Logical operator {} is not supported: must be one of {}.'.format(
                    op, JaxRDDLCompiler.LOGICAL_OPS) + 
                '\n' + JaxRDDLCompiler._print_stack_trace(expr))
        
        jaxexprs = [self._jax(e, bool) for e in expr.args]
        
        if n == 1 and op == '~':
            return self._jax_eval_unary_op(jnp.logical_not, dtype, *jaxexprs)
        elif n == 2:
            jaxop = JaxRDDLCompiler.LOGICAL_OPS[op]
            return self._jax_eval_binary_op(jaxop, dtype, *jaxexprs)
            
        raise RDDLInvalidNumberOfArgumentsError(
            'Logical operator {} does not have the required number of args, got {}.'.format(op, n) + 
            '\n' + JaxRDDLCompiler._print_stack_trace(expr))
    
    # control flow
    def _jax_control(self, expr, op, dtype):
        if op not in {'if'}:
            raise RDDLNotImplementedError(
                'Control flow type {} is not supported: must be one of {}.'.format(
                    op, {'if'}) + 
                '\n' + JaxRDDLCompiler._print_stack_trace(expr))
        
        jax_pred = self._jax(expr.args[0], bool)
        jax_true = self._jax(expr.args[1], dtype)
        jax_false = self._jax(expr.args[2], dtype)
        
        def _f(x, key):
            val, key = jax_pred(x, key)
            return jax.lax.cond(val, jax_true, jax_false, x, key)
        
        return _f
        
    # random variables        
    def _jax_random(self, expr, name, dtype):
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
        elif name == 'Bernoulli':
            return self._jax_bernoulli(expr)
        elif name in {'Poisson', 'Gamma'}:
            
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
        return self._jax(expr.args[0], jnp.int32)
    
    def _jax_dirac(self, expr):
        return self._jax(expr.args[0], jnp.float32)
    
    def _jax_uniform(self, expr):
        jaxlb, jaxub = [self._jax(e, jnp.float32) for e in expr.args]
        
        # U(a, b) = a + (b - a) * xi, where xi ~ U(0, 1)
        def _f(x, key):
            ax, key = jaxlb(x, key)
            bx, key = jaxub(x, key)
            key, subkey = jrng.split(key)
            U = jrng.uniform(key=subkey)
            sample = ax + (bx - ax) * U
            return sample, key

        return _f

    def _jax_normal(self, expr):
        jaxmean, jaxvar = [self._jax(e, jnp.float32) for e in expr.args]

        # N(mu, s^2) = mu + s * N(0, 1)         
        def _f(x, key):
            meanx, key = jaxmean(x, key)
            varx, key = jaxvar(x, key)
            stdx = jnp.sqrt(varx)
            key, subkey = jrng.split(key)
            Z = jrng.normal(key=subkey)
            sample = meanx + stdx * Z
            return sample, key
            
        return _f
    
    def _jax_exponential(self, expr):
        jaxscale = self._jax(expr.args[0], jnp.float32)
        
        # Exp(scale) = scale * Exp(1)
        def _f(x, key):
            scalex, key = jaxscale(x, key)
            key, subkey = jrng.split(key)
            Exp1 = jrng.exponential(key=subkey)
            sample = scalex * Exp1
            return sample, key

        return _f
    
    def _jax_weibull(self, expr):
        jaxshape, jaxscale = [self._jax(e, jnp.float32) for e in expr.args]
        
        # W(shape, scale) = scale * (-log(1 - U(0, 1)))^{1 / shape}
        # TODO: make this numerically stable
        def _f(x, key):
            kx, key = jaxshape(x, key)
            lx, key = jaxscale(x, key)
            key, subkey = jrng.split(key)
            U = jrng.uniform(key=subkey)
            sample = lx * jnp.power(-jnp.log(1 - U), 1 / kx)
            return sample, key
        
        return _f
    
    def _jax_bernoulli(self, expr):
        jaxprob = self._jax(expr.args[0], jnp.float32)
        
        def _f(x, key):
            px, key = jaxprob(x, key)
            key, subkey = jrng.split(key)
            U = jrng.uniform(key=subkey)
            sample = jnp.less(U, px)
            return sample, key
        
        return _f
