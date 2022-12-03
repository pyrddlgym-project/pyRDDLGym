from typing import Sized
import sympy as sp

from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidNumberOfArgumentsError, RDDLMissingCPFDefinitionError, RDDLNotImplementedError, RDDLTypeError, RDDLUndefinedCPFError

from xaddpy.xadd import XADD

from pyRDDLGym.Core.Parser.expr import Expression


VALID_RELATIONAL_OPS = {'>=', '>', '<=', '<', '==', '~='}
OP_TO_XADD_OP = {
    '*': 'prod', '+': 'add', '-': 'subtract', '/': 'div',
    '|': 'or', '^': 'and', '~=': '!='
}
PRIME = '\''


class RDDLXADD:
    def __init__(self, rddl_ast):
        self.AST = rddl_ast
        self._context: XADD = XADD()
        self._var_to_node_id = {}
        self._op_to_node_id = {}
        self._node_id_to_op = {}
        self._var_name_to_p_types = {}
        self._aggr_to_scope = {}
        self._pvar_name_to_var_type = {}
        self.get_pvariable_info()
        self.convert_cpfs_to_xadds()
    
    def get_pvariable_info(self):
        for pvariable in self.AST.domain.pvariables:
            name = pvariable.name
            p_range = pvariable.range
            p_type = pvariable.param_types
            self._var_name_to_p_types[name] = p_type
            self._pvar_name_to_var_type[name] = p_range
            
    def convert_cpfs_to_xadds(self):
        ast = self.AST
        for pvar in ast.domain.pvariables:
            pvar_name = pvar.name
            fluent_type = pvar.fluent_type
            if fluent_type == 'non-fluent' or fluent_type == 'action-fluent':
                continue
            assert fluent_type in {'state-fluent', 'derived-fluent', 'interm-fluent', 'observ_fluent'}

            # Get the CPF of this pvariable
            cpf = None
            pvar_name = pvar_name + PRIME if fluent_type == 'state-fluent' else pvar_name
            for cpfs in self.AST.domain.cpfs[1]:
                if cpfs.pvar[1][0] == pvar_name:
                    cpf = cpfs
                    break
            if cpf is None:
                raise RDDLMissingCPFDefinitionError(
                    f'CPF <{pvar_name}> is missing a valid definition.'
                )
            
            expr = cpf.expr
            expr_xadd_node_id = self.expr_to_xadd(expr)
            pvar_sp = sp.symbols(pvar_name)
            self.ns[pvar_name] = pvar_sp
            self._var_to_node_id[pvar_sp] = expr_xadd_node_id
                     
        
        # Reward
        expr = ast.domain.reward
        expr_xadd_node_id = self.expr_to_xadd(expr)
        ast.domain.reward = expr_xadd_node_id

        # Terminal condition
        terminals = []
        for i, terminal in enumerate(ast.domain.terminals):
            expr = terminal
            expr_xadd_node_id = self.expr_to_xadd(expr)
            terminals.append(expr_xadd_node_id)
        ast.domain.terminals = terminals
        
        # Preconditions
        preconditions = []
        for i, precondition in enumerate(ast.domain.preconditions):
            expr = precondition
            expr_xadd_node_id = self.expr_to_xadd(expr)
            preconditions.append(expr_xadd_node_id)
        ast.domainmodel.preconditions = preconditions
        
        # Invariants
        invariants = []
        for i, invariant in enumerate(ast.domain.invariants):
            expr = invariant
            expr_xadd_node_id = self.expr_to_xadd(expr)
            invariants.append(expr_xadd_node_id)
        ast.domain.invariants = invariants      
    
    def expr_to_xadd(self, expr: Expression) -> int:
        # TODO: how to modify lst so that only branch out when encountering 
        # a decision expression (be it a boolean variable or an expression)
        node_id = self._op_to_node_id.get(expr)
        if node_id is not None:
            return node_id

        etype, op = expr.etype
        if etype == "constant":
            return self.constant_to_xadd(expr)
        elif etype == "pvar":
            return self.pvar_to_xadd(expr)
        elif etype == "aggregation":
            return self.aggr_to_xadd(expr)
        elif etype == "control":
            return self.control_to_xadd(expr)
        elif etype == "randomvar":
            return self.randomvar_to_xadd(expr)
        elif etype == "func":
            return self.func_to_xadd(expr)
        elif etype == "arithmetic":
            return self.arithmetic_to_xadd(expr)
        elif etype == "relational":
            return self.relational_to_xadd(expr)
        elif etype == "boolean":
            return self.bool_to_xadd(expr)
        else:
            raise Exception(f'Internal error: type {etype} is not supported.')
        
    def constant_to_xadd(self, expr: Expression) -> int:
        assert expr.etype[0] == 'constant'
        const = sp.sympify(expr.args, locals=self.ns)
        return self._context.convert_to_xadd(const)
    
    def pvar_to_xadd(self, expr: Expression) -> int:
        assert expr.etype[0] == 'pvar'
        var, args = expr.args
        var_type = self._pvar_name_to_var_type[var]
        args = '__' + '__'.join([arg for arg in args]) if len(args) >= 1 else ''
        var_ = self.ns.setdefault(
            var + args,
            sp.Symbol(
                var.replace('-', '_')+args,
                bool=var_type == 'bool'
            )
        )
        node_id = self._context.convert_to_xadd(var_)
        self._var_to_node_id[var_] = node_id
        return node_id

    def aggr_to_xadd(self, expr: Expression) -> int:
        """
        For lifted variables, we cannot evaluate aggregations such as 
            'exists', 'forall', 'sum', 'prod', 'avg', 'minimum', 'maximum'.
        Thus, these will be evaluated explicitly when grounding. 
        For now, it is sufficient to return a node with the right type.
        Additionally, we need to know what variables are given as arguments to
        this expression.
        
        For example, for an 'exists' operation,
            1) create a unique sympy variable associated with this expression
                The variable name should be unique but should be identical for 
                identical operations). 
                An exists operation can be uniquely identified by 
                    * the argument(s) over which we aggregate
                        these are ?x and ?y from exists_{?x: xpos, ?y: ypos}
                    * the operand... Can we recognize identical expressions?
                
            2) 
        """
        assert expr.etype[0] == 'aggregation'
        etype, op = expr.etype
        args = expr.args
        arg_vars = []
        for arg in args:
            if arg[0] == 'typed_var':
                arg_vars += list(arg[1]) # TODO: handle these
        assert len(arg_vars) >= 2

        num_aggregations = len(self._aggr_to_scope)
        postfix = '__' + '__'.join([a for a in arg_vars[::2]])
        
        # Aggregations that return Booleans
        if op == 'exists' or op == 'forall':
            var_sp = sp.Symbol(f'exists{postfix}_{num_aggregations}', bool=True)
        # Aggregations that return int/float values
        elif op in ('sum', 'prod', 'avg', 'minimum', 'maximum'):
            var_sp = sp.Symbol(f'{op}{postfix}_{num_aggregations}')
        else:
            raise RDDLNotImplementedError
        # TODO: Scope doesn't contain alls
        self._aggr_to_scope[var_sp] = [self.ns[var.split('/')[0]] for var in expr.scope]
        node_id = self._context.get_leaf_node(var_sp)
        return node_id        

    def control_to_xadd(self, expr: Expression) -> int:
        assert expr.etype[0] == 'control'
        args = list(map(self.expr_to_xadd, expr.args))
        return
    
    def randomvar_to_xadd(self, expr: Expression) -> int:
        assert expr.etype[0] == 'randomvar'
        args = list(map(self.expr_to_xadd, expr.args))
        return
    
    def func_to_xadd(self, expr: Expression) -> int:
        assert expr.etype[0] == 'func'
        etype, op = expr.etype
        args = list(map(self.expr_to_xadd, expr.args))
        if op == 'pow':
            assert len(args) == 2
            pow = expr.args[1].value
            node_id = self._context.unary_op(args[0], op, pow)
        elif op == 'max' or op == 'min':
            assert len(args) == 2
            node_id = self._context.apply(args[0], args[1], op)
        else:
            assert len(args) == 1
            node_id = self._context.unary_op(args[0], op)
        return node_id

    def arithmetic_to_xadd(self, expr: Expression) -> int:
        assert expr.etype[0] == 'arithmetic'
        etype, op = expr.etype
        args = list(map(self.expr_to_xadd, expr.args))
        if len(args) == 1:          # Unary operation
            node_id = self._context.unary_op(args[0], op)
        elif len(args) == 2:
            node_id = self._context.apply(args[0], args[1], OP_TO_XADD_OP.get(op, op))
        else:
            raise ValueError("Operations with XADD nodes should be unary or binary")
        return node_id

    def relational_to_xadd(self, expr: Expression) -> int:
        assert expr.etype[0] == 'relational'
        etype, op = expr.etype
        if op not in VALID_RELATIONAL_OPS:
            raise RDDLNotImplementedError(
                f'Relational operator {op} is not supported: must be one of {VALID_RELATIONAL_OPS}'
            )   #TODO: print stack trace?

        args = list(map(self.expr_to_xadd, expr.args))            
        if not isinstance(args, Sized):
            raise RDDLTypeError(
                f'Internal error: expected Sized, got {type(args)}'
            )   #TODO: print stack trace?
        elif len(args) != 2:
            raise RDDLInvalidNumberOfArgumentsError(
                f'Relational operator {op} requires 2 args, got {len(args)}'
            )   #TODO: print stack trace?
        
        node_id = self._context.apply(args[0], args[1], OP_TO_XADD_OP.get(op, op))
        return node_id

    def bool_to_xadd(self, expr: Expression) -> int:    
        assert expr.etype[0] == 'boolean'
        etype, op = expr.etype
        args = list(map(self.expr_to_xadd, expr.args))
        if len(args) == 1 and op == '~':
            node_id = self._context.unary_op(args[0], OP_TO_XADD_OP.get(op, op))
            return node_id
        elif len(args) >= 2:
            if op == '|' or op == '^':
                node_id = args[0]
                for arg in args[1:]:
                    node_id = self._context.apply(node_id, args[1], OP_TO_XADD_OP.get(op, op))
                return node_id
            elif len(args) == 2:
                if op == '~':
                    return  # When does this happen? 
                elif op == '=>':
                    return
                elif op == '<=>':
                    return
        
        raise RDDLInvalidNumberOfArgumentsError(
            f'Logical operator {op} does not have the required number of args, got {len(args)}' +
            f'\n{expr}'     # TODO: print stack trace?
        )
    
    @property
    def ns(self):
        if not hasattr(self, '_ns'):
            self._ns = {}
        return self._ns

    def print(self, node_id):
        print(self._context.get_exist_node(node_id))