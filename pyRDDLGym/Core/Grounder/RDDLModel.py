from abc import ABCMeta
from typing import Sized
import sympy as sp
from xaddpy.xadd import XADD, ControlFlow

from pyRDDLGym.Core.ErrorHandling.RDDLException import (
    RDDLInvalidNumberOfArgumentsError, RDDLMissingCPFDefinitionError, 
    RDDLNotImplementedError, RDDLTypeError, RDDLUndefinedCPFError
)
from pyRDDLGym.Core.Parser.expr import Expression

VALID_RELATIONAL_OPS = {'>=', '>', '<=', '<', '==', '~='}
OP_TO_XADD_OP = {
    '*': 'prod', '+': 'add', '-': 'subtract', '/': 'div',
    '|': 'or', '^': 'and', '~=': '!='
}
PRIME = '\''
UNIFORM_VAR_NAME = '#_UNIFORM_{num}'
GAUSSIAN_VAR_NAME = '#_GAUSSIAN_{num}'
EXPONENTIAL_VAR_NAME = '#_EXPONENTIAL_{num}'


class PlanningModel(metaclass=ABCMeta):
    def __init__(self):
        self._AST = None
        self._nonfluents = None
        self._states = None
        self._statesranges = None
        self._nextstates = None
        self._prevstates = None
        self._initstate = None
        self._actions = None
        self._objects = None
        self._actionsranges = None
        self._derived = None
        self._interm = None
        self._observ = None
        self._observranges = None
        self._cpfs = None
        self._cpforder = None
        self._reward = None
        self._terminals = None
        self._preconditions = None
        self._invariants = None
        self._gvar_to_type = None

        #new definitions
        self._max_allowed_actions = None
        self._horizon = None
        self._discount = None

    def SetAST(self, AST):
        self._AST = AST

    @property
    def objects(self):
        return self._objects

    @objects.setter
    def objects(self, value):
        self._objects = value

    @property
    def nonfluents(self):
        return self._nonfluents

    @nonfluents.setter
    def nonfluents(self, val):
        self._nonfluents = val

    @property
    def states(self):
        return self._states

    @states.setter
    def states(self, val):
        self._states = val

    @property
    def statesranges(self):
        return self._statesranges

    @statesranges.setter
    def statesranges(self, value):
        self._statesranges = value

    @property
    def next_state(self):
        return self._nextstates

    @next_state.setter
    def next_state(self, val):
        self._nextstates = val

    @property
    def prev_state(self):
        return self._prevstates

    @prev_state.setter
    def prev_state(self, val):
        self._prevstates = val

    @property
    def init_state(self):
        return self._initstate

    @init_state.setter
    def init_state(self, val):
        self._initstate = val

    @property
    def actions(self):
        return self._actions

    @actions.setter
    def actions(self, val):
        self._actions = val

    @property
    def actionsranges(self):
        return self._actionsranges

    @actionsranges.setter
    def actionsranges(self, value):
        self._actionsranges = value

    @property
    def derived(self):
        return self._derived

    @derived.setter
    def derived(self, val):
        self._derived = val

    @property
    def interm(self):
        return self._interm

    @interm.setter
    def interm(self, val):
        self._interm = val

    @property
    def observ(self):
        return self._observ

    @observ.setter
    def observ(self, value):
        self._observ = value

    @property
    def observranges(self):
        return self._observranges

    @observranges.setter
    def observranges(self, value):
        self._observranges = value

    @property
    def cpfs(self):
        return self._cpfs

    @cpfs.setter
    def cpfs(self, val):
        self._cpfs = val

    @property
    def cpforder(self):
        return self._cpforder

    @cpforder.setter
    def cpforder(self, val):
        self._cpforder = val

    @property
    def reward(self):
        return self._reward

    @reward.setter
    def reward(self, val):
        self._reward = val

    @property
    def terminals(self):
        return self._terminals

    @terminals.setter
    def terminals(self, value):
        self._terminals = value

    @property
    def preconditions(self):
        return self._preconditions

    @preconditions.setter
    def preconditions(self, val):
        self._preconditions = val

    @property
    def invariants(self):
        return self._invariants

    @invariants.setter
    def invariants(self, val):
        self._invariants = val
    
    @property
    def gvar_to_type(self):
        return self._gvar_to_type

    @gvar_to_type.setter
    def gvar_to_type(self, val):
        self._gvar_to_type = val
    
    @property
    def pvar_to_type(self):
        return self._pvar_to_type

    @pvar_to_type.setter
    def pvar_to_type(self, val):
        self._pvar_to_type = val

    @property
    def gvar_to_pvar(self):
        return self._gvar_to_pvar

    @gvar_to_pvar.setter
    def gvar_to_pvar(self, val):
        self._gvar_to_pvar = val

    @property
    def discount(self):
        return self._discount

    @discount.setter
    def discount(self, val):
        self._discount = val

    @property
    def horizon(self):
        return self._horizon

    @horizon.setter
    def horizon(self, val):
        self._horizon = val

    @property
    def max_allowed_actions(self):
        return self._max_allowed_actions

    @max_allowed_actions.setter
    def max_allowed_actions(self, val):
        self._max_allowed_actions = val


class RDDLModel(PlanningModel):
    def __init__(self):
        super().__init__()


class RDDLModelWXADD(PlanningModel):
    def __init__(self, ast):
        super().__init__()
        self.AST = ast
        self._context: XADD = XADD()
        self._var_name_to_node_id = {}
        self._sympy_var_to_node_id = {}
        self._sympy_var_name_to_var_name = {}
        self._var_name_to_sympy_var_name = {}
        self._op_to_node_id = {}
        self._node_id_to_op = {}
        self._curr_pvar = None
        self.rvs = self._context._random_var_set
    
    def compile(self):
        self.reset_dist_var_num()
        self.convert_cpfs_to_xadds()
    
    def reset_dist_var_num(self):
        self._num_uniform = 0
        self._num_gaussian = 0
        self._num_exponential = 0
    
    def convert_cpfs_to_xadds(self):
        cpfs = sorted(self.cpfs)
        
        # Handle state-fluent
        for cpf in cpfs:
            expr = self.cpfs[cpf]
            pvar_name = self.gvar_to_pvar[cpf]
            if pvar_name != self._curr_pvar:
                self._curr_pvar = pvar_name
                self.reset_dist_var_num()
            expr_xadd_node_id = self.expr_to_xadd(expr)
            self.cpfs[cpf] = expr_xadd_node_id
        self.cpfs = self.cpfs

        # Reward
        expr = self.reward
        expr_xadd_node_id = self.expr_to_xadd(expr)
        self.reward = expr_xadd_node_id

        # Terminal condition
        terminals = []
        for i, terminal in enumerate(self.terminals):
            expr = terminal
            expr_xadd_node_id = self.expr_to_xadd(expr)
            terminals.append(expr_xadd_node_id)
        self.terminals = terminals
        
        # Skip preconditions
        # preconditions = []
        # for i, precondition in enumerate(self.preconditions):
        #     expr = precondition
        #     expr_xadd_node_id = self.expr_to_xadd(expr)
        #     preconditions.append(expr_xadd_node_id)
        # self.preconditions = preconditions
        
        # Also skip invariants
        # invariants = []
        # for i, invariant in enumerate(self.invariants):
        #     expr = invariant
        #     expr_xadd_node_id = self.expr_to_xadd(expr)
        #     invariants.append(expr_xadd_node_id)
        # self.invariants = invariants      
    
    def expr_to_xadd(self, expr: Expression) -> int:
        node_id = self._op_to_node_id.get(expr)
        if node_id is not None:
            return node_id

        etype, op = expr.etype
        if etype == "constant":
            node_id = self.constant_to_xadd(expr)
        elif etype == "pvar":
            node_id = self.pvar_to_xadd(expr)
        elif etype == "aggregation":
            node_id = self.aggr_to_xadd(expr)
        elif etype == "control":
            node_id = self.control_to_xadd(expr)
        elif etype == "randomvar":
            node_id = self.randomvar_to_xadd(expr)
        elif etype == "func":
            node_id = self.func_to_xadd(expr)
        elif etype == "arithmetic":
            node_id = self.arithmetic_to_xadd(expr)
        elif etype == "relational":
            node_id = self.relational_to_xadd(expr)
        elif etype == "boolean":
            node_id = self.bool_to_xadd(expr)
        else:
            raise Exception(f'Internal error: type {etype} is not supported.')
        node_id = self._context.make_canonical(node_id)
        return node_id
    
    def constant_to_xadd(self, expr: Expression) -> int:
        assert expr.etype[0] == 'constant'
        const = sp.sympify(expr.args, locals=self.ns)
        return self._context.convert_to_xadd(const)
    
    def pvar_to_xadd(self, expr: Expression) -> int:
        assert expr.etype[0] == 'pvar'
        var, args = expr.args
        var_type = self.gvar_to_type[var]
        if var in self.nonfluents:
            var_ = self.nonfluents[var]
            node_id = self._context.convert_to_xadd(sp.S(var_))
            self._var_name_to_node_id[var] = node_id            
        else:
            var_ = self.ns.setdefault(
                var,
                sp.Symbol(
                    var.replace('-', '_'),
                    bool=var_type == 'bool'
                )
            )
            node_id = self._context.convert_to_xadd(var_)
            self._sympy_var_to_node_id[var_] = node_id
            self._var_name_to_node_id[var] = node_id
            self._sympy_var_name_to_var_name[str(var_)] = var
            self._var_name_to_sympy_var_name[var] = str(var_)
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
        """
        Control corresponds to if-else if-else statements.
        For ifs and else ifs, the first argument is the condition, 
        the second is the true branch, and the third argument is the false branch
        (else can be absorbed into the false branch of the last else if statement).
        
        Let's say the condition is represented by an XADD node n1;
        the true branch n2; and the false branch n3.
        Then, the final node can be obtained by the leaf operation ControlFlow,
        which goes to the leaf nodes of n1 and creates a decision node whose 
        high branch corresponds to the node n2 and whose low branch is the node n3.
        """
        assert expr.etype[0] == 'control'
        args = list(map(self.expr_to_xadd, expr.args))
        condition = args[0]
        true_branch = args[1]
        false_branch = args[2]
        leaf_op = ControlFlow(
            true_branch=true_branch,
            false_branch=false_branch,
            context=self._context
        )
        node_id = self._context.reduce_process_xadd_leaf(
            condition,
            leaf_op=leaf_op,
            decisions=[],
            decision_values=[]
        )
        return node_id
    
    def randomvar_to_xadd(self, expr: Expression) -> int:
        assert expr.etype[0] == 'randomvar'
        dist = expr.etype[1].lower()
        args = list(map(self.expr_to_xadd, expr.args))
        if dist == 'bernoulli':
            assert len(args) == 1
            proba = args[0]
            num_rv = self._num_uniform
            unif_rv = sp.Symbol(UNIFORM_VAR_NAME.format(num=num_rv), random=True)
            uniform = self._context.convert_to_xadd(
                unif_rv,
                params=(0, 1),     # rv ~ Uniform(0, 1)
                type='UNIFORM'
            )
            node_id = self._context.apply(uniform, proba, '<=')
            self._num_uniform += 1
            return node_id
        elif dist == 'binomial':
            pass
        elif dist == 'exponential':
            assert len(args) == 1
            num_rv = self._num_uniform
            unif_rv = sp.Symbol(UNIFORM_VAR_NAME.format(num=num_rv), random=True)
            uniform = self._context.convert_to_xadd(
                unif_rv,
                params=(0, 1),      # rv ~ Uniform(0, 1)
                type='UNIFORM'
            )
            # '-log(1 - U) * scale' is an exponential sample reparameterized with Uniform
            minus_u = self._context.unary_op(uniform, '-')
            log1_minus_u = self._context.unary_op(minus_u, 'log1p')
            neg_log1_minus_u = self._context.unary_op(log1_minus_u, '-')
            scale = args[0]
            node_id = self._context.apply(neg_log1_minus_u, scale, 'prod')
            self._num_uniform += 1
            return node_id
        elif dist == 'normal':
            assert len(args) == 2
            mean, var = args
            num_rv = self._num_gaussian
            gauss_rv = sp.Symbol(GAUSSIAN_VAR_NAME.format(num=num_rv), random=True)
            gaussian = self._context.convert_to_xadd(
                gauss_rv,
                params=(0, 1),    # rv ~ Normal(0, 1)
                type='NORMAL',
            )
            # mean + sqrt(var) * epsilon
            std = self._context.unary_op(var, 'sqrt')
            scaled = self._context.apply(std, gaussian, 'prod')
            node_id = self._context.apply(mean, scaled, 'add')
            self._num_gaussian += 1
            return node_id
        else:
            raise RDDLNotImplementedError(
                f'Distribution {dist} does not allow reparameterization'
            )   # TODO: print stack trace?
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
        node_id = self._context.make_canonical(node_id)
        return node_id

    def arithmetic_to_xadd(self, expr: Expression) -> int:
        assert expr.etype[0] == 'arithmetic'
        etype, op = expr.etype
        args = list(map(self.expr_to_xadd, expr.args))
        if len(args) == 1:          # Unary operation
            node_id = self._context.unary_op(args[0], op)
        elif len(args) == 2:
            node_id = self._context.apply(args[0], args[1], OP_TO_XADD_OP.get(op, op))
        elif len(args) > 2:
            node_id = args[0]
            for arg in args[1:]:
                node_id = self._context.apply(node_id, arg, OP_TO_XADD_OP.get(op, op))
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
                    node_id = self._context.apply(node_id, arg, OP_TO_XADD_OP.get(op, op))
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


def main():
    AST = "dfdf"
    M = RDDLModel()
    M.SetAST(AST)
    print(M.objects)
    M.objects["2"]=3
    print(M.objects)
    M.objects = {1 : 3, 4 : 5}
    print(M.objects)
    print("hello")




if __name__ == "__main__":
    main()
