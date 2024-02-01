from typing import Set, Sized, Tuple

import sympy as sp
from xaddpy.xadd import XADD
from xaddpy.xadd.xadd import ControlFlow, DeltaFunctionSubstitution

from pyRDDLGym.Core.Compiler.RDDLModel import PlanningModel, RDDLGroundedModel
from pyRDDLGym.Core.ErrorHandling.RDDLException import (
    RDDLInvalidNumberOfArgumentsError,
    RDDLNotImplementedError,
    RDDLTypeError,
)
from pyRDDLGym.Core.Parser.expr import Expression
from pyRDDLGym.XADD.helper import get_bernoulli_node_id


VALID_RELATIONAL_OPS = {'>=', '>', '<=', '<', '==', '~='}
OP_TO_XADD_OP = {
    '*': 'prod', '+': 'add', '-': 'subtract', '/': 'div',
    '|': 'or', '^': 'and', '~=': '!='
}
PRIME = '\''
UNIFORM_VAR_NAME = '#_UNIFORM_{num}'
GAUSSIAN_VAR_NAME = '#_GAUSSIAN_{num}'
EXPONENTIAL_VAR_NAME = '#_EXPONENTIAL_{num}'


class RDDLModelWXADD(PlanningModel):

    def __init__(
            self,
            model: RDDLGroundedModel,
            context: XADD = None,
            simulation: bool = True,
    ):
        super().__init__()
        self.model = model
        self._context: XADD = XADD() if context is None else context
        self._var_name_to_node_id = {}
        self._sympy_var_to_node_id = {}
        self._sympy_var_name_to_var_name = {}
        self._var_name_to_sympy_var_name = {}
        self._op_to_node_id = {}
        self._node_id_to_op = {}
        self._curr_pvar = None
        self.rvs = self._context._random_var_set

        # Get grounded variables
        self.states = model.states
        self.actions = model.actions
        self.nonfluents = model.nonfluents
        self.next_state = model.next_state
        self.prev_state = model.prev_state
        self.init_state = model.init_state
        self.cpfs = model.cpfs
        self.cpforder = model.cpforder
        self.gvar_to_cpforder = model.gvar_to_cpforder
        self.reward = model.reward
        self.terminals = model.terminals
        self.preconditions = model.preconditions
        self.invariants = model.invariants
        self.derived = model.derived
        self.interm = model.interm
        self.observ = model.observ
        self.observranges = model.observranges
        self.objects = model.objects
        self.actionsranges = model.actionsranges
        self.statesranges = model.statesranges
        self.gvar_to_type = model.gvar_to_type
        self.pvar_to_type = model.pvar_to_type
        self.gvar_to_pvar = model.gvar_to_pvar
        self.max_allowed_actions = model.max_allowed_actions
        self.horizon = model.horizon
        self.discount = model.discount
        self.compiled = False
        self.simulation = simulation
        self._need_postprocessing = False
        self._postprocessing_kwargs = {}

    def compile(self, simulation=None):
        if simulation is not None and isinstance(simulation, bool):
            self.simulation = simulation
        self.reset_dist_var_num()
        self.convert_cpfs_to_xadds()
        self.compiled = True

    def reset_dist_var_num(self):
        self._num_uniform = 0
        self._num_gaussian = 0
        self._num_exponential = 0

    def convert_cpfs_to_xadds(self):
        
        # the API for accessing CPFs is now dict of {name: (objects, expr)}
        # see Mike pull request from Dec 18
        cpfs = {name: expr for name, (_, expr) in self.cpfs.items()}
        cpfs = sorted(cpfs)
        
        # Handle state-fluent
        for cpf in cpfs:
            self._need_postprocessing = False
            _, expr = self.cpfs[cpf]
            pvar_name = self.gvar_to_pvar[cpf]
            if pvar_name != self._curr_pvar:
                self._curr_pvar = pvar_name
                self.reset_dist_var_num()
            expr_xadd_node_id = self.expr_to_xadd(expr)
            # Post-processing for Bernoulli CPFs
            if self._need_postprocessing:
                expr_xadd_node_id = self.postprocess(
                    expr_xadd_node_id,
                    **self._postprocessing_kwargs)
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
            var_, node_id = self.add_sympy_var(var, var_type)
        return node_id

    def add_sympy_var(
            self,
            var_name: str,
            var_type: str,
            random: bool = False,
            **kwargs,
    ) -> Tuple[sp.Symbol, int]:
        if var_name in self._var_name_to_sympy_var_name:
            var_ = self.ns[var_name]
            node_id = self._var_name_to_node_id[var_name]
            return var_, node_id
        var_ = self.ns.setdefault(
            var_name,
            sp.Symbol(
                var_name.replace('-', '_'),
                bool=var_type == 'bool',
                random=random
            )
        )
        node_id = self._context.convert_to_xadd(var_, **kwargs)
        self._sympy_var_to_node_id[var_] = node_id
        self._var_name_to_node_id[var_name] = node_id
        self._sympy_var_name_to_var_name[str(var_)] = var_name
        self._var_name_to_sympy_var_name[var_name] = str(var_)
        return var_, node_id

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
                arg_vars += list(arg[1])  # TODO: handle these
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
        """Converts a random variable expression to an XADD node.

        When `self.simulation` is True, we reparameterize RVs for sampling purposes. 
          That is, a Bernoulli rv can be sampled by sampling a uniform random value 
          within the range [0, 1] and check whether the sampled value is greater/smaller 
          than the Bernoulli parameter.
        On the other hand, when `self.simulation` is False, we will create a node 
            something like the following:
                ([Bernoulli_params_1_123])
            where `123` is the node ID associated with the Bernoulli parameter,
            and `1` denotes the number of parameters, which is introduced to help with
            parsing later in a downstream task.

        Currently, `self.simulation` should be set to True for all rvs except for 
        Bernoulli. Furthermore, we assume that the Bernoulli node is used only at a
        leaf node of the XADD.
        """
        assert expr.etype[0] == 'randomvar'
        dist = expr.etype[1]
        args = list(map(self.expr_to_xadd, expr.args))
        
        if dist != 'Bernoulli':
            assert self.simulation, ("Currently, only Bernoulli" 
                                     "can be used for non-simulation.")

        if dist == 'Bernoulli':
            assert len(args) == 1
            proba = args[0]
            # Sample a Bernoulli rv by sampling a uniform rv
            if self.simulation:
                num_rv = self._num_uniform
                unif_rv = sp.Symbol(UNIFORM_VAR_NAME.format(num=num_rv), random=True)
                uniform = self._context.convert_to_xadd(
                    unif_rv,
                    params=(0, 1),  # rv ~ Uniform(0, 1)
                    type='UNIFORM'
                )
                node_id = self._context.apply(uniform, proba, '<=')
                self._num_uniform += 1
            # Create a Bernoulli node
            # TODO: How to assert that a Bernoulli node can only come at a leaf?
            else:
                name = f'{dist}_params_{len(args)}_{proba}'
                rv, node_id = self.add_sympy_var(name,
                                                 'real',
                                                 random=True,
                                                 params=(proba,),
                                                 type='BERNOULLI')
                # Need to postprocess the node to make leaf nodes 
                # represent Bernoulli probabilities
                self._need_postprocessing = True
            return node_id
        
        elif dist == 'Binomial':
            pass
        
        elif dist == 'krondelta':
            # Change Boolean leaf values to 0s and 1s
            assert len(args) == 1
            arg, = args
            node_id = self._context.unary_op(arg, op='int')
            return node_id

        elif dist == 'exponential':
            assert len(args) == 1
            num_rv = self._num_uniform
            unif_rv = sp.Symbol(UNIFORM_VAR_NAME.format(num=num_rv), random=True)
            uniform = self._context.convert_to_xadd(
                unif_rv,
                params=(0, 1),  # rv ~ Uniform(0, 1)
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
                params=(0, 1),  # rv ~ Normal(0, 1)
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
            )  # TODO: print stack trace?
            
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
        if len(args) == 1:  # Unary operation
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
            )  # TODO: print stack trace?

        args = list(map(self.expr_to_xadd, expr.args))            
        if not isinstance(args, Sized):
            raise RDDLTypeError(
                f'Internal error: expected Sized, got {type(args)}'
            )  # TODO: print stack trace?
        elif len(args) != 2:
            raise RDDLInvalidNumberOfArgumentsError(
                f'Relational operator {op} requires 2 args, got {len(args)}'
            )  # TODO: print stack trace?
        
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
            f'\n{expr}'  # TODO: print stack trace?
        )

    def postprocess(self, node_id: int, **kwargs) -> int:
        """Post processes a CPF XADD node.
        
        Currently, this method exists solely to handle the Bernoulli
        CPFs when the simulation flag is set to False. In this case,
        we replace the Bernoulli leaf nodes with their corresponding
        parameter nodes.
        """
        var_set = self._context.collect_vars(node_id)
        bernoulli_set = set()
        for v in var_set:
            name = str(v)
            if name.startswith('Bernoulli'):
                bernoulli_set.add(v)
        for rv in bernoulli_set:
            bern_param_node_id = get_bernoulli_node_id(rv)
            leaf_op = DeltaFunctionSubstitution(
                sub_var=rv, xadd_sub_at_leaves=node_id, context=self._context)
            node_id = self._context.reduce_process_xadd_leaf(node_id=bern_param_node_id,
                                                             leaf_op=leaf_op,
                                                             decisions=[],
                                                             decision_values=[])
        try:
            node_id = self._context.reduce_lp(node_id)
        except:
            pass
        return node_id

    @property
    def ns(self):
        if not hasattr(self, '_ns'):
            self._ns = {}
        return self._ns

    def print(self, node_id):
        print(self._context.get_exist_node(node_id))

    def collect_vars(self, node_id: int) -> Set[str]:
        """Returns the set containing variables existing in the current node (in str type)"""
        var_set = self._context.collect_vars(node_id)
        var_set = var_set.difference(self._context._random_var_set)
        return set(self._sympy_var_name_to_var_name[str(v)] for v in var_set)
    
    @property
    def vars_in_rew(self) -> Set[str]:
        if not hasattr(self, '_vars_in_rew'):
            self._vars_in_rew = self.collect_vars(self.reward)
        return self._vars_in_rew
    
    @vars_in_rew.setter
    def vars_in_rew(self, var: Set[str]):
        self._vars_in_rew = var

    @property
    def simulation(self) -> bool:
        return self._simulation

    @simulation.setter
    def simulation(self, sim: bool):
        self._simulation = sim

    @property
    def compiled(self) -> bool:
        return self._compiled

    @compiled.setter
    def compiled(self, comp: bool):
        self._compiled = comp
