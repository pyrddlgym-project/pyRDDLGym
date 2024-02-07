import numpy as np

from pyRDDLGym.core.debug.exception import print_stack_trace, raise_warning
from pyRDDLGym.core.simulator import RDDLSimulator


class RDDLConstraints:
    '''Provides added functionality to understand a set of action-preconditions
    and state invariants in a RDDL file.
    '''
    
    def __init__(self, simulator: RDDLSimulator,
                 max_bound: float=np.inf,
                 inequality_tol: float=1e-5,
                 vectorized: bool=False) -> None:
        '''Creates a new set of state and action constraints.
        
        :param simulator: the RDDL simulator to evaluate potential non-fluent
        expressions nested in constraints
        :param max_bound: initial value for maximum possible bounds
        :param inequality_tol: tolerance for inequality > and < comparisons
        :param vectorized: whether bounds are represented as pairs of numpy arrays
        corresponding to lifted fluent names (if True), or as pairs of scalars for 
        grounded fluent names (if False)
        '''        
        self.sim = simulator
        self.rddl = rddl = simulator.rddl
        self.BigM = max_bound
        self.epsilon = inequality_tol
        self.vectorized = vectorized
        
        # initialize the bounds to [-inf, inf]
        self._bounds = {}
        for (var, vtype) in rddl.variable_types.items():
            if vtype in {'state-fluent', 'observ-fluent', 'action-fluent'}:
                ptypes = rddl.variable_params[var]
                shape = rddl.object_counts(ptypes)
                if shape:
                    self._bounds[var] = [
                        np.full(shape=shape, fill_value=-self.BigM),
                        np.full(shape=shape, fill_value=+self.BigM)
                    ]
                else:
                    self._bounds[var] = [-self.BigM, +self.BigM]

        # actions and states bounds extraction for gym's action and state spaces
        # currently supports only linear inequality constraints
        self._is_box_precond = []
        for (index, precond) in enumerate(rddl.preconditions):
            tag = f'Action precondition {index + 1}'
            is_box = self._parse_bounds(tag, precond, [], rddl.action_fluents)
            self._is_box_precond.append(is_box)
                    
        self._is_box_invariant = []
        for (index, invariant) in enumerate(rddl.invariants):
            tag = f'State invariant {index + 1}'
            is_box = self._parse_bounds(tag, invariant, [], rddl.state_fluents)
            self._is_box_invariant.append(is_box)

        for (name, bounds) in self._bounds.items():
            RDDLSimulator._check_bounds(*bounds, f'Variable <{name}>', bounds)
        
        # ground the bounds if not vectorized
        if self.vectorized:
            self._bounds = {name: tuple(value) 
                            for (name, value) in self._bounds.items()}
        else:
            new_bounds = {}
            for (var, (lower, upper)) in self._bounds.items():
                lower = np.ravel(lower, order='C')
                upper = np.ravel(upper, order='C')
                gvars = rddl.variable_groundings[var]
                assert (len(gvars) == len(lower) == len(upper))
                new_bounds.update(zip(gvars, zip(lower, upper)))      
            self._bounds = new_bounds        
        
        # log bounds to file
        if simulator.logger is not None:
            bounds_info = '\n\t'.join(f'{k}: {v}' 
                                      for (k, v) in self._bounds.items())
            simulator.logger.log(f'[info] computed simulation bounds:\n' 
                                 f'\t{bounds_info}\n')
        
    def _parse_bounds(self, tag, expr, objects, search_vars):
        etype, op = expr.etype
        
        # for aggregation can only parse forall, since it is equivalent to a 
        # set of independent constraints determined by the function aggregated
        if etype == 'aggregation' and op == 'forall':
            * pvars, arg = expr.args
            new_objects = objects + [pvar for (_, pvar) in pvars]
            return self._parse_bounds(tag, arg, new_objects, search_vars)
        
        # for logical expression can only parse conjunction
        # same rationale as forall discussed above
        elif etype == 'boolean' and op == '^':
            success = True
            for arg in expr.args:
                success_arg = self._parse_bounds(tag, arg, objects, search_vars)
                success = success and success_arg
            return success
        
        # relational operation in constraint at the top level, i.e. constraint
        # LHS <= RHS for example
        elif etype == 'relational':
            rddl = self.rddl
            var, lim, loc, slices = self._parse_bounds_relational(
                tag, expr, objects, search_vars)
            success = var is not None and loc is not None
            if success: 
                op = np.minimum if loc == 1 else np.maximum
                if slices:
                    self._bounds[var][loc][slices] = op(
                        self._bounds[var][loc][slices], lim)
                else:
                    self._bounds[var][loc] = op(self._bounds[var][loc], lim)
            return success
        
        # not possible to parse as a box constraint
        else:
            return False
               
    def _parse_bounds_relational(self, tag, expr, objects, search_vars):
        left, right = expr.args    
        _, op = expr.etype
        left_pvar = left.is_pvariable_expression() and left.args[0] in search_vars
        right_pvar = right.is_pvariable_expression() and right.args[0] in search_vars
        
        # both LHS and RHS are pvariable expressions, or relational operator 
        # cannot be simplified further
        if (left_pvar and right_pvar) or op not in {'<=', '<', '>=', '>'}:
            raise_warning(
                f'{tag} does not have a structure of '
                f'<action or state fluent> <op> <rhs>, where ' 
                f'<op> is one of {{<=, <, >=, >}} and '
                f'<rhs> is a deterministic function of non-fluents only, '
                f'and will be ignored.\n' + 
                print_stack_trace(expr), 'red')
            return None, 0.0, None, []
        
        # neither side is a pvariable, nothing to do
        elif not left_pvar and not right_pvar:
            return None, 0.0, None, []
        
        # only one of LHS and RHS are pvariable expressions, other side constant
        else:
            
            # which one is the constant expression?
            if left_pvar:
                var, args = left.args
                const_expr = right
            else:
                var, args = right.args
                const_expr = left
            if args is None:
                args = []
                
            if not self.rddl.is_non_fluent_expression(const_expr):
                raise_warning(
                    f'{tag} contains a fluent expression '
                    f'(a nondeterministic operation or fluent variable) '
                    f'on both sides of an (in)equality, and will be ignored.\n' + 
                    print_stack_trace(const_expr), 'red')
                return None, 0.0, None, []
            
            # use the simulator to evaluate the constant side of the comparison
            # this is likened to a constant-folding operation
            const = self.sim._sample(const_expr, self.sim.subs)
            eps, loc = self._get_op_code(op, left_pvar)
            lim = const + eps
            
            # finally, since the pvariable expression may contain literals,
            # construct a slice that would assign the above constant value to it
            rddl = self.rddl
            slices = []
            for arg in args:
                if rddl.is_literal(arg):
                    arg = rddl.strip_literal(arg)
                    slices.append(rddl.object_to_index[arg])
                else:
                    slices.append(slice(None))
            slices = tuple(slices)
            
            return var, lim, loc, slices
            
    def _get_op_code(self, op, is_right):
        eps = 0.0
        if is_right:
            if op in ['<=', '<']:
                loc = 1
                if op == '<':
                    eps = -self.epsilon
            elif op in ['>=', '>']:
                loc = 0
                if op == '>':
                    eps = self.epsilon
        else:
            if op in ['<=', '<']:
                loc = 0
                if op == '<':
                    eps = self.epsilon
            elif op in ['>=', '>']:
                loc = 1
                if op == '>':
                    eps = -self.epsilon
        return eps, loc

    @property
    def bounds(self):
        return self._bounds

    @bounds.setter
    def bounds(self, value):
        self._bounds = value
    
    @property
    def is_box_preconditions(self):
        return self._is_box_precond
    
    @property
    def is_box_invariants(self):
        return self._is_box_invariant
    
