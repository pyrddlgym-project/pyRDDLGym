import numpy as np
import warnings

from pyRDDLGym.Core.ErrorHandling.RDDLException import print_stack_trace

from pyRDDLGym.Core.Simulator.RDDLSimulator import RDDLSimulator


class RDDLConstraints:
    '''Provides added functionality to understand a set of action-preconditions
    and state invariants in a RDDL file.'''
    
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
        self.rddl = simulator.rddl
        self.BigM = max_bound
        self.epsilon = inequality_tol
        self.vectorized = vectorized
        
        self._bounds = {}
        for (var, vtype) in self.rddl.variable_types.items():
            if vtype in {'state-fluent', 'observ-fluent', 'action-fluent'}:
                ptypes = self.rddl.param_types[var]
                if vectorized:
                    shape = self.rddl.object_counts(ptypes)
                    self._bounds[var] = [-self.BigM * np.ones(shape),
                                         +self.BigM * np.ones(shape)]
                else:
                    for gname in self.rddl.ground_names(var, ptypes):
                        self._bounds[gname] = [-self.BigM, +self.BigM]

        # actions and states bounds extraction for gym's action and state spaces
        # currently supports only linear inequality constraints
        self._is_box_precond = []
        for precond in self.rddl.preconditions:
            is_box = self._parse_bounds(precond, [], self.rddl.actions)
            self._is_box_precond.append(is_box)
                    
        self._is_box_invariant = []
        for invariant in self.rddl.invariants:
            is_box = self._parse_bounds(invariant, [], self.rddl.states)
            self._is_box_invariant.append(is_box)

        for (name, bounds) in self._bounds.items():
            RDDLSimulator._check_bounds(*bounds, f'Variable <{name}>', bounds)
            
        # log bounds to file
        if simulator.logger is not None:
            bounds_info = '\n\t'.join(
                f'{k}: {v}' for (k, v) in self._bounds.items())
            message = (f'[info] computed simulation bounds:\n' 
                       f'\t{bounds_info}\n')
            simulator.logger.log(message)
        
    def _parse_bounds(self, expr, objects, search_vars):
        etype, op = expr.etype
        
        # for aggregation can only parse forall, since it is equivalent to a 
        # set of independent constraints determined by the function aggregated
        if etype == 'aggregation' and op == 'forall':
            * pvars, arg = expr.args
            new_objects = objects + [pvar for (_, pvar) in pvars]
            return self._parse_bounds(arg, new_objects, search_vars)
        
        # for logical expression can only parse conjunction
        # same rationale as forall discussed above
        elif etype == 'boolean' and op == '^':
            success = True
            for arg in expr.args:
                success_arg = self._parse_bounds(arg, objects, search_vars)
                success = success and success_arg
            return success
        
        # relational operation in constraint at the top level, i.e. constraint
        # LHS <= RHS for example
        elif etype == 'relational':
            var, lim, loc, active = self._parse_bounds_relational(
                expr, objects, search_vars)
            success = var is not None and loc is not None
            if success: 
                if objects:
                    if self.vectorized:
                        op = np.minimum if loc == 1 else np.maximum
                        self._bounds[var][loc] = op(self._bounds[var][loc], lim)
                    else:
                        op = min if loc == 1 else max
                        ptypes = [ptype for (_, ptype) in objects]
                        variations = self.rddl.variations(ptypes)
                        lims = np.ravel(lim, order='C')
                        for (args, lim) in zip(variations, lims):
                            active_args = [args[i] for i in active]
                            key = self.rddl.ground_name(var, active_args)
                            self._bounds[key][loc] = op(self._bounds[key][loc], lim)
                else:
                    op = min if loc == 1 else max
                    self._bounds[var][loc] = op(self._bounds[var][loc], lim)
            return success
        
        # not possible to parse as a box constraint
        else:
            return False
               
    def _parse_bounds_relational(self, expr, objects, search_vars):
        left, right = expr.args    
        _, op = expr.etype
        is_left_pvar = left.is_pvariable_expression() and left.args[0] in search_vars
        is_right_pvar = right.is_pvariable_expression() and right.args[0] in search_vars
        
        # both LHS and RHS are pvariable expressions, or relational operator 
        # cannot be simplified further
        if (is_left_pvar and is_right_pvar) or op not in ['<=', '<', '>=', '>']:
            warnings.warn(
                f'Constraint does not have a structure of '
                f'<action or state fluent> <op> <rhs>, where:' 
                    f'\n<op> is one of {{<=, <, >=, >}}'
                    f'\n<rhs> is a deterministic function of '
                    f'non-fluents or constants only.\n' + 
                    print_stack_trace(expr))
            return None, 0.0, None, []
        
        # neither side is a pvariable, nothing to do
        elif not is_left_pvar and not is_right_pvar:
            return None, 0.0, None, []
        
        # only one of LHS and RHS are pvariable expressions, other side constant
        else:
            
            # which one is the constant expression?
            if is_left_pvar:
                var, args = left.args
                const_expr = right
            else:
                var, args = right.args
                const_expr = left
            if args is None:
                args = []
                
            if not self.rddl.is_non_fluent_expression(const_expr):
                warnings.warn(
                    f'Bound must be a deterministic function of '
                    f'non-fluents or constants only.\n' + 
                    print_stack_trace(const_expr))
                return None, 0.0, None, []
            
            # use the simulator to evaluate the constant side of the comparison
            # this is likened to a constant-folding operation
            const = self.sim._sample(const_expr, self.sim.subs)
            eps, loc = self._get_op_code(op, is_left_pvar)
            lim = const + eps
            
            arg_to_index = {obj[0]: i for (i, obj) in enumerate(objects)}
            active = [arg_to_index[arg] for arg in args if arg in arg_to_index]

            return var, lim, loc, active
            
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
    
