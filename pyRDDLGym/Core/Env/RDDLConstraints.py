import numpy as np
import warnings

from pyRDDLGym.Core.Simulator.RDDLSimulator import RDDLSimulator


class RDDLConstraints:

    def __init__(self, simulator: RDDLSimulator, max_bound: float=np.inf) -> None:
        self.sim = simulator
        self.rddl = simulator.rddl
        self.BigM = max_bound
        self.epsilon = 0.001
        
        self._bounds = {}
        for (var, vtype) in self.rddl.variable_types.items():
            if vtype in {'state-fluent', 'observ-fluent', 'action-fluent'}:
                ptypes = self.rddl.param_types[var]
                for gname in self.rddl.ground_names(var, ptypes):
                    self._bounds[gname] = [-self.BigM, +self.BigM]

        # actions and states bounds extraction for gym's action and state spaces
        # currently supports only linear inequality constraints
        for precond in self.rddl.preconditions:
            self._parse_bounds(precond, [], self.rddl.actions)
            
        for invariant in self.rddl.invariants:
            self._parse_bounds(invariant, [], self.rddl.states)

        for (name, bounds) in self._bounds.items():
            RDDLSimulator._check_bounds(*bounds, f'Variable <{name}>', bounds)
            
        # log bounds to file
        if simulator.logger is not None:
            bounds_info = '\n\t'.join(f'{k}: {v}' 
                                      for (k, v) in self._bounds.items())
            message = (f'computed simulation bounds:\n' 
                       f'\t{bounds_info}\n')
            simulator.logger.log(message)
        
    def _parse_bounds(self, expr, objects, search_vars):
        etype, op = expr.etype
        
        if etype == 'aggregation' and op == 'forall':
            * pvars, arg = expr.args
            new_objects = objects + [pvar for (_, pvar) in pvars]
            self._parse_bounds(arg, new_objects, search_vars)
            
        elif etype == 'boolean' and op == '^':
            for arg in expr.args:
                self._parse_bounds(arg, objects, search_vars)
                
        elif etype == 'relational':
            var, lim, loc, active = self._parse_bounds_relational(
                expr, objects, search_vars)
            if var is not None and loc is not None: 
                if objects:
                    ptypes = [ptype for (_, ptype) in objects]
                    variations = self.rddl.variations(ptypes)
                    lims = np.ravel(lim, order='C')
                    for (args, lim) in zip(variations, lims):
                        active_args = [args[i] for i in active]
                        key = self.rddl.ground_name(var, active_args)
                        self._update_bound(key, loc, lim)
                else:
                    self._update_bound(var, loc, lim)
    
    def _update_bound(self, key, loc, lim):
        if loc == 1:
            if self._bounds[key][loc] > lim:
                self._bounds[key][loc] = lim
        else:
            if self._bounds[key][loc] < lim:
                self._bounds[key][loc] = lim
        
    def _parse_bounds_relational(self, expr, objects, search_vars):
        left, right = expr.args    
        _, op = expr.etype
        is_left_pvar = left.is_pvariable_expression() and left.args[0] in search_vars
        is_right_pvar = right.is_pvariable_expression() and right.args[0] in search_vars
        
        if (is_left_pvar and is_right_pvar) or op not in ['<=', '<', '>=', '>']:
            warnings.warn(
                f'Constraint does not have a structure of '
                f'<action or state fluent> <op> <rhs>, where:' 
                    f'\n<op> is one of {{<=, <, >=, >}}'
                    f'\n<rhs> is a deterministic function of '
                    f'non-fluents or constants only.\n' + 
                RDDLSimulator._print_stack_trace(expr))
            return None, 0.0, None, []
            
        elif not is_left_pvar and not is_right_pvar:
            return None, 0.0, None, []
        
        else:
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
                    RDDLSimulator._print_stack_trace(const_expr))
                return None, 0.0, None, []
            
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