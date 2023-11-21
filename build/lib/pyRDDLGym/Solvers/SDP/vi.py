"""Defines the Value Iteration solver."""
from typing import Dict, Tuple

import sympy as sp
from xaddpy.xadd.xadd import XADD, XADDLeafMinOrMax

from pyRDDLGym.Solvers.SDP.base import SymbolicSolver
from pyRDDLGym.Solvers.SDP.helper import Action, MDP


class ValueIteration(SymbolicSolver):

    def solve(self) -> int:
        """See the base class."""
        # Reset the counter.
        self.n_curr_iter = 0

        # Initialize the value function.
        value_dd = self.context.ZERO

        # Perform VI for the set number of iterations.
        while self.n_curr_iter < self.n_max_iter:
            self.n_curr_iter += 1

            # Cache the current value function.
            _prev_dd = value_dd

            # Perform the Bellman backup.
            value_dd = self.bellman_backup(value_dd)

            # Check for convergence.
            if self.enable_early_convergence and value_dd == _prev_dd:
                print(f'\nVI: Converged to solution early, at iteration {self.n_curr_iter}')
                break

        self.flush_caches()
        return value_dd

    def bellman_backup(self, dd: int) -> int:
        """Performs the VI Bellman backup.
        
        Args:
            dd: The current value function XADD ID to which Bellman back is applied.

        Returns:
            The new value function XADD ID.
        """
        max_dd = None

        # Iterate over each action.
        for a_name, action in self.mdp.actions.items():
            regr = self.regress(dd, action, action.atype == 'real')

            if max_dd is None:
                max_dd = regr
            else:
                max_dd = self.context.apply(max_dd, regr, op='max')
                max_dd = self.mdp.standardize(max_dd)

            self.flush_caches()
        return max_dd

    def regress(self, dd: int, a: Action, regress_cont: bool = True) -> int:
        # Prime the value function.
        q = self.context.substitute(dd, self.mdp.prime_subs)

        # Discount.
        if self.mdp.discount < 1.0:
            q = self.context.scalar_op(q, self.mdp.discount, op='prod')

        # Add the reward if it contains primed vars that need to be regressed.
        i_and_ns_vars_in_reward = self.filter_i_and_ns_vars(
            self.context.collect_vars(a.reward)
        )
        if len(i_and_ns_vars_in_reward) > 0:
            q = self.context.apply(q, a.reward, op='add')

        # Get variables to eliminate.
        vars_to_regress = self.filter_i_and_ns_vars(
            self.context.collect_vars(q), allow_bool=True, allow_cont=True)
        var_order = self.sort_var_set(vars_to_regress)

        # Regress each variable in the topological order.
        for v in var_order:
            if v in self.mdp.cont_ns_vars or v in self.mdp.cont_i_vars:
                q = self.regress_cvars(q, a, v)
            elif v in self.mdp.bool_ns_vars or v in self.mdp.bool_i_vars:
                q = self.regress_bvars(q, a, v)

        # Add the reward.
        if len(i_and_ns_vars_in_reward) == 0:
            q = self.context.apply(q, a.reward, op='add')

        # Continuous action parameter.
        if regress_cont:
            q = self.regress_action(q, a)

        # Standardize the node.
        q = self.mdp.standardize(q)
        return q

    def regress_action(self, dd: int, a: Action) -> int:
        """Regresses a continuous action parameter."""
        # Max out the action.
        q_vars = self.context.collect_vars(dd)
        bounds = self.mdp.get_bounds(a)

        # Can skip if the action variable is not included in the value function.
        if a.symbol not in q_vars:
            return dd

        # Max out the action variable.
        dd = self.max_out_var(dd, a.symbol, bound_dict={a.symbol: bounds})
        # TODO: need to flush caches?
        return dd

    def max_out_var(
            self, dd: int, v: sp.Symbol, bound_dict: Dict[sp.Symbol, Tuple[float, float]]
    ) -> int:
        """Maxes out a continuous variable from an XADD node."""
        max_op = XADDLeafMinOrMax(var=v, is_max=True, bound_dict=bound_dict, context=self.context)
        _ = self.context.reduce_process_xadd_leaf(dd, max_op, [], [])
        return max_op._running_result
