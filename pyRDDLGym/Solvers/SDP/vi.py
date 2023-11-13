"""Defines the Value Iteration solver."""
from typing import Dict, Tuple

import sympy as sp
from xaddpy.xadd.xadd import XADD, XADDLeafMinOrMax

from pyRDDLGym.Solvers.SDP.base import SymbolicSolver
from pyRDDLGym.Solvers.SDP.helper import Action, MDP


class ValueIteration(SymbolicSolver):

    def solve(self):
        # Reset the counter.
        self._n_curr_iter = 0

        # Initialize the value function.
        value_dd = self.context.ZERO

        # Perform VI for the set number of iterations.
        while self._n_curr_iter < self._n_max_iter:
            # Cache
            _prev_dd = value_dd

            # Perform the Bellman backup.
            value_dd, 

    def bellman_backup(self, dd: int) -> int:
        """Performs the VI Bellman backup.
        
        Args:
            dd: The current value function XADD ID to which Bellman back is applied.

        Returns:
            The new value function XADD ID.
        """
        max_dd = None

        # Iterate over each action.
        for a in self.mdp.actions:
            regr = self.regress(dd, a)

    def regress(self, dd: int, a: Action, regress_cont: bool = True) -> int:
        # Prime the value function.
        q = self.context.substitute(dd, self.mdp.prime_subs)

        # Discount.
        if self.mdp.discount < 1.0:
            q = self.context.scalar_op(q, self.mdp.discount, op='*')

        # Add the reward if it contains primed vars that need to be regressed.
        i_and_ns_vars_in_reward = self.filter_i_and_ns_vars(
            self.context.collect_vars(a.reward)
        )
        if len(i_and_ns_vars_in_reward) > 0:
            q = self.context.apply(q, a.reward, op='+')

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
            q = self.context.apply(q, a.reward, op='+')

        # Continuous action parameter.
        if regress_cont:
            q = self.regress_action(q, a)

        # Standardize the node.
        q = self.mdp.standardize_node(q)
        return q

    def regress_action(self, dd: int, a: Action) -> int:
        """Regresses a continuous action parameter."""
        # No action parameters to maximize over.
        if len(a.params) == 0:
            return dd

        # Max out the action.
        q_vars = self.context.collect_vars(dd)
        bound_dict = self.mdp.get_bounds(a)

        # Can skip if the action variable is not included in the value function.
        if a.symbol not in q_vars:
            return dd

        # Max out the action variable.
        q = self.max_out_var(q, a.symbol, bound_dict)
        # TODO: need to flush caches?
        return q

    def max_out_var(
            self, dd: int, v: sp.Symbol, bound_dict: Dict[sp.Symbol, Tuple[float, float]]
    ) -> int:
        """Maxes out a continuous variable from an XADD node."""
        max_op = XADDLeafMinOrMax(var=v, is_max=True, bound_dict=bound_dict, context=self.context)
        _ = self.context.reduce_process_xadd_leaf(dd, max_op, [], [])
        return max_op._running_result
