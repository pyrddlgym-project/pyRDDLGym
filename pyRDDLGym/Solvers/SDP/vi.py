"""Defines the Value Iteration solver."""
from typing import Dict, List, Optional, Set, Tuple

import sympy as sp
from xaddpy.xadd.xadd import XADDLeafMinOrMax

from pyRDDLGym.Solvers.SDP.base import SymbolicSolver
from pyRDDLGym.Solvers.SDP.helper import CAction


class ValueIteration(SymbolicSolver):
    """Value Iteration solver."""

    def __init__(self, annotate: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.annotate = annotate

    def bellman_backup(self, dd: int) -> int:
        """Performs the VI Bellman backup.
        
        Args:
            dd: The current value function XADD ID to which Bellman back is applied.

        Returns:
            The new value function XADD ID.
        """
        max_dd = None

        # Iterate over each Boolean action.
        for a_name, action in self.mdp.bool_actions.items():
            regr = self.regress(dd, action.reward, action.cpfs)

            if max_dd is None:
                max_dd = regr
            else:
                max_dd = self.context.apply(max_dd, regr, op='max')
                max_dd = self.mdp.standardize(max_dd)

            self.flush_caches()

        # Handle the case when there are no boolean actions.
        if max_dd is None:
            max_dd = self.regress(dd, self.mdp.reward, self.mdp.cpfs)
        # Max out continuous actions.
        max_dd = self.regress_continuous_actions(max_dd)
        return max_dd

    def regress(self, value_dd: int, reward: int, cpfs: Dict[sp.Symbol, int]) -> int:
        """Regresses the value function.
        
        Args:
            value_dd: The current value function XADD ID.
            reward: The reward XADD ID.
            cpfs: The dictionary mapping variables to their corresponding CPF XADD IDs.
        Returns:
            The ID of the regressed value function.
        """
        # Prime the value function.
        q = self.context.substitute(value_dd, self.mdp.prime_subs)

        # Discount.
        if self.mdp.discount < 1.0:
            q = self.context.scalar_op(q, self.mdp.discount, op='prod')

        # Add the reward if it contains primed vars that need to be regressed.
        i_and_ns_vars_in_reward = self.filter_i_and_ns_vars(
            self.context.collect_vars(reward)
        )
        if len(i_and_ns_vars_in_reward) > 0:
            q = self.context.apply(q, reward, op='add')

        # Get variables to eliminate.
        all_vars = self.context._bool_var_set.union(self.context._cont_var_set)
        vars_to_regress = self.filter_i_and_ns_vars(
            all_vars, allow_bool=True, allow_cont=True)
        var_order = self.sort_var_set(vars_to_regress)

        # Regress each variable.
        for v in var_order:
            # If not in the current value function, skip.
            var_set = self.context.collect_vars(q)
            if v not in var_set:
                continue

            # Otherwise, regress.
            if v in self.mdp.cont_ns_vars or v in self.mdp.cont_i_vars:
                q = self.regress_cvars(q, cpfs[v], v)
            elif v in self.mdp.bool_ns_vars or v in self.mdp.bool_i_vars:
                q = self.regress_bvars(q, cpfs[v], v)

        # Add the reward.
        if len(i_and_ns_vars_in_reward) == 0:
            q = self.context.apply(q, reward, op='add')

        # Standardize the node.
        q = self.mdp.standardize(q)
        return q

    # TODO: Implement heuristic ordering.
    def get_action_variable_ordering(self, actions: Set[CAction]) -> List[CAction]:
        return list(actions)

    def regress_continuous_actions(self, dd: Optional[int] = None) -> int:
        """Regresses a continuous action parameter."""
        # Get the continuous action variables.
        actions = set(self.mdp.cont_actions.values())

        # Determine the elimination ordering.
        ordered = self.get_action_variable_ordering(actions)

        # Get the variables in the value DD.
        dd_vars = self.context.collect_vars(dd)

        # SVE: max out each action sequentially.
        for action in ordered:
            # Get the lower and upper bounds of the action variable.
            bounds = self.mdp.get_bounds(action.symbol)

            # Can skip if the action variable is not included in the value function.
            if action.symbol not in dd_vars:
                continue

            # Otherwise, max out the action variable.
            dd = self.max_out_var(dd, action.symbol, bound_dict={action.symbol: bounds})
        # TODO: need to flush caches?
        return dd

    def max_out_var(
            self, dd: int, v: sp.Symbol, bound_dict: Dict[sp.Symbol, Tuple[float, float]]
    ) -> int:
        """Maxes out a continuous variable from an XADD node."""
        max_op = XADDLeafMinOrMax(
            var=v,
            is_max=True,
            bound_dict=bound_dict,
            context=self.context,
            annotate=self.annotate,
        )
        _ = self.context.reduce_process_xadd_leaf(dd, max_op, [], [])
        return max_op._running_result
