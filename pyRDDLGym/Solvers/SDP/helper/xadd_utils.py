"""Implements some utility functions for XADDs."""

from typing import Dict, List, Set, Union

import sympy as sp
import sympy.core.relational as relational
from sympy.logic import boolalg
from xaddpy.utils.util import get_bound
from xaddpy.xadd.xadd import XADD, XADDLeafOperation


class BoundAnalysis(XADDLeafOperation):
    """Leaf operation that configures bounds over set variables.
    
    Args:
        context: The XADD context.
        var_set: The set of variables to configure bounds for.
    """

    def __init__(
            self,
            context: XADD,
            var_set: Set[sp.Symbol],
    ):
        super().__init__(context)
        self.var_set = var_set
        self.lb_dict: Dict[sp.Symbol, Union[sp.Basic, int, float]] = {}
        self.ub_dict: Dict[sp.Symbol, Union[sp.Basic, int, float]] = {}

    def process_xadd_leaf(
            self,
            decisions: List[sp.Basic],
            decision_values: List[bool],
            leaf_val: sp.Basic,
    ) -> int:
        """Processes an XADD partition to configure bounds.

        Args:
            decisions: The list of decisions.
            decision_values: The list of decision values.
            leaf_val: The leaf value.
        Returns:
            None.
        """
        assert isinstance(leaf_val, boolalg.BooleanAtom) or isinstance(leaf_val, bool)
        # False leaf node represents an invalid partition.
        if not leaf_val:
            return self._context.get_leaf_node(leaf_val)

        # Iterate over the decisions and decision values.
        for dec_expr, is_true in zip(decisions, decision_values):
            # Iterate over the variables.
            for v in self.var_set:
                if v not in dec_expr.atoms():
                    continue
                assert dec_expr not in self._context._bool_var_set
                lhs, rhs, gt = dec_expr.lhs, dec_expr.rhs, isinstance(dec_expr, relational.GreaterThan)
                gt = (gt and is_true) or (not gt and not is_true)
                expr = lhs >= rhs if gt else lhs <= rhs

                # Get bounds over `v`.
                bound_expr, is_ub = get_bound(v, expr)
                if is_ub:
                    ub = self.ub_dict.setdefault(v, bound_expr)
                    # Get the tightest upper bound.
                    self.ub_dict[v] = min(ub, bound_expr)
                else:
                    lb = self.lb_dict.setdefault(v, bound_expr)
                    # Get the tighest lower bound.
                    self.lb_dict[v] = max(lb, bound_expr)

        return self._context.get_leaf_node(leaf_val)
