"""Defines the base class for a symbolic solver."""
import abc
from typing import Set

import sympy as sp
from xaddpy import XADD
from xaddpy.xadd.xadd import DeltaFunctionSubstitution

from pyRDDLGym.Solvers.SDP.helper import Action, MDP
from pyRDDLGym.XADD.RDDLLevelAnalysisXADD import RDDLLevelAnalysisWXADD


class SymbolicSolver:
    """Base class for a symbolic solver."""

    def __init__(
            self,
            mdp: MDP,
            max_iter: int = 100,
            enable_early_convergence: bool = False,
    ):
        self._mdp = mdp
        self.n_curr_iter = 0
        self.n_max_iter = max_iter
        self.enable_early_convergence = enable_early_convergence

        self.level_analyzer = RDDLLevelAnalysisWXADD(self.mdp.model)
        self.levels = self.level_analyzer.compute_levels()
        self.var_to_level = {}
        for l, var_set in self.levels.items():
            for v in var_set:
                self.var_to_level[v] = l

    @abc.abstractmethod
    def solve(self) -> int:
        """Returns the solution XADD ID of the value function."""

    @abc.abstractmethod
    def bellman_backup(self) -> int:
        """Performs the Bellman backup."""

    @abc.abstractmethod
    def regress(self, *args, **kwargs):
        """Regresses the value function."""

    @property
    def mdp(self) -> MDP:
        """Returns the MDP."""
        return self._mdp

    @property
    def context(self) -> XADD:
        """Returns the XADD context."""
        return self._mdp.context

    def filter_i_and_ns_vars(
            self, var_set: set, allow_bool: bool = True, allow_cont: bool = True
    ) -> Set[str]:
        """Returns the set of interm and next state variables in the given var set."""
        filtered_vars = set()
        for v in var_set:
            if allow_cont and (v in self.mdp.cont_ns_vars or v in self.mdp.cont_i_vars):
                filtered_vars.add(v)
            elif allow_bool and (v in self.mdp.bool_ns_vars or v in self.mdp.bool_i_vars):
                filtered_vars.add(v)
        return filtered_vars

    def sort_var_set(self, var_set):
        """Sorts the given variable set by level."""
        return sorted(
            var_set,
            key=lambda v: self.var_to_level.get(v, float('inf')),
            reverse=True,
        )

    def regress_cvars(self, q: int, a: Action, v: sp.Symbol) -> int:
        """Regress a continuous variable from the value function `q`."""
        # Get the CPF for the variable.
        cpf = a.get_cpf(v)

        # Check the regression cache.
        key = (str(v), cpf, q)
        res = self.mdp.cont_regr_cache.get(key)
        if res is not None:
            return res

        # Perform regression via Delta function substitution.
        leaf_op = DeltaFunctionSubstitution(v, q, self.context, is_linear=self.mdp.is_linear)
        q = self.context.reduce_process_xadd_leaf(cpf, leaf_op, [], [])

        # Simplify the resulting XADD if possible.
        if self.mdp.is_linear:
            q = self.context.reduce_lp(q)

        # Cache and return the result.
        self.mdp.cont_regr_cache[key] = q
        return q

    def regress_bvars(self, q: int, a: Action, v: sp.Symbol) -> int:
        """Regress a boolean variable from the value function `q`."""
        # Get the CPF for the variable.
        cpf = a.get_cpf(v)
        dec_id = self.context._expr_to_id[self.mdp.model.ns[str(v)]]

        # Convert nodes to 1 and 0.
        cpf = self.context.unary_op(cpf, 'int')

        # Marginalize out the boolean variable.
        q = self.context.apply(q, cpf, op='prod')
        restrict_high = self.context.op_out(q, dec_id, op='restrict_high')
        restrict_low = self.context.op_out(q, dec_id, op='restrict_low')
        q = self.context.apply(restrict_high, restrict_low, op='add')
        return q

    def flush_caches(self):
        """Flush cache objects."""

    def print(self, dd: int):
        """Prints the value function."""
        self.mdp.model.print(dd)
