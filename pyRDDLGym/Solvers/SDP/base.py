"""Defines the base class for a symbolic solver."""
import abc
from typing import Dict, Set

import sympy as sp
from xaddpy import XADD
from xaddpy.xadd.xadd import DeltaFunctionSubstitution

from pyRDDLGym.Solvers.SDP.helper import SingleAction, MDP
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

    @abc.abstractmethod
    def bellman_backup(self, dd: int) -> int:
        """Performs the Bellman backup."""

    @abc.abstractmethod
    def regress_bool_actions(self, *args, **kwargs):
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

    def regress_cvars(self, q: int, cpf: int, v: sp.Symbol) -> int:
        """Regress a continuous variable from the value function `q`."""

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

    def regress_bvars(self, q: int, cpf: int, v: sp.Symbol) -> int:
        """Regress a boolean variable from the value function `q`."""
        dec_id = self.context._expr_to_id[self.mdp.model.ns[str(v)]]

        # Convert leaf nodes to float values.
        cpf = self.context.unary_op(cpf, 'float')

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

    def export(self, dd: int, fname: str):
        """Exports the decision diagram to a text file."""
        self.mdp.context.export_xadd(fname)
