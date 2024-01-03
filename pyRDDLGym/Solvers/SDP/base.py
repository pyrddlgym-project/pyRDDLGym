"""Defines the base class for a symbolic solver."""
import abc
import time
import psutil
from typing import Any, Dict, List, Optional, Set

import symengine.lib.symengine_wrapper as core
from xaddpy import XADD
from xaddpy.utils.symengine import BooleanVar
from xaddpy.xadd.xadd import DeltaFunctionSubstitution, VAR_TYPE

from pyRDDLGym.Solvers.SDP.helper import MDP
from pyRDDLGym.XADD.RDDLLevelAnalysisXADD import RDDLLevelAnalysisWXADD
from pyRDDLGym.XADD.helper import get_bernoulli_node_id


FLUSH_PERCENT_MINIMUM = 0.3


class SymbolicSolver:
    """Base class for a symbolic solver."""

    def __init__(
            self,
            mdp: MDP,
            max_iter: int = 100,
            enable_early_convergence: bool = False,
            perform_reduce_lp: bool = True,
    ):
        self._mdp = mdp
        self._prev_dd = None
        self._max_dd = None
        self._value_dd = self.context.ZERO
        self.n_curr_iter = 0
        self.n_max_iter = max_iter
        self.enable_early_convergence = enable_early_convergence
        self.perform_reduce_lp = perform_reduce_lp

        self.level_analyzer = RDDLLevelAnalysisWXADD(self.mdp.model)
        self.levels = self.level_analyzer.compute_levels()
        self.var_to_level = {}
        for l, var_set in self.levels.items():
            for v in var_set:
                self.var_to_level[v] = l

        self.special_nodes = []

    @property
    def value_dd(self) -> int:
        """Returns the value function XADD ID at current iteration."""
        return self._value_dd

    @value_dd.setter
    def value_dd(self, value_dd: int):
        """Sets the value function XADD ID at current iteration."""
        self._value_dd = value_dd

    def solve(self) -> Dict[str, Any]:
        """See the base class."""
        # Result dict.
        res = dict(
            value_dd=[],    # Value function ID per iteration.
            time=[],        # Time per iteration.
        )

        # Reset the counter.
        self.n_curr_iter = 0

        # Initialize the value function.
        self.value_dd = self.context.ZERO

        # Update the reduce_lp setting.
        self.context.perform_reduce_lp = self.perform_reduce_lp

        # Perform VI for the set number of iterations.
        while self.n_curr_iter < self.n_max_iter:
            self.n_curr_iter += 1

            # Cache the current value function.
            self._prev_dd = self.value_dd

            # Perform the Bellman backup.
            # Time the Bellman backup.
            stime = time.time()
            self.value_dd = self.bellman_backup(self.value_dd)
            etime = time.time()

            # Record the results.
            self.special_nodes.append(self.value_dd)
            res['value_dd'].append(self.value_dd)
            res['time'].append(etime - stime)
            # Print out the intermediate results.
            print(f'{self.__class__.__name__}: Iteration {self.n_curr_iter}, Time: {etime - stime}')

            # Check for convergence.
            if self.enable_early_convergence and self.value_dd == self._prev_dd:
                print(f'\nVI: Converged to solution early, at iteration {self.n_curr_iter}')
                break

            # Flush caches.
            self.flush_caches()

        self.flush_caches()
        return res

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

    def regress(self, value_dd: int, reward: int, cpfs: Dict[VAR_TYPE, int]) -> int:
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
        q = self.standardize(q)
        return q

    def regress_cvars(self, q: int, cpf: int, v: core.Symbol) -> int:
        """Regress a continuous variable from the value function `q`.
        
        After regressing the continuous variable via DeltaFunctionSubstitution,
        we check whether the resulting XADD contains Bernoulli variables.

        If so, we do the following to average out the Bernoulli variables:
            for each Bernoulli variable:
                Q <- p * Q subs(bern_var = 1) + (1 - p) * Q subs(bern_var = 0)
            where p is the Bernoulli parameter, which itself can be an XADD whose
            leaf values are the Bernoulli probabilities. `subs(bern_var = val)` 
            denotes the substitution operation that replaces all Bernoulli
            variables with val, which is either 0 or 1.

        Args:
            q: The value function XADD ID.
            cpf: The CPF XADD ID.
            v: The continuous variable to regress.
        Returns:
            The ID of the regressed value function.
        """

        # Check the regression cache.
        key = (str(v), cpf, q)
        res = self.mdp.cont_regr_cache.get(key)
        if res is not None:
            return res

        # Perform regression via Delta function substitution.
        leaf_op = DeltaFunctionSubstitution(v, q, self.context, is_linear=self.mdp.is_linear)
        q = self.context.reduce_process_xadd_leaf(cpf, leaf_op, [], [])
        q = self.standardize(q)

        # Handle Bernoulli variables if there is one.
        var_set = self.context.collect_vars(q)
        bern_var_set = {v for v in var_set if str(v).startswith('Bernoulli')}
        for v in bern_var_set:
            # Get the Bernoulli parameter.
            bern_p = get_bernoulli_node_id(v)
            bern_1_p = self.context.apply(self.context.ONE, bern_p, op='subtract')

            # Check if the Bernoulli variable is used as a decision.
            bern_as_dec = v in self.context._id_to_expr.values()

            # Substitute the Bernoulli variable with 1 and multiply by the Bernoulli parameter.
            q_v_true = self.context.substitute(
                q, {v: True if bern_as_dec else self.context.ONE})
            q_v_true = self.standardize(q_v_true)
            q_v_true = self.context.apply(q_v_true, bern_p, op='prod')
            q_v_true = self.standardize(q_v_true)

            # Substitute the Bernoulli variable with 0 and multiply by (1 - Bernoulli parameter).
            q_v_false = self.context.substitute(
                q, {v: False if bern_as_dec else self.context.ZERO})
            q_v_false = self.standardize(q_v_false)
            q_v_false = self.context.apply(q_v_false, bern_1_p, op='prod')
            q_v_false = self.standardize(q_v_false)

            # Marginalize out the Bernoully variable.
            q = self.context.apply(q_v_true, q_v_false, op='add')

            # Standardize the node.
            q = self.standardize(q)

        # Cache and return the result.
        self.mdp.cont_regr_cache[key] = q
        return q

    def regress_bvars(self, q: int, cpf: int, v: BooleanVar) -> int:
        """Regress a boolean variable from the value function `q`."""
        v_name = self.mdp.model._sym_var_name_to_var_name[str(v)]
        dec_id = self.context._expr_to_id[self.mdp.model.ns[v_name]]

        # Convert leaf nodes to float values.
        cpf = self.context.unary_op(cpf, 'float')

        # Marginalize out the boolean variable.
        q = self.context.apply(q, cpf, op='prod')
        restrict_high = self.context.op_out(q, dec_id, op='restrict_high')
        restrict_low = self.context.op_out(q, dec_id, op='restrict_low')
        q = self.context.apply(restrict_high, restrict_low, op='add')
        return q

    def flush_caches(self, special_nodes: Optional[List[int]] = None) -> None:
        """Flush cache objects."""
        # Check whether the memory usage is too high.
        available_memory = psutil.virtual_memory().available
        total_memory = psutil.virtual_memory().total
        available_memory_ratio = available_memory / total_memory
        if available_memory_ratio > FLUSH_PERCENT_MINIMUM:
            return  # No need to flush.

        # Print out current memory usage information.
        print(
            (
                f"Before flush: {len(self.context._id_to_node)} nodes in use,"
                f" free memory: {available_memory / 10e6} MB ="
                f" {available_memory_ratio * 100:.2f}% available memory."
            )
        )

        # Add special nodes.
        self.context.clear_special_nodes()
        if special_nodes is not None:
            for n in special_nodes:
                self.context.add_special_node(n)

        for n in self.special_nodes:
            self.context.add_special_node(n)

        for a_name, action in self.mdp.actions.items():
            self.context.add_special_node(action.reward)
            for var_name, cpf in action.cpfs.items():
                self.context.add_special_node(cpf)

        # Add value function XADDs.
        if self._prev_dd is not None:
            self.context.add_special_node(self._prev_dd)
        self.context.add_special_node(self.value_dd)

        # Add the max XADD if not None.
        if self._max_dd is not None:
            self.context.add_special_node(self._max_dd)

        # Flush the caches.
        self.mdp.cont_regr_cache.clear()
        self.context.flush_caches()

        # Print out memory usage after flushing.
        available_memory = psutil.virtual_memory().available
        total_memory = psutil.virtual_memory().total
        available_memory_ratio = available_memory / total_memory
        print(
            (
                f"After flush: {len(self.context._id_to_node)} nodes in use,"
                f" free memory: {available_memory / 10e6} MB ="
                f" {available_memory_ratio * 100:.2f}% available memory."
            )
        )

    def standardize(self, dd: int) -> int:
        dd = self.context.make_canonical(dd)
        if self.mdp.is_linear and self.perform_reduce_lp:
            dd = self.context.reduce_lp(dd)
        return dd

    def print(self, dd: int):
        """Prints the value function."""
        self.mdp.model.print(dd)

    def export(self, dd: int, fname: str):
        """Exports the decision diagram to a text file."""
        self.mdp.context.export_xadd(dd, fname)
