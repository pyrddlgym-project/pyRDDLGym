"""Implements a parser that converts an XADD RDDL model into an MDP."""

import itertools
from typing import Dict, List, Set, Tuple, Union

import sympy as sp
from xaddpy import XADD

from pyRDDLGym.Solvers.SDP.helper.action import SingleAction, CAction, BActions
from pyRDDLGym.Solvers.SDP.helper.mdp import MDP
from pyRDDLGym.Solvers.SDP.helper.xadd_utils import BoundAnalysis
from pyRDDLGym.XADD.RDDLModelXADD import RDDLModelWXADD


def _truncated_powerset(iterable, max_size: int, include_noop: bool):
    """Returns the powerset of an iterable."""
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(1 - int(include_noop), max_size + 1)
    )


class MDPParser:

    def parse(
            self,
            model: RDDLModelWXADD,
            discount: float = 1.0,
            concurrency: Union[str, int] = 1,
            is_linear: bool = False,
            include_noop: bool = True,
            is_vi: bool = True,
    ) -> MDP:
        """Parses the RDDL model into an MDP object.

        Args:
            model: The RDDL model compiled in XADD.
            discount: The discount factor.
            concurrency: The number of concurrent boolean actions.
            is_linear: Whether the MDP is linear or not.
            include_noop: Whether to include the no-op action or not. Defaults to True.
            is_vi: Whether solving Value Iteration (VI).

        Returns:
            The MDP object.
        """
        if concurrency == 'pos-inf':
            concurrency = int(1e9)
        mdp = MDP(model, is_linear, discount, concurrency)

        # Configure the bounds of continuous states.
        cont_s_vars = set()
        for s in model.states:
            if model.gvar_to_type[s] != 'real':
                continue
            cont_s_vars.add(model.ns[s])
        cont_state_bounds = self.configure_bounds(mdp, model.invariants, cont_s_vars)
        mdp.cont_state_bounds = cont_state_bounds

        # Go throuah all actions and get corresponding CPFs and rewards.
        # For Boolean actions, we have to handle concurrency using `ConcurrentBAction`
        # class.
        actions = model.actions
        bool_actions, cont_actions = [], []
        for name, val in actions.items():
            atype = 'bool' if isinstance(val, bool) else 'real'
            a_symbol = model.ns.get(name)
            if a_symbol is None:
                print(f'Warning: action {name} not found in RDDLModelWXADD.actions')
                a_symbol, a_node_id = model.add_sympy_var(name, atype)
            if atype == 'bool':
                bool_actions.append(SingleAction(name, a_symbol, model, 'bool'))
            else:
                cont_actions.append(CAction(name, a_symbol, model))            

        # Add concurrent actions for Boolean actions.
        if is_vi:
            # Need to consider all combinations of boolean actions.
            # Note: there's always an implicit no-op action with which
            # none of the boolean actions are set to True.
            total_bool_actions = tuple(
                _truncated_powerset(
                    bool_actions,
                    mdp.max_allowed_actions,
                    include_noop=include_noop,
            ))
            for actions in total_bool_actions:
                names = tuple(a.name for a in actions)
                symbols = tuple(a.symbol for a in actions)
                action = BActions(names, symbols, model)
                mdp.add_action(action)
        else:
            for action in bool_actions:
                name, symbol = action.name, action.symbol
                mdp.add_action(
                    BActions((name,), (symbol,), model)
                )

        # Add continuous actions.
        for action in cont_actions:
            mdp.add_action(action)

        # Configure the bounds of continuous actions.
        if len(mdp.cont_a_vars) > 0:
            cont_action_bounds = self.configure_bounds(mdp, model.preconditions, mdp.cont_a_vars)
            mdp.cont_action_bounds = cont_action_bounds

        # Update the state variable sets.
        mdp.update_state_var_sets()

        # Update the next state and interm variable sets.
        mdp.update_i_and_ns_var_sets()

        # Go through the actions and update the corresponding CPF XADDs.
        mdp.update(is_vi=is_vi)
        return mdp

    def configure_bounds(
            self, mdp: MDP, conditions: List[int], var_set: Set[sp.Symbol],
    ) -> Dict[CAction, Tuple[float, float]]:
        """Configures the bounds over continuous variables."""
        context = mdp.context

        # Bounds dictionaries to be updated.
        lb_dict: Dict[sp.Symbol, List[sp.Basic]] = {}
        ub_dict: Dict[sp.Symbol, List[sp.Basic]] = {}

        # Iterate over conditions (state invariants or action preconditions).
        for p in conditions:
            # Instantiate the leaf operation object.
            leaf_op = BoundAnalysis(context=mdp.context, var_set=var_set)

            # Perform recursion.
            context.reduce_process_xadd_leaf(p, leaf_op, [], [])

            # Retrieve bounds.
            lb_d = leaf_op.lb_dict
            ub_d = leaf_op.ub_dict
            for v in lb_d:
                v_max_lb = lb_dict.setdefault(v, lb_d[v])
                v_max_lb = max(v_max_lb, lb_d[v])   # Get the largest lower bound.
                lb_dict[v] = v_max_lb
            for v in ub_d:
                v_min_ub = ub_dict.setdefault(v, ub_d[v])
                v_min_ub = min(v_min_ub, ub_d[v])   # Get the smallest upper bound.
                ub_dict[v] = v_min_ub

        # Convert the bounds dictionaries into a dictionary of tuples.
        bounds = {}
        for v in var_set:
            lb = lb_dict.get(v, -float('inf'))
            ub = ub_dict.get(v, float('inf'))
            lb, ub = float(lb), float(ub)
            bounds[v] = (lb, ub)
        context.update_bounds(bounds)
        return bounds
