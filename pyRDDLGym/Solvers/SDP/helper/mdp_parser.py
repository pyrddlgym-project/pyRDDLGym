"""Implements a parser that converts an XADD RDDL model into an MDP."""

from typing import Dict, List, Set, Tuple

import sympy as sp
from xaddpy import XADD

from pyRDDLGym.Solvers.SDP.helper.action import BAction, CAction
from pyRDDLGym.Solvers.SDP.helper.mdp import MDP
from pyRDDLGym.Solvers.SDP.helper.xadd_utils import BoundAnalysis
from pyRDDLGym.XADD.RDDLModelXADD import RDDLModelWXADD


class Parser:

    def parse(
            self, model: RDDLModelWXADD, discount: float = 1.0, is_linear: bool = False,
    ) -> MDP:
        """Parses the RDDL model into an MDP object."""
        mdp = MDP(model, is_linear, discount)
        # Configure the bounds of continuous states.
        cont_s_vars = set()
        for s in model.states:
            if model.gvar_to_type[s] != 'real':
                continue
            cont_s_vars.add(model.ns[s])
        cont_state_bounds = self.configure_bounds(mdp, model.invariants, cont_s_vars)
        mdp.cont_state_bounds = cont_state_bounds

        # Go throuah all actions and get corresponding CPFs and rewards.
        actions = model.actions
        action_type = {}
        for name, val in actions.items():
            atype = 'bool' if isinstance(val, bool) else 'real'
            a_symbol = model.ns.get(name)
            if a_symbol is None:
                print(f'Warning: action {name} not found in RDDLModelWXADD.actions')
                a_symbol, a_node_id = model.add_sympy_var(name, atype)
            action_type[name] = atype
            a_cls = BAction if atype == 'bool' else CAction
            action = a_cls(name, a_symbol, model)

            # Add the action to the MDP.
            mdp.add_action(action)

        # Configure the bounds of continuous actions.
        if len(mdp.cont_a_vars) > 0:
            cont_action_bounds = self.configure_bounds(mdp, model.preconditions, mdp.cont_a_vars)
            mdp.cont_action_bounds = cont_action_bounds

        # Update the next state and interm variable sets.
        mdp.update_i_and_ns_var_sets()

        # Go through the actions and update the corresponding CPF XADDs.
        mdp.update()
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
