"""Implements a parser that converts an XADD RDDL model into an MDP."""

from typing import Dict, List, Tuple

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

        # Go throuah all actions and get corresponding CPFs and rewards.
        actions = model.actions
        action_dict = {}    # To be used to restrict a CPF with a Boolean action.
        action_type = {}
        for name, val in actions.items():
            atype = 'bool' if isinstance(val, bool) else 'real'
            a_symbol = model.ns.get(name)
            if a_symbol is None:
                print(f'Warning: action {name} not found in RDDLModelWXADD.actions')
                a_symbol, a_node_id = model.add_sympy_var(name, atype)
            if atype == 'bool':
                action_dict[a_symbol] = False
            action_type[name] = atype

        for name, val in actions.items():
            atype = action_type[name]
            a_symbol = model.ns[name]
            subst_dict = action_dict.copy()

            if atype == 'bool':
                action = BAction(name, a_symbol, mdp.context)

                # For Boolean actions, we can restrict CPFs with the action.
                subst_dict[a_symbol] = True

                # Also, rewards can be restricted.
                reward = action.restrict(model.reward, subst_dict)
            else:
                action = CAction(name, a_symbol, mdp.context)
                reward = model.reward

            for state_fluent, cpf in model.cpfs.items():
                var_ = model.ns[state_fluent]
                if atype == 'bool':
                    cpf = action.restrict(cpf, subst_dict)
                action.add_cpf(var_, cpf)            
            action.reward = reward

            # Add the action to the MDP.
            mdp.add_action(action)

        # Configure the bounds of continuous actions.
        if len(mdp.cont_a_vars) > 0:
            cont_action_bounds = self.configure_bounds(mdp, model.preconditions)
            mdp.cont_action_bounds = cont_action_bounds

        # Update the next state and interm variable sets.
        mdp.update_var_sets()
        return mdp

    def configure_bounds(
            self, mdp: MDP, preconditions: List[int]
    ) -> Dict[CAction, Tuple[float, float]]:
        """Configures the bounds over continuous actions."""
        context = mdp.context

        # Get the continuous action variables.
        a_var_set = mdp.cont_a_vars

        # Bounds dictionaries to be updated.
        lb_dict: Dict[sp.Symbol, List[sp.Basic]] = {}
        ub_dict: Dict[sp.Symbol, List[sp.Basic]] = {}

        # Iterate over preconditions.
        for p in preconditions:
            # Instantiate the leaf operation object.
            leaf_op = BoundAnalysis(context=mdp.context, var_set=a_var_set)

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
        cont_action_bounds = {}
        for v in a_var_set:
            lb = lb_dict.get(v, -float('inf'))
            ub = ub_dict.get(v, float('inf'))
            lb, ub = float(lb), float(ub)
            action = mdp.a_var_to_action[v]
            cont_action_bounds[action] = (lb, ub)
        return cont_action_bounds
