"""Implements a parser that converts an XADD RDDL model into an MDP."""

from typing import Dict, List, Tuple

import sympy as sp
from xaddpy import XADD

from pyRDDLGym.Solvers.SDP.helper.action import BAction, CAction
from pyRDDLGym.Solvers.SDP.helper.mdp import MDP
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
            action_dict[a_symbol] = False
            action_type[name] = atype

        for name, val in actions.items():
            atype = action_type[name]
            a_symbol = model.ns[name]

            if atype == 'bool':
                action = BAction(name, a_symbol, mdp.context)

                # For Boolean actions, we can restrict CPFs with the action.
                subst_dict = action_dict.copy()
                subst_dict[a_symbol] = True
                cpfs = model.cpfs
                for state_fluent, cpf in cpfs.items():
                    cpf = action.restrict(cpf, subst_dict)
                    var_ = model.ns[state_fluent]
                    action.add_cpf(var_, cpf)

                # Also, rewards can be restricted.
                reward = action.restrict(model.reward, subst_dict)
                action.reward = reward

            else:
                action = CAction(name, a_symbol, mdp.context)

            # Add the action to the MDP.
            mdp.add_action(action)

        # Configure the bounds of continuous actions.
        if len(mdp.cont_a_vars) > 0:
            cont_action_bounds = self.configure_bounds(mdp, model.preconditions)
            mdp.cont_action_bounds = cont_action_bounds

        return mdp

    # TODO: Implement this: should use recursion to retrieve valid bounds over actions.
    def configure_bounds(self, mdp: MDP, preconditions: List[int]) -> Dict[CAction, Tuple[float, float]]:
        """Configures the bounds over continuous actions."""
