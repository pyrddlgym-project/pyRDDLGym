"""Defines the MDP class."""

import sympy as sp
from typing import Dict, Tuple

from pyRDDLGym.XADD.RDDLModelXADD import RDDLModelWXADD
from pyRDDLGym.Solvers.SDP.helper import Action, BAction, CAction


class MDP:
    """Defines the MDP class.
    
    Args:
        model: The RDDL model compiled in XADD.
        is_linear: Whether the MDP is linear or not.
        discount: The discount factor.
    """
    def __init__(
            self,
            model: RDDLModelWXADD,
            is_linear: bool = False,
            discount: float = 1.0,
    ):
        self.model = model
        self.context = model._context
        self.is_linear = is_linear
        self.discount = discount
        self._prime_subs = self.get_prime_subs()

        self.cont_ns_vars = set()
        self.bool_ns_vars = set()
        self.cont_i_vars = set()
        self.bool_i_vars = set()
        self.bool_s_vars = set()
        self.bool_a_vars = set()
        self.cont_s_vars = set()
        self.cont_a_vars = set()

        self.actions: Dict[str, Action] = {}
        self.a_var_to_action: Dict[sp.Symbol, Action] = {}
        self.action_to_a_var: Dict[Action, sp.Symbol] = {}
        self.cont_action_bounds: Dict[CAction, Tuple[float, float]] = {}

        # Cache
        self.cont_regr_cache: Dict[Tuple[str, int, int], int] = {}

    def get_prime_subs(self) -> Dict[sp.Symbol, sp.Symbol]:
        """Returns the substitution dictionary for the primed variables."""
        m = self.model
        s_to_ns = m.next_state
        prime_subs = {}
        for s, ns in s_to_ns.items():
            s_var = m.ns[s]
            ns_var, var_node_id = m.add_sympy_var(ns, m.gvar_to_type[ns])
            prime_subs[s_var] = ns_var
        return prime_subs

    def update_var_sets(self):
        m = self.model
        for v, vtype in m.gvar_to_type.items():
            if v in m.nonfluents:
                continue
            v_, v_node_id = m.add_sympy_var(v, vtype)
            if v in m.next_state.values():
                if vtype == 'bool':
                    self.bool_ns_vars.add(v_)
                else:
                    self.cont_ns_vars.add(v_)
            elif v in m.interm:
                if vtype == 'bool':
                    self.bool_i_vars.add(v_)
                else:
                    self.cont_i_vars.add(v_)

    def add_action(self, action: Action):
        """Adds an action to the MDP."""
        self.actions[action.name] = action
        if isinstance(action, BAction):
            self.bool_a_vars.add(action.symbol)
        else:
            self.cont_a_vars.add(action.symbol)
        self.a_var_to_action[action.symbol] = action
        self.action_to_a_var[action] = action.symbol

    @property
    def cpfs(self):
        return self.model.cpfs

    @property
    def prime_subs(self):
        return self._prime_subs

    @prime_subs.setter
    def prime_subs(self, prime_subs: Dict[sp.Symbol, sp.Symbol]):
        self._prime_subs = prime_subs

    def get_bounds(self, a: CAction) -> Tuple[float, float]:
        return self.cont_action_bounds[a]

    def standardize(self, dd: int) -> int:
        """Standardizes the given XADD node."""
        dd = self.context.make_canonical(dd)
        if self.is_linear:
            dd = self.context.reduce_lp(dd)
        return dd
