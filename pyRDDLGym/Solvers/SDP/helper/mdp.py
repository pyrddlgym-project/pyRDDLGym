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
        self.max_allowed_actions = self.model.max_allowed_actions
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
        self.cont_action_bounds: Dict[sp.Symbol, Tuple[float, float]] = {}

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

    def update_i_and_ns_var_sets(self):
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

    def update(self) -> None:
        """Goes through all CPFs and actions and updates them."""
        dual_action_cpfs = {}
        for a_name, act in self.actions.items():
            # Update CPFs.
            for v_name, cpf in self.model.cpfs.items():
                # Handle Boolean next state and interm variables.
                v = self.context._str_var_to_var[v_name]
                if v._assumptions['bool'] and (v_name in self.model.next_state.values() or v_name in self.model.interm):
                    cpf_a = dual_action_cpfs.get(v_name)
                    if cpf_a is None:
                        var_id, _ = self.context.get_dec_expr_index(v, create=False)
                        high = cpf
                        low = self.context.apply(self.context.ONE, high, op='subtract')
                        cpf_a = self.context.get_inode_canon(var_id, low, high)
                        dual_action_cpfs[v_name] = cpf_a
                    cpf = cpf_a
                # Boolean actions can be restricted.
                if isinstance(act, BAction):
                    cpf_a = act.restrict(True, cpf)
                    act.add_cpf(v, cpf_a)
                else:
                    act.add_cpf(v, cpf)
            # Update reward CPFs.
            reward = self.model.reward
            if isinstance(act, BAction):
                reward = act.restrict(True, reward)
            act.reward = reward
            
    @property
    def cpfs(self):
        return self.model.cpfs

    @property
    def prime_subs(self):
        return self._prime_subs

    @prime_subs.setter
    def prime_subs(self, prime_subs: Dict[sp.Symbol, sp.Symbol]):
        self._prime_subs = prime_subs

    def get_bounds(self, a: sp.Symbol) -> Tuple[float, float]:
        return self.cont_action_bounds[a]

    def standardize(self, dd: int) -> int:
        """Standardizes the given XADD node."""
        dd = self.context.make_canonical(dd)
        if self.is_linear:
            dd = self.context.reduce_lp(dd)
        return dd
