"""Defines the MDP class."""

import sympy as sp
from typing import Dict, Tuple, Union

from pyRDDLGym.XADD.RDDLModelXADD import RDDLModelWXADD
from pyRDDLGym.Solvers.SDP.helper import Action, BAction, CAction, ConcurrentAction


class MDP:
    """Defines the MDP class.
    
    Args:
        model: The RDDL model compiled in XADD.
        is_linear: Whether the MDP is linear or not.
        discount: The discount factor.
        concurrency: The number of concurrent boolean actions.
    """
    def __init__(
            self,
            model: RDDLModelWXADD,
            is_linear: bool = False,
            discount: float = 1.0,
            concurrency: int = 1,
    ):
        self.model = model
        self.context = model._context
        self.is_linear = is_linear
        self.discount = discount
        self.cpfs = {}
        self.max_allowed_actions = concurrency
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
        self.bool_actions: Dict[str, Union[BAction, ConcurrentAction]] = {}
        self.cont_actions: Dict[str, CAction] = {}
        self.a_var_to_action: Dict[sp.Symbol, Action] = {}
        self.action_to_a_var: Dict[Action, sp.Symbol] = {}

        # Bounds
        self.cont_state_bounds: Dict[sp.Symbol, Tuple[float, float]] = {}
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
            self.bool_actions[action.name] = action
        elif isinstance(action, ConcurrentAction):
            self.bool_a_vars.update(set(action.symbol))
            self.bool_actions[action.name] = action
        elif isinstance(action, CAction):
            self.cont_a_vars.add(action.symbol)
            self.cont_actions[action.name] = action
        else:
            raise ValueError(f'{action} is not a valid action type.')
        # self.a_var_to_action[action.symbol] = action
        # self.action_to_a_var[action] = action.symbol

    def update(self) -> None:
        """Goes through all CPFs and actions and updates them."""
        dual_cpfs_bool = {}
        action_subst_dict = {a: False for a in self.bool_a_vars}
        for v_name, cpf in self.model.cpfs.items():
            # Handle Boolean next state and interm variables.
            v = self.context._str_var_to_var[v_name]
            if v._assumptions['bool'] and (v_name in self.model.next_state.values() or v_name in self.model.interm):
                cpf_ = dual_cpfs_bool.get(v_name)
                if cpf_ is None:
                    var_id, _ = self.context.get_dec_expr_index(v, create=False)
                    high = cpf
                    low = self.context.apply(self.context.ONE, high, op='subtract')
                    cpf_ = self.context.get_inode_canon(var_id, low, high)
                    dual_cpfs_bool[v_name] = cpf_
                cpf = cpf_

            # Update CPFs and reward.
            self.cpfs[v] = cpf
            for a_name, act in self.actions.items():
                # Boolean actions can be restricted.
                if isinstance(act, BAction) or isinstance(act, ConcurrentAction):
                    cpf_ = act.restrict(cpf, action_subst_dict)
                    act.add_cpf(v, cpf_)
                    reward = act.restrict(self.reward, action_subst_dict)
                else:
                    act.add_cpf(v, cpf)
                    reward = self.reward
                act.reward = reward

    @property
    def prime_subs(self):
        return self._prime_subs

    @property
    def reward(self) -> int:
        return self.model.reward

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
