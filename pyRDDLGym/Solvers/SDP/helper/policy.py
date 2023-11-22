from typing import Dict

from .mdp import MDP
from .action import Action

import sympy as sp


class Policy:
    """A policy class to support policy evaluation"""
    def __init__(self, mdp: MDP):
        self.mdp = mdp
        self._dist: Dict[Action, int] = {}
    
    def get_policy_xadd(self, action: Action) -> int:
        """Returns the policy XADD for the specified action.
        
        Args:
            action (Action): The action.
        
        Returns:
            int: The policy DD.
        """
        return self._dist.get(action)

    def load_policy(self, policy: Dict[Action, int]):
        """Loads an XADD per each action.
        
        Args:
            policy (Dict[Action, int]): The policy.
        """
        for action, node_id in policy.items():
            self._dist[action] = node_id
    
    def compile_policy(self):
        policy_id_all = self.context.ONE

        for action, policy_id_true in self._dist.items():
            if action._atype != 'bool':
                raise NotImplementedError("Continuous actions are not supported yet")
            
            # get action in int form
            dec_id, is_reversed = self.context.get_dec_expr_index(action._symbol, create=True)
            high: int = self.context.get_leaf_node(sp.S(1))
            low: int = self.context.get_leaf_node(sp.S(0))
            if is_reversed:
                low, high = high, low
            # get indicator version of true and false action
            action_id_true: int = self.context.get_internal_node(dec_id, low=low, high=high)
            action_id_false = self.context.apply(self.context.ONE, action_id_true, 'subtract')
            # get the policy where false action are taken
            policy_id_false = self.context.apply(self.context.ONE, policy_id_true, 'subtract')
            # integrate action indicator in policy
            action_policy_id_true = self.context.apply(policy_id_true, action_id_true, 'prod')
            action_policy_id_false = self.context.apply(policy_id_false, action_id_false, 'prod')
            # combine true and false action policy
            action_policy_id_all = self.context.apply(action_policy_id_true, action_policy_id_false, 'max')
            # use the max operator to combien polcies of different actions
            policy_id_all = self.context.apply(policy_id_all, action_policy_id_all, 'min')

        if self.mdp._is_linear:
            policy_id_all = self.mdp.standardize_dd(policy_id_all)
        
        return policy_id_all
            

    @property
    def context(self):
        return self.mdp.context
