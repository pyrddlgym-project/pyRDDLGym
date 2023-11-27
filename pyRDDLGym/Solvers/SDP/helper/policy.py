
from typing import Dict, List

import sympy as sp


class Policy:
    """Defines the policy class."""

    def __init__(
            self,
            policy_dict: Dict[sp.Symbol, int],
            policy_fname: str,
    ):
        self._policy_dict = policy_dict
        self.policy_fname = policy_fname

    def __getitem__(self, item: sp.Symbol) -> int:
        return self._policy_dict[item]

    @property
    def actions(self) -> List[sp.Symbol]:
        return list(self._policy_dict.keys())
