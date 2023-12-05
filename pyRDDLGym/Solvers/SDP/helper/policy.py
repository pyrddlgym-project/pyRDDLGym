
from typing import Dict, List

from xaddpy.xadd.xadd import VAR_TYPE


class Policy:
    """Defines the policy class."""

    def __init__(
            self,
            policy_dict: Dict[VAR_TYPE, int],
            policy_fname: str,
    ):
        self._policy_dict = policy_dict
        self.policy_fname = policy_fname

    def __getitem__(self, item: VAR_TYPE) -> int:
        return self._policy_dict[item]

    @property
    def actions(self) -> List[VAR_TYPE]:
        return list(self._policy_dict.keys())
