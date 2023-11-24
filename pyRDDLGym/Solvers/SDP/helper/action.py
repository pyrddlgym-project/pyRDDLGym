"""Defines the Action class."""

from typing import Dict, List, Optional

import sympy as sp
from xaddpy import XADD


from pyRDDLGym.XADD.RDDLModelXADD import RDDLModelWXADD


class Action:
    """Base Action class."""

    def __init__(
            self,
            name: str,
            symbol: sp.Symbol,
            model: RDDLModelWXADD,
            atype: str,
    ):
        self.name = name
        self.symbol = symbol
        self.model = model
        self.context = model._context
        self.atype = atype      # action type: 'bool' or 'real'
        self.cpfs = {}
        self._reward = None

    def add_cpf(self, v: sp.Symbol, cpf: int):
        """Adds a CPF XADD node ID to the action."""
        self.cpfs[v] = cpf

    def get_cpf(self, v: sp.Symbol) -> int:
        """Gets the CPF XADD node ID given a variable."""
        return self.cpfs[v]

    @property
    def reward(self) -> int:
        return self._reward

    @reward.setter
    def reward(self, reward: int):
        self._reward = reward

    def __repr__(self) -> str:
        return self.name


class BAction(Action):
    """Boolean Action class."""
    def __init__(self, name: str, symbol: sp.Symbol, model: RDDLModelWXADD):
        super().__init__(name, symbol, model, 'bool')          

    def restrict(self, b: bool, dd: int) -> int:
        """Restrict a given XADD with the specified value of the action."""
        # Get the variable set of the XADD.
        var_set = self.context.collect_vars(dd)
        # Skip if the action variable is not in the XADD.
        if self.symbol not in var_set:
            return dd
        # Prepare the substitution dictionary.
        subst_dict = {self.symbol: b}
        # Perform the substitution.
        return self.context.substitute(dd, subst_dict)

    @property
    def cpfs(self) -> Dict[sp.Symbol, int]:
        return self.cpfs


class CAction(Action):
    """Continuous Action class."""
    def __init__(self, name: str, symbol: sp.Symbol, model: RDDLModelWXADD):
        super().__init__(name, symbol, model, 'real')
        self.need_bound_analysis = True

    @property
    def ub(self) -> float:
        """Returns the upper bound of the continuous action."""
        return self._ub

    @ub.setter
    def ub(self, ub: float):
        self._ub = ub

    @property
    def lb(self) -> float:
        """Returns the lower bound of the continuous action."""
        return self._lb

    @lb.setter
    def lb(self, lb: float):
        self._lb = lb
