"""Defines the Action class."""

import abc
from typing import Dict, List, Optional

import sympy as sp
from xaddpy import XADD


class Action:
    def __init__(
            self,
            name: str,
            symbol: sp.Symbol,
            context: XADD,
            atype: str,
            params: Optional[List] = None,
    ):
        self.name = name
        self.symbol = symbol
        self.context = context
        self.atype = atype      # action type: 'bool' or 'real'
        self.params = [] if params is None else params
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
    def __init__(self, name: str, symbol: sp.Symbol, context: XADD):
        super().__init__(name, symbol, context, 'bool')

    def restrict(self, cpf: int, subst_dict: Dict[sp.Symbol, bool]):
        """Restricts a given CPF with this Boolean action."""
        assert subst_dict[self.symbol]
        return self.context.substitute(cpf, subst_dict)


class CAction(Action):
    """Continuous Action class."""
    def __init__(self, name: str, symbol: sp.Symbol, context: XADD):
        super().__init__(name, symbol, context, 'real')
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
