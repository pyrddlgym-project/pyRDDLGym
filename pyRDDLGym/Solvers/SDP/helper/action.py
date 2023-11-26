"""Defines the Action class."""

import abc
from typing import Dict, Tuple, Union

import sympy as sp
from xaddpy import XADD

from pyRDDLGym.XADD.RDDLModelXADD import RDDLModelWXADD


class Action(metaclass=abc.ABCMeta):
    """The base action class."""

    def add_cpf(self, v: sp.Symbol, cpf: int):
        """Adds a CPF XADD node ID to the action."""
        self.cpfs[v] = cpf

    def get_cpf(self, v: sp.Symbol) -> int:
        """Gets the CPF XADD node ID given a variable."""
        return self.cpfs[v]

    @property
    def symbol(self) -> Union[sp.Symbol, Tuple[sp.Symbol]]:
        assert hasattr(self, '_symbol'), 'Action symbol not set'
        return self._symbol

    @property
    def reward(self) -> int:
        assert hasattr(self, '_reward'), 'Action reward not set'
        return self._reward

    @reward.setter
    def reward(self, reward: int):
        self._reward = reward

    def __repr__(self) -> str:
        return self.name

    @property
    def cpfs(self) -> Dict[sp.Symbol, int]:
        if not hasattr(self, '_cpfs'):
            self._cpfs = {}
        return self._cpfs


class SingleAction(Action):
    """Base Action class."""

    def __init__(
            self,
            name: str,
            symbol: sp.Symbol,
            model: RDDLModelWXADD,
            atype: str,
    ):
        self.name = name
        self._symbol = symbol
        self.model = model
        self.context = model._context
        self.atype = atype      # action type: 'bool' or 'real'
        self._reward = None



class BAction(SingleAction):
    """Boolean Action class."""
    def __init__(self, name: str, symbol: sp.Symbol, model: RDDLModelWXADD):
        super().__init__(name, symbol, model, 'bool')          

    def restrict(self, dd: int, subst_dict: Dict[sp.Symbol, bool]) -> int:
        """Restrict a given XADD by setting the action value as True."""
        # Get the variable set of the XADD.
        var_set = self.context.collect_vars(dd)
        # Skip if no action variable is in the XADD.
        if len(set(subst_dict.keys()).intersection(var_set)) == 0:
            return dd
        # Prepare the substitution dictionary.
        subst_dict.update({self.symbol: True})
        # Perform the substitution.
        return self.context.substitute(dd, subst_dict)


class CAction(SingleAction):
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


class ConcurrentAction(Action):
    """Concurrent Boolean Action class."""
    def __init__(
            self,
            names: Tuple[str],
            symbols: Tuple[sp.Symbol],
            model: RDDLModelWXADD,
    ):
        self.name = '_'.join(names)
        self._symbol = symbols
        self.model = model
        self.context = model._context
        self._reward = None

    def add_cpf(self, *args, **kwargs):
        return super().add_cpf(*args, **kwargs)

    def get_cpf(self, *args, **kwargs):
        return super().get_cpf(*args, **kwargs)

    def restrict(self, dd: int, subst_dict: Dict[sp.Symbol, bool]) -> int:
        """Restrict a given XADD by setting the action values as True."""
        # Make a copy to prevent in-place updates.
        subst_dict = subst_dict.copy()
        # Get the variable set of the XADD.
        var_set = self.context.collect_vars(dd)
        # Skip if no action variable is in the XADD.
        if len(set(subst_dict.keys()).intersection(var_set)) == 0:
            return dd
        # Prepare the substitution dictionary.
        subst_dict.update({s: True for s in self.symbol})
        # Perform the substitution.
        return self.context.substitute(dd, subst_dict)
