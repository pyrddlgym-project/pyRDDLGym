# This file is part of pyrddl.

# pyrddl is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# pyrddl is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with pyrddl. If not, see <http://www.gnu.org/licenses/>.


from pyrddl.expr import Expression
from pyrddl.pvariable import PVariable

from typing import Tuple, List


PVarExpr = Tuple[str, Tuple[str, List[str]]]


class CPF(object):
    '''Conditional Probability Function.
    Note:
        This class is intended to be solely used by the parser and compiler.
        Do not attempt to directly use this class to build a CPF object.
    Args:
        pvar: CPF's parameterized variable.
        expr: CPF's expression.
    Attributes:
        pvar (:obj:`PVariable`): CPF's parameterized variable.
        expr (:obj:`Expression`): CPF's expression.
    '''

    def __init__(self, pvar: PVarExpr, expr: Expression) -> None:
        self.pvar = pvar
        self.expr = expr

    @property
    def name(self) -> str:
        '''Returns the CPF's pvariable name.'''
        return Expression._pvar_to_name(self.pvar[1])

    def __repr__(self) -> str:
        '''Returns the CPF's canonical representation.'''
        cpf = '{} =\n{};'.format(str(self.pvar), str(self.expr))
        return cpf
