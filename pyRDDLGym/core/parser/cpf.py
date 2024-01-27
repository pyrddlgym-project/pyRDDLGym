# This file is based on thiago pbueno's pyrddl.
# https://github.com/thiagopbueno/pyrddl
# it was adapted and extended for pyRDDLGym


from pyRDDLGym.Core.Parser.expr import Expression

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
