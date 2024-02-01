from typing import Union

import sympy as sp
from sympy.logic import boolalg


def get_bernoulli_node_id(
    expr_or_var: Union[sp.Symbol, boolalg.BooleanAtom]) -> int:
    """Returns the node ID associated with the Bernoulli param."""
    # Note: the pattern is f'{dist}_params_{len(args)}_{proba}'.
    if isinstance(expr_or_var, boolalg.BooleanAtom):
        return 1 if expr_or_var else 0
    name = str(expr_or_var)
    dist, _, len_args, param = name.split('_')
    assert dist == 'Bernoulli', (
        "The leaf node must be a Bernoulli node."
    )
    assert len_args == '1', (
        "The length of the arguments must be 1."
    )
    assert param.isdigit(), (
        "The parameter must be a digit corresponding to"
        "the node ID of the Bernoulli parameter."
    )
    return int(param)
