from typing import List, Union

import sympy as sp
from sympy.logic import boolalg
from xaddpy.xadd.xadd import XADDLeafOperation


class TransformToBernoulliXADD(XADDLeafOperation):
    """Defines the leaf operation that transforms to a Bernoulli node."""
    def __init__(
            self,
            context,
    ):
        super().__init__(context)
        self._require_canonical = True

        def _get_bernoulli_node_id(
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
        self._get_bernoulli_node_id = _get_bernoulli_node_id

    def process_xadd_leaf(
            self,
            decs: List[sp.Basic],
            dec_vals: List[bool],
            leaf_val: sp.Basic,
            **kwargs
    ) -> int:
        """Transforms the leaf node to a Bernoulli node."""
        assert (isinstance(leaf_val, sp.Symbol) or 
                isinstance(leaf_val, boolalg.BooleanAtom)), (
            "The leaf node must be a Symbol or a BooleanAtom."
        )
        node_id = self._get_bernoulli_node_id(leaf_val)
        return node_id
