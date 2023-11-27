"""Implements a parser for Policy class."""

import json
import os
from typing import Dict, Optional, Tuple
import sympy as sp
from sympy.logic import boolalg

from pyRDDLGym.Solvers.SDP.helper.mdp import MDP
from pyRDDLGym.Solvers.SDP.helper.policy import Policy
from pyRDDLGym.Solvers.SDP.helper.xadd_utils import ValueAssertion


class PolicyParser:
    """Parses a policy from a json file and its associated policy XADDs."""

    def parse(
            self,
            mdp: MDP,
            policy_fname: str,
            assert_concurrency: bool = True,
            concurrency: Optional[int] = None,
    ) -> Policy:
        """Parses the policy from a given json file.

        Args:
            mdp: The MDP object.
            policy_fname: The policy file name.
            assert_concurrency: Whether to assert concurrency. If set to True,
                this method will raise an error when the concurrency condition does not
                match with the given MDP configuration.
            concurrency: If assert_concurrency is set to True, this argument specifies
                the maximum concurrency number.
        
        Returns:
            The parsed policy object.
        """
        self.mdp = mdp
        assert policy_fname.endswith('.json'), 'Policy file must be a json file.'
        assert os.path.exists(policy_fname), 'Policy file does not exist.'
        try:
            policy_dict = json.load(open(policy_fname, 'r'))
        except Exception as e:
            raise RuntimeError(f'Failed to load policy from {policy_fname}.') from e
        assert not assert_concurrency or (assert_concurrency and concurrency is not None)

        parsed_policy_dict = {}

        try:
            # Get the action fluents and validate them.
            a_vars = policy_dict.pop('action-fluents')
            self._validate_action_fluents(a_vars)
            self._validate_policy_json(policy_dict)

            for a_name in a_vars:
                a_symbol, a_xadd = self._parse_policy_for_single_action(policy_dict, a_name)
                parsed_policy_dict[a_symbol] = a_xadd
            policy = Policy(parsed_policy_dict, policy_fname)
        except:
            raise RuntimeError(f'Failed to load policy from {policy_fname}.')

        # Assert concurrency.
        if assert_concurrency:
            self._assert_concurrency(parsed_policy_dict)
        return policy

    def _validate_action_fluents(self, a_vars):
        """Validates the action fluents."""
        for a_var in a_vars:
            if a_var not in self.model.actions:
                raise ValueError(f'Action fluent {a_var} is not defined in the model.')

    def _validate_policy_json(self, policy_dict):
        """Validates the policy json."""
        action_set_from_model = set(self.model.actions)
        action_set_from_policy = set(policy_dict.keys())
        assert len(action_set_from_model.symmetric_difference(action_set_from_policy)) == 0, \
            'Action set from the policy does not match with the model.'
        for a, val in policy_dict:
            assert isinstance(val, str), f'Value for action {a} must be a string file path.'
            assert val.endswith('.xadd'), f'Value for action {a} must be a string file path ending with .xadd.'
            assert os.path.exists(val), f'Value for action {a} must be a string file path that exists.'

    def _parse_policy_for_single_action(self, policy_dict, a_name) -> Tuple[sp.Symbol, int]:
        """Parses the policy for a single action."""
        assert a_name in policy_dict, f'Action {a_name} not found in the policy.'
        path = policy_dict[a_name]
        a_type = self.model.gvar_to_type[a_name]

        # Parse the XADD from file.
        a_dd = self.context.import_xadd(path)
        a_var = self.model.ns[a_name]

        # Boolean action should condition on itself being True or False.
        if a_type == 'bool':
            a_var_id, _ = self.context.get_dec_expr_index(a_var, create=False)
            high = a_dd
            low = self.context.apply(self.context.ONE, high, op='subtract')
            a_dd = self.context.get_inode_canon(a_var_id, low, high)

        # Validate the given action XADD.
        self._validate_action(a_dd, a_type)
        return a_var, a_dd

    def _validate_action(self, a_dd: int, a_type: str):
        """Validates the action XADD."""
        if a_type == 'bool':
            self._validate_bool_action_dd(a_dd)
        elif a_type == 'real':
            self._validate_cont_action_dd(a_dd)
        else:
            raise ValueError(f'Action type {a_type} not supported.')

    def _validate_bool_action_dd(self, a_dd: int):
        """Validates the boolean action XADD."""
        var_set = self.context.collect_vars(a_dd)

        # Boolean action should only depend on state fluents.
        s_vars = self.mdp.cont_s_vars.union(self.mdp.bool_s_vars)
        assert len(var_set.difference(s_vars)) == 0, \
            'Boolean action should only depend on state fluents.'

        # Check leaf value types.
        leaf_op = ValueAssertion(
            self.context,
            fn=lambda x: int(x) >= 0 and int(x) <= 1,
            msg='Boolean action leaf {leaf_val} is not within range [0, 1]',
        )
        self.context.reduce_process_xadd_leaf(a_dd, leaf_op, [], [])

    def _validate_cont_action_dd(self, a_dd: int):
        """Validates the continuous action XADD."""
        var_set = self.context.collect_vars(a_dd)

        # Continuous action should depend on state fluents and boolean actions.
        s_vars = self.mdp.cont_s_vars.union(self.mdp.bool_s_vars)
        bool_a_vars = self.mdp.bool_a_vars
        assert len(var_set.difference(s_vars.union(bool_a_vars))) == 0, \
            'Continuous action should only depend on state fluents and boolean actions.'

        # Check leaf value types.
        leaf_op = ValueAssertion(
            self.context,
            fn=lambda x: not isinstance(x, boolalg.BooleanAtom),
            msg='Continuous action leaf {leaf_val} is a Boolean value.'
        )
        self.context.reduce_process_xadd_leaf(a_dd, leaf_op, [], [])

    def _assert_concurrency(
            self,
            policy: Dict[sp.Symbol, int],
            concurrency: int,
    ):
        """Asserts the concurrency is satisfied for the given policy."""
        bool_dd = self.context.ZERO

        for a, dd in policy.items():
            a_name = self.model._sympy_var_name_to_var_name[str(a)]
            a_type = self.model.gvar_to_type[a_name]
            if a_type == 'bool':
                a_true_dd = self.context.unary_op(dd, op='ceil')
                bool_dd = self.context.apply(bool_dd, a_true_dd, 'add')

        # Assert concurrency.
        if bool_dd != self.context.ZERO:
            leaf_op = ValueAssertion(
                self.context,
                fn=lambda x: float(x) <= concurrency,
                msg='Concurrency condition is not satisfied with {leaf_val}.'
            )
            self.context.reduce_process_xadd_leaf(bool_dd, leaf_op, [], [])

    @property
    def model(self):
        return self.mdp.model

    @property
    def context(self):
        return self.mdp.context
