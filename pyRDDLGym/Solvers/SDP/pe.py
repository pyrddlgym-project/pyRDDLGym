"""Defines the Policy Evaluation solver."""

from typing import Optional

from xaddpy.xadd.xadd import DeltaFunctionSubstitution

from pyRDDLGym.Solvers.SDP.base import SymbolicSolver
from pyRDDLGym.Solvers.SDP.helper import Policy


class PolicyEvaluation(SymbolicSolver):
    """Policy Evaluation solver."""

    def __init__(self, policy: Policy, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = policy
        self.reward: Optional[int] = None
        self._embed_policy_to_cpfs()

    def _embed_policy_to_cpf(self, cpf: int, policy: Policy) -> int:
        """Embeds the policy into a single CPF."""
        cpf_ = cpf
        var_set = self.context.collect_vars(cpf)

        for a_var in policy.actions:
            pi_a = policy[a_var]    # pi(a|s).
            # Boolean actions.
            if a_var in self.mdp.bool_a_vars:
                # The decision ID of the boolean variable node.
                var_id = self.context._expr_to_id[a_var]

                # Make leaf values numbers.
                pi_a = self.context.unary_op(pi_a, 'float')

                # Marginalize out the boolean action variable.
                cpf_ = self.context.apply(cpf_, pi_a, op='prod')
                restrict_high = self.context.op_out(cpf_, var_id, op='restrict_high')
                restrict_low = self.context.op_out(cpf_, var_id, op='restrict_low')
                cpf_ = self.context.apply(restrict_high, restrict_low, op='add')
            else:
                if a_var not in var_set:
                    continue
                leaf_op = DeltaFunctionSubstitution(
                    a_var, cpf_, self.context, is_linear=self.mdp.is_linear)
                cpf_ = self.context.reduce_process_xadd_leaf(pi_a, leaf_op, [], [])
        return cpf_

    def _embed_policy_to_cpfs(self):
        """Embeds the policy into the CPFs."""
        cpfs = self.mdp.cpfs
        policy = self.policy

        # Update next state and interm variable CPFs.
        for v, cpf in cpfs.items():
            cpfs[v] = self._embed_policy_to_cpf(cpf, policy)

        # Update the reward CPF.
        self.reward = self._embed_policy_to_cpf(self.mdp.reward, policy)

    def bellman_backup(self, dd: int) -> int:
        """Performs the policy evaluation Bellman backup.

        Args:
            dd: The current value function XADD ID to which Bellman back is applied.
        
        Returns:
            The new value function XADD ID.
        """
        # Regress the value function.
        regr = self.regress(dd, reward=self.reward, cpfs=self.mdp.cpfs)
        return regr
