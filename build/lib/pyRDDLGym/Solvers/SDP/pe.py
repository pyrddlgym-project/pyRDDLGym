"""Defines the Policy Evaluation solver."""
from pyRDDLGym.Solvers.SDP.base import SymbolicSolver

from pyRDDLGym.Solvers.SDP.base import SymbolicSolver
from pyRDDLGym.Solvers.SDP.helper import Action, MDP

import sympy as sp

from xaddpy.xadd.xadd import DeltaFunctionSubstitution

from typing import Set

from pyRDDLGym.Core.Compiler.RDDLLevelAnalysis import _topological_sort

class PolicyEvaluation(SymbolicSolver):

    def load_policy(self, policy):
        self.policy = policy
        # generate policy cpfs precomputed
        self.policy_cpfs, self.policy_reward = self.preload_policy()


    def solve(self) -> int:
        
        # Initialize the iteration counter
        self._cur_iter = 0

        # Initialize the value function to be the zero node
        value_dd = self.context.ZERO

        # Perform policy evaluation for the specified number of iterations, or until convergence
        while self.n_curr_iter < self.n_max_iter:
            self.n_curr_iter += 1

            self._prev_dd = value_dd
            
            # Compute the next value function
            value_dd = self.bellman_backup(value_dd)

            if self.mdp.is_linear:
                value_dd = self.mdp.standardize(value_dd)
        
        return value_dd
    

    def bellman_backup(self, dd: int) -> int:
        """Performs the VI Bellman backup.
        
        Args:
            dd: The current value function XADD ID to which Bellman back is applied.

        Returns:
            The new value function XADD ID.
        """
        # Compute the action value function
        regr = self.regress(dd)

        if self.mdp.is_linear:
            regr = self.mdp.standardize(regr)
        
        return regr

    def regress(self, value_dd: int, regress_cont: bool = False) -> int:
        # Prime the value function
        subst_dict = self.mdp.prime_subs

        q = self.context.substitute(value_dd, subst_dict)

        # Discount
        if self.mdp.discount < 1.0:
            q = self.context.scalar_op(q, self.mdp.discount, 'prod')

        # Add reward *if* it contains primed vars that need to be regressed
        i_and_ns_vars_in_reward = self.filter_i_and_ns_vars(self.context.collect_vars(self.mdp.model.reward))

        if len(i_and_ns_vars_in_reward) > 0:
            q = self.context.apply(q, self.mdp.model.reward, 'add')

        # # Get variables to eliminate
        ## old method of filtering vars to regress only in q
        vars_to_regress = self.filter_i_and_ns_vars(self.context.collect_vars(q), True, True)

        # vars_to_regress = list(set.union(self.mdp.cont_i_vars, self.mdp.bool_i_vars, 
        #                             self.mdp.cont_ns_vars, self.mdp.bool_ns_vars))
        # call_graph = self.level_analyzer.build_call_graph()
        # sorted_graph = _topological_sort(call_graph)

        sorted_vars_to_regress = self.sort_var_set(vars_to_regress)

        # for s in self.sorted_levels:
        for v in sorted_vars_to_regress:
            if v in self.mdp.cont_ns_vars or v in self.mdp.cont_i_vars:
                q = self.regress_cvars(q, v)
            elif v in self.mdp.bool_ns_vars or v in self.mdp.bool_i_vars:
                q = self.regress_bvars(q, v)

               
        # # # bernoulli noise
        # ber_vars_to_regress = self.filter_ber_vars(self.context.collect_vars(q))
        
        # for v in ber_vars_to_regress:
        #     q = self.regress_ber_vars(q, v)

        # Add the rewards
        if len(i_and_ns_vars_in_reward) == 0:
            q = self.context.apply(q, self.mdp.model.reward, 'sum')

        # Continuous noise?
        # TODO
        # q = self.regress_noise(q, action)

        # # Continuous parameter
        # if regress_cont:
        #     q = self.regress_action(q, action)

        if self.mdp.is_linear:
            q = self.mdp.standardize(q)

        return q

    def regress_cvars(self, q: int, v: sp.Symbol) -> int:
        """Regress a continuous variable from the value function `q`."""
        # Get the CPF for the variable.
        cpf = self.policy_cpfs[str(v)]

        # Check the regression cache.
        key = (str(v), cpf, q)
        res = self.mdp.cont_regr_cache.get(key)
        if res is not None:
            return res

        # Perform regression via Delta function substitution.
        leaf_op = DeltaFunctionSubstitution(v, q, self.context, is_linear=self.mdp.is_linear)
        q = self.context.reduce_process_xadd_leaf(cpf, leaf_op, [], [])

        # Simplify the resulting XADD if possible.
        if self.mdp.is_linear:
            q = self.context.reduce_lp(q)

        # Cache and return the result.
        self.mdp.cont_regr_cache[key] = q
        return q

    def regress_bvars(self, q: int, v: sp.Symbol) -> int:
        """Regress a boolean variable from the value function `q`."""
        # Get the CPF for the variable.
        cpf = self.policy_cpfs[str(v)]
        dec_id = self.context._expr_to_id[self.mdp.model.ns[str(v)]]

        # Convert nodes to 1 and 0.
        cpf = self.context.unary_op(cpf, 'float')
        cpf_true = cpf
        cpf_false = self.context.apply(self.context.ONE, cpf_true, 'subtract')



        # Marginalize out the boolean variable.
        # q = self.context.apply(q, cpf, op='prod')
        restrict_high = self.context.op_out(q, dec_id, op='restrict_high')
        restrict_low = self.context.op_out(q, dec_id, op='restrict_low')

        q_true = self.context.apply(cpf_true, restrict_high, 'prod')
        q_false = self.context.apply(cpf_false, restrict_low, 'prod')

        # q = self.context.apply(restrict_high, restrict_low, op='add')

        q = self.context.apply(q_true, q_false, 'add')

        return q
    
    
    # def regress_ber_vars(self, q: int, v: sp.Symbol) -> int:
    #     prob_id = int(str(v).split('_')[-1])
    #     not_prob_id = self.context.apply(self.context.ONE, prob_id, 'subtract')

    #     dec_id = self.context._expr_to_id[self.mdp.model.ns[str(v)]]

    #     restrict_high = self.context.op_out(q, dec_id, 'restrict_high')
    #     restrict_low = self.context.op_out(q, dec_id, 'restrict_low')

    #     true_prob_id = self.context.apply(restrict_high, prob_id, 'prod')
    #     false_prob_id = self.context.apply(restrict_low, not_prob_id, 'prod')

    #     q = self.context.apply(true_prob_id, false_prob_id, 'add')

    #     return q

    # def filter_ber_vars(
    #         self, var_set: set) -> Set[str]:
    #     filtered_vars = set()
    #     for v in var_set:
    #         if str(v).startswith('#_UNIFORM'):
    #             filtered_vars.add(v)
    #     return filtered_vars
    
    def preload_policy(self):
        policy_cpfs = {}

        # get transtion cpfs according to policy
        for cpf_name, cpf_id in self.mdp.model.cpfs.items():
            # collect all action vars in the cpfs
            action_vars = set([str(i) for i in self.context.collect_vars(cpf_id) if str(i) in self.mdp.model.actions.keys()])
            policy_cpf = cpf_id

            for a in action_vars:
                a_symbol = self.mdp.model.ns[str(a)]
                a_type = self.mdp.actions[a].atype
                if a_type == 'bool':
                    # get state space where policy is True and False seperately
                    a_policy_true = self.policy._dist[self.mdp.actions[a]]
                    a_policy_false = self.context.apply(self.context.ONE, a_policy_true, 'subtract')
                    # subsitute action with true and false in cpfs
                    true_aciton_cpf = self.context.substitute(policy_cpf, {a_symbol:True})
                    false_aciton_cpf = self.context.substitute(policy_cpf, {a_symbol:False})
                    # find states where transtion happens
                    true_cpf = self.context.apply(true_aciton_cpf, a_policy_true, 'prod')
                    false_cpf = self.context.apply(false_aciton_cpf, a_policy_false, 'prod')
                    # marginalize all the true and false
                    policy_cpf = self.context.apply(true_cpf, false_cpf, 'add')
                    policy_cpf = self.mdp.standardize(policy_cpf)
                elif a_type == 'real':
                    a_policy = self.policy._dist[self.mdp.actions[a]]
                    # substitue values to cpfs
                    leaf_op = DeltaFunctionSubstitution(a_symbol, policy_cpf, self.context, self.mdp.is_linear)
                    policy_cpf = self.context.reduce_process_xadd_leaf(a_policy, leaf_op, [], [])
                    if self.mdp.is_linear:
                        policy_cpf = self.mdp.standardize(policy_cpf)   
                else:
                    raise ValueError('{0} type not defined'.format(a_type))
            
                
            policy_cpfs[cpf_name] = policy_cpf

        return policy_cpfs, None

