    #
    # def _compile_fluent_update(self, jax_cpfs, jax_reward, jax_policy):
    #     '''Given the current values of cpfs x, and RNG state key:
    #        produces a dict that maps cpf names (including primed state)
    #        to (hopefully traced) jax expressions representing their values
    #        at the next decision epoch
    #     '''        
    #
    #     def _fluent_update(x, key, t, theta):
    #
    #         # evaluate actions based on the current state
    #         actions, key = jax_policy(x, key, t)
    #         x = {**x, **actions}
    #
    #         # evaluate all CPFs and reward in topological order
    #         for order in self._order_cpfs:
    #             for cpf in self.cpforder[order]: 
    #                 if cpf in self._model.next_state:
    #                     primed_cpf = self._model.next_state[cpf]
    #                     jax_cpf = jax_cpfs[primed_cpf]
    #                     sample, key = jax_cpf(x, key)
    #                     x = {**x, primed_cpf: sample}
    #                 else:
    #                     jax_cpf = jax_cpfs[cpf]
    #                     sample, key = jax_cpf(x, key)
    #                     x = {**x, cpf: sample}
    #         reward, key = jax_reward(x, key)
    #
    #         # update all state fluents
    #         next_x = {**self._nonfluents, **actions}
    #         for order in self._order_cpfs:
    #             for cpf in self.cpforder[order]: 
    #                 if cpf in self._model.next_state:
    #                     primed_cpf = self._model.next_state[cpf]
    #                     next_x[cpf] = x[primed_cpf]
    #                 else:
    #                     next_x[cpf] = x[cpf]
    #
    #         return next_x, reward, key
    #
    #     return _fluent_update
    #
    # def compile_return(self, jax_policy, n_steps: int):
    #     jax_cpfs = self.compile_cpfs()
    #     jax_reward = self.compile_reward()        
    #     state_update = self._compile_fluent_update(jax_cpfs, jax_reward, jax_policy)
    #
    #     def _iterate(t, carry):
    #         x, R, key = carry
    #         next_x, next_r, next_key = state_update(x, key, t)
    #         next_R = R + next_r
    #         next_carry = (next_x, next_R, next_key)
    #         return next_carry            
    #
    #     def _return(x, key):
    #         actions, key = jax_policy(x, key, 0)
    #         x = {**x, **actions}
    #         init_state = (x, 0., key)
    #         x, R, key = jax.lax.fori_loop(0, n_steps, _iterate, init_state)
    #         return R, (x, key)
    #
    #     return jax.jit(_return)
    #
