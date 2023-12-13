import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

class RejectionSampler:
    def __init__(self,
                 batch_size,
                 action_dim,
                 policy,
                 config):
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.policy = policy
        self.config = config
        self.init_state = None
        self.step_size = None

        self.sample_shape = (self.action_dim, 2,
                             1, self.action_dim)
        self.actions_shape = self.sample_shape[:-1]
        self.batch_shape = (self.batch_size,) + self.sample_shape

    def prep(self,
             key,
             target_log_prob_fn,
             unconstraining_bijector):
        self.target_log_prob_fn = target_log_prob_fn
        self.unconstraining_bijector = unconstraining_bijector
        return key

    def generate_initial_state(self, key):
        return key

    def generate_step_size(self, key):
        return key

    def cond_fn(self, val):
        _, _, _, _, passed = val
        return jnp.logical_not(jnp.all(passed))

    def body_fn(self, val):
        key, M, theta, sample, passed = val
        key, *subkeys = jax.random.split(key, num=4)
        actions = self.policy.sample(subkeys[0], theta, self.actions_shape)
        policy_pdf = self.policy.pdf(subkeys[1], theta, actions)[..., 0]
        instrumental_pdf = jnp.exp(self.target_log_prob_fn(actions))

        u = jax.random.uniform(subkeys[2], shape=policy_pdf.shape)
        acceptance_criterion = u < instrumental_pdf / (M * policy_pdf)

        # accept if meet criterion and not previously accepted
        acceptance_criterion = jnp.logical_and(acceptance_criterion,
                                               jnp.logical_not(passed))
        # mark newly accepted actions
        passed = jnp.logical_or(passed, acceptance_criterion)

        acceptance_criterion = acceptance_criterion.reshape(self.action_dim,2,1,1)
        sample = jnp.where(acceptance_criterion, actions, sample)
        return (key, M, theta, sample, passed)

    def sample(self, key, theta):
        def _gen_one_sample_per_param(key, theta, M):
            sample = jnp.empty(shape=self.sample_shape)
            passed = jnp.zeros(shape=(self.action_dim, 2)).astype(bool)
            init_val = (key, M, theta, sample, passed)
            _, _, _, sample, _ = jax.lax.while_loop(self.cond_fn, self.body_fn, init_val)
            return sample

        key, *subkeys = jax.random.split(key, num=self.batch_size+1)
        subkeys = jnp.asarray(subkeys)
        samples = jax.jit(jax.vmap(_gen_one_sample_per_param, (0, None, None), 0))(
            subkeys, theta, self.config['rejection_threshold'])

        return key, samples, jnp.array([1])
