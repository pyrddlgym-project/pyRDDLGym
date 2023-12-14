import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

VALID_SAMPLING_PDF_TYPES = (
    'cur_policy',
    'uniform',
)

class RejectionSampler:
    def __init__(self,
                 n_iters,
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

        self.sampling_pdf_type = self.config['sampling_pdf_type']
        assert self.sampling_pdf_type in VALID_SAMPLING_PDF_TYPES

        self.sample_shape = (self.action_dim, 2,
                             1, self.action_dim)
        self.actions_shape = self.sample_shape[:-1]
        self.batch_shape = (self.batch_size,) + self.sample_shape

        self.stats = {}

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
        if self.sampling_pdf_type == 'cur_policy':
            actions = self.policy.sample(subkeys[0], theta, self.actions_shape)
            sampling_pdf = self.policy.pdf(subkeys[1], theta, actions)[..., 0]
        elif self.sampling_pdf_type == 'uniform':
            actions = jax.random.uniform(subkeys[1], shape=self.sample_shape, minval=-5.0, maxval=5.0)
            sampling_pdf = 0.1**(self.action_dim) * jnp.ones(shape=(self.action_dim, 2))
        instrumental_pdf = jnp.exp(self.target_log_prob_fn(actions))

        u = jax.random.uniform(subkeys[2], shape=sampling_pdf.shape)
        acceptance_criterion = u < instrumental_pdf / (M * sampling_pdf)

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
            subkeys, theta, self.config['rejection_rate'])

        return key, samples, None

    def update_stats(self, it, samples, is_accepted):
        """Included to have a consistent interface with that of HMC"""
        pass

    def print_report(self, it):
        print(f'Rejection Sampler :: Batch={self.batch_size} :: Rej.rate={self.config["rejection_rate"]}')
