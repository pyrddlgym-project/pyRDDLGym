import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

VALID_PROPOSAL_PDF_TYPES = (
    'cur_policy',
    'uniform',
)
VALID_SHAPE_TYPES = (
    'one_sample_per_parameter',
    'one_sample_per_dJ_summand',
)
VALID_REJECTION_RATE_TYPES = (
    'constant',
    'linear_ramp',
)

def rejection_rate_schedule(it, type_, params):
    if type_ == 'constant':
        return params['value']
    elif type_ == 'linear_ramp':
        return params['from'] + it * params['delta']


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

        self.proposal_pdf_type = self.config['proposal_pdf_type']
        assert self.proposal_pdf_type in VALID_PROPOSAL_PDF_TYPES

        self.shape_type = self.config['sample_shape_type']
        assert self.shape_type in VALID_SHAPE_TYPES

        self.rejection_rate_type = self.config['rejection_rate']['type']
        assert self.rejection_rate_type in VALID_REJECTION_RATE_TYPES
        self.rejection_rate_params = self.config['rejection_rate']['params']
        if self.rejection_rate_type == 'linear_ramp':
            self.rejection_rate_params['delta'] = (self.rejection_rate_params['to'] - self.rejection_rate_params['from']) / n_iters
        self.rejection_rate = None

        self.n_distinct_samples_used_buffer = []
        self.stats = {
            'n_distinct_samples_used': []
        }

    def prep(self,
             key,
             it,
             target_log_prob_fn,
             unconstraining_bijector):
        self.target_log_prob_fn = target_log_prob_fn
        self.unconstraining_bijector = unconstraining_bijector
        self.rejection_rate = rejection_rate_schedule(it, self.rejection_rate_type, self.rejection_rate_params)
        return key

    def generate_initial_state(self, key, it, samples):
        """Included to have a consistent interface with that of HMC"""
        return key

    def generate_step_size(self, key):
        """Included to have a consistent interface with that of HMC"""
        return key

    def cond_fn(self, val):
        _, _, _, _, _, is_sampled_matrix = val
        return jnp.logical_not(jnp.all(is_sampled_matrix))

    def body_fn(self, val):
        key, M, theta, samples, n, is_sampled_matrix = val
        key, *subkeys = jax.random.split(key, num=4)

        if self.proposal_pdf_type == 'cur_policy':
            single_proposed_action = self.policy.sample(subkeys[0], theta, (1,))
            proposal_density_val = self.policy.pdf(subkeys[1], theta, single_proposed_action)
        elif self.proposal_pdf_type == 'uniform':
            minval, maxval = -8.0, 8.0
            single_proposed_action = jax.random.uniform(subkeys[1], shape=(self.action_dim,), minval=minval, maxval=maxval)
            proposal_density_val = (1/(maxval - minval))**(self.action_dim) * jnp.ones(shape=(1,))

        if self.shape_type == 'one_sample_per_parameter':
            # stack multiple copies of the proposed action to test against
            # the acceptance criterion for each instrumental density
            proposed_action = jnp.stack([jnp.stack([single_proposed_action] * 2)] * self.action_dim)
        elif self.shape_type == 'one_sample_per_dJ_summand':
            proposed_action = single_proposed_action

        instrumental_density_val = jnp.exp(self.target_log_prob_fn(proposed_action))

        # sample independent uniform variables to test the acceptance criterion
        u = jax.random.uniform(subkeys[2], shape=instrumental_density_val.shape)
        acceptance_criterion = u < (instrumental_density_val / (M * proposal_density_val))

        # accept if meet criterion and not previously accepted
        acceptance_criterion = jnp.logical_and(acceptance_criterion,
                                               jnp.logical_not(is_sampled_matrix))

        # update used sample count
        n = n + jnp.any(acceptance_criterion)

        # mark newly accepted actions
        is_sampled_matrix = jnp.logical_or(is_sampled_matrix, acceptance_criterion)

        if self.shape_type == 'one_sample_per_parameter':
            acceptance_criterion = jnp.broadcast_to(
                acceptance_criterion[:, :, jnp.newaxis, jnp.newaxis],
                shape=(self.action_dim, 2, 1, self.action_dim))

        samples = jnp.where(acceptance_criterion, proposed_action, samples)

        return (key, M, theta, samples, n, is_sampled_matrix)

    def sample(self, key, theta):
        def _accept_reject(key, theta, M):
            if self.shape_type == 'one_sample_per_parameter':
                samples = jnp.empty(shape=(self.action_dim, 2, 1, self.action_dim))
                is_sampled_matrix = jnp.zeros(shape=(self.action_dim, 2)).astype(bool)
            elif self.shape_type == 'one_sample_per_dJ_summand':
                samples = jnp.empty(shape=(1, self.action_dim))
                is_sampled_matrix = jnp.zeros(shape=(1,)).astype(bool)

            init_val = (key, M, theta, samples, 0, is_sampled_matrix)
            _, _, _, samples, n_distinct, _ = jax.lax.while_loop(self.cond_fn, self.body_fn, init_val)
            return samples, n_distinct

        # run rejection sampling over a batch
        key, *batch_subkeys = jax.random.split(key, num=self.batch_size+1)
        batch_subkeys = jnp.asarray(batch_subkeys)
        samples, n_distinct = jax.jit(jax.vmap(_accept_reject, (0, None, None), 0))(
            batch_subkeys, theta, self.rejection_rate)

        # keep in buffer until statistics for current iteration
        # are updated
        self.n_distinct_samples_used_buffer.append(n_distinct[0])

        return key, samples, None

    def update_stats(self, it, samples, is_accepted):
        self.stats['n_distinct_samples_used'].extend(self.n_distinct_samples_used_buffer)
        self.n_distinct_samples_used_buffer.clear()

    def print_report(self, it):
        print(f'Rejection Sampler'
              f' :: Batch={self.batch_size}'
              f' :: Rej.rate type={self.rejection_rate_type},'
              f' cur.val.={self.rejection_rate}'
              f' :: Proposal pdf={self.config["proposal_pdf_type"]}'
              f' :: # Distinct samples={self.stats["n_distinct_samples_used"][-1]}')
