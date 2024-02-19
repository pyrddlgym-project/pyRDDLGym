import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
from pyRDDLGym.PolicyGradient.samplers.rejection_sampler import FixedNumTrialsRejectionSampler

VALID_INIT_DISTR_TYPES = (
    'uniform',
    'normal',
)

VALID_STEP_SIZE_DISTR_TYPES = (
    'constant',
    'uniform',
    'discrete_uniform',
    'log_uniform',
)

VALID_REINIT_STRATEGY_TYPES = (
    'random_sample',
    'random_prev_chain_elt',
    'cur_policy',
    'rejection_sampler',
)

def compute_next_sample_corr(D):
    sample_size = len(D)
    D = D - jnp.mean(D)
    K = jnp.correlate(D, D, 'full')
    K = K / K[sample_size-1]
    return K[sample_size]



class HMCSampler:
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

        self.config['num_iters_per_chain'] = int(batch_size / self.config['num_chains'])
        assert self.config['num_iters_per_chain'] > 0, (
            f'[HMCSampler] Please check that batch_size >= sampler["num_chains"]. '
            f'batch_size={batch_size}, sampler["num_chains"]={config["num_chains"]}')

        if self.config['init_distribution']['type'] == 'normal':
            self.config['init_distribution']['std'] = jnp.sqrt(self.config['init_distribution']['var'])

        self.reinit_strategy = self.config['reinit_strategy']

        assert self.config['init_distribution']['type'] in VALID_INIT_DISTR_TYPES
        assert self.config['step_size_distribution']['type'] in VALID_STEP_SIZE_DISTR_TYPES
        assert self.config['reinit_strategy']['type'] in VALID_REINIT_STRATEGY_TYPES

        self.divergence_threshold = self.config.get('divergence_threshold', 10.)
        self.track_next_sample_correlation = self.config.get('track_next_sample_correlation', False)

        self.stats = {
            'step_size': jnp.empty(shape=(n_iters,)),
            'acceptance_rate': jnp.empty(shape=(n_iters,)),
            'num_divergent_chains': jnp.empty(shape=(n_iters,)),
        }

        if self.track_next_sample_correlation:
            self.stats.update({
                'next_sample_correlation_per_chain': jnp.empty(shape=(n_iters, self.config['num_chains'], action_dim, 2, action_dim)),
                'next_sample_correlation_min': jnp.empty(shape=(n_iters,)),
                'next_sample_correlation_mean': jnp.empty(shape=(n_iters,)),
                'next_sample_correlation_max': jnp.empty(shape=(n_iters,)),
            })

        # @@@@@
        self.rej_subsampler = FixedNumTrialsRejectionSampler(
            n_iters,
            1,
            action_dim,
            policy,
            config={
                'proposal_pdf_type': 'cur_policy',
                'sample_shape_type': 'one_sample_per_parameter',
                'rejection_rate_schedule': {
                    'type': 'constant',
                    'params': {
                        'value': 250
                    }
                }
            })
         #@@@@@

    def generate_initial_state(self, key, it, prev_samples):
        key, subkey = jax.random.split(key)

        #@@@@
        if it == 0:
            shape = (self.config['num_chains'], self.action_dim, 2, 1, self.action_dim)
            config = self.config['init_distribution']
            if config['type'] == 'uniform':
                self.init_state = jax.random.uniform(
                    subkey,
                    shape=shape,
                    minval=config['min'],
                    maxval=config['max'])
            elif config['type'] == 'normal':
                self.init_state = jax.random.normal(
                    subkey,
                    shape=shape)
                self.init_state = config['mean'] + self.init_state * config['std']

        #@@@@
        else:
            if self.config['reinit_strategy']['type'] == 'random_prev_chain_elt':
                self.init_state = jax.random.choice(
                    subkey,
                    a=prev_samples,
                    shape=(self.config['num_chains'],),
                    replace=False)

            elif self.config['reinit_strategy']['type'] == 'random_sample':
                shape = (self.config['num_chains'], self.action_dim, 2, 1, self.action_dim)
                config = self.config['init_distribution']
                if config['type'] == 'uniform':
                    self.init_state = jax.random.uniform(
                        subkey,
                        shape=shape,
                        minval=config['min'],
                        maxval=config['max'])
                elif config['type'] == 'normal':
                    self.init_state = jax.random.normal(
                        subkey,
                        shape=shape)
                    self.init_state = config['mean'] + self.init_state * config['std']

            elif self.config['reinit_strategy']['type'] == 'cur_policy':
                shape = (self.config['num_chains'], self.action_dim, 2, 1, self.action_dim)
                self.init_state = self.policy.sample(subkey, self.policy.theta, shape[:-1])

            elif self.config['reinit_strategy']['type'] == 'rejection_sampler':
                key, self.init_state, _ = self.rej_subsampler.sample(key, self.policy.theta)

        return key

    def set_initial_state(self, state):
        self.init_state = state

    def generate_step_size(self, key):
        key, subkey = jax.random.split(key)

        config = self.config['step_size_distribution']
        if config['type'] == 'constant':
            self.step_size = config['value']
        elif config['type'] == 'uniform':
            self.step_size = jax.random.uniform(
                subkey,
                minval=config['min'],
                maxval=config['max'])
        elif config['type'] == 'discrete_uniform':
            index = jax.random.randint(
                subkey,
                shape=(),
                minval=0, maxval=len(config['values']))
            self.step_size = config['values'][index]
        elif config['type'] == 'log_uniform':
            log_step_size = jax.random.uniform(
                subkey,
                shape=(),
                minval=jnp.log(config['min']),
                maxval=jnp.log(config['max']))
            self.step_size = jnp.exp(log_step_size)
        return key


    def prep(self,
             key,
             it,
             target_log_prob_fn,
             unconstraining_bijector):

        num_leapfrog_steps = self.config['num_leapfrog_steps']
        num_burnin_steps = self.config['num_burnin_iters_per_chain']
        num_adaptation_steps = self.config.get('num_adaptation_steps')

        if num_adaptation_steps is None:
            num_adaptation_steps = int(num_burnin_steps * 0.8)

        parallel_log_density_over_chains = jax.vmap(target_log_prob_fn, 0, 0)

        hmc_sampler = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=parallel_log_density_over_chains,
            step_size=self.step_size,
            num_leapfrog_steps=num_leapfrog_steps)

        hmc_sampler_with_adaptive_step_size = tfp.mcmc.SimpleStepSizeAdaptation(
            inner_kernel=hmc_sampler,
            num_adaptation_steps=num_adaptation_steps)

        self.sampler = tfp.mcmc.TransformedTransitionKernel(
            inner_kernel=hmc_sampler_with_adaptive_step_size,
            bijector=unconstraining_bijector)

        #@@@@
        key = self.rej_subsampler.prep(key, it, target_log_prob_fn, unconstraining_bijector)
        #@@@@
        return key

    def sample(self, key, theta):
        key, subkey = jax.random.split(key)
        return key, *tfp.mcmc.sample_chain(
            seed=subkey,
            num_results=self.config['num_iters_per_chain'],
            num_burnin_steps=self.config['num_burnin_iters_per_chain'],
            current_state=self.init_state,
            kernel=self.sampler,
            trace_fn=lambda _, pkr: pkr.inner_results.inner_results.is_accepted)

    def update_stats(self, it, samples, is_accepted):
        self.stats['acceptance_rate'] = self.stats['acceptance_rate'].at[it].set(jnp.mean(is_accepted))
        self.stats['step_size'] = self.stats['step_size'].at[it].set(self.step_size)

        num_chains = self.config['num_chains']
        num_samples_per_chain = self.config['num_iters_per_chain']

        samples = samples.reshape(num_chains, num_samples_per_chain, self.action_dim, 2, self.action_dim)
        num_divergent_samples_per_chain = jnp.sum(jnp.abs(samples) > self.divergence_threshold, axis=1)
        num_divergent_chains = jnp.sum(num_divergent_samples_per_chain > 0)
        self.stats['num_divergent_chains'] = self.stats['num_divergent_chains'].at[it].set(num_divergent_chains)

        if self.track_next_sample_correlation:
            next_sample_correlation = jnp.apply_along_axis(
                compute_next_sample_corr,
                axis=1,
                arr=samples)
            self.stats['next_sample_correlation_per_chain'] = self.stats['next_sample_correlation_per_chain'].at[it].set(next_sample_correlation)
            self.stats['next_sample_correlation_min'] = self.stats['next_sample_correlation_min'].at[it].set(next_sample_correlation)
            self.stats['next_sample_correlation_mean'] = self.stats['next_sample_correlation_mean'].at[it].set(next_sample_correlation)
            self.stats['next_sample_correlation_max'] = self.stats['next_sample_correlation_max'].at[it].set(next_sample_correlation)

    def print_report(self, it):
        print(f'HMC :: Batch={self.batch_size} :: Chains={self.config["num_chains"]} :: Init.distr={self.config["init_distribution"]["type"]}')
        print(f'       Burnin={self.config["num_burnin_iters_per_chain"]} :: Step size={self.stats["step_size"][it]} :: Num.leapfrog={self.config["num_leapfrog_steps"]}')
        print(f'       Acceptance rate={self.stats["acceptance_rate"][it]} :: Num.div.chains={self.stats["num_divergent_chains"][it]}')
        if self.track_next_sample_correlation:
            print(f'Next sample corr.: {self.stats["next_sample_correlation_min"][it]} <= (Mean) {self.stats["next_sample_correlation_mean"][it]} <= {self.stats["next_sample_correlation_max"][it]}')


class NoUTurnSampler(HMCSampler):
    def prep(self,
             key,
             it,
             target_log_prob_fn,
             unconstraining_bijector):
        num_burnin_iters_per_chain = self.config['num_burnin_iters_per_chain']
        num_adaptation_steps = self.config.get('num_adaptation_steps')
        max_tree_depth = self.config['max_tree_depth']

        if num_adaptation_steps is None:
            num_adaptation_steps = int(num_burnin_iters_per_chain * 0.8)

        parallel_log_density_over_chains = jax.vmap(target_log_prob_fn, 0, 0)

        nuts = tfp.mcmc.NoUTurnSampler(
            target_log_prob_fn=parallel_log_density_over_chains,
            step_size=self.step_size,
            max_tree_depth=max_tree_depth)

        nuts_with_adaptive_step_size = tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=nuts,
            num_adaptation_steps=num_adaptation_steps)

        self.sampler = tfp.mcmc.TransformedTransitionKernel(
            inner_kernel=nuts_with_adaptive_step_size,
            bijector=unconstraining_bijector)

        key = self.rej_subsampler.prep(key, it, target_log_prob_fn, unconstraining_bijector)
        return key

    def print_report(self, it):
        print(f'NUTS :: Batch={self.batch_size} :: Chains={self.config["num_chains"]} :: Init.distr={self.config["init_distribution"]["type"]}')
        print(f'        Burnin={self.config["num_burnin_iters_per_chain"]} :: Step size={self.stats["step_size"][it]} :: Max.tree depth={self.config["max_tree_depth"]}')
        print(f'        Acceptance rate={self.stats["acceptance_rate"][it]} :: Num.div.chains={self.stats["num_divergent_chains"][it]}')
