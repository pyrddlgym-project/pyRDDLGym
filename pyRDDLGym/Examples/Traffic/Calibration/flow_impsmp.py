import jax
import optax
import jax.numpy as jnp
import numpy as np
import haiku as hk
from tensorflow_probability.substrates import jax as tfp

import functools
from time import perf_counter as timer
from datetime import datetime
import json

import pyRDDLGym.Examples.Traffic.Calibration.instances.inflow_calibration_models as models

from jax.config import config as jconfig
useGPU = True
platform_name = 'gpu' if useGPU else 'cpu'
use64bit = True # tfp.bijectors.SoftClip seems to have a bug with float64
jconfig.update('jax_debug_nans', True)
jconfig.update('jax_platform_name', platform_name)
jconfig.update('jax_enable_x64', use64bit)
jnp.set_printoptions(
    linewidth=9999,
    formatter={'float': lambda x: "{0:0.3f}".format(x)})


SEEDS = (3264, 42, 1010, 1234, 91291)
from sys import argv

key = jax.random.PRNGKey(SEEDS[int(argv[2])])

# The batched and sequential rollout samplers have different
# shapes of the argument and return arrays. (The batched versions
# have an additional dimension for the batch.) Because jit-compilation
# requires static array shapes, the two sampler types are compiled
# separately

# number of shards per batch (e.g. for splitting the computation on a GPU)
n_shards = 4
# number of rollouts per shard
n_rollouts_per_shard = int(argv[1])
# sample size
batch_size = n_rollouts_per_shard * n_shards

eval_n_rollouts = 32


# true inflow rate at each source
# (the number of inflow rates may be <= number of sources)
true_rates = jnp.array([0.3, 0.2, 0.1, 0.4, 0.1, 0.2] * 10)
true_rates = true_rates[:12]
action_dim = len(true_rates)

model_cls = models.InflowCalibration3x3Model

hmc_model = model_cls(action_dim)
base_model = model_cls(action_dim)
eval_model = model_cls(action_dim)


key, subkey = jax.random.split(key)

hmc_model.compile_relaxed(
    n_rollouts=1,
    use64bit=True)
base_model.compile(
    n_rollouts=n_rollouts_per_shard,
    use64bit=True)
eval_model.compile(
    n_rollouts=eval_n_rollouts,
    use64bit=True)

hmc_model.compute_ground_truth(subkey, true_rates)
base_model.compute_ground_truth(subkey, true_rates)
eval_model.compute_ground_truth(subkey, true_rates)



def parametrized_policy(input):
    """The parametrized policy pi_theta.

    Currently, the policy is assumed to be parameterized as a multivariate
    normal policy N(0, Sigma) with 0 mean and a diagonal covariance matrix Sigma.
    This parametrization is unconstrained, and the constraints are later enforced
    using a smooth bijection (actually diffeomorphism).

    The parametrization is currently a MLP with 32x32 hidden nodes.

    The dm-haiku library stores the parameters theta structured as a dictionary,
    with the weights of each layer and the biases of each layer getting
    a separate dictionary key. To work with theta, it is often necessary
    to make use of the jax.tree_util module."""

    mlp = hk.Sequential([
#        hk.Linear(32), jax.nn.relu,
#        hk.Linear(32), jax.nn.relu,
        hk.Linear(2, with_bias=False)
    ])
    output = mlp(input)
    mean, cov = (jnp.squeeze(x) for x in jnp.split(output, 2, axis=1))
    return mean, jnp.diag(jax.nn.softplus(cov))

policy = hk.transform(parametrized_policy)
policy_apply = jax.jit(policy.apply)
key, subkey = jax.random.split(key)
one_hot_inputs = jnp.eye(action_dim)
theta = policy.init(subkey, one_hot_inputs)

n_params = sum(leaf.flatten().shape[0] for leaf in jax.tree_util.tree_leaves(theta))


# Set up the bijector
#bijector_obj = tfp.bijectors.IteratedSigmoidCentered()
class SimplexBijector(tfp.bijectors.IteratedSigmoidCentered):
    # Wraps the IteratedSigmoidCentered bijector, adding
    # a projection onto the first N (out of N+1) coordinates
    def __init__(self, action_dim, max_rate):
        super().__init__()
        self.action_dim = action_dim
        self.max_rate = max_rate

        self._inverse_jac = jax.jacrev(self._inverse)

    def _forward(self, x):
        y = super()._forward(x[..., jnp.newaxis])
        return y[..., 0] * self.max_rate

    def _inverse(self, y):
        y = (y/self.max_rate)[..., jnp.newaxis]
        s = jnp.sum(y, axis=-1)[..., jnp.newaxis]
        return super()._inverse(
            jnp.concatenate((y,s), axis=-1))[..., 0]

    def _inverse_det_jacobian(self, y):
        y = jnp.squeeze(y)
        return jnp.abs(jnp.linalg.det(self._inverse_jac(y)))

    def _inverse_log_det_jacobian(self, y):
        return jnp.log(self._inverse_det_jacobian(y))

bijector_obj = SimplexBijector(
    action_dim=action_dim,
    max_rate=0.4)

def policy_pdf(theta, rng, actions):
    mean, cov = policy_apply(theta, rng, one_hot_inputs)
    unconstrained_actions = bijector_obj.inverse(actions)
    normal_pdf = jax.scipy.stats.multivariate_normal.pdf(unconstrained_actions, mean=mean, cov=cov)
    density_correction = jnp.apply_along_axis(bijector_obj._inverse_det_jacobian, axis=1, arr=actions)
    return normal_pdf * density_correction

def policy_sample(theta, rng, n_rollouts):
    mean, cov = policy_apply(theta, rng, one_hot_inputs)
    action_sample = jax.random.multivariate_normal(
        rng, mean, cov,
        shape=(n_rollouts,))
    action_sample = bijector_obj.forward(action_sample)
    return action_sample



epsilon = jnp.array([1e-6], dtype=base_model.compiler.REAL)

def scalar_mult(alpha, v):
    return alpha * v

weighting_map = jax.vmap(
    scalar_mult, in_axes=0, out_axes=0)

def flatten_dJ_shard(dJ_shard, si, per_shard, flat_dJ_shard):
    ip0 = 0
    for leaf in jax.tree_util.tree_leaves(dJ_shard):
        leaf = leaf.reshape(per_shard, -1)
        ip1 = ip0 + leaf.shape[1]
        flat_dJ_shard = flat_dJ_shard.at[si*per_shard:(si+1)*per_shard, ip0:ip1].set(leaf)
        ip0 = ip1
    return flat_dJ_shard


def parse_optimizer(config):
    if config['optimizer'] == 'rmsprop':
        optimizer = optax.rmsprop(learning_rate=config['lr'], momentum=config['momentum'])
    elif config['optimizer'] == 'adam':
        optimizer = optax.adam(learning_rate=config['lr'])
    elif config['optimizer'] == 'adagrad':
        optimizer = optax.adagrad(learning_rate=config['lr'])
    elif config['optimizer'] == 'adabelief':
        optimizer = optax.adabelief(learning_rate=config['lr'])
    elif config['optimizer'] == 'sgd_with_momentum':
        optimizer = optax.sgd(learning_rate=config['lr'], momentum=config['momentum'])
    elif config['optimizer'] == 'sgd':
        optimizer = optax.sgd(learning_rate=config['lr'], momentum=0.)
    else:
        raise ValueError
    return optimizer

# === REINFORCE ===
#reinforce_config = {
#    'optimizer': 'rmsprop',
#    'lr': 1e-3,
#    'momentum': 0.1,
#}

#reinforce_config = {
#    'optimizer': 'sgd',
#    'lr': 1e-2,
#    'verbose': True,
#}

reinforce_config = {
    'optimizer': 'adagrad',
    'lr': 0.7,
    'verbose': True
}

@jax.jit
def reinforce_inner_loop(key, theta):
    """Computes an estimate of dJ^pi over a sample B using the REINFORCE formula

    dJ hat = 1/|B| * sum_{b in B} r(a_b) * grad pi(a_b) / pi(a_b)

    where dJ denotes the gradient of J^pi with respect to theta

    Args:
        key: JAX random key
        theta: Current policy pi parameters
        subs: Values of the RDDL nonfluents used to run the RDDL model

    Returns:
        key: Mutated JAX random key
        dJ_hat: dJ estimator
        batch_stats: Dictionary of statistics for the current sample
    """

    # initialize the estimator
    dJ_hat = jax.tree_util.tree_map(lambda leaf: jnp.zeros(leaf.shape), theta)

    # initialize the batch stats
    batch_stats = {
        'actions': jnp.empty(shape=(batch_size, action_dim)),
        'dJ': jnp.empty(shape=(batch_size, n_params)),
        'dJ_covar': None
    }

    # compute dJ hat for the current sample, dividing the computation up
    # into shards of size n_rollouts_per_shard (for example, this can be
    # useful for fiting the computation into GPU VRAM)
    for si in range(n_shards):
        key, subkey = jax.random.split(key)
        actions = policy_sample(theta, subkey, n_rollouts_per_shard)
        pi = policy_pdf(theta, subkey, actions)
        dpi = jax.jacrev(policy_pdf, argnums=0)(theta, subkey, actions)
        rewards = base_model.compute_loss(subkey, actions)
        factors = rewards / (pi + 1e-6)

        dJ_shard = jax.tree_util.tree_map(lambda dpi_term: weighting_map(factors, dpi_term), dpi)

        # update the dJ estimator dJ_hat
        dJ_shard_find_mean_fn = lambda x: jnp.mean(x, axis=0)
        accumulant = jax.tree_util.tree_map(dJ_shard_find_mean_fn, dJ_shard)
        dJ_hat = jax.tree_util.tree_map(lambda x, y: x+(y/n_shards), dJ_hat, accumulant)

        # collect statistics for the batch
        batch_stats['dJ'] = flatten_dJ_shard(dJ_shard, si, n_rollouts_per_shard, batch_stats['dJ'])
        batch_stats['actions'] = batch_stats['actions'].at[si*n_rollouts_per_shard:(si+1)*n_rollouts_per_shard].set(actions)

    batch_stats['dJ_covar'] = jnp.cov(batch_stats['dJ'], rowvar=False)

    return key, dJ_hat, batch_stats

def update_reinforce_stats(key, it, algo_stats, batch_stats, theta):
    """Updates the REINFORCE statistics using the statistics returned
    from the computation of dJ_hat for the current sample as well as
    the current policy mean/cov """
    dJ_hat = jnp.mean(batch_stats['dJ'], axis=0)

    algo_stats['dJ'][it] = batch_stats['dJ']
    algo_stats['dJ_hat_max'][it] = jnp.max(dJ_hat)
    algo_stats['dJ_hat_min'][it] = jnp.min(dJ_hat)
    algo_stats['dJ_hat_norm'][it] = jnp.linalg.norm(dJ_hat)
    algo_stats['dJ_covar_max'][it] = jnp.max(batch_stats['dJ_covar'])
    algo_stats['dJ_covar_min'][it] = jnp.min(batch_stats['dJ_covar'])
    algo_stats['dJ_covar_diag_max'][it] = jnp.max(jnp.diag(batch_stats['dJ_covar']))
    algo_stats['dJ_covar_diag_min'][it] = jnp.min(jnp.diag(batch_stats['dJ_covar']))

    key, subkey = jax.random.split(key)
    policy_mean, policy_cov = policy_apply(theta, subkey, one_hot_inputs)
    actions = policy_sample(theta, subkey, eval_n_rollouts)
    rewards = eval_model.compute_loss(subkey, actions)

    algo_stats['reward_mean'][it] = jnp.mean(rewards)
    algo_stats['reward_std'][it] = jnp.std(rewards)
    algo_stats['reward_sterr'][it] = algo_stats['reward_std'][it] / jnp.sqrt(eval_n_rollouts)
    algo_stats['policy_mean'][it] = policy_mean
    algo_stats['policy_cov'][it] = jnp.diag(policy_cov)
    algo_stats['transformed_policy_mean'][it] = jnp.squeeze(jnp.mean(batch_stats['actions'], axis=0))
    algo_stats['transformed_policy_cov'][it] = jnp.squeeze(jnp.std(batch_stats['actions'], axis=0))**2
    return key, algo_stats

def print_reinforce_report(it, algo_stats, subt0, subt1):
    """Prints out the results for the current REINFORCE iteration to console"""
    print(f'Iter {it} :: REINFORCE :: Runtime={subt1-subt0}s')
    print(f'Untransformed parametrized policy [Mean, Diag(Cov)] =')
    print(algo_stats['policy_mean'][it])
    print(algo_stats['policy_cov'][it])
    print(f'Transformed action sample statistics [[Means], [StDevs]] =')
    print(algo_stats['transformed_policy_mean'][it])
    print(algo_stats['transformed_policy_cov'][it])
    print(algo_stats['dJ_hat_min'][it], '<= dJ <=', algo_stats['dJ_hat_max'][it])
    print('|dJ_hat|', algo_stats['dJ_hat_norm'][it])
    print(algo_stats['dJ_covar_diag_min'][it], '<= diag(cov(dJ)) <=', algo_stats["dJ_covar_diag_max"][it])
    print(f'Eval. reward={algo_stats["reward_mean"][it]:.3f} \u00B1 {algo_stats["reward_sterr"][it]:.3f}\n')


def reinforce(key, n_iters, theta, config):
    """Runs the REINFORCE algorithm"""
    # initialize stats collection
    algo_stats = {
        'n_iters': n_iters,
        'algorithm': 'REINFORCE',
        'config': config,
        'action_dim': action_dim,
        'batch_size': batch_size,
        'reward_mean': np.empty(shape=(n_iters,)),
        'reward_std': np.empty(shape=(n_iters,)),
        'reward_sterr': np.empty(shape=(n_iters,)),
        'policy_mean': np.empty(shape=(n_iters, action_dim)),
        'policy_cov': np.empty(shape=(n_iters, action_dim)),
        'transformed_policy_mean': np.empty(shape=(n_iters, action_dim)),
        'transformed_policy_cov': np.empty(shape=(n_iters, action_dim)),
        'dJ': np.empty(shape=(n_iters, batch_size, n_params)),
        'dJ_hat_max': np.empty(shape=(n_iters,)),
        'dJ_hat_min': np.empty(shape=(n_iters,)),
        'dJ_hat_norm': np.empty(shape=(n_iters,)),
        'dJ_covar_max': np.empty(shape=(n_iters,)),
        'dJ_covar_min': np.empty(shape=(n_iters,)),
        'dJ_covar_diag_max': np.empty(shape=(n_iters,)),
        'dJ_covar_diag_min': np.empty(shape=(n_iters,)),
    }

    # initialize optimizer
    optimizer = parse_optimizer(config)
    opt_state = optimizer.init(theta)

    # run REINFORCE
    for it in range(n_iters):
        subt0 = timer()

        key, dJ_hat, batch_stats = reinforce_inner_loop(key, theta)
        updates, opt_state = optimizer.update(dJ_hat, opt_state)
        theta = optax.apply_updates(theta, updates)

        # update statistics and print out report for current iteration
        key, algo_stats = update_reinforce_stats(key, it, algo_stats, batch_stats, theta)
        if config.get('verbose', False):
            print_reinforce_report(it, algo_stats, subt0, timer())

    return algo_stats


# === REINFORCE with Importance Sampling ====
#impsmp_config = {
#    'optimizer': 'rmsprop',
#    'lr': 1e-3,
#    'momentum': 0.1,
#}
impsmp_config = {
    'hmc_num_iters': int(batch_size),
    'hmc_step_size_init': 0.1,
    'hmc_num_leapfrog_steps': 30,
    'optimizer': 'sgd',
#    'lr': 3e-3,
#    'lr': 1e-2,
    'optimizer': 'adagrad',
    'lr': float(argv[3]),
}
impsmp_config['hmc_num_burnin_steps'] = int(impsmp_config['hmc_num_iters']/2)

@jax.jit
def unnormalized_rho(key, theta, a):
    dpi = jax.jacrev(policy_pdf, argnums=0)(theta, key, a)
    dpi_norm = jax.tree_util.tree_reduce(lambda x, y: jnp.maximum(x, jnp.max(jnp.abs(y))), dpi, initializer=jnp.array([-jnp.inf]))

    reward = hmc_model.compute_loss(key, a)
    density = jnp.abs(reward[0]) * dpi_norm[0]
    return density

@jax.jit
def unnormalized_log_rho(key, theta, a):
    return jnp.log(unnormalized_rho(key, theta, a) + epsilon)

batch_unnormalized_rho = jax.jit(jax.vmap(
    unnormalized_rho, (None, None, 0), 0))

def cos_sim(A, B):
    return jnp.dot(A, B) / (jnp.linalg.norm(A) * jnp.linalg.norm(B))

@jax.jit
def impsmp_inner_loop(key, theta, samples):
    batch_stats = {
        'sample_rewards': jnp.empty(shape=(batch_size,)),
        'dJ': jnp.empty(shape=(batch_size, n_params)),
    }

    dJ_hat = jax.tree_util.tree_map(lambda theta_item: jnp.zeros(theta_item.shape), theta)
    Zinv = 0.


    for si in range(n_shards):
        key, subkey = jax.random.split(key)
        ifrom, ito = si*n_rollouts_per_shard, (si+1)*n_rollouts_per_shard

        # evaluate the dJ approximation over the full trajectory using subsampling_model
        # (typically, the subsampling_model will be the Relaxed RDDL model)
        actions = samples[ifrom:ito]

        pi = policy_pdf(theta, key, actions)
        dpi = jax.jacrev(policy_pdf, argnums=0)(theta, key, actions)
        rewards = base_model.compute_loss(key, actions)
        rho = batch_unnormalized_rho(key, theta, actions[:,jnp.newaxis,:])
        factors = rewards / (rho + epsilon)
        jax.tree_util.tree_map(print_shapes, dpi)
        scores = jax.tree_util.tree_map(lambda dpi_term: weighting_map(factors, dpi_term), dpi)

        # the mean of scores along axis 0 is equal to
        #    (dJ_fr + dJ_(fr+1) + ... + dJ_to) * (1/n_rollouts_per_shard)
        dJ_hat = jax.tree_util.tree_map(lambda x, y: x+jnp.mean(y, axis=0)/n_shards, dJ_hat, scores)

        # estimate of the normalizing factor Z of the density rho
        Zinv += jnp.sum(pi/rho)

        # collect statistics for the shard
        batch_stats['dJ'] = flatten_dJ_shard(scores, si, n_rollouts_per_shard, batch_stats['dJ'])
        batch_stats['sample_rewards'] = batch_stats['sample_rewards'].at[ifrom:ito].set(rewards)

    # scale by Zinv
    Zinv += epsilon
    dJ_hat = jax.tree_util.tree_map(lambda x: x / Zinv, dJ_hat)
    batch_stats['dJ'] = jax.tree_util.tree_map(lambda x: x / Zinv, batch_stats['dJ'])

    return key, dJ_hat, batch_stats

def update_impsmp_stats(key, it, algo_stats, batch_stats, theta):
    """Updates the REINFORCE with Importance Sampling statistics
    using the statistics returned from the computation of dJ_hat
    for the current sample as well as the current policy params theta"""
    key, subkey = jax.random.split(key)
    policy_mean, policy_cov = policy_apply(theta, subkey, one_hot_inputs)
    eval_actions = policy_sample(theta, subkey, eval_n_rollouts)
    eval_rewards = eval_model.compute_loss(subkey, eval_actions)

    # calculate the covariance matrices
    dJ_hat = jnp.mean(batch_stats['dJ'], axis=0)
    dJ_covar = jnp.cov(batch_stats['dJ'], rowvar=False)

    algo_stats['dJ'][it] = batch_stats['dJ']
    algo_stats['dJ_hat_max'][it] = jnp.max(dJ_hat)
    algo_stats['dJ_hat_min'][it] = jnp.min(dJ_hat)
    algo_stats['dJ_hat_norm'][it] = jnp.linalg.norm(dJ_hat)
    algo_stats['dJ_covar'][it] = dJ_covar
    algo_stats['dJ_covar_max'][it] = jnp.max(dJ_covar)
    algo_stats['dJ_covar_min'][it] = jnp.min(dJ_covar)
    algo_stats['dJ_covar_diag_max'][it] = jnp.max(jnp.diag(dJ_covar))
    algo_stats['dJ_covar_diag_min'][it] = jnp.min(jnp.diag(dJ_covar))

    algo_stats['policy_mean'][it] = policy_mean
    algo_stats['policy_cov'][it] = jnp.diag(policy_cov)
    algo_stats['transformed_policy_mean'][it] = jnp.mean(eval_actions, axis=0)
    algo_stats['transformed_policy_cov'][it] = jnp.std(eval_actions, axis=0)**2
    algo_stats['reward_mean'][it] = jnp.mean(eval_rewards)
    algo_stats['reward_std'][it] = jnp.std(eval_rewards)
    algo_stats['reward_sterr'][it] = algo_stats['reward_std'][it] / jnp.sqrt(eval_n_rollouts)

    algo_stats['sample_rewards'][it] = batch_stats['sample_rewards']
    algo_stats['sample_reward_mean'][it] = jnp.mean(batch_stats['sample_rewards'])
    algo_stats['sample_reward_std'][it] = jnp.std(batch_stats['sample_rewards'])
    algo_stats['sample_reward_sterr'][it] = algo_stats['sample_reward_std'][it] / jnp.sqrt(batch_size)
    algo_stats['acceptance_rate'][it] = batch_stats['acceptance_rate']
    algo_stats['hmc_step_size'][it] = batch_stats['hmc_step_size']

    return key, algo_stats

def print_impsmp_report(it, algo_stats, is_accepted, subt0, subt1):
    """Prints out the results for the current REINFORCE with Importance Sampling iteration"""
    print(f'Iter {it} :: Importance Sampling :: Runtime={subt1-subt0}s')
    print(f'HMC step size={algo_stats["hmc_step_size"][it]:.4f} :: HMC acceptance rate={algo_stats["acceptance_rate"][it]*100:.2f}%')
    print('Untransformed parametrized policy [Mean, Diag(Cov)] =')
    print(algo_stats['policy_mean'][it])
    print(algo_stats['policy_cov'][it])
    print('Transformed action sample statistics [Mean, StDev] =')
    print(algo_stats['transformed_policy_mean'][it])
    print(algo_stats['transformed_policy_cov'][it])
    print(f'{algo_stats["dJ_hat_min"][it]} <= dJ_hat <= {algo_stats["dJ_hat_max"][it]} :: Norm={algo_stats["dJ_hat_norm"][it]}')
    print(algo_stats['dJ_covar_diag_min'][it], '<= diag(cov(dJ)) <=', algo_stats['dJ_covar_diag_max'][it])
    print(f'HMC sample reward={algo_stats["sample_reward_mean"][it]:.3f} \u00B1 {algo_stats["sample_reward_sterr"][it]:.3f} '
          f':: Eval reward={algo_stats["reward_mean"][it]:.3f} \u00B1 {algo_stats["reward_sterr"][it]:.3f}\n')


def impsmp(key, n_iters, theta, config):
    """Runs the REINFORCE with Importance Sampling algorithm"""
    # initialize stats collection
    algo_stats = {
        'algorithm': 'ImpSmp',
        'config': config,
        'n_iters': n_iters,
        'batch_size': batch_size,
        'action_dim': action_dim,
        'policy_mean': np.empty(shape=(n_iters, action_dim)),
        'policy_cov': np.empty(shape=(n_iters, action_dim)),
        'transformed_policy_mean': np.empty(shape=(n_iters, action_dim)),
        'transformed_policy_cov': np.empty(shape=(n_iters, action_dim)),
        'reward_mean': np.empty(shape=(n_iters,)),
        'reward_std': np.empty(shape=(n_iters,)),
        'reward_sterr': np.empty(shape=(n_iters,)),
        'sample_rewards': np.empty(shape=(n_iters, batch_size)),
        'sample_reward_mean': np.empty(shape=(n_iters,)),
        'sample_reward_std': np.empty(shape=(n_iters,)),
        'sample_reward_sterr': np.empty(shape=(n_iters,)),
        'acceptance_rate': np.empty(shape=(n_iters,)),
        'hmc_step_size': np.empty(shape=(n_iters,)),
        'dJ': np.empty(shape=(n_iters, batch_size, n_params)),
        'dJ_hat_max': np.empty(shape=(n_iters,)),
        'dJ_hat_min': np.empty(shape=(n_iters,)),
        'dJ_hat_norm': np.empty(shape=(n_iters,)),
        'dJ_covar': np.empty(shape=(n_iters, n_params, n_params)),
        'dJ_covar_max': np.empty(shape=(n_iters,)),
        'dJ_covar_min': np.empty(shape=(n_iters,)),
        'dJ_covar_diag_max': np.empty(shape=(n_iters,)),
        'dJ_covar_diag_min': np.empty(shape=(n_iters,)),
    }

    # initialize HMC
    key, subkey = jax.random.split(key)
    hmc_initializer = jax.random.uniform(
        subkey,
        shape=(1,action_dim),
        minval=0.05, maxval=0.35)
    hmc_step_size = config['hmc_step_size_init']

    # initialize optimizer
    optimizer = parse_optimizer(config)
    opt_state = optimizer.init(theta)

    # initialize unconstraining bijector
    unconstraining_bijector = [
        bijector_obj
    ]

    # run REINFORCE with Importance Sampling
    for it in range(n_iters):
        subt0 = timer()
        key, *subkeys = jax.random.split(key, num=5)

        log_density = functools.partial(
            unnormalized_log_rho, subkeys[1], theta)

        adaptive_hmc_kernel = tfp.mcmc.TransformedTransitionKernel(
            inner_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
                inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
                    target_log_prob_fn=log_density,
                    num_leapfrog_steps=config['hmc_num_leapfrog_steps'],
                    step_size=hmc_step_size),
                num_adaptation_steps=int(config['hmc_num_burnin_steps'] * 0.8)),
            bijector=unconstraining_bijector)

        samples, is_accepted = tfp.mcmc.sample_chain(
            seed=subkeys[2],
            num_results=config['hmc_num_iters'],
            num_burnin_steps=config['hmc_num_burnin_steps'],
            current_state=hmc_initializer,
            kernel=adaptive_hmc_kernel,
            trace_fn=lambda _, pkr: pkr.inner_results.inner_results.is_accepted)

        # compute dJ_hat
        samples = jnp.squeeze(samples)
        key, dJ_hat, batch_stats = impsmp_inner_loop(
            key, theta, samples)

        updates, opt_state = optimizer.update(dJ_hat, opt_state)
        theta = optax.apply_updates(theta, updates)

        # initialize the next chain at a random point of the current chain
        hmc_intializer = jax.random.choice(subkeys[3], samples)

        # update stats and printout
        batch_stats['acceptance_rate'] = jnp.mean(is_accepted)
        batch_stats['hmc_step_size'] = hmc_step_size
        key, algo_stats = update_impsmp_stats(key, it, algo_stats, batch_stats, theta)
        print_impsmp_report(it, algo_stats, is_accepted, subt0, timer())
    return algo_stats

# === Debugging utils ===
def print_shapes(x):
    print(x.shape)

def eval_single_rollout(a):
    a = jnp.asarray(a)
    rewards = reward_seqtl(key, subs_hmc, a)
    return rewards

class SimpleNumpyToJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

if __name__ == '__main__':
    method = 'impsmp'
    assert method in ('reinforce', 'impsmp')
    n_iters = 75

    # if disable=True, runs without jit (much slower, but can inspect the data
    # in the middle of a jitted computation). If disable=False, runs with jit
    with jax.disable_jit(disable=False):
        key, subkey = jax.random.split(key)
        if method == 'impsmp':
            algo_stats = impsmp(key=subkey, n_iters=n_iters, theta=theta, config=impsmp_config)
        elif method == 'reinforce':
            algo_stats = reinforce(key=subkey, n_iters=n_iters, theta=theta, config=reinforce_config)
        else: raise KeyError

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f'{timestamp}_{method}_iters{n_iters}_b{batch_size}'
    path = f'tmp/{filename}.json'

    with open(path, 'w') as file:
        json.dump(algo_stats, file, cls=SimpleNumpyToJSONEncoder)
    print('Saved results to', path)
