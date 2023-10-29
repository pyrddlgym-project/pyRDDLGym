import jax
import optax
import jax.numpy as jnp
import numpy as np
import haiku as hk
from tensorflow_probability.substrates import jax as tfp

import functools
from sys import argv
from time import perf_counter as timer
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import json
from math import ceil

from pyRDDLGym import ExampleManager
from pyRDDLGym import RDDLEnv
from pyRDDLGym.Core.Jax.JaxRDDLLogic import FuzzyLogic
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxRDDLCompilerWithGrad

t0 = timer()

from jax.config import config as jconfig
useGPU = False
platform_name = 'gpu' if useGPU else 'cpu'
use64bit = True # tfp.bijectors.SoftClip seems to have a bug with float64
jconfig.update('jax_debug_nans', True)
jconfig.update('jax_platform_name', platform_name)
jconfig.update('jax_enable_x64', use64bit)
jnp.set_printoptions(
    linewidth=9999,
    formatter={'float': lambda x: "{0:0.3f}".format(x)})


key = jax.random.PRNGKey(3264)


# specify the model
EnvInfo = ExampleManager.GetEnvInfo('traffic4phase')
myEnv = RDDLEnv.RDDLEnv(domain=EnvInfo.get_domain(),
#                        instance='instances/instance01.rddl')
                        instance='instances/instance_2x2.rddl')

model = myEnv.model
N = myEnv.numConcurrentActions

init_state_subs = myEnv.sampler.subs
t_warmup = 0
t_plan = myEnv.horizon
rollout_horizon = t_plan - t_warmup

compiler = JaxRDDLCompilerWithGrad(
    rddl=model,
    use64bit=use64bit,
    logic=FuzzyLogic(weight=15))
compiler.compile()


# Define policy fn
# Every traffic light is assumed to follow a fixed-time plan
all_red_stretch = [1.0, 0.0, 0.0, 0.0]
protected_left_stretch = [1.0] + [0.0]*19
through_stretch = [1.0] + [0.0]*59
full_cycle = (all_red_stretch + protected_left_stretch + all_red_stretch + through_stretch)*2
BASE_PHASING = jnp.array(full_cycle*10, dtype=compiler.REAL)
FIXED_TIME_PLAN = jnp.broadcast_to(
    BASE_PHASING[..., jnp.newaxis],
    shape=(BASE_PHASING.shape[0], N))

def policy_fn(key, policy_params, hyperparams, step, states):
    return {'advance': FIXED_TIME_PLAN[step]}


# The batched and sequential rollout samplers have different
# shapes of the argument and return arrays. (The batched versions
# have an additional dimension for the batch.) Because jit-compilation
# requires static array shapes, the two sampler types are compiled
# separately

# number of shards per batch (e.g. for splitting the computation on a GPU)
n_shards = 1
# number of rollouts per shard
n_rollouts_per_shard = 128
batch_size = n_rollouts_per_shard * n_shards

sampler = compiler.compile_rollouts(policy=policy_fn,
                                    n_steps=rollout_horizon,
                                    n_batch=n_rollouts_per_shard)
sampler_hmc = compiler.compile_rollouts(policy=policy_fn,
                                        n_steps=rollout_horizon,
                                        n_batch=1)


# Set up initial states
subs_train = {}
subs_hmc = {}
for (name, value) in init_state_subs.items():
    value = jnp.array(value)[jnp.newaxis, ...]
    train_value = jnp.repeat(value, repeats=n_rollouts_per_shard, axis=0)
    train_value = train_value.astype(compiler.REAL)
    subs_train[name] = train_value

    hmc_value = jnp.repeat(value, repeats=1, axis=0)
    hmc_value = hmc_value.astype(compiler.REAL)
    subs_hmc[name] = hmc_value
for (state, next_state) in model.next_state.items():
    subs_train[next_state] = subs_train[state]
    subs_hmc[next_state] = subs_hmc[state]


# Set up ground truth inflow rates
source_indices = range(16,24)
true_source_rates = jnp.array([0.3, 0.2, 0.1, 0.4, 0.1, 0.2, 0.3])
num_nonzero_sources = 7
source_rates = true_source_rates[:num_nonzero_sources]
s0, s1, s2 = source_indices[0], source_indices[num_nonzero_sources], source_indices[-1]

# Prepare the ground truth inflow rates
batch_source_rates = jnp.broadcast_to(
    source_rates[jnp.newaxis, ...],
    shape=(n_rollouts_per_shard,s1-s0))

# Prepare the ground truth for batched rollouts
subs_train['SOURCE-ARRIVAL-RATE'] = subs_train['SOURCE-ARRIVAL-RATE'].at[:,s0:s1].set(batch_source_rates)
subs_train['SOURCE-ARRIVAL-RATE'] = subs_train['SOURCE-ARRIVAL-RATE'].at[:,s1:s2].set(0.)
rollouts = sampler(
        key,
        policy_params=None,
        hyperparams=None,
        subs=subs_train,
        model_params=compiler.model_params)
GROUND_TRUTH_LOOPS = rollouts['pvar']['flow-into-link'][:,:,:]

# Prepare the ground truth for sequential rollouts
subs_hmc['SOURCE-ARRIVAL-RATE'] = subs_hmc['SOURCE-ARRIVAL-RATE'].at[:,s0:s1].set(source_rates)
subs_hmc['SOURCE-ARRIVAL-RATE'] = subs_hmc['SOURCE-ARRIVAL-RATE'].at[:,s1:s2].set(0.)
rollout_hmc = sampler_hmc(
        key,
        policy_params=None,
        hyperparams=None,
        subs=subs_hmc,
        model_params=compiler.model_params)
GROUND_TRUTH_LOOPS_HMC = rollout_hmc['pvar']['flow-into-link'][:,:,:]


one_hot_inputs = jnp.eye(num_nonzero_sources)

def parametrized_policy(input):
    """The parametrized policy pi_theta.

    Currently, the policy is assumed to be parameterized by a multivariate
    normal policy with a diagonal covariance matrix

    The parametrization is currently a MLP with 32x32 hidden nodes.

    The haiku library stores the parameters theta structured as a dictionary
    with the weights of each layer and the biases of each layer getting
    a separate dictionary key. To work with theta, it is often necessary
    to make use of the jax.tree_util module."""

    mlp = hk.Sequential([
        hk.Linear(32), jax.nn.relu,
        hk.Linear(32), jax.nn.relu,
        hk.Linear(2)
    ])
    output = mlp(input)
    mean, cov = (jnp.squeeze(x) for x in jnp.split(output, 2, axis=1))
    return mean, jnp.diag(jax.nn.softplus(cov))

policy = hk.transform(parametrized_policy)
policy_apply = jax.jit(policy.apply)
key, subkey = jax.random.split(key)
theta = policy.init(subkey, one_hot_inputs)

n_params = sum(leaf.flatten().shape[0] for leaf in jax.tree_util.tree_leaves(theta))


# Set up the bijector
#bijector_obj = tfp.bijectors.IteratedSigmoidCentered()
class SimplexBijector(tfp.bijectors.IteratedSigmoidCentered):
    # Wraps the IteratedSigmoidCentered bijector, adding
    # a projection onto the first N (out of N+1) coordinates
    def __init__(self, num_nonzero_sources, max_rate):
        super().__init__()
        self.num_nonzero_sources = num_nonzero_sources
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
    num_nonzero_sources=num_nonzero_sources,
    max_rate=0.4)

def policy_pdf(theta, rng, actions):
    mean, cov = policy_apply(theta, rng, one_hot_inputs)
    unconstrained_actions = bijector_obj.inverse(actions)
    normal_pdf = jax.scipy.stats.multivariate_normal.pdf(unconstrained_actions, mean=mean, cov=cov)
    density_correction = jnp.apply_along_axis(bijector_obj._inverse_det_jacobian, axis=1, arr=actions)
    return normal_pdf * density_correction

def policy_sample(theta, rng):
    mean, cov = policy_apply(theta, rng, one_hot_inputs)
    action_sample = jax.random.multivariate_normal(
        rng, mean, cov,
        shape=(n_rollouts_per_shard,))
    action_sample = bijector_obj.forward(action_sample)
    return action_sample

def reward_batched(rng, subs, actions):
    subs['SOURCE-ARRIVAL-RATE'] = subs['SOURCE-ARRIVAL-RATE'].at[:,s0:s1].set(actions)
    rollouts = sampler(
        rng,
        policy_params=None,
        hyperparams=None,
        subs=subs,
        model_params=compiler.model_params)
    delta = rollouts['pvar']['flow-into-link'][:,:,:] - GROUND_TRUTH_LOOPS
    return jnp.sum(delta*delta, axis=(1,2))

def reward_seqtl(rng, subs, action):
    subs['SOURCE-ARRIVAL-RATE'] = subs['SOURCE-ARRIVAL-RATE'].at[:,s0:s1].set(action)
    rollout = sampler_hmc(
        rng,
        policy_params=None,
        hyperparams=None,
        subs=subs,
        model_params=compiler.model_params)
    delta = rollout['pvar']['flow-into-link'][:,:,:] - GROUND_TRUTH_LOOPS_HMC
    return jnp.sum(delta*delta, axis=(1,2))

clip_val = jnp.array([1e-3], dtype=compiler.REAL)

def scalar_mult(alpha, v):
    return alpha * v

weighting_map = jax.vmap(
    scalar_mult, in_axes=0, out_axes=0)



def parse_optimizer(config):
    if config['optimizer'] == 'rmsprop':
        optimizer = optax.rmsprop(learning_rate=config['lr'], momentum=config['momentum'])
    elif config['optimizer'] == 'adam':
        optimizer = optax.adam(learning_rate=config['lr'])
    elif config['optimizer'] == 'sgd_with_momentum':
        optimizer = optax.sgd(learning_rate=config['lr'], momentum=config['momentum'])
    elif config['optimizer'] == 'sgd':
        optimizer = optax.sgd(learning_rate=config['lr'])
    else:
        raise ValueError
    return optimizer

# === REINFORCE ===
reinforce_config = {
#    'optimizer': 'rmsprop',
    'optimizer': 'sgd',
    'lr': 1e-4,
    'momentum': 0.1,
}

@jax.jit
def reinforce_inner_loop(key, theta, subs):
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
        'rewards': 0.,
        'actions': jnp.empty(shape=(batch_size, num_nonzero_sources)),
        'dJ': jnp.empty(shape=(batch_size, n_params)),
        'dJ_covar': None
    }

    # compute dJ hat for the current sample, dividing the computation up
    # into shards of size n_rollouts_per_shard (for example, this can be
    # useful for fiting the computation into GPU VRAM)
    for si in range(n_shards):
        key, subkey = jax.random.split(key)
        actions = policy_sample(theta, subkey)
        pi = policy_pdf(theta, subkey, actions)
        dpi = jax.jacrev(policy_pdf, argnums=0)(theta, subkey, actions)
        rewards = reward_batched(subkey, subs, actions)
        factors = rewards / (pi + 1e-6)

        dJ_shard = jax.tree_util.tree_map(lambda dpi_term: weighting_map(factors, dpi_term), dpi)

        # update the dJ  estimator dJ_hat
        dJ_shard_find_mean_fn = lambda x: jnp.mean(x, axis=0)
        accumulant = jax.tree_util.tree_map(dJ_shard_find_mean_fn, dJ_shard)
        dJ_hat = jax.tree_util.tree_map(lambda x, y: x+(y/n_shards), dJ_hat, accumulant)

        # collect statistics for the batch
        # collect flattened dJ (0th index = batch index)
        ip0 = 0
        for leaf in jax.tree_util.tree_leaves(dJ_shard):
            leaf = leaf.reshape(n_rollouts_per_shard, -1)
            ip1 = ip0 + leaf.shape[1]
            batch_stats['dJ'] = batch_stats['dJ'].at[si*n_rollouts_per_shard:(si+1)*n_rollouts_per_shard, ip0:ip1].set(leaf)
            ip0 = ip1

        # collect other statistics
        batch_stats['rewards'] += jnp.mean(rewards)/n_shards
        batch_stats['actions'] = batch_stats['actions'].at[si*n_rollouts_per_shard:(si+1)*n_rollouts_per_shard].set(actions)

    batch_stats['dJ_covar'] = jnp.cov(batch_stats['dJ'])

    return key, dJ_hat, batch_stats

def update_reinforce_stats(it, algo_stats, batch_stats, policy_mean, policy_cov):
    """Updates the REINFORCE statistics using the statistics returned
    from the computation of dJ_hat for the current sample as well as
    the current policy mean/cov """
    algo_stats['dJ_covar_max'][it] = jnp.max(batch_stats['dJ_covar'])
    algo_stats['dJ_covar_min'][it] = jnp.min(batch_stats['dJ_covar'])
    algo_stats['dJ_covar_diag_max'][it] = jnp.max(jnp.diag(batch_stats['dJ_covar']))
    algo_stats['dJ_covar_diag_min'][it] = jnp.min(jnp.diag(batch_stats['dJ_covar']))
    algo_stats['rewards'][it] = batch_stats['rewards']
    algo_stats['policy_mean'][it] = policy_mean
    algo_stats['policy_cov'][it] = jnp.diag(policy_cov)
    algo_stats['transformed_policy_mean'][it] = jnp.squeeze(jnp.mean(batch_stats['actions'], axis=0))
    algo_stats['transformed_policy_cov'][it] = jnp.squeeze(jnp.std(batch_stats['actions'], axis=0))**2
    return algo_stats

def print_reinforce_report(it, algo_stats, subt0, subt1):
    """Prints out the results for the current REINFORCE iteration to console"""
    print(f'Iter {it} :: REINFORCE :: Runtime={subt1-subt0}s')
    print(f'Untransformed parametrized policy [Mean, Diag(Cov)] =')
    print(algo_stats['policy_mean'][it])
    print(algo_stats['policy_cov'][it])
    print(f'Transformed action sample statistics [[Means], [StDevs]] =')
    print(algo_stats['transformed_policy_mean'][it])
    print(algo_stats['transformed_policy_cov'][it])
    print(algo_stats['dJ_covar_diag_min'][it], '<= diag(cov(dJ)) <=', algo_stats["dJ_covar_diag_max"][it])
    print(f'Eval. reward={algo_stats["rewards"][it]}\n')


def reinforce(key, n_iters, theta, subs, config):
    """Runs the REINFORCE algorithm"""
    # initialize stats collection
    algo_stats = {
        'n_iters': n_iters,
        'algorithm': 'REINFORCE',
        'config': config,
        'action_dim': num_nonzero_sources,
        'rewards': np.empty(shape=(n_iters,)),
        'policy_mean': np.empty(shape=(n_iters, num_nonzero_sources)),
        'policy_cov': np.empty(shape=(n_iters, num_nonzero_sources)),
        'transformed_policy_mean': np.empty(shape=(n_iters, num_nonzero_sources)),
        'transformed_policy_cov': np.empty(shape=(n_iters, num_nonzero_sources)),
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

        key, subkey = jax.random.split(key)
        mean, cov = policy_apply(theta, subkey, one_hot_inputs)

        key, dJ_hat, batch_stats = reinforce_inner_loop(key, theta, subs)
        updates, opt_state = optimizer.update(dJ_hat, opt_state)
        theta = optax.apply_updates(theta, updates)

        # update statistics and print out report for current iteration
        algo_stats = update_reinforce_stats(it, algo_stats, batch_stats, mean, cov)
        print_reinforce_report(it, algo_stats, subt0, timer())

    return algo_stats


# === REINFORCE with Importance Sampling ====
impsmp_config = {
    'hmc_num_iters': int(batch_size),
    'hmc_step_size': 0.1,
    'hmc_num_leapfrog_steps': 30,
    'optimizer': 'sgd',
    'lr': 1e-2,
    'momentum': 0.1,
}
impsmp_config['hmc_num_burnin_steps'] = int(impsmp_config['hmc_num_iters']/8)

@jax.jit
def unnormalized_log_rho(key, theta, subs, a):
    dpi = jax.jacrev(policy_pdf, argnums=0)(theta, key, a)
    #dpi_norm = jax.tree_util.tree_reduce(lambda x,y: x + jnp.sum((y/100)**2), dpi, initializer=jnp.zeros(1))
    #dpi_norm = 100*jnp.sqrt(dpi_norm)
    dpi_norm = jax.tree_util.tree_reduce(lambda x, y: jnp.maximum(x, jnp.max(jnp.abs(y))), dpi, initializer=jnp.array([-jnp.inf]))
    rewards = reward_seqtl(key, subs, a)
    density = jnp.abs(rewards) * dpi_norm
    return jnp.log(density)[0]
    #return jnp.log(jnp.maximum(density, clip_val))[0]

batch_unnormalized_log_rho = jax.jit(jax.vmap(
    unnormalized_log_rho, (None, None, None, 0), 0))

@jax.jit
def impsmp_inner_loop(key, theta, subs_train, subs_hmc, samples):
    key, subkey = jax.random.split(key)

    # initialize the estimator of dJ
    dJ_hat = jax.tree_util.tree_map(lambda theta_item: jnp.zeros(theta_item.shape), theta)

    # initialize the batch stats
    batch_stats = {
        'rewards': 0.,
        'dJ': jnp.empty(shape=(batch_size, n_params)),
        'dJ_covar': None
    }

    # initialize estimate of the normalization factor
    Zinv = 0.

    for si in range(n_shards):
        actions = samples[si*n_rollouts_per_shard:(si+1)*n_rollouts_per_shard]
        pi = policy_pdf(theta, subkey, actions)
        dpi = jax.jacrev(policy_pdf, argnums=0)(theta, subkey, actions)
        rewards = reward_batched(subkey, subs_train, actions)
        rho = jnp.exp(batch_unnormalized_log_rho(subkey, theta, subs_hmc, actions[:,jnp.newaxis,:]))
        factors = rewards / (rho + 1e-8)

        Zinv += jnp.sum(pi/rho)

        dJ_shard = jax.tree_util.tree_map(lambda dpi_term: weighting_map(factors, dpi_term), dpi)

        # update the dJ estimator dJ_hat
        dJ_shard_find_mean_fn = lambda x: jnp.mean(x, axis=0)
        accumulant = jax.tree_util.tree_map(dJ_shard_find_mean_fn, dJ_shard)
        dJ_hat = jax.tree_util.tree_map(lambda x, y: x+(y/n_shards), dJ_hat, accumulant)

        # collect statistics for the batch
        # collect flattened dJ (0th index = batch index)
        ip0 = 0
        for leaf in jax.tree_util.tree_leaves(dJ_shard):
            leaf = leaf.reshape(n_rollouts_per_shard, -1)
            ip1 = ip0 + leaf.shape[1]
            batch_stats['dJ'] = batch_stats['dJ'].at[si*n_rollouts_per_shard:(si+1)*n_rollouts_per_shard, ip0:ip1].set(leaf)
            ip0 = ip1

        # collect other statistics
        batch_stats['rewards'] += jnp.mean(rewards)/n_shards

    # multiply by Z (= divide by Zinv)
    dJ_hat = jax.tree_util.tree_map(lambda x: x / jnp.maximum(Zinv, clip_val), dJ_hat)
    batch_stats['dJ'] = jax.tree_util.tree_map(lambda x: x / jnp.maximum(Zinv, clip_val), batch_stats['dJ'])

    # calculate the covariance matrix for the sample
    batch_stats['dJ_covar'] = jnp.cov(batch_stats['dJ'])
    return key, dJ_hat, batch_stats

def update_impsmp_stats(key, it, algo_stats, batch_stats, theta):
    """Updates the REINFORCE with Importance Sampling statistics
    using the statistics returned from the computation of dJ_hat
    for the current sample as well as the current policy params theta"""
    key, subkey = jax.random.split(key)
    policy_mean, policy_cov = policy_apply(theta, subkey, one_hot_inputs)
    eval_actions = policy_sample(theta, subkey)
    eval_rewards = jnp.mean(reward_batched(subkey, subs_train, eval_actions))

    algo_stats['dJ_covar_max'][it] = jnp.max(batch_stats['dJ_covar'])
    algo_stats['dJ_covar_min'][it] = jnp.min(batch_stats['dJ_covar'])
    algo_stats['dJ_covar_diag_max'][it] = jnp.max(jnp.diag(batch_stats['dJ_covar']))
    algo_stats['dJ_covar_diag_min'][it] = jnp.min(jnp.diag(batch_stats['dJ_covar']))

    algo_stats['policy_mean'][it] = policy_mean
    algo_stats['policy_cov'][it] = jnp.diag(policy_cov)
    algo_stats['transformed_policy_mean'][it] = jnp.mean(eval_actions, axis=0)
    algo_stats['transformed_policy_cov'][it] = jnp.std(eval_actions, axis=0)**2
    algo_stats['sample_rewards'][it] = batch_stats['rewards']
    algo_stats['eval_rewards'][it] = eval_rewards

    return key, algo_stats

def print_impsmp_report(it, algo_stats, is_accepted, subt0, subt1):
    """Prints out the results for the current REINFORCE with Importance Sampling iteration"""
    print(f'Iter {it} :: Importance Sampling :: Runtime={subt1-subt0}s :: HMC acceptance rate={jnp.mean(is_accepted)*100:.2f}%')
    print('Untransformed parametrized policy [Mean, Diag(Cov)] =')
    print(algo_stats['policy_mean'][it])
    print(algo_stats['policy_cov'][it])
    print('Transformed action sample statistics [Mean, StDev] =')
    print(algo_stats['transformed_policy_mean'][it])
    print(algo_stats['transformed_policy_cov'][it])
    print(algo_stats['dJ_covar_diag_min'][it], '<= diag(cov(dJ)) <=', algo_stats['dJ_covar_diag_max'][it])
    print(f'HMC sample reward={algo_stats["sample_rewards"][it]} :: Eval reward={algo_stats["eval_rewards"][it]}\n')


def impsmp(key, n_iters, theta, subs_train, subs_hmc, config):
    """Runs the REINFORCE with Importance Sampling algorithm"""
    # initialize stats collection
    algo_stats = {
        'algorithm': 'ImpSmp',
        'config': config,
        'n_iters': n_iters,
        'action_dim': num_nonzero_sources,
        'policy_mean': np.empty(shape=(n_iters, num_nonzero_sources)),
        'policy_cov': np.empty(shape=(n_iters, num_nonzero_sources)),
        'transformed_policy_mean': np.empty(shape=(n_iters, num_nonzero_sources)),
        'transformed_policy_cov': np.empty(shape=(n_iters, num_nonzero_sources)),
        'sample_rewards': np.empty(shape=(n_iters,)),
        'eval_rewards': np.empty(shape=(n_iters,)),
        'dJ_covar_max': np.empty(shape=(n_iters,)),
        'dJ_covar_min': np.empty(shape=(n_iters,)),
        'dJ_covar_diag_max': np.empty(shape=(n_iters,)),
        'dJ_covar_diag_min': np.empty(shape=(n_iters,)),
    }

    # initialize HMC
    key, subkey = jax.random.split(key)
    hmc_initializer = jax.random.uniform(
        subkey,
        shape=(1,num_nonzero_sources),
        minval=0.05, maxval=0.35)

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
            unnormalized_log_rho, subkeys[1], theta, subs_hmc)

        adaptive_hmc_kernel = tfp.mcmc.TransformedTransitionKernel(
            inner_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
                inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
                    target_log_prob_fn=log_density,
                    num_leapfrog_steps=config['hmc_num_leapfrog_steps'],
                    step_size=config['hmc_step_size']),
                num_adaptation_steps=int(config['hmc_num_burnin_steps'] * 0.8)),
            bijector=unconstraining_bijector)

        samples, is_accepted = tfp.mcmc.sample_chain(
            seed=subkeys[2],
            num_results=config['hmc_num_iters'],
            num_burnin_steps=config['hmc_num_burnin_steps'],
            current_state=hmc_initializer,
            kernel=adaptive_hmc_kernel,
            trace_fn=lambda _, pkr: pkr.inner_results.inner_results.is_accepted)

        samples = jnp.squeeze(samples)

        key, dJ_hat, batch_stats = impsmp_inner_loop(
            key, theta, subs_train, subs_hmc, samples)

        updates, opt_state = optimizer.update(dJ_hat, opt_state)
        theta = optax.apply_updates(theta, updates)

        # initialize the next chain at a random point of the current chain
        hmc_intializer = jax.random.choice(subkeys[3], samples)

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
    method = 'reinforce'
    n_iters = 75

    # if disable=True, runs without jit (much slower, but can inspect the data
    # in the middle of a jitted computation). If disable=False, runs with jit
    with jax.disable_jit(disable=False):
        key, subkey = jax.random.split(key)
        if method == 'impsmp':
            method_config = impsmp_config
            algo_stats = impsmp(key=subkey, n_iters=n_iters, theta=theta, subs_train=subs_train, subs_hmc=subs_hmc, config=method_config)
        elif method == 'reinforce':
            method_config = reinforce_config
            algo_stats = reinforce(key=subkey, n_iters=n_iters, theta=theta, subs=subs_train, config=method_config)
        else: raise KeyError

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f'{timestamp}_{method}_iters{n_iters}'

    with open(f'tmp/{filename}.json', 'w') as file:
        json.dump(algo_stats, file, cls=SimpleNumpyToJSONEncoder)
