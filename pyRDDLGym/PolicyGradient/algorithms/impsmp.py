import jax
import optax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp
from enum import Enum
import matplotlib.pyplot as plt

import functools
from time import perf_counter as timer


def cos_sim(A, B):
    return jnp.dot(A, B) / (jnp.linalg.norm(A) * jnp.linalg.norm(B))

def scalar_mult(alpha, v):
    return alpha * v

weighting_map_inner = jax.vmap(scalar_mult, in_axes=0, out_axes=0)
weighting_map = jax.vmap(weighting_map_inner, in_axes=0, out_axes=0)

def flatten_dJ_shard(dJ_shard, ifrom, ito, flat_dJ_shard):
    ip0 = 0
    per_shard = ito - ifrom
    for leaf in jax.tree_util.tree_leaves(dJ_shard):
        leaf = leaf.reshape(per_shard, -1)
        ip1 = ip0 + leaf.shape[1]
        flat_dJ_shard = flat_dJ_shard.at[ifrom:ito, ip0:ip1].set(leaf)
        ip0 = ip1
    return flat_dJ_shard

def init_hmc_state(key, shape, config):
    if config['type'] == 'uniform':
        return jax.random.uniform(
            key,
            shape=shape,
            minval=config['min'],
            maxval=config['max'])
    elif config['type'] == 'normal':
        hmc_initializer = jax.random.normal(
            key,
            shape=shape)
        return config['mean'] + hmc_initializer * config['std']
    else:
        raise ValueError('[init_hmc_state] Unrecognized distribution type')



@functools.partial(jax.jit, static_argnames=('policy', 'model'))
def unnormalized_rho(key, theta, policy, model, a):
    dpi = jax.jacrev(policy.pdf, argnums=1)(key, theta, a)
    dpi_norm = jax.tree_util.tree_reduce(lambda x, y: jnp.maximum(x, jnp.max(jnp.abs(y))), dpi, initializer=jnp.array([-jnp.inf]))
    losses = model.compute_loss(key, a)
    density = jnp.abs(losses) * dpi_norm
    return density

@functools.partial(jax.jit, static_argnames=('policy', 'model', 'epsilon'))
def unnormalized_log_rho(key, theta, policy, model, epsilon, a):
    density = unnormalized_rho(key, theta, policy, model, a)
    return jnp.log(density + epsilon)


@functools.partial(
    jax.jit,
    static_argnames=('n_shards',
                     'batch_size',
                     'epsilon',
                     'est_Z',
                     'policy',
                     'hmc_model',
                     'train_model'))
def impsmp_inner_loop(key, theta, samples,
                      n_shards, batch_size, epsilon, est_Z, policy, hmc_model, train_model):
    batch_stats = {
        'sample_rewards': jnp.empty(shape=(batch_size,)),
        'dJ': jnp.empty(shape=(batch_size, policy.n_params)),
    }

    dJ_hat = jax.tree_util.tree_map(lambda theta_item: jnp.zeros(theta_item.shape), theta)
    Zinv = 0.

    num_chains = samples.shape[1]

    for si in range(n_shards):
        key, subkey = jax.random.split(key)

        actions = samples[si*train_model.n_rollouts:(si+1)*train_model.n_rollouts]

        pi = policy.pdf(key, theta, actions)
        dpi = jax.jacrev(policy.pdf, argnums=1)(key, theta, actions)

        def squeezed_loss(key, actions):
            return jnp.squeeze(train_model.compute_loss(key, actions))
        batch_compute_loss = jax.vmap(squeezed_loss, (None, 1), 1)
        losses = batch_compute_loss(key, actions)

        batch_unnormalized_rho = jax.vmap(unnormalized_rho, (None, None, None, None, 0), 0)
        rho = batch_unnormalized_rho(key, theta, policy, hmc_model, actions)

        factors = losses / (rho + epsilon)
        scores = jax.tree_util.tree_map(lambda dpi_term: weighting_map(factors, dpi_term), dpi)

        # the mean of scores along axis 0 is equal to
        #    (dJ_fr + dJ_(fr+1) + ... + dJ_to) * (1/model.n_rollouts)
        # additionally dividing the mean by 'n_shards' results in overall division by 'batch_size'
        dJ_hat = jax.tree_util.tree_map(lambda x, y: x+jnp.mean(y, axis=(0,1))/n_shards, dJ_hat, scores)

        # estimate of the normalizing factor Z of the density rho
        if est_Z:
            Zinv += jnp.sum(pi/(rho + epsilon))

        # collect statistics for the shard
        per_shard = num_chains * train_model.n_rollouts
        ifrom, ito = si*per_shard, (si+1)*per_shard
        batch_stats['dJ'] = flatten_dJ_shard(scores, ifrom, ito, batch_stats['dJ'])
        batch_stats['sample_rewards'] = batch_stats['sample_rewards'].at[ifrom:ito].set(losses.flatten())

    if est_Z:
        Zinv = (Zinv / batch_size) + epsilon
        dJ_hat = jax.tree_util.tree_map(lambda x: x / Zinv, dJ_hat)
        batch_stats['dJ'] = jax.tree_util.tree_map(lambda x: x / Zinv, batch_stats['dJ'])

    return key, dJ_hat, batch_stats

@functools.partial(jax.jit, static_argnames=('batch_size', 'eval_n_shards', 'eval_batch_size', 'policy', 'model'))
def update_impsmp_stats(key, it, batch_size, eval_n_shards, eval_batch_size, algo_stats, batch_stats, theta, policy, model):
    """Updates the REINFORCE with Importance Sampling statistics
    using the statistics returned from the computation of dJ_hat
    for the current sample as well as the current policy params theta"""
    key, subkey = jax.random.split(key)
    policy_mean, policy_cov = policy.apply(subkey, theta)
    eval_actions = policy.sample(subkey, theta, (model.n_rollouts,))
    eval_rewards = model.compute_loss(subkey, eval_actions)

    # calculate the covariance matrices
    dJ_hat = jnp.mean(batch_stats['dJ'], axis=0)
    dJ_covar = jnp.cov(batch_stats['dJ'], rowvar=False)

    algo_stats['dJ']                 = algo_stats['dJ'].at[it].set(batch_stats['dJ'])
    algo_stats['dJ_hat_max']         = algo_stats['dJ_hat_max'].at[it].set(jnp.max(dJ_hat))
    algo_stats['dJ_hat_min']         = algo_stats['dJ_hat_min'].at[it].set(jnp.min(dJ_hat))
    algo_stats['dJ_hat_norm']        = algo_stats['dJ_hat_norm'].at[it].set(jnp.linalg.norm(dJ_hat))
    algo_stats['dJ_covar']           = algo_stats['dJ_covar'].at[it].set(dJ_covar)
    algo_stats['dJ_covar_max']       = algo_stats['dJ_covar_max'].at[it].set(jnp.max(dJ_covar))
    algo_stats['dJ_covar_min']       = algo_stats['dJ_covar_min'].at[it].set(jnp.min(dJ_covar))
    algo_stats['dJ_covar_diag_max']  = algo_stats['dJ_covar_diag_max'].at[it].set(jnp.max(jnp.diag(dJ_covar)))
    algo_stats['dJ_covar_diag_mean'] = algo_stats['dJ_covar_diag_mean'].at[it].set(jnp.mean(jnp.diag(dJ_covar)))
    algo_stats['dJ_covar_diag_min']  = algo_stats['dJ_covar_diag_min'].at[it].set(jnp.min(jnp.diag(dJ_covar)))

    algo_stats['policy_mean']             = algo_stats['policy_mean'].at[it].set(policy_mean)
    algo_stats['policy_cov']              = algo_stats['policy_cov'].at[it].set(jnp.diag(policy_cov))
    algo_stats['transformed_policy_mean'] = algo_stats['transformed_policy_mean'].at[it].set(jnp.mean(eval_actions, axis=0))
    algo_stats['transformed_policy_cov']  = algo_stats['transformed_policy_cov'].at[it].set(jnp.std(eval_actions, axis=0)**2)
    algo_stats['reward_mean']             = algo_stats['reward_mean'].at[it].set(jnp.mean(eval_rewards))
    algo_stats['reward_std']              = algo_stats['reward_std'].at[it].set(jnp.std(eval_rewards))
    algo_stats['reward_sterr']            = algo_stats['reward_sterr'].at[it].set(algo_stats['reward_std'][it] / jnp.sqrt(model.n_rollouts))

    algo_stats['sample_rewards']      = algo_stats['sample_rewards'].at[it].set(batch_stats['sample_rewards'])
    algo_stats['sample_reward_mean']  = algo_stats['sample_reward_mean'].at[it].set(jnp.mean(batch_stats['sample_rewards']))
    algo_stats['sample_reward_std']   = algo_stats['sample_reward_std'].at[it].set(jnp.std(batch_stats['sample_rewards']))
    algo_stats['sample_reward_sterr'] = algo_stats['sample_reward_sterr'].at[it].set(algo_stats['sample_reward_std'][it] / jnp.sqrt(batch_size))
    algo_stats['acceptance_rate']     = algo_stats['acceptance_rate'].at[it].set(batch_stats['acceptance_rate'])
    algo_stats['hmc_step_size']       = algo_stats['hmc_step_size'].at[it].set(batch_stats['hmc_step_size'])

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
    print(algo_stats['dJ_covar_diag_min'][it], '<=', algo_stats['dJ_covar_diag_mean'][it], '<=', algo_stats['dJ_covar_diag_max'][it])
    print(f'HMC sample reward={algo_stats["sample_reward_mean"][it]:.3f} \u00B1 {algo_stats["sample_reward_sterr"][it]:.3f} '
          f':: Eval reward={algo_stats["reward_mean"][it]:.3f} \u00B1 {algo_stats["reward_sterr"][it]:.3f}\n')


def impsmp(key, n_iters, config, bijector, policy, optimizer, models):
    """Runs the REINFORCE with Importance Sampling algorithm"""

    # parse config
    action_dim = policy.action_dim
    batch_size = config['batch_size']
    eval_batch_size = config['eval_batch_size']

    hmc_model = models['hmc_model']
    train_model = models['train_model']
    eval_model = models['eval_model']

    n_shards = int(batch_size // (hmc_model.n_rollouts * train_model.n_rollouts))
    eval_n_shards = int(eval_batch_size // eval_model.n_rollouts)
    assert n_shards > 0, (
         '[reinforce] Please check that batch_size >= (hmc_model.n_rollouts * train_model.n_rollouts).'
        f' batch_size={batch_size}, hmc_model.n_rollouts={hmc_model.n_rollouts}, train_model.n_rollouts={train_model.n_rollouts}')
    assert eval_n_shards > 0, (
        '[reinforce] Please check that eval_batch_size >= eval_model.n_rollouts.'
        f' eval_batch_size={eval_batch_size}, eval_model.n_rollouts={eval_model.n_rollouts}')

    hmc_config = config['hmc']
    hmc_config['num_chains'] = hmc_model.n_rollouts
    hmc_config['num_iters_per_chain'] = int(batch_size // hmc_model.n_rollouts)
    assert hmc_config['num_iters_per_chain'] > 0

    epsilon = config.get('epsilon', 1e-12)

    est_Z = config.get('est_Z', True)
    if hmc_config['init_distribution']['type'] == 'normal':
        hmc_config['init_distribution']['std'] = jnp.sqrt(hmc_config['init_distribution']['var'])

    # initialize stats collection
    algo_stats = {
        'action_dim': action_dim,
        'batch_size': batch_size,
        'eval_batch_size': eval_batch_size,
        'policy_mean': jnp.empty(shape=(n_iters, action_dim)),
        'policy_cov': jnp.empty(shape=(n_iters, action_dim)),
        'transformed_policy_mean': jnp.empty(shape=(n_iters, action_dim)),
        'transformed_policy_cov': jnp.empty(shape=(n_iters, action_dim)),
        'reward_mean': jnp.empty(shape=(n_iters,)),
        'reward_std': jnp.empty(shape=(n_iters,)),
        'reward_sterr': jnp.empty(shape=(n_iters,)),
        'sample_rewards': jnp.empty(shape=(n_iters, batch_size)),
        'sample_reward_mean': jnp.empty(shape=(n_iters,)),
        'sample_reward_std': jnp.empty(shape=(n_iters,)),
        'sample_reward_sterr': jnp.empty(shape=(n_iters,)),
        'acceptance_rate': jnp.empty(shape=(n_iters,)),
        'hmc_step_size': jnp.empty(shape=(n_iters,)),
        'dJ': jnp.empty(shape=(n_iters, batch_size, policy.n_params)),
        'dJ_hat_max': jnp.empty(shape=(n_iters,)),
        'dJ_hat_min': jnp.empty(shape=(n_iters,)),
        'dJ_hat_norm': jnp.empty(shape=(n_iters,)),
        'dJ_covar': jnp.empty(shape=(n_iters, policy.n_params, policy.n_params)),
        'dJ_covar_max': jnp.empty(shape=(n_iters,)),
        'dJ_covar_min': jnp.empty(shape=(n_iters,)),
        'dJ_covar_diag_max': jnp.empty(shape=(n_iters,)),
        'dJ_covar_diag_mean': jnp.empty(shape=(n_iters,)),
        'dJ_covar_diag_min': jnp.empty(shape=(n_iters,)),
    }

    # initialize HMC
    key, subkey = jax.random.split(key)
    hmc_initializer = init_hmc_state(subkey, (hmc_config['num_chains'], action_dim), hmc_config['init_distribution'])
    hmc_step_size = hmc_config['init_step_size']

    # initialize optimizer
    opt_state = optimizer.init(policy.theta)

    # initialize unconstraining bijector
    unconstraining_bijector = [
        bijector
    ]

    # run REINFORCE with Importance Sampling
    for it in range(n_iters):
        try:
            subt0 = timer()
            key, *subkeys = jax.random.split(key, num=5)

            log_density = functools.partial(
                unnormalized_log_rho, subkeys[1], policy.theta, policy, hmc_model, epsilon)

            adaptive_hmc_kernel = tfp.mcmc.TransformedTransitionKernel(
                inner_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
                    inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
                        target_log_prob_fn=log_density,
                        num_leapfrog_steps=hmc_config['num_leapfrog_steps'],
                        step_size=hmc_step_size),
                    num_adaptation_steps=int(hmc_config['num_burnin_iters_per_chain'] * 0.8)),
                bijector=unconstraining_bijector)

            samples, is_accepted = tfp.mcmc.sample_chain(
                seed=subkeys[2],
                num_results=hmc_config['num_iters_per_chain'],
                num_burnin_steps=hmc_config['num_burnin_iters_per_chain'],
                current_state=hmc_initializer,
                kernel=adaptive_hmc_kernel,
                trace_fn=lambda _, pkr: pkr.inner_results.inner_results.is_accepted)

            # compute dJ_hat
            key, dJ_hat, batch_stats = impsmp_inner_loop(
                key, policy.theta, samples,
                n_shards, batch_size, epsilon, est_Z, policy, hmc_model, train_model)

            updates, opt_state = optimizer.update(dJ_hat, opt_state)
            policy.theta = optax.apply_updates(policy.theta, updates)

            # initialize the next chain at a random point of the current chain
            if hmc_config['reinit_strategy'] == 'random_sample':
                hmc_initializer = init_hmc_state(subkeys[3], (hmc_config['num_chains'], action_dim), hmc_config['init_distribution'])
            elif hmc_config['reinit_strategy'] == 'random_prev_chain_elt':
                hmc_initializer = jax.random.choice(subkeys[3], samples)
            else:
                raise ValueError('[impsmp] Unrecognized HMC reinitialization strategy '
                                f'{hmc_config["reinit_strategy"]}. Expect "random_sample" '
                                 'or "random_prev_chain_elt"')

            # update stats and printout results for the current iteration
            batch_stats['acceptance_rate'] = jnp.mean(is_accepted)
            batch_stats['hmc_step_size'] = hmc_step_size
            key, algo_stats = update_impsmp_stats(key, it, batch_size, eval_n_shards, eval_batch_size,
                                                  algo_stats, batch_stats, policy.theta,
                                                  policy, eval_model)
            print_impsmp_report(it, algo_stats, is_accepted, subt0, timer())
        except FloatingPointError:
            print('[impsmp] Caught FloatingPointError exception. Saving results up to current iteration and exiting')
            break

    algo_stats.update({
        'algorithm': 'ImpSmp',
        'n_iters': n_iters,
        'config': config,
    })
    return key, algo_stats





# ===== Visualization utilities =====
def impsmp_analyze_1d_samples(key, n_iters, config, bijector, policy, optimizer, models):
    """Analyzes the one-dimensional instrumental density computation and HMC sampling"""

    # parse config
    action_dim = policy.action_dim
    assert action_dim == 1
    batch_size = config['batch_size']
    eval_batch_size = config['eval_batch_size']

    hmc_model = models['hmc_model']
    train_model = models['train_model']
    eval_model = models['eval_model']
    assert hmc_model.n_rollouts == 1

    hmc_config = config['hmc']
    hmc_config['num_chains'] = hmc_model.n_rollouts
    hmc_config['num_iters_per_chain'] = int(batch_size // hmc_model.n_rollouts)
    assert hmc_config['num_iters_per_chain'] > 0

    epsilon = config.get('epsilon', 1e-12)

    if hmc_config['init_distribution']['type'] == 'normal':
        hmc_config['init_distribution']['std'] = jnp.sqrt(hmc_config['init_distribution']['var'])

    # initialize HMC
    key, subkey = jax.random.split(key)
    hmc_initializer = init_hmc_state(subkey, (hmc_config['num_chains'], action_dim), hmc_config['init_distribution'])
    hmc_step_size = hmc_config['init_step_size']

    # initialize unconstraining bijector
    unconstraining_bijector = [
        bijector
    ]

    # produce the 1d visualization
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(14, 7)

    key, *subkeys = jax.random.split(key, num=3)

    log_density = functools.partial(
        unnormalized_log_rho, subkeys[0], policy.theta, policy, hmc_model, epsilon)

    lowerbnd = config.get('lowerbnd', -5.)
    upperbnd = config.get('upperbnd', 5.)
    step = config.get('step', 0.1)
    n_pts = int((upperbnd-lowerbnd)/step)

    test_range = jnp.arange(lowerbnd, upperbnd, step).reshape(n_pts, hmc_config["num_chains"], 1)

    batch_compute_loss = jax.vmap(hmc_model.compute_loss, (None, 0), 0)
    losses = batch_compute_loss(subkeys[0], test_range).flatten()
    losses = jnp.abs(losses)

    jacobian = jax.jacrev(policy.pdf, argnums=1)
    batch_dpi = jax.vmap(jacobian, (None, None, 0), 0)
    dpi = batch_dpi(subkeys[0], policy.theta, test_range)['linear']['w']
    def get_infty_norm(dpi):
        return jnp.maximum(
            jnp.abs(dpi[0,0,0]),
            jnp.abs(dpi[0,0,1])
        )
    batch_dpi_infty_norm = jax.vmap(get_infty_norm, 0, 0)
    dpi_infty_norm = batch_dpi_infty_norm(dpi)

    batch_compute_density = jax.vmap(log_density, (0,), 0)
    rho = batch_compute_density(test_range).flatten()
    rho = jnp.exp(rho)

    assert jnp.isclose(losses * dpi_infty_norm, rho).all()

    ax[0].plot(losses, label='|relaxed loss|')
    ax[0].plot(dpi_infty_norm, label='|dpi|')
    ax[0].plot(rho, label='Instrumental density')

    ax[0].legend()
    ax[0].grid(visible=True)

    adaptive_hmc_kernel = tfp.mcmc.TransformedTransitionKernel(
        inner_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
            inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=log_density,
                num_leapfrog_steps=hmc_config['num_leapfrog_steps'],
                step_size=hmc_step_size),
            num_adaptation_steps=int(hmc_config['num_burnin_iters_per_chain'] * 0.8)),
        bijector=unconstraining_bijector)

    samples, is_accepted = tfp.mcmc.sample_chain(
        seed=subkeys[1],
        num_results=hmc_config['num_iters_per_chain'],
        num_burnin_steps=hmc_config['num_burnin_iters_per_chain'],
        current_state=hmc_initializer,
        kernel=adaptive_hmc_kernel,
        trace_fn=lambda _, pkr: pkr.inner_results.inner_results.is_accepted)


    ax[1].hist(samples.flatten())
    ax[1].set_xlim(lowerbnd, upperbnd)

    plt.suptitle(
        f'Comparison of the instrumental density and the distribution of {batch_size} HMC samples\n'
        f'Env: Dim=1, Summands={train_model.n_summands}\n'
        f'HMC Settings: '
        f'Burnin={hmc_config["num_burnin_iters_per_chain"]}, '
        f'Step size={hmc_step_size}, '
        f'Num leapfrog steps={hmc_config["num_leapfrog_steps"]}'
    )

    plt.tight_layout()

    img_path = config.get('img_path', '1d_HMC_results.png')
    plt.savefig(img_path)
    print('Saved 1d instrumental density and HMC sample histogram to', img_path)
    return key, {}



def impsmp_analyze_2d_samples(key, n_iters, config, bijector, policy, optimizer, models):
    """Analyzes the two-dimensional instrumental density and HMC sampling"""

    # parse config
    action_dim = policy.action_dim
    assert action_dim == 2
    batch_size = config['batch_size']
    eval_batch_size = config['eval_batch_size']

    hmc_model = models['hmc_model']
    train_model = models['train_model']
    eval_model = models['eval_model']
    assert hmc_model.n_rollouts == 1

    hmc_config = config['hmc']
    hmc_config['num_chains'] = hmc_model.n_rollouts
    hmc_config['num_iters_per_chain'] = int(batch_size // hmc_model.n_rollouts)
    assert hmc_config['num_iters_per_chain'] > 0

    epsilon = config.get('epsilon', 1e-12)

    if hmc_config['init_distribution']['type'] == 'normal':
        hmc_config['init_distribution']['std'] = jnp.sqrt(hmc_config['init_distribution']['var'])

    # initialize HMC
    key, subkey = jax.random.split(key)
    hmc_initializer = init_hmc_state(subkey, (hmc_config['num_chains'], action_dim), hmc_config['init_distribution'])
    hmc_step_size = hmc_config['init_step_size']

    # initialize unconstraining bijector
    unconstraining_bijector = [
        bijector
    ]

    # produce the 1d visualization
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(14, 7)

    key, *subkeys = jax.random.split(key, num=3)

    log_density = functools.partial(
        unnormalized_log_rho, subkeys[0], policy.theta, policy, hmc_model, epsilon)

    lowerbnd = config.get('lowerbnd', -5.)
    upperbnd = config.get('upperbnd', 5.)
    step = config.get('step', 0.1)
    n_pts = int((upperbnd-lowerbnd)/step)

    test_range = jnp.arange(lowerbnd, upperbnd, step)

    z = np.zeros(shape=(n_pts, n_pts))
    for ix, x in enumerate(test_range):
        for iy, y in enumerate(test_range):
            a = jnp.array([x,y]).reshape(1,2)
            z[ix,iy] = log_density(a)

    z = np.exp(z)
    ax[0].imshow(z, interpolation='none', origin='lower', extent=(lowerbnd,upperbnd,lowerbnd,upperbnd))
    ax[0].set_title('Instrumental density')

    adaptive_hmc_kernel = tfp.mcmc.TransformedTransitionKernel(
        inner_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
            inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=log_density,
                num_leapfrog_steps=hmc_config['num_leapfrog_steps'],
                step_size=hmc_step_size),
            num_adaptation_steps=int(hmc_config['num_burnin_iters_per_chain'] * 0.8)),
        bijector=unconstraining_bijector)


    samples, is_accepted = tfp.mcmc.sample_chain(
        seed=subkeys[1],
        num_results=hmc_config['num_iters_per_chain'],
        num_burnin_steps=hmc_config['num_burnin_iters_per_chain'],
        current_state=hmc_initializer,
        kernel=adaptive_hmc_kernel,
        trace_fn=lambda _, pkr: pkr.inner_results.inner_results.is_accepted)

    bins = np.zeros(shape=(n_pts, n_pts))
    coords = np.floor((samples - lowerbnd)/step).astype(np.int32)
    print(samples[:100])
    for sample, (cx, cy) in zip(samples, np.squeeze(coords)):
        bins[cx, cy] += 1
    ax[1].set_title(str(batch_size) + ' Samples')
    ax[1].imshow(bins, interpolation='none', origin='lower', extent=(lowerbnd,upperbnd,lowerbnd,upperbnd))


    plt.suptitle(
        f'Comparison of the instrumental density and the distribution of {batch_size} HMC samples\n'
        f'Env: Dim=2, Summands={train_model.n_summands}\n'
        f'HMC Settings: '
        f'Burnin={hmc_config["num_burnin_iters_per_chain"]}, '
        f'Step size={hmc_step_size}, '
        f'Num leapfrog steps={hmc_config["num_leapfrog_steps"]}'
    )

    plt.tight_layout()

    img_path = config.get('img_path', '2d_HMC_results.png')
    plt.savefig(img_path)
    print('Saved 2d instrumental density and HMC histogram to', img_path)
    return key, {}

