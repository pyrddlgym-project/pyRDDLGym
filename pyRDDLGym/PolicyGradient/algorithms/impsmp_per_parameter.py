import jax
import optax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp
from enum import Enum

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

def init_hmc_state(key, n_chains, action_dim, policy, config):
    shape = (n_chains, action_dim, 2, 1, action_dim)
    if config['type'] == 'uniform':
        return jax.random.uniform(
            key,
            shape=shape,
            minval=config['min'],
            maxval=config['max'])
    elif config['type'] == 'normal':
        init_state = jax.random.normal(
            key,
            shape=shape)
        return config['mean'] + init_state * config['std']
    elif config['type'] == 'current_policy':
        return policy.sample(key, policy.theta, shape[:-1])
    else:
        raise ValueError('[init_hmc_state] Unrecognized distribution type')



@functools.partial(jax.jit, static_argnames=('policy', 'model'))
def unnormalized_rho(key, theta, policy, model, a):
    dpi = jax.jacrev(policy.pdf, argnums=1)(key, theta, a)
    dpi = jax.tree_util.tree_map(lambda x: jnp.diagonal(x, axis1=0, axis2=3), dpi)
    dpi = jax.tree_util.tree_map(lambda x: jnp.diagonal(x, axis1=0, axis2=2), dpi)
    dpi = jax.tree_util.tree_map(lambda x: x[0], dpi)
    dpi_abs = jax.tree_util.tree_map(lambda x: jnp.abs(x), dpi)

    # compute losses over the first three axes, which index
    # (chain_idx, action_dim_idx, mean_or_var_idx, 1, action_dim_idx)
    #            |<---   parameter indices   --->|

    compute_loss_axis_0 = jax.vmap(model.compute_loss, (None, 0), 0)
    compute_loss_axes_0_1 = jax.vmap(compute_loss_axis_0, (None, 0), 0)

    losses = compute_loss_axes_0_1(key, a)[..., 0]
    losses = jnp.abs(losses)
    density = losses * dpi_abs['linear']['w']
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
def impsmp_per_parameter_inner_loop(key, theta, samples,
                                    n_shards, batch_size, epsilon, est_Z,
                                    policy, hmc_model, train_model):
    batch_stats = {
        'sample_rewards': jnp.empty(shape=(batch_size, policy.n_params)),
        'scores': jnp.empty(shape=(batch_size, policy.n_params)),
        'dJ': jnp.empty(shape=(batch_size, policy.n_params)),
    }

    dJ_hat = jax.tree_util.tree_map(lambda theta_item: jnp.zeros(theta_item.shape), theta)
    Zinv = 0.

    jacobian = jax.jacrev(policy.pdf, argnums=1)

    for si in range(n_shards):
        key, subkey = jax.random.split(key)

        actions = samples[si*train_model.n_rollouts:(si+1)*train_model.n_rollouts]

        pi = policy.pdf(key, theta, actions)[..., 0]
        dpi = jacobian(key, theta, actions)
        dpi = jax.tree_util.tree_map(lambda x: jnp.diagonal(x, axis1=1, axis2=4), dpi)
        dpi = jax.tree_util.tree_map(lambda x: jnp.diagonal(x, axis1=1, axis2=3), dpi)
        dpi = jax.tree_util.tree_map(lambda x: x[:,0,:,:], dpi)
        dpi_abs = jax.tree_util.tree_map(lambda x: jnp.abs(x), dpi)

        compute_loss_axis_1 = jax.vmap(train_model.compute_loss, (None, 1), 1)
        compute_loss_axes_1_2 = jax.vmap(compute_loss_axis_1, (None, 1), 1)
        batch_compute_loss = jax.vmap(compute_loss_axes_1_2, (None, 1), 1)

        losses = batch_compute_loss(key, actions)[..., 0]

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
            Zinv += jnp.sum(pi/(rho + epsilon), axis=(0,))

        # collect statistics for the shard
        per_shard = train_model.n_rollouts
        ifrom, ito = si*per_shard, (si+1)*per_shard
        batch_stats['dJ'] = flatten_dJ_shard(scores, ifrom, ito, batch_stats['dJ'])
        batch_stats['scores'] = batch_stats['scores'].at[ifrom:ito].set(scores['linear']['w'].reshape(-1, policy.n_params))
        batch_stats['sample_rewards'] = batch_stats['sample_rewards'].at[ifrom:ito].set(losses.reshape(-1, policy.n_params))

    if est_Z:
        Zinv = (Zinv / batch_size) + epsilon
        dJ_hat = jax.tree_util.tree_map(lambda x: x / Zinv, dJ_hat)
        batch_stats['dJ'] = jax.tree_util.tree_map(lambda x: x / (Zinv.reshape(policy.n_params,)), batch_stats['dJ'])

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

    algo_stats['scores']              = algo_stats['scores'].at[it].set(batch_stats['scores'])
    algo_stats['mean_abs_score']       = algo_stats['mean_abs_score'].at[it].set(jnp.mean(jnp.abs(batch_stats['scores'])))

    return key, algo_stats

def print_impsmp_report(it, algo_stats, is_accepted, batch_size, num_chains, subt0, subt1):
    """Prints out the results for the current REINFORCE with Importance Sampling iteration"""
    print(f'Iter {it} :: Importance Sampling :: Runtime={subt1-subt0}s')
    print(f'Batch size={batch_size} :: Num.Chains={num_chains} :: HMC step size={algo_stats["hmc_step_size"][it]:.4f} :: HMC acceptance rate={algo_stats["acceptance_rate"][it]*100:.2f}%')
    print('Untransformed parametrized policy [Mean, Diag(Cov)] =')
    print(algo_stats['policy_mean'][it])
    print(algo_stats['policy_cov'][it])
    print('Transformed action sample statistics [Mean, StDev] =')
    print(algo_stats['transformed_policy_mean'][it])
    print(algo_stats['transformed_policy_cov'][it])
    print(f'{algo_stats["dJ_hat_min"][it]} <= dJ_hat <= {algo_stats["dJ_hat_max"][it]} :: Norm={algo_stats["dJ_hat_norm"][it]}')
    print(f'dJ cov: {algo_stats["dJ_covar_diag_min"][it]} <= Mean {algo_stats["dJ_covar_diag_mean"][it]} <= {algo_stats["dJ_covar_diag_max"][it]}')
    print(f'Mean abs. score={algo_stats["mean_abs_score"][it]}')
    print(f'HMC sample reward={algo_stats["sample_reward_mean"][it]:.3f} \u00B1 {algo_stats["sample_reward_sterr"][it]:.3f} '
          f':: Eval reward={algo_stats["reward_mean"][it]:.3f} \u00B1 {algo_stats["reward_sterr"][it]:.3f}\n')


def impsmp_per_parameter(key, n_iters, config, bijector, policy, optimizer, models):
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
         '[impsmp_per_parameter] Please check that batch_size >= (hmc_model.n_rollouts * train_model.n_rollouts).'
        f' batch_size={batch_size}, hmc_model.n_rollouts={hmc_model.n_rollouts}, train_model.n_rollouts={train_model.n_rollouts}')
    assert eval_n_shards > 0, (
        '[impsmp_per_parameter] Please check that eval_batch_size >= eval_model.n_rollouts.'
        f' eval_batch_size={eval_batch_size}, eval_model.n_rollouts={eval_model.n_rollouts}')

    hmc_config = config['hmc']
    hmc_config['num_iters_per_chain'] = int(batch_size // hmc_config["num_chains"])
    assert hmc_config['num_iters_per_chain'] > 0, (
        f'[impsmp_per_parameter] Please check that batch_size >= hmc["num_chains"].'
        f' batch_size={batch_size}, hmc["num_chains"]={hmc_config["num_chains"]}')

    epsilon = config.get('epsilon', 1e-12)

    est_Z = config.get('est_Z', True)
    if hmc_config['init_distribution']['type'] == 'normal':
        hmc_config['init_distribution']['std'] = jnp.sqrt(hmc_config['init_distribution']['var'])

    # initialize stats collection
    algo_stats = {
        'policy_mean': jnp.empty(shape=(n_iters, action_dim)),
        'policy_cov': jnp.empty(shape=(n_iters, action_dim)),
        'transformed_policy_mean': jnp.empty(shape=(n_iters, action_dim)),
        'transformed_policy_cov': jnp.empty(shape=(n_iters, action_dim)),
        'reward_mean': jnp.empty(shape=(n_iters,)),
        'reward_std': jnp.empty(shape=(n_iters,)),
        'reward_sterr': jnp.empty(shape=(n_iters,)),
        'sample_rewards': jnp.empty(shape=(n_iters, batch_size, policy.n_params)),
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
        'scores': jnp.empty(shape=(n_iters, batch_size, policy.n_params)),
        'mean_abs_score': jnp.empty(shape=(n_iters,)),
    }

    # initialize HMC
    key, subkey = jax.random.split(key)
    hmc_initializer = init_hmc_state(subkey, hmc_config['num_chains'], action_dim, policy, hmc_config['init_distribution'])
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
            parallel_log_density_over_chains = jax.vmap(log_density, 0, 0)

            adaptive_hmc_kernel = tfp.mcmc.TransformedTransitionKernel(
                inner_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
                    inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
                        target_log_prob_fn=parallel_log_density_over_chains,
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

            samples = samples.reshape(batch_size, action_dim, 2, 1, action_dim)

            # compute dJ_hat
            key, dJ_hat, batch_stats = impsmp_per_parameter_inner_loop(
                key, policy.theta, samples,
                n_shards, batch_size, epsilon, est_Z,
                policy, hmc_model, train_model)

            updates, opt_state = optimizer.update(dJ_hat, opt_state)
            policy.theta = optax.apply_updates(policy.theta, updates)

            # initialize the next chain at a random point of the current chain
            if hmc_config['reinit_strategy'] == 'random_sample':
                hmc_initializer = init_hmc_state(subkeys[3], hmc_config['num_chains'], action_dim, policy, hmc_config['init_distribution'])
            elif hmc_config['reinit_strategy'] == 'random_prev_chain_elt':
                hmc_initializer = jax.random.choice(
                    subkeys[3],
                    a=samples,
                    shape=(hmc_config["num_chains"],),
                    replace=False)
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
            print_impsmp_report(it, algo_stats, is_accepted,
                                batch_size, hmc_config["num_chains"],
                                subt0, timer())
        except FloatingPointError:
            hmc_step_size = hmc_step_size / 2
            if hmc_step_size < 1e-4:
                break
            for stat_name, stat in algo_stats.items():
                algo_stats[stat_name] = algo_stats[stat_name].at[it].set(stat[it-1])
            print(f'[impsmp] Iteration {it}. Caught FloatingPointError exception. Reducing step size to {hmc_step_size}')

    algo_stats.update({
        'algorithm': 'ImpSmpPerParameter',
        'n_iters': n_iters,
        'config': config,
        'action_dim': action_dim,
        'batch_size': batch_size,
        'eval_batch_size': eval_batch_size,
    })
    return key, algo_stats
