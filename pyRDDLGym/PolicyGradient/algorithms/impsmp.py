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



@functools.partial(jax.jit, static_argnames=('policy', 'model'))
def unnormalized_rho(key, theta, policy, model, a):
    dpi = jax.jacrev(policy.pdf, argnums=1)(key, theta, a)
    dpi_norm = jax.tree_util.tree_reduce(
#        lambda x, y: jnp.maximum(x, jnp.max(jnp.abs(y))),
        lambda x, y: jnp.minimum(x, jnp.min(jnp.abs(y))),
        dpi,
        initializer=jnp.array([-jnp.inf]))

    losses = model.compute_loss(key, a, True)
    losses = jnp.abs(losses)
    density = losses * dpi_norm
    return density

@functools.partial(jax.jit, static_argnames=('policy', 'model', 'log_cutoff'))
def unnormalized_log_rho(key, theta, policy, model, log_cutoff, a):
    density = unnormalized_rho(key, theta, policy, model, a)
    return jnp.maximum(log_cutoff, jnp.log(density))


@functools.partial(
    jax.jit,
    static_argnames=(
        'batch_size',
        'subsample_size',
        'n_shards',
        'epsilon',
        'est_Z',
        'policy',
        'sampling_model',
        'train_model',))
def impsmp_inner_loop(key, theta, samples,
                      batch_size, subsample_size, n_shards, epsilon, est_Z,
                      policy, sampling_model, train_model):

    batch_stats = {}
    jacobian = jax.jacrev(policy.pdf, argnums=1)

    def compute_dJ_hat_summands_for_shard(key, actions):
        key, subkey = jax.random.split(key)

        pi = policy.pdf(subkey, theta, actions)
        dpi = jacobian(subkey, theta, actions)

        batch_compute_loss = jax.vmap(train_model.compute_loss, (None, 1, None), 1)
        losses = batch_compute_loss(subkey, actions, True)

        batch_unnormalized_rho = jax.vmap(unnormalized_rho, (None, None, None, None, 0), 0)
        rho = batch_unnormalized_rho(subkey, theta, policy, sampling_model, actions)

        factors = losses / (rho + epsilon)
        dJ_summands = jax.tree_util.tree_map(lambda dpi_term: weighting_map(factors, dpi_term), dpi)

        return key, (pi, rho, losses, dJ_summands)

    sharded_actions = jnp.split(samples, n_shards)
    sharded_actions = jnp.stack(sharded_actions)
    key, (pi, rho, losses, dJ_summands) = jax.lax.scan(compute_dJ_hat_summands_for_shard, key, sharded_actions)

    key, subkey = jax.random.split(key)
    dJ_summands = jax.tree_util.tree_map(lambda term: term.reshape(batch_size, policy.action_dim, 2), dJ_summands)
    dJ_summands = jax.tree_util.tree_map(lambda term: jax.random.choice(key, term, shape=(subsample_size,), replace=False, axis=0), dJ_summands)
    #FIXME: The below assumes a linear policy parametrization
    batch_stats['scores'] = dJ_summands['linear']['w'].reshape(subsample_size, policy.n_params)

    # estimate of the normalizing factors Z
    if est_Z:
        Zinv = jnp.mean(pi/(rho + epsilon), axis=(0,1))
        dJ_summands = jax.tree_util.tree_map(lambda item: item / Zinv, dJ_summands)
        batch_stats['Zinv'] = Zinv[0]

    dJ_hat = jax.tree_util.tree_map(lambda item: jnp.mean(item, axis=(0,)), dJ_summands)

    #FIXME: The below assumes a linear policy parametrization
    batch_stats['dJ'] = dJ_summands['linear']['w'].reshape(subsample_size, policy.n_params)
    batch_stats['sample_losses'] = losses.reshape(batch_size,)

    return key, dJ_hat, batch_stats

@functools.partial(
    jax.jit,
    static_argnames=(
        'batch_size',
        'eval_n_shards',
        'eval_batch_size',
        'policy',
        'model',
        'est_Z',
        'save_dJ'))
def update_impsmp_stats(
    key,
    it,
    batch_size,
    eval_n_shards,
    eval_batch_size,
    algo_stats,
    batch_stats,
    samples,
    is_accepted,
    theta,
    policy,
    model,
    est_Z,
    save_dJ):
    """Updates the REINFORCE with Importance Sampling statistics
    using the statistics returned from the computation of dJ_hat
    for the current sample as well as the current policy params theta"""

    key, subkey = jax.random.split(key)
    policy_mean, policy_cov = policy.apply(subkey, theta)
    eval_actions = policy.sample(subkey, theta, (model.n_rollouts,))
    eval_rewards = model.compute_loss(subkey, eval_actions, False)

    dJ_hat = jnp.mean(batch_stats['dJ'], axis=0)
    dJ_covar = jnp.cov(batch_stats['dJ'], rowvar=False).reshape(batch_size, batch_size)

    if save_dJ:
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

    algo_stats['sample_losses']     = algo_stats['sample_losses'].at[it].set(batch_stats['sample_losses'])
    algo_stats['sample_loss_mean']  = algo_stats['sample_loss_mean'].at[it].set(jnp.mean(batch_stats['sample_losses']))
    algo_stats['sample_loss_std']   = algo_stats['sample_loss_std'].at[it].set(jnp.std(batch_stats['sample_losses']))
    algo_stats['sample_loss_sterr'] = algo_stats['sample_loss_sterr'].at[it].set(algo_stats['sample_loss_std'][it] / jnp.sqrt(batch_size))

    algo_stats['mean_abs_score']      = algo_stats['mean_abs_score'].at[it].set(jnp.mean(jnp.abs(batch_stats['scores'])))
    algo_stats['std_abs_score']       = algo_stats['std_abs_score'].at[it].set(jnp.std(jnp.abs(batch_stats['scores'])))

    if est_Z:
        algo_stats['Zinv'] = algo_stats['Zinv'].at[it].set(batch_stats['Zinv'])

    return key, algo_stats

def print_impsmp_report(it, algo_stats, batch_size, sampler, est_Z, subt0, subt1):
    """Prints out the results for the current REINFORCE with Importance Sampling iteration"""
    print(f'Iter {it} :: Importance Sampling :: Runtime={subt1-subt0}s')
    sampler.print_report(it)
    print(f'Untransformed parametrized policy [Mean, Diag(Cov)] =')
    print(algo_stats['policy_mean'][it])
    print(algo_stats['policy_cov'][it])
    print(f'Transformed action sample statistics [Mean, StDev] =')
    print(algo_stats['transformed_policy_mean'][it])
    print(algo_stats['transformed_policy_cov'][it])
    print(f'{algo_stats["dJ_hat_min"][it]} <= dJ_hat <= {algo_stats["dJ_hat_max"][it]} :: Norm={algo_stats["dJ_hat_norm"][it]}')
    print(f'dJ cov: {algo_stats["dJ_covar_diag_min"][it]} <= (Mean) {algo_stats["dJ_covar_diag_mean"][it]} <= {algo_stats["dJ_covar_diag_max"][it]}')
    if est_Z:
        print(f'Z={1/algo_stats["Zinv"][it]}')
    print(f'Mean abs. score={algo_stats["mean_abs_score"][it]} \u00B1 {algo_stats["std_abs_score"][it]:.3f}')
    print(f'Sample loss={algo_stats["sample_loss_mean"][it]:.3f} \u00B1 {algo_stats["sample_loss_sterr"][it]:.3f} '
          f':: Eval loss={algo_stats["reward_mean"][it]:.3f} \u00B1 {algo_stats["reward_sterr"][it]:.3f}\n')


def impsmp(key, n_iters, config, bijector, policy, sampler, optimizer, models):
    """Runs the REINFORCE with Importance Sampling algorithm"""
    # parse config
    action_dim = policy.action_dim
    batch_size = config['batch_size']
    subsample_size = config.get('subsample_size', batch_size)
    eval_batch_size = config['eval_batch_size']

    sampling_model = models['sampling_model']
    train_model = models['train_model']
    eval_model = models['eval_model']

    n_shards = int(batch_size / (sampling_model.n_rollouts * train_model.n_rollouts))
    eval_n_shards = int(eval_batch_size / eval_model.n_rollouts)
    assert n_shards > 0, (
         '[impsmp] Please check that batch_size >= (sampling_model.n_rollouts * train_model.n_rollouts).'
        f' batch_size={batch_size}, sampling_model.n_rollouts={sampling_model.n_rollouts}, train_model.n_rollouts={train_model.n_rollouts}')
    assert eval_n_shards > 0, (
        '[impsmp] Please check that eval_batch_size >= eval_model.n_rollouts.'
        f' eval_batch_size={eval_batch_size}, eval_model.n_rollouts={eval_model.n_rollouts}')

    est_Z = config.get('est_Z', True)

    epsilon = config.get('epsilon', 1e-12)
    log_cutoff = config.get('log_cutoff', -1000.)

    save_dJ = config.get('save_dJ', False)

    # initialize stats collection
    algo_stats = {
        'policy_mean': jnp.empty(shape=(n_iters, action_dim)),
        'policy_cov': jnp.empty(shape=(n_iters, action_dim)),
        'transformed_policy_mean': jnp.empty(shape=(n_iters, action_dim)),
        'transformed_policy_cov': jnp.empty(shape=(n_iters, action_dim)),
        'reward_mean': jnp.empty(shape=(n_iters,)),
        'reward_std': jnp.empty(shape=(n_iters,)),
        'reward_sterr': jnp.empty(shape=(n_iters,)),
        'sample_losses': jnp.empty(shape=(n_iters, batch_size,)),
        'sample_loss_mean': jnp.empty(shape=(n_iters,)),
        'sample_loss_std': jnp.empty(shape=(n_iters,)),
        'sample_loss_sterr': jnp.empty(shape=(n_iters,)),
        'dJ_hat_max': jnp.empty(shape=(n_iters,)),
        'dJ_hat_min': jnp.empty(shape=(n_iters,)),
        'dJ_hat_norm': jnp.empty(shape=(n_iters,)),
        'dJ_covar': jnp.empty(shape=(n_iters, policy.n_params, policy.n_params)),
        'dJ_covar_max': jnp.empty(shape=(n_iters,)),
        'dJ_covar_min': jnp.empty(shape=(n_iters,)),
        'dJ_covar_diag_max': jnp.empty(shape=(n_iters,)),
        'dJ_covar_diag_mean': jnp.empty(shape=(n_iters,)),
        'dJ_covar_diag_min': jnp.empty(shape=(n_iters,)),
        'Zinv': jnp.empty(shape=(n_iters,)),
        'mean_abs_score': jnp.empty(shape=(n_iters,)),
        'std_abs_score': jnp.empty(shape=(n_iters,)),
    }

    if save_dJ:
        algo_stats['dJ'] = jnp.empty(shape=(n_iters, subsample_size, policy.n_params))


    # initialize sampler
    key = sampler.generate_initial_state(key)
    key = sampler.generate_step_size(key)

    # initialize optimizer
    opt_state = optimizer.init(policy.theta)

    # initialize unconstraining bijector
    unconstraining_bijector = [
        bijector
    ]

    # run REINFORCE with Importance Sampling
    for it in range(n_iters):
        subt0 = timer()
        key, *subkeys = jax.random.split(key, num=5)

        log_density = functools.partial(
            unnormalized_log_rho, subkeys[1], policy.theta, policy, sampling_model, log_cutoff)

        key = sampler.generate_step_size(key)
        key = sampler.prep(key,
                           target_log_prob_fn=log_density,
                           unconstraining_bijector=unconstraining_bijector)
        try:
            key, samples, is_accepted = sampler.sample(key, policy.theta)
        except FloatingPointError:
            for stat_name, stat in algo_stats.items():
                algo_stats[stat_name] = algo_stats[stat_name].at[it].set(stat[it-1])

            key = sampler.generate_initial_state(key)
            key = sampler.generate_step_size(key)

            print(f'[impsmp] Iteration {it}. Caught FloatingPointError exception.'
                  f' The sampler has been reinitialized.')

        else:
            samples = samples.reshape(batch_size, 1, action_dim)

            # compute dJ_hat
            key, dJ_hat, batch_stats = impsmp_inner_loop(
                key, policy.theta, samples, batch_size, subsample_size,
                n_shards, epsilon, est_Z,
                policy, sampling_model, train_model)

            updates, opt_state = optimizer.update(dJ_hat, opt_state)
            policy.theta = optax.apply_updates(policy.theta, updates)

            # initialize the next chain
            if config['reinit_strategy'] == 'random_sample':
                key = sampler.generate_initial_state(key)

            elif config['reinit_strategy'] == 'random_prev_chain_elt':
                sampler.set_initial_state(
                    jax.random.choice(
                        subkeys[3],
                        a=samples,
                        shape=(sampler.config["num_chains"],),
                        replace=False))

            #elif config['reinit_strategy'] == 'random_prev_chain_elt_with_intermixing':
            #    hmc_initializer = jax.random.choice(
            #        subkeys[3],
            #        a=samples,
            #        shape=(sampler.config["num_chains"],),
            #        replace=False)

            #    log_density = functools.partial(
            #        unnormalized_log_rho, subkeys[1], policy.theta, policy, models['mixing_model'], log_cutoff)
            #    parallel_log_density_over_chains = jax.vmap(log_density, 0, 0)

            #    intermediate_hmc_kernel = sampler_factory(
            #        target_log_prob_fn=parallel_log_density_over_chains,
            #        step_size=sampler.config['reinit_step_size'],
            #        unconstraining_bijector=unconstraining_bijector,
            #        **hmc_config)

            #    hmc_initializer, _ = tfp.mcmc.sample_chain(
            #        seed=subkeys[3],
            #        num_results=1,
            #        num_burnin_steps=sampler.config['reinit_num_burnin_iters_per_chain'],
            #        current_state=hmc_initializer,
            #        kernel=intermediate_hmc_kernel,
            #        trace_fn=lambda _, pkr: pkr.inner_results.inner_results.is_accepted)
            #    hmc_initializer = hmc_initializer[0]
            else:
                raise ValueError('[impsmp] Unrecognized sampler reinitialization strategy '
                                f'{hmc_config["reinit_strategy"]}. Expect "random_sample" '
                                 'or "random_prev_chain_elt"')

            # update stats and printout results for the current iteration
            key, algo_stats = update_impsmp_stats(key, it,
                                                  batch_size,
                                                  eval_n_shards, eval_batch_size,
                                                  algo_stats, batch_stats,
                                                  samples, is_accepted,
                                                  policy.theta, policy, eval_model,
                                                  est_Z, save_dJ)
            sampler.update_stats(it, samples, is_accepted)
            if config['verbose']:
                print_impsmp_report(it, algo_stats, batch_size, sampler, est_Z, subt0, timer())

    algo_stats.update({
        'algorithm': 'ImpSmpPerParameter',
        'n_iters': n_iters,
        'config': config,
        'action_dim': action_dim,
        'batch_size': batch_size,
        'eval_batch_size': eval_batch_size,
        'sampling_model_weight': sampling_model.weight,
        'sampler_stats': sampler.stats,
    })
    return key, algo_stats
