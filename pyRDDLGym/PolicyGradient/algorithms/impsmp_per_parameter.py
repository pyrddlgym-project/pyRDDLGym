"""Importance Sampling with a single density per parameter.

The samples are split into positive and negative parts."""

import jax
import optax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp
import matplotlib.pyplot as plt
import functools
from time import perf_counter as timer
import warnings


def scalar_mult(alpha, v):
    return alpha * v

weighting_map_inner = jax.vmap(scalar_mult, in_axes=0, out_axes=0)
weighting_map = jax.vmap(weighting_map_inner, in_axes=0, out_axes=0)


def diagonal_of_jacobian(key, pi, theta, a):
    """The following computation of the diagonal of the Jacobian matrix
    uses JAX primitives, and works with any policy parametrization, but
    computes the entire Jacobian before taking the diagonal. Therefore,
    the computation time scales rather poorly with increasing dimension.
    There is apparently no natural way in JAX of computing the diagonal
    only without also computing all of the off-diagonal terms.

    See also:
        https://stackoverflow.com/questions/70956578/jacobian-diagonal-computation-in-jax
    """
    dpi = jax.jacrev(pi, argnums=1)(key, theta, a)
    dpi = jax.tree_util.tree_map(lambda x: jnp.diagonal(x, axis1=0, axis2=3), dpi)
    dpi = jax.tree_util.tree_map(lambda x: jnp.diagonal(x, axis1=0, axis2=2), dpi)
    dpi = jax.tree_util.tree_map(lambda x: x[0], dpi)
    return dpi

def analytic_diagonal_of_jacobian(key, pi, theta, a):
    """The following computes the diagonal of the Jacobian analytically.
    It valid ONLY when the policy is parametrized by a normal distribution
    with parameters

        mu_i
        sigma_i^2 = softplus(u_i)

    In this case, it is possible to compute the partials in closed form,
    and avoid computing the off-diagonal terms in the Jacobian.

    The scaling of computation time with increasing dimension seems much
    improved.
    """
    pi_val = pi(key, theta, a)[..., 0]

    theta = theta['linear']['w']
    mu = theta[:, 0]
    u = theta[:, 1]
    sigsq = jax.nn.softplus(u)

    softplus_correction = 1 - (1/(1 + jnp.exp(u)))

    mu_mult = (jnp.diag(a[:,0,0,:]) - mu) / sigsq
    sigsq_mult = 0.5 * softplus_correction * (((jnp.diag(a[:,1,0,:]) - mu) / sigsq) - 1) / sigsq

    partials = jnp.stack([mu_mult, sigsq_mult], axis=1) * pi_val
    return partials


@functools.partial(
    jax.jit,
    static_argnames=(
        'policy',
        'model',
        'importance_weight_upper_cap'))
def unnormalized_rho(key, theta, policy, model, importance_weight_upper_cap, a):
    pi = policy.pdf(key, theta, a)[..., 0]

    #dpi = diagonal_of_jacobian(key, policy.pdf, theta, a)
    #dpi = dpi['linear']['w']
    dpi = analytic_diagonal_of_jacobian(key, policy.pdf, theta, a)

    # compute losses over the first three axes, which index
    # (chain_idx, action_dim_idx, mean_or_var_idx, 1, action_dim_idx)
    #            |<---   parameter indices   --->|

    compute_loss_axis_0 = jax.vmap(model.compute_loss, (None, 0, None), 0)
    compute_loss_axes_0_1 = jax.vmap(compute_loss_axis_0, (None, 0, None), 0)

    losses = compute_loss_axes_0_1(key, a, True)[..., 0]

    rho = losses * dpi

    rho = jnp.abs(rho)
    cutoff_criterion = jnp.logical_and(
        rho > 0.,
        pi / rho >= importance_weight_upper_cap)
    rho = jnp.where(
        cutoff_criterion,
        pi / importance_weight_upper_cap,
        rho)
    return rho

@functools.partial(
    jax.jit,
    static_argnames=(
        'policy',
        'model',
        'importance_weight_upper_cap'))
def unnormalized_log_rho(key, theta, policy, model, importance_weight_upper_cap, a):
    rho = unnormalized_rho(key, theta, policy, model, importance_weight_upper_cap, a)
    return jnp.log(rho)



@functools.partial(
    jax.jit,
    static_argnames=(
        'unnormalized_instrumental_density',
        'batch_size',
        'subsample_size',
        'n_shards',
        'epsilon',
        'importance_weight_upper_cap',
        'Z_est_type',
        'Z_est_sample_size',
        'policy',
        'sampling_model',
        'train_model',))
def impsmp_per_parameter_inner_loop(key, theta, samples, unnormalized_instrumental_density, accepted_matrix,
                                    batch_size, subsample_size, n_shards,
                                    epsilon, importance_weight_upper_cap,
                                    Z_est_type, Z_est_sample_size,
                                    policy, sampling_model, train_model):

    batch_stats = {}

    jacobian = jax.jacrev(policy.pdf, argnums=1)

    compute_loss_axis_1 = jax.vmap(train_model.compute_loss, (None, 1, None), 1)
    compute_loss_axes_1_2 = jax.vmap(compute_loss_axis_1, (None, 1, None), 1)
    batch_compute_loss = jax.vmap(compute_loss_axes_1_2, (None, 1, None), 1)

    batch_unnormalized_rho = jax.vmap(unnormalized_instrumental_density, (None, None, None, None, None, 0), 0)

    sharded_actions = jnp.split(samples, n_shards)
    sharded_actions = jnp.stack(sharded_actions)

    def _compute_pre_dJ_terms_sharded(key, actions):
        # By definition, the "pre_dJ" terms have the form
        #     r(a) * dpi/dtheta_i(a)
        # These are the terms that are necessary to evaluate to determine the
        # signs of the summands that go into the estimate of dJ.
        #
        # The "complete" form for coordinate i is
        #     Z_i * (r(a) / rho_i(a)) * dpi/dtheta_i(a)
        # Z_i will be estimated using samples drawn from pi (performed below),
        # and rho is enough to compute after subsampling (also performed below)
        #
        # Q: the Jacobian for computed for both accepted and rejected samples
        # how can this be made more efficient?
        key, subkey = jax.random.split(key)

        pi = policy.pdf(subkey, theta, actions)[..., 0]
        #dpi = jax.vmap(diagonal_of_jacobian, (None, None, None, 0), 0)(subkey, policy.pdf, theta, actions)
        dpi = jax.vmap(analytic_diagonal_of_jacobian, (None, None, None, 0), 0)(subkey, policy.pdf, theta, actions)
        dpi = {'linear': {'w': dpi}}

        losses = batch_compute_loss(subkey, actions, True)[..., 0]
        pre_dJ_terms = jax.tree_util.tree_map(lambda dpi_term: weighting_map(losses, dpi_term), dpi)

        return key, (pi, losses, pre_dJ_terms)

    key, (pi, losses, pre_dJ_terms) = jax.lax.scan(_compute_pre_dJ_terms_sharded, key, sharded_actions)
    pre_dJ_terms = jax.tree_util.tree_map(lambda term: term.reshape(batch_size, policy.action_dim, 2), pre_dJ_terms)

    def _pos_neg_in_coord(key, pre_dJ, accepted, samples):
        dJ_is_pos = jnp.logical_and((pre_dJ >= 0), accepted)
        dJ_is_neg = jnp.logical_and((pre_dJ < 0), accepted)
        pos_dJ_ind = jnp.argwhere(dJ_is_pos, size=batch_size, fill_value=-1)[..., 0]
        neg_dJ_ind = jnp.argwhere(dJ_is_neg, size=batch_size, fill_value=-1)[..., 0]

        pos_tot = jnp.sum(dJ_is_pos)
        neg_tot = jnp.sum(dJ_is_neg)
        accepted_tot = jnp.sum(accepted)

        pos_correction = pos_tot / accepted_tot
        neg_correction = neg_tot / accepted_tot

        pos_pre_dJ = jnp.take(pre_dJ, pos_dJ_ind[:subsample_size], axis=0)
        pos_subsamples = jnp.take(samples, pos_dJ_ind[:subsample_size], axis=0)
        neg_pre_dJ = jnp.take(pre_dJ, neg_dJ_ind[:subsample_size], axis=0)
        neg_subsamples = jnp.take(samples, neg_dJ_ind[:subsample_size], axis=0)

        return (pos_correction, pos_subsamples, pos_pre_dJ,
                neg_correction, neg_subsamples, neg_pre_dJ)

    key, subkey = jax.random.split(key)
    split_pos_neg_by_coord_axis_1 = jax.vmap(_pos_neg_in_coord, (None, 1, 1, 1), -1)
    split_pos_neg_by_coord = jax.vmap(split_pos_neg_by_coord_axis_1, (None, 1, 1, 1), -2)
    pos_correction, pos_subsamples, pos_pre_dJ, neg_correction, neg_subsamples, neg_pre_dJ = split_pos_neg_by_coord(
        subkey, pre_dJ_terms['linear']['w'], accepted_matrix, samples)

    pos_subsamples = jnp.transpose(pos_subsamples, axes=(0, 3, 4, 1, 2))
    neg_subsamples = jnp.transpose(neg_subsamples, axes=(0, 3, 4, 1, 2))

    key, *subkeys = jax.random.split(key, num=3)
    pos_rho = batch_unnormalized_rho(subkeys[0], theta, policy, sampling_model, importance_weight_upper_cap, pos_subsamples)
    neg_rho = batch_unnormalized_rho(subkeys[1], theta, policy, sampling_model, importance_weight_upper_cap, neg_subsamples)

    dJ_hat = (pos_correction * jnp.mean((pos_pre_dJ / pos_rho), axis=0)
              + neg_correction * jnp.mean((neg_pre_dJ / neg_rho), axis=0))
    dJ_hat = {'linear': {'w': dJ_hat}}



#    key, subkey = jax.random.split(key)
#    dJ_terms = jax.tree_util.tree_map(lambda term: jax.random.choice(key, term, shape=(subsample_size,), replace=False, axis=0), dJ_terms)

    #FIXME: The below assumes a linear policy parametrization
    #batch_stats['scores'] = dJ_summands['linear']['w'].reshape(subsample_size, policy.n_params)

    # estimate of the normalizing factors Z
    if Z_est_type == 'forward':
        def draw_Z_estimate(init_, xs):
            key, accumulant = init_
            key, *subkeys = jax.random.split(key, num=4)
            Z_sample = policy.sample(subkeys[0], theta, (1,))
            pi = policy.pdf(subkeys[1], theta, Z_sample)

            # stack identical copies of the sample for evaluating the instrumental densities rho
            Z_sample = jnp.stack([jnp.stack([Z_sample] * 2)] * policy.action_dim)
            rho = unnormalized_instrumental_density(subkeys[2], theta, policy, sampling_model, importance_weight_upper_cap, Z_sample)

            accumulant = accumulant + (rho / pi)
            return (key, accumulant), None

        (key, accumulant), _ = jax.lax.scan(draw_Z_estimate, (key, jnp.zeros(shape=(policy.action_dim, 2))), xs=None, length=Z_est_sample_size)
        Z = accumulant / Z_est_sample_size
    elif Z_est_type == 'reverse':
        importance_weights = pi / (rho + epsilon)
        Zinv = jnp.mean(importance_weights, axis=(0,1))
        Z = 1 / Zinv

        batch_stats['effective_sample_size'] = jnp.sum(importance_weights, axis=(0,1))**2 / jnp.sum(importance_weights*importance_weights, axis=(0,1))
        batch_stats['importance_weight_range'] = jnp.array([jnp.min(importance_weights), jnp.max(importance_weights)])
    elif Z_est_type == 'none':
        Z = 1.0

    dJ_hat = jax.tree_util.tree_map(lambda item: Z * item, dJ_hat)

    #dJ_summands = jax.tree_util.tree_map(lambda item: item * Z, dJ_summands)
    #dJ_hat = jax.tree_util.tree_map(lambda item: jnp.mean(item, axis=(0,)), dJ_summands)

    batch_stats['Z'] = Z.reshape(policy.n_params)
    #FIXME: The below assumes a linear policy parametrization
    #batch_stats['dJ'] = dJ_summands['linear']['w'].reshape(subsample_size, policy.n_params)
    batch_stats['sample_losses'] = losses.reshape(batch_size, policy.n_params)

    return key, dJ_hat, batch_stats

@functools.partial(
    jax.jit,
    static_argnames=(
        'batch_size',
        'subsample_size',
        'eval_n_shards',
        'eval_batch_size',
        'policy',
        'model',
        'Z_est_type',
        'save_dJ'))
def update_impsmp_stats(
    key,
    it,
    batch_size,
    subsample_size,
    eval_n_shards,
    eval_batch_size,
    algo_stats,
    batch_stats_plus,
    batch_stats_minus,
    samples_plus,
    samples_minus,
    is_accepted_plus,
    is_accepted_minus,
    theta,
    policy,
    model,
    Z_est_type,
    save_dJ):
    """Updates the REINFORCE with Importance Sampling statistics
    using the statistics returned from the computation of dJ_hat
    for the current sample as well as the current policy params theta"""

    key, subkey = jax.random.split(key)
    policy_mean, policy_cov = policy.apply(subkey, theta)
    eval_actions = policy.sample(subkey, theta, (model.n_rollouts,))
    eval_rewards = model.compute_loss(subkey, eval_actions, False)

    algo_stats['policy_mean'] = algo_stats['policy_mean'].at[it].set(policy_mean)
    algo_stats['policy_cov'] = algo_stats['policy_cov'].at[it].set(jnp.diag(policy_cov))
    algo_stats['transformed_policy_mean'] = algo_stats['transformed_policy_mean'].at[it].set(jnp.mean(eval_actions, axis=0))
    algo_stats['transformed_policy_cov'] = algo_stats['transformed_policy_cov'].at[it].set(jnp.std(eval_actions, axis=0)**2)
    algo_stats['reward_mean'] = algo_stats['reward_mean'].at[it].set(jnp.mean(eval_rewards))
    algo_stats['reward_std'] = algo_stats['reward_std'].at[it].set(jnp.std(eval_rewards))
    algo_stats['reward_sterr'] = algo_stats['reward_sterr'].at[it].set(algo_stats['reward_std'][it] / jnp.sqrt(model.n_rollouts))

    return key, algo_stats


    dJ_estimates = jnp.concatenate([batch_stats_plus['dJ'], batch_stats_minus['dJ']], axis=0)
    sample_losses = jnp.concatenate([batch_stats_plus['sample_losses'], batch_stats_minus['sample_losses']], axis=0)
    Z = jnp.stack([batch_stats_plus['Z'], batch_stats_minus['Z']])

    dJ_hat = jnp.mean(dJ_estimates, axis=0)
    dJ_covar = jnp.cov(dJ_estimates, rowvar=False)

    if save_dJ:
        algo_stats['dJ'] = algo_stats['dJ'].at[it].set(dJ_estimates)
    algo_stats['dJ_hat_max'] = algo_stats['dJ_hat_max'].at[it].set(jnp.max(dJ_hat))
    algo_stats['dJ_hat_min'] = algo_stats['dJ_hat_min'].at[it].set(jnp.min(dJ_hat))
    algo_stats['dJ_hat_norm'] = algo_stats['dJ_hat_norm'].at[it].set(jnp.linalg.norm(dJ_hat))
    algo_stats['dJ_covar'] = algo_stats['dJ_covar'].at[it].set(dJ_covar)
    algo_stats['dJ_covar_max'] = algo_stats['dJ_covar_max'].at[it].set(jnp.max(dJ_covar))
    algo_stats['dJ_covar_min'] = algo_stats['dJ_covar_min'].at[it].set(jnp.min(dJ_covar))
    algo_stats['dJ_covar_diag_max']  = algo_stats['dJ_covar_diag_max'].at[it].set(jnp.max(jnp.diag(dJ_covar)))
    algo_stats['dJ_covar_diag_mean'] = algo_stats['dJ_covar_diag_mean'].at[it].set(jnp.mean(jnp.diag(dJ_covar)))
    algo_stats['dJ_covar_diag_min']  = algo_stats['dJ_covar_diag_min'].at[it].set(jnp.min(jnp.diag(dJ_covar)))

    algo_stats['sample_losses'] = algo_stats['sample_losses'].at[it].set(sample_losses)
    algo_stats['sample_loss_mean'] = algo_stats['sample_loss_mean'].at[it].set(jnp.mean(sample_losses))
    algo_stats['sample_loss_std'] = algo_stats['sample_loss_std'].at[it].set(jnp.std(sample_losses))
    algo_stats['sample_loss_sterr'] = algo_stats['sample_loss_sterr'].at[it].set(algo_stats['sample_loss_std'][it] / jnp.sqrt(batch_size))

    algo_stats['mean_plus_score'] = algo_stats['mean_plus_score'].at[it].set(jnp.mean(jnp.abs(batch_stats_plus['scores'])))
    algo_stats['std_plus_score'] = algo_stats['std_plus_score'].at[it].set(jnp.std(jnp.abs(batch_stats_plus['scores'])))
    algo_stats['mean_minus_score'] = algo_stats['mean_minus_score'].at[it].set(jnp.mean(jnp.abs(batch_stats_minus['scores'])))
    algo_stats['std_minus_score'] = algo_stats['std_minus_score'].at[it].set(jnp.std(jnp.abs(batch_stats_minus['scores'])))

    algo_stats['Z'] = algo_stats['Z'].at[it].set(Z)
    if Z_est_type == 'reverse':
        algo_stats['effective_sample_size_plus'] = algo_stats['effective_sample_size_plus'].at[it].set(batch_stats_plus['effective_sample_size'])
        algo_stats['importance_weight_range_plus'] = algo_stats['importance_weight_range_plus'].at[it].set(batch_stats_plus['importance_weight_range'])
        algo_stats['effective_sample_size_minus'] = algo_stats['effective_sample_size_minus'].at[it].set(batch_stats_minus['effective_sample_size'])
        algo_stats['importance_weight_range_minus'] = algo_stats['importance_weight_range_minus'].at[it].set(batch_stats_minus['importance_weight_range'])

    return key, algo_stats

def print_impsmp_report(it, algo_stats, batch_size, sampler, Z_est_type, subt0, subt1):
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
    print(f'Z={algo_stats["Z"][it]}')
    print(f'Mean plus score={algo_stats["mean_plus_score"][it]} \u00B1 {algo_stats["std_plus_score"][it]:.3f}')
    print(f'Mean minus score={algo_stats["mean_minus_score"][it]} \u00B1 {algo_stats["std_minus_score"][it]:.3f}')

    print(f'Z est. type={Z_est_type}')
    if Z_est_type == 'reverse':
        for sign_str in ('plus', 'minus'):
            ess = algo_stats[f'effective_sample_size_{sign_str}'][it]
            ess_min, ess_mean, ess_max = jnp.min(ess), jnp.mean(ess), jnp.max(ess)
            importance_weight_range = algo_stats[f'importance_weight_range_{sign_str}'][it]
            print(f'Effective sample size {sign_str}: {ess_min} <= (Mean) {ess_mean} <= {ess_max}')
            print(f'Importance weight range {sign_str}: {importance_weight_range}')
    print(f'Sample loss={algo_stats["sample_loss_mean"][it]:.3f} \u00B1 {algo_stats["sample_loss_sterr"][it]:.3f} '
          f':: Eval loss={algo_stats["reward_mean"][it]:.3f} \u00B1 {algo_stats["reward_sterr"][it]:.3f}\n')


def impsmp_per_parameter(key, n_iters, config, bijector, policy, sampler, optimizer, models):
    """Runs the REINFORCE with Importance Sampling algorithm"""
    # parse config
    action_dim = policy.action_dim
    batch_size = config['batch_size']
    subsample_size = config.get('subsample_size', batch_size)
    eval_batch_size = config['eval_batch_size']

    sampling_model = models['sampling_model']
    train_model = models['train_model']
    eval_model = models['eval_model']

    assert batch_size % (sampling_model.n_rollouts * train_model.n_rollouts) == 0
    n_shards = int(batch_size / (sampling_model.n_rollouts * train_model.n_rollouts))
    assert n_shards > 0, (
         '[impsmp_per_parameter] Please check that batch_size >= (sampling_model.n_rollouts * train_model.n_rollouts).'
        f' batch_size={batch_size}, sampling_model.n_rollouts={sampling_model.n_rollouts}, train_model.n_rollouts={train_model.n_rollouts}')

    assert eval_batch_size % eval_model.n_rollouts == 0
    eval_n_shards = int(eval_batch_size / eval_model.n_rollouts)
    assert eval_n_shards > 0, (
        '[impsmp_per_parameter] Please check that eval_batch_size >= eval_model.n_rollouts.'
        f' eval_batch_size={eval_batch_size}, eval_model.n_rollouts={eval_model.n_rollouts}')

    # parse the technique for estimating Z
    Z_est = config['Z_est']
    Z_est_type = Z_est['type']
    if Z_est_type == 'forward':
        Z_est_sample_size = Z_est['params']['sample_size']
    elif Z_est_type == 'reverse':
        Z_est_sample_size = -1.0
    elif Z_est_type == 'none':
        Z_est_sample_size = -1.0

    epsilon = config.get('epsilon', 1e-12)
    importance_weight_upper_cap = config['importance_weight_upper_cap']

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
        'sample_losses': jnp.empty(shape=(n_iters, 2*batch_size, policy.n_params)),
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
        'Z': jnp.empty(shape=(n_iters, 2, policy.n_params)),
        'mean_plus_score': jnp.empty(shape=(n_iters,)),
        'std_plus_score': jnp.empty(shape=(n_iters,)),
        'mean_minus_score': jnp.empty(shape=(n_iters,)),
        'std_minus_score': jnp.empty(shape=(n_iters,)),
    }
    if Z_est_type == 'reverse':
        algo_stats['effective_sample_size_plus'] = jnp.empty(shape=(n_iters, action_dim, 2))
        algo_stats['importance_weight_range_plus'] = jnp.empty(shape=(n_iters, 2))
        algo_stats['effective_sample_size_minus'] = jnp.empty(shape=(n_iters, action_dim, 2))
        algo_stats['importance_weight_range_minus'] = jnp.empty(shape=(n_iters, 2))
    if save_dJ:
        algo_stats['dJ'] = jnp.empty(shape=(n_iters, 2*subsample_size, policy.n_params)),

    # initialize optimizer
    opt_state = optimizer.init(policy.theta)

    # initialize unconstraining bijector
    unconstraining_bijector = [
        bijector
    ]

    samples = None

    # run REINFORCE with Importance Sampling
    for it in range(n_iters):
        subt0 = timer()
        key, *subkeys = jax.random.split(key, num=5)

        log_density = functools.partial(
            unnormalized_log_rho,
            subkeys[1],
            policy.theta,
            policy,
            sampling_model,
            importance_weight_upper_cap)

        key = sampler.generate_step_size(key)
        key = sampler.prep(key,
                           it,
                           target_log_prob_fn=log_density,
                           unconstraining_bijector=unconstraining_bijector)
        key = sampler.generate_initial_state(key, it, samples)
        try:
            key, samples, accepted_matrix = sampler.sample(key, policy.theta)
        except FloatingPointError:
            warning_msg = f'[impsmp_per_parameter] Iteration {it}. Caught FloatingPointError exception during sampling of log_rho.'
            warnings.warn(warning_msg)
            for stat_name, stat in algo_stats.items():
                algo_stats[stat_name] = algo_stats[stat_name].at[it].set(stat[it-1])
            continue
        else:
            samples = samples.reshape(batch_size, action_dim, 2, 1, action_dim)
            accepted_matrix = accepted_matrix.reshape(batch_size, action_dim, 2)
            is_accepted = jnp.copy(accepted_matrix)
            #@@@
            accepted_matrix = True * jnp.ones(shape=accepted_matrix.shape)
            #@@@


        # compute dJ_hat
        key, dJ_hat, batch_stats = impsmp_per_parameter_inner_loop(
            unnormalized_instrumental_density=unnormalized_rho,
            key=key,
            theta=policy.theta,
            samples=samples,
            accepted_matrix=accepted_matrix,
            # constant parameters:
            batch_size=batch_size,
            subsample_size=subsample_size,
            n_shards=n_shards,
            epsilon=epsilon,
            importance_weight_upper_cap=importance_weight_upper_cap,
            Z_est_type=Z_est_type,
            Z_est_sample_size=Z_est_sample_size,
            policy=policy,
            sampling_model=sampling_model,
            train_model=train_model)

        updates, opt_state = optimizer.update(dJ_hat, opt_state)
        policy.theta = optax.apply_updates(policy.theta, updates)

        theta_lower_cap = np.log(np.exp(0.05) - 1)
        policy.theta['linear']['w'] = policy.theta['linear']['w'].at[:,1].set(jnp.maximum(policy.theta['linear']['w'][:,1], theta_lower_cap))

        # update stats and printout results for the current iteration
        key, algo_stats = update_impsmp_stats(key, it,
                                              batch_size, subsample_size,
                                              eval_n_shards, eval_batch_size,
                                              algo_stats, batch_stats, batch_stats,
                                              samples, accepted_matrix,
                                              samples, accepted_matrix,
                                              policy.theta, policy, eval_model,
                                              Z_est_type, save_dJ)
        sampler.update_stats(it, samples, is_accepted)
        if config['verbose']:
            print_impsmp_report(it, algo_stats, batch_size, sampler, Z_est_type, subt0, timer())

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
