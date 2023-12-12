import numpy as np
import jax
import jax.numpy as jnp
import optax
import functools
from time import perf_counter as timer

def scalar_mult(alpha, v):
    return alpha * v

weighting_map = jax.vmap(
    scalar_mult, in_axes=0, out_axes=0)

def flatten_dJ_shard(dJ_shard, shard_fr, shard_to, flat_dJ_shard):
    ip0 = 0
    for leaf in jax.tree_util.tree_leaves(dJ_shard):
        leaf = leaf.reshape(shard_to-shard_fr, -1)
        ip1 = ip0 + leaf.shape[1]
        flat_dJ_shard = flat_dJ_shard.at[shard_fr:shard_to, ip0:ip1].set(leaf)
        ip0 = ip1
    return flat_dJ_shard


@functools.partial(jax.jit, static_argnames=('n_shards', 'batch_size', 'epsilon', 'policy', 'model'))
def reinforce_inner_loop(key, theta, n_shards, batch_size, epsilon, policy, model):
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
        'actions': jnp.empty(shape=(batch_size, policy.action_dim)),
        'dJ': jnp.empty(shape=(batch_size, policy.n_params)),
    }

    jacobian = jax.jacrev(policy.pdf, argnums=1)

    # compute dJ hat for the current sample, dividing the computation up
    # into shards of size model.n_rollouts (for example, this can be
    # useful for fiting the computation into GPU VRAM)
    for si in range(n_shards):
        shard_fr, shard_to = si*(model.n_rollouts), (si+1)*(model.n_rollouts)

        key, subkey = jax.random.split(key)
        actions = policy.sample(subkey, theta, (model.n_rollouts,))
        pi = policy.pdf(subkey, theta, actions)
        dpi = jacobian(subkey, theta, actions)
        rewards = jnp.squeeze(model.compute_loss(subkey, actions))
        factors = rewards / (pi + epsilon)

        dJ_shard = jax.tree_util.tree_map(lambda dpi_term: weighting_map(factors, dpi_term), dpi)

        # update the dJ estimator dJ_hat
        dJ_shard_find_mean_fn = lambda x: jnp.mean(x, axis=0)
        accumulant = jax.tree_util.tree_map(dJ_shard_find_mean_fn, dJ_shard)
        dJ_hat = jax.tree_util.tree_map(lambda x, y: x+(y/n_shards), dJ_hat, accumulant)

        # collect statistics for the batch
        batch_stats['dJ'] = flatten_dJ_shard(dJ_shard, shard_fr, shard_to, batch_stats['dJ'])
        batch_stats['actions'] = batch_stats['actions'].at[shard_fr:shard_to].set(actions)

    return key, dJ_hat, batch_stats

@functools.partial(jax.jit, static_argnames=('eval_n_shards', 'eval_batch_size', 'policy', 'model'))
def evaluate_policy(key, it, algo_stats, eval_n_shards, eval_batch_size, theta, policy, model):
    key, subkey = jax.random.split(key)
    policy_mean, policy_cov = policy.apply(subkey, theta)
    actions = policy.sample(subkey, theta, (model.n_rollouts,))
    rewards = model.compute_loss(subkey, actions)

    algo_stats['reward_mean']        = algo_stats['reward_mean'].at[it].set(jnp.mean(rewards))
    algo_stats['reward_std']         = algo_stats['reward_std'].at[it].set(jnp.std(rewards))
    algo_stats['reward_sterr']       = algo_stats['reward_sterr'].at[it].set(algo_stats['reward_std'][it] / jnp.sqrt(model.n_rollouts))
    algo_stats['policy_mean']        = algo_stats['policy_mean'].at[it].set(policy_mean)
    algo_stats['policy_cov']         = algo_stats['policy_cov'].at[it].set(jnp.diag(policy_cov))
    return key, algo_stats

@jax.jit
def update_reinforce_stats(key, it, algo_stats, batch_stats):
    """Updates the REINFORCE statistics using the statistics returned
    from the computation of dJ_hat for the current sample as well as
    the current policy mean/cov """
    dJ_hat = jnp.mean(batch_stats['dJ'], axis=0)
    dJ_covar = jnp.cov(batch_stats['dJ'], rowvar=False)

    algo_stats['dJ']                 = algo_stats['dJ'].at[it].set(batch_stats['dJ'])
    algo_stats['dJ_hat_max']         = algo_stats['dJ_hat_max'].at[it].set(jnp.max(dJ_hat))
    algo_stats['dJ_hat_min']         = algo_stats['dJ_hat_min'].at[it].set(jnp.min(dJ_hat))
    algo_stats['dJ_hat_norm']        = algo_stats['dJ_hat_norm'].at[it].set(jnp.linalg.norm(dJ_hat))
    algo_stats['dJ_covar_max']       = algo_stats['dJ_covar_max'].at[it].set(jnp.max(dJ_covar))
    algo_stats['dJ_covar_min']       = algo_stats['dJ_covar_min'].at[it].set(jnp.min(dJ_covar))
    algo_stats['dJ_covar_diag_max']  = algo_stats['dJ_covar_diag_max'].at[it].set(jnp.max(jnp.diag(dJ_covar)))
    algo_stats['dJ_covar_diag_min']  = algo_stats['dJ_covar_diag_min'].at[it].set(jnp.min(jnp.diag(dJ_covar)))
    algo_stats['dJ_covar_diag_mean'] = algo_stats['dJ_covar_diag_mean'].at[it].set(jnp.mean(jnp.diag(dJ_covar)))
    algo_stats['transformed_policy_sample_mean'] = algo_stats['transformed_policy_sample_mean'].at[it].set(jnp.squeeze(jnp.mean(batch_stats['actions'], axis=0)))
    algo_stats['transformed_policy_sample_cov']  = algo_stats['transformed_policy_sample_cov'].at[it].set(jnp.squeeze(jnp.std(batch_stats['actions'], axis=0))**2)
    return key, algo_stats

def print_reinforce_report(it, algo_stats, subt0, subt1):
    """Prints out the results for the current REINFORCE iteration to console"""
    print(f'Iter {it} :: REINFORCE :: Runtime={subt1-subt0}s')
    print(f'Untransformed parametrized policy [Mean, Diag(Cov)] =')
    print(algo_stats['policy_mean'][it])
    print(algo_stats['policy_cov'][it])
    print(f'Transformed action sample statistics [[Means], [StDevs]] =')
    print(algo_stats['transformed_policy_sample_mean'][it])
    print(algo_stats['transformed_policy_sample_cov'][it])
    print(algo_stats['dJ_hat_min'][it], '<= dJ <=', algo_stats['dJ_hat_max'][it], f':: dJ norm={algo_stats["dJ_hat_norm"][it]}')
    print('dJ covar:', algo_stats['dJ_covar_diag_min'][it], '<= Mean', algo_stats["dJ_covar_diag_mean"][it], ' <=', algo_stats["dJ_covar_diag_max"][it])
    print(f'Eval. reward={algo_stats["reward_mean"][it]:.3f} \u00B1 {algo_stats["reward_sterr"][it]:.3f}\n')


def reinforce(key, n_iters, config, bijector, policy, optimizer, models):
    """Runs the REINFORCE algorithm"""

    action_dim = policy.action_dim
    batch_size = config['batch_size']
    eval_batch_size = config['eval_batch_size']

    train_model = models['train_model']
    eval_model = models['eval_model']

    n_shards = int(batch_size // train_model.n_rollouts)
    eval_n_shards = int(eval_batch_size // eval_model.n_rollouts)
    assert n_shards > 0, (
         '[reinforce] Please check that batch_size > train_model.n_rollouts.'
        f' batch_size={batch_size}, train_model.n_rollouts={train_model.n_rollouts}')
    assert eval_n_shards > 0, (
        '[reinforce] Please check that eval_batch_size > eval_model.n_rollouts.'
        f' eval_batch_size={eval_batch_size}, eval_model.n_rollouts={eval_model.n_rollouts}')

    epsilon = config.get('epsilon', 1e-12)

    # initialize stats collection
    algo_stats = {
        'action_dim': action_dim,
        'batch_size': batch_size,
        'eval_batch_size': eval_batch_size,
        'reward_mean':        jnp.empty(shape=(n_iters,)),
        'reward_std':         jnp.empty(shape=(n_iters,)),
        'reward_sterr':       jnp.empty(shape=(n_iters,)),
        'policy_mean':        jnp.empty(shape=(n_iters, action_dim)),
        'policy_cov':         jnp.empty(shape=(n_iters, action_dim)),
        'transformed_policy_sample_mean': jnp.empty(shape=(n_iters, action_dim)),
        'transformed_policy_sample_cov':  jnp.empty(shape=(n_iters, action_dim)),
        'dJ':                 jnp.empty(shape=(n_iters, batch_size, policy.n_params)),
        'dJ_hat_max':         jnp.empty(shape=(n_iters,)),
        'dJ_hat_min':         jnp.empty(shape=(n_iters,)),
        'dJ_hat_norm':        jnp.empty(shape=(n_iters,)),
        'dJ_covar_max':       jnp.empty(shape=(n_iters,)),
        'dJ_covar_min':       jnp.empty(shape=(n_iters,)),
        'dJ_covar_diag_max':  jnp.empty(shape=(n_iters,)),
        'dJ_covar_diag_min':  jnp.empty(shape=(n_iters,)),
        'dJ_covar_diag_mean': jnp.empty(shape=(n_iters,)),
    }

    # initialize optimizer
    opt_state = optimizer.init(policy.theta)

    # run REINFORCE
    for it in range(n_iters):
        subt0 = timer()

        key, algo_stats = evaluate_policy(key, it, algo_stats, eval_n_shards, eval_batch_size, policy.theta, policy, eval_model)

        key, dJ_hat, batch_stats = reinforce_inner_loop(
            key, policy.theta,
            n_shards, batch_size, epsilon, policy, train_model)

        updates, opt_state = optimizer.update(dJ_hat, opt_state)
        policy.theta = optax.apply_updates(policy.theta, updates)

        # update statistics and print out report for current iteration
        key, algo_stats = update_reinforce_stats(key, it, algo_stats, batch_stats)
        if config.get('verbose', False):
            print_reinforce_report(it, algo_stats, subt0, timer())

    algo_stats.update({
        'algorithm': 'REINFORCE',
        'n_iters': n_iters,
        'config': config,
    })
    return key, algo_stats
