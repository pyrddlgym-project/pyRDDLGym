import jax
import optax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp
import matplotlib.pyplot as plt
import functools

def init_hmc_state(key, n_chains, action_dim, config):
    shape = (n_chains, action_dim)
    if config['type'] == 'uniform':
        init_state = jax.random.uniform(
            key,
            shape=shape,
            minval=config['min'],
            maxval=config['max'])
    elif config['type'] == 'normal':
        init_state = jax.random.normal(
            key,
            shape=shape)
        init_state = config['mean'] + init_state * config['std']
    else:
        raise ValueError('[init_hmc_state] Unrecognized distribution type')
    # repeat the initializer per-parameter
    # this adds two axes to the initial HMC state,
    # one corresponding the action_dim, and one distinguishing mean vs. var
    # (together, they parametrize m1, ..., md, v1, ..., vd)
    init_state = jnp.stack([init_state] * action_dim, axis=1)
    init_state = jnp.stack([init_state] * 2, axis=2)
    return init_state

@functools.partial(jax.jit, static_argnames=('policy', 'model'))
def unnormalized_rho(key, theta, policy, model, a):
    dpi = jax.jacrev(policy.pdf, argnums=1)(key, theta, a)
    dpi = jax.tree_util.tree_map(lambda x: jnp.diagonal(x, axis1=1, axis2=3), dpi)
    dpi = jax.tree_util.tree_map(lambda x: jnp.diagonal(x, axis1=1, axis2=2), dpi)
    dpi_abs = jax.tree_util.tree_map(lambda x: jnp.abs(x), dpi)

    # compute losses over the first three axes, which index
    # (chain_idx, action_dim_idx, mean_or_var_idx, action_dim_idx)
    #            |<---   parameter indices   --->|

    compute_loss_axis_1 = jax.vmap(model.compute_loss, (None, 1), 1)
    compute_loss_axes_1_2 = jax.vmap(compute_loss_axis_1, (None, 1), 1)
    losses = compute_loss_axes_1_2(key, a)[...,0]
    losses = jnp.abs(losses)

    density = losses * dpi_abs['linear']['w']
    return density

@functools.partial(jax.jit, static_argnames=('policy', 'model', 'epsilon'))
def unnormalized_log_rho(key, theta, policy, model, epsilon, a):
    density = unnormalized_rho(key, theta, policy, model, a)
    return jnp.log(density)


def impsmp_per_parameter_analyze_2d_sampling(key, n_iters, config, bijector, policy, optimizer, models):
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

    # initialize HMC
    key, subkey = jax.random.split(key)
    hmc_initializer = init_hmc_state(subkey, hmc_config['num_chains'], action_dim, hmc_config['init_distribution'])
    hmc_step_size = hmc_config['init_step_size']

    # initialize unconstraining bijector
    unconstraining_bijector = [
        bijector
    ]

    # run REINFORCE with Importance Sampling
    key, *subkeys = jax.random.split(key, num=5)

    log_density = functools.partial(
        unnormalized_log_rho, subkeys[1], policy.theta, policy, hmc_model, epsilon)

    lb, ub, step = -5., 5., 0.2
    n_pts = int((ub-lb)/step)

    test_range = jnp.arange(lb, ub, step=step)
    test_range_extended = jnp.broadcast_to(test_range[jnp.newaxis, jnp.newaxis, jnp.newaxis, jnp.newaxis, :], shape=(1, 2, 2, 1, n_pts))

    IDs = np.zeros(shape=(n_pts,n_pts,2,2))

    for ix in range(n_pts):
        x = test_range_extended[...,ix]
        for iy in range(n_pts):
            y = test_range_extended[...,iy]
            z = jnp.concatenate((x,y), axis=3)
            IDs[ix,iy] = jnp.exp(log_density(z)[0])

    fig, ax = plt.subplots(2,6)
    fig.set_size_inches(28,14)

    for i in range(2):
        for j in range(2):
            ax[i,j].imshow(IDs[...,i,j], interpolation='none', origin='lower', extent=(lb,ub,lb,ub))
            ax[i,j].set_title(f'Instrumental density. Param ({i},{j})')

    adaptive_hmc_kernel = tfp.mcmc.TransformedTransitionKernel(
        inner_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
            inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=log_density,
                num_leapfrog_steps=hmc_config['num_leapfrog_steps'],
                step_size=hmc_step_size),
            num_adaptation_steps=int(hmc_config['num_burnin_iters_per_chain'] * 0.8)),
        bijector=unconstraining_bijector)


    for (batch, index_delta) in [(1024, 2), (32768, 4)]:
        samples, is_accepted = tfp.mcmc.sample_chain(
            seed=subkeys[2],
            num_results=batch,
            num_burnin_steps=hmc_config['num_burnin_iters_per_chain'],
            current_state=hmc_initializer,
            kernel=adaptive_hmc_kernel,
            trace_fn=lambda _, pkr: pkr.inner_results.inner_results.is_accepted)

        bins = np.zeros(shape=(n_pts, n_pts, 2, 2))
        coords = np.floor((samples - lb)/step).astype(np.int32)

        for ib in range(batch):
            for i in range(2):
                for j in range(2):
                    cx, cy = coords[ib, 0, i, j]
                    bins[cx, cy, i, j] += 1


        for i in range(2):
            for j in range(2):
                ax[i,j+index_delta].imshow(bins[...,i,j], interpolation='none', origin='lower', extent=(lb,ub,lb,ub))
                ax[i,j+index_delta].set_title(f'{batch} Samples. Param ({i},{j})')



    plt.suptitle(f'Comparing the instrumental densities and the distributions of HMC samples for the four policy parameters\n'
                  'Env: Dim=2, Summands=10\n'
                  'HMC Settings:'
                 f' Burnin={hmc_config["num_burnin_iters_per_chain"]},'
                 f' Step size={hmc_step_size},'
                 f' Num. leapfrog={hmc_config["num_leapfrog_steps"]}')

    plt.tight_layout()
    plt.savefig('tmp.png')

    return key, {}
