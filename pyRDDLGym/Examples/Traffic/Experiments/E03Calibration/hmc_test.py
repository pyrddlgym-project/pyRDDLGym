from tensorflow_probability.substrates import jax as tfp
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt


# Log-density definitions (unnormalized)
# Target distribution is proportional to: `exp(-x (1 + x))`
# Note: exp(-0.5*((x+0.5)/sqrt(1/2))^2) = Cexp(-x^2 - x)
# So this is proportional to the density of N(-0.5, 1/sqrt(2)^2)
def unnormalized_log_prob(x):
    return -x - x**2

# N(0,1^2)
def unnormalized_log_prob2(x):
    return -0.5*x**2

# Assume density factors as product
# log(rho(x,y)) = log(rho1(x)rho2(y)) = log(rho1(x)) + log(rho2(y))
def two_dim_unnormalized_log_prob(X):
    return unnormalized_log_prob(X[0]) + unnormalized_log_prob2(X[1])

v_2d_rho = jax.vmap(two_dim_unnormalized_log_prob, in_axes=0, out_axes=0)

def batch_two_dim_unnormalized_log_prob(X):
    return jnp.sum(v_2d_rho(X), axis=-1)



# Runs the chain (with burn-in) with one-dimensional density rho(x)
def run_chain(seed, initial_state, num_results):

    num_burnin_steps = int(num_results/10)

    adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
        tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=unnormalized_log_prob2,
            num_leapfrog_steps=3,
            step_size=1.),
        num_adaptation_steps=int(num_burnin_steps * 0.8))

    samples, is_accepted = tfp.mcmc.sample_chain(
        seed=seed,
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=initial_state,
        kernel=adaptive_hmc,
        trace_fn=lambda _, pkr: pkr.inner_results.is_accepted)

    return samples, is_accepted

#Runs chain with the two-dimensional density rho(X) = rho(x,y)
def run_chain_2d_density(seed, initial_state, num_results):

    num_burnin_steps = int(num_results/10)

    adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
        tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=batch_two_dim_unnormalized_log_prob,
            num_leapfrog_steps=3,
            step_size=1.),
        num_adaptation_steps=int(num_burnin_steps * 0.8))

    samples, is_accepted = tfp.mcmc.sample_chain(
        seed=seed,
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=initial_state,
        kernel=adaptive_hmc,
        trace_fn=lambda _, pkr: pkr.inner_results.is_accepted)

    return samples, is_accepted




def test_1d_density_chain(num_results, num_parallel_chains, filter=False):
    """ \int_{-inf}^{inf} exp(-x(1+x)) = e^(1/4) * pi^(1/2)
        Let X be a r.v. with density e^(-1/4) * pi^(-1/2) * exp(-x(1+x))
            E(X) = -0.5
            StDev(X) = 1/sqrt(2) ~ 0.707107...
    """
    key = jax.random.PRNGKey(3264)
    key, subkey = jax.random.split(key, num=2)
    initial_states = jax.random.uniform(
        subkey,
        shape=(num_parallel_chains,1),
        minval=-0.5,
        maxval=0.5)

    samples, is_accepted = run_chain(key, initial_states, num_results)

    if filter:
        samples = samples[is_accepted]

    print(f'Testing HMC with a one-dimensional density with {num_results} steps per chain (with {num_parallel_chains} parallel chains). Filter={filter}')
    print('Estimated means (expected -0.5):')
    print(jnp.mean(samples, axis=0))
    print('Estimated StDevs (expected 1/sqrt(2) ~ 0.707107...)')
    print(jnp.std(samples, axis=0))
    print('Combined estimates:')
    print('Mean:', jnp.mean(samples))
    print('StDev:', jnp.std(samples))

    X = jnp.squeeze(samples*samples - jnp.mean(samples, axis=0))
    print(jnp.mean(X, axis=0))
    print(jnp.sqrt(jnp.mean(X, axis=0)))
    print(X.shape)

def test_2d_density_chain(num_results, num_parallel_chains, filter=False):
    """ \int_{-inf}^{inf} exp(-x(1+x)) dx = e^(1/4) * pi^(1/2)
        \int_{-inf}^{inf} exp(-x^2/2) dx = (2*pi)^{1/2}
        Let Y be a r.v. with density 2^(-1/2) * e^(-1/4) * pi^(-1) * exp(-x(1+x) - y^2/2)
            E(Y) = [-0.5, 0]
            Covar(Y) = [ 1/2  0 ]
                       [  0   1 ]
    """
    key = jax.random.PRNGKey(3264)
    key, subkey = jax.random.split(key, num=2)
    initial_states = jax.random.uniform(
        subkey,
        shape=(num_parallel_chains,2),
        minval=-0.5,
        maxval=0.5)

    samples, is_accepted = run_chain_2d_density(key, initial_state=initial_states, num_results=num_results)

    print(f'Testing HMC with a two-dimensional density with {num_results} steps per chain (and {num_parallel_chains} parallel chains). Filter={filter}')
    print('Estimated means (E[X]=-0.5 E[Y]=0)')
    print(jnp.mean(samples, axis=0))
    print('Estimated Covar matrix')
    print('Covar((X,Y)) = [ 1/2   0 ]\n'
          '               [  0    1 ]')
    samples = jnp.swapaxes(samples, 0, 1)
    for s in samples:
        print(jnp.cov(s, rowvar=False))

if __name__ == '__main__':
    #test_1d_density_chain(num_results=1e3, num_parallel_chains=4, filter=False)
    test_2d_density_chain(num_results=1e4, num_parallel_chains=1, filter=False)

