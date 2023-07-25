from tensorflow_probability.substrates import jax as tfp
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt


# Target distribution is proportional to: `exp(-x (1 + x))`
def unnormalized_log_prob(x):
    return -x - x**2

key = jax.random.PRNGKey(3264)

def run_chain(seed, num_results):
    # Run the chain (with burn-in).
    num_burnin_steps = int(num_results/10)

    # Initialize the HMC transition kernel.
    adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
        tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=unnormalized_log_prob,
            num_leapfrog_steps=3,
            step_size=1.),
        num_adaptation_steps=int(num_burnin_steps * 0.8))

    samples, is_accepted = tfp.mcmc.sample_chain(
        seed=seed,
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=1.,
        kernel=adaptive_hmc,
        trace_fn=lambda _, pkr: pkr.inner_results.is_accepted)

    return samples, is_accepted

samples, is_accepted = run_chain(seed=key, num_results=1e4)

# int_-inf^inf exp(-x(1+x)) = e^(1/4) * pi^(1/2)
print('Expected: Mean=-0.5, StDev=1/sqrt(2)=0.707107...')
print('Before filtering by accepted/unaccepted')

print(jnp.mean(samples), jnp.std(samples), jnp.sqrt(jnp.mean(samples*samples - jnp.mean(samples)**2)))

fig, ax = plt.subplots(1,2)
fig.set_size_inches(14,7)

X = np.arange(-5.0, stop=5., step=0.1)
Z = np.exp(0.25) * np.pi**(0.5)
Y = 1/Z * np.exp(-X*(1+X)) * samples.shape[0]
n, bins, patches = ax[0].hist(samples)
ax[1].plot(X, Y)
plt.show()

samples = samples[is_accepted]

print('After filtering')
print(jnp.mean(samples), jnp.std(samples))


iters = [10, 50, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000, 25000, 50000, 75000, 100000]
seeds = np.random.randint(low=1, high=60000, size=(10,))
results = np.zeros(shape=(len(iters), len(seeds), 4))

for ni, n in enumerate(iters):
    print(ni, n)
    for si, s in enumerate(seeds):
        key = jax.random.PRNGKey(s)
        samples, is_accepted = run_chain(seed=key, num_results=n)
        results[ni,si,0] = np.mean(samples)
        results[ni,si,1] = np.std(samples)
        samples = samples[is_accepted]
        results[ni,si,2] = np.mean(samples)
        results[ni,si,3] = np.std(samples)

averages = np.mean(results, axis=1)

fig, ax = plt.subplots(1,2)
fig.set_size_inches(14,7)
ax[0].plot(iters, averages[:,0], label='Unfiltered')
ax[0].plot(iters, averages[:,2], label='Filtered')
ax[1].plot(iters, averages[:,1], label='Unfiltered')
ax[1].plot(iters, averages[:,3], label='Filtered')
ax[0].plot(iters, [-0.5] * len(iters), linestyle='dashed')
ax[1].plot(iters, [0.707107] * len(iters), linestyle='dashed')
ax[0].set_title('HMC Mean convergence (average of 10 runs)')
ax[1].set_title('HMC StDev convergence (average of 10 runs)')
ax[0].legend()
ax[1].legend()
plt.tight_layout()
plt.show()
