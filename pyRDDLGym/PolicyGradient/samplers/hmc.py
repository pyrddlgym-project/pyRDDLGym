from tensorflow_probability.substrates import jax as tfp

def init_hmc_sampler(
    target_log_prob_fn,
    step_size,
    num_leapfrog_steps,
    unconstraining_bijector,
    num_adaptation_steps=None,
    **kwargs):
    """Initializes an HMC sampler with SimpleStepSizeAdaptation"""
    if num_adaptation_steps is None:
        num_adaptation_steps = int(num_leapfrog_steps * 0.8)
    sampler = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        step_size=step_size,
        num_leapfrog_steps=num_leapfrog_steps)

    sampler_with_adaptive_step_size = tfp.mcmc.SimpleStepSizeAdaptation(
        inner_kernel=sampler,
        num_adaptation_steps=num_adaptation_steps)

    return tfp.mcmc.TransformedTransitionKernel(
        inner_kernel=sampler_with_adaptive_step_size,
        bijector=unconstraining_bijector)
