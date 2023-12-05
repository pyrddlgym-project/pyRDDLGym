from tensorflow_probability.substrates import jax as tfp

def init_nuts_sampler(
    target_log_prob_fn,
    step_size,
    unconstraining_bijector,
    max_tree_depth,
    num_burnin_iters_per_chain,
    num_adaptation_steps=None,
    **kwargs):
    """Initializes a NUTS sampler with DualAveragingSizeAdaptation"""
    if num_adaptation_steps is None:
        num_adaptation_steps = int(num_burnin_iters_per_chain * 0.8)
    sampler = tfp.mcmc.NoUTurnSampler(
        target_log_prob_fn=target_log_prob_fn,
        step_size=step_size,
        max_tree_depth=max_tree_depth)

    sampler_with_adaptive_step_size = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=sampler,
        num_adaptation_steps=num_adaptation_steps)

    return tfp.mcmc.TransformedTransitionKernel(
        inner_kernel=sampler_with_adaptive_step_size,
        bijector=unconstraining_bijector)
