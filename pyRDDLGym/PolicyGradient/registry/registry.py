import optax
from tensorflow_probability.substrates import jax as tfp
import pyRDDLGym.PolicyGradient.algorithms
import pyRDDLGym.PolicyGradient.bijectors
import pyRDDLGym.PolicyGradient.policies
import pyRDDLGym.PolicyGradient.samplers
import pyRDDLGym.Examples.Traffic.Calibration.instances.inflow_calibration_models
import pyRDDLGym.Examples.SumOfHalfSpaces.instances.model

model_lookup_table = {
    'inflow_calibration': pyRDDLGym.Examples.Traffic.Calibration.instances.inflow_calibration_models.InflowCalibrationModel,
    'inflow_calibration_grid_2x2': pyRDDLGym.Examples.Traffic.Calibration.instances.inflow_calibration_models.InflowCalibration2x2GridModel,
    'inflow_calibration_grid_3x3': pyRDDLGym.Examples.Traffic.Calibration.instances.inflow_calibration_models.InflowCalibration3x3GridModel,
    'inflow_calibration_grid_4x4': pyRDDLGym.Examples.Traffic.Calibration.instances.inflow_calibration_models.InflowCalibration4x4GridModel,
    'inflow_calibration_grid_6x6': pyRDDLGym.Examples.Traffic.Calibration.instances.inflow_calibration_models.InflowCalibration6x6GridModel,
    'sum_of_half_spaces': pyRDDLGym.Examples.SumOfHalfSpaces.instances.model.SumOfHalfSpacesModel,
}

bijector_lookup_table = {
    'identity': pyRDDLGym.PolicyGradient.bijectors.identity.Identity,
    'simplex': pyRDDLGym.PolicyGradient.bijectors.simplex.SimplexBijector,
}

sampler_lookup_table = {
    'hmc': pyRDDLGym.PolicyGradient.samplers.hmc.HMCSampler,
    'nuts': pyRDDLGym.PolicyGradient.samplers.hmc.NoUTurnSampler,
    'rejection_sampler': pyRDDLGym.PolicyGradient.samplers.rejection_sampler.RejectionSampler,
}

policy_lookup_table = {
    'multivar_normal_with_linear_parametrization': pyRDDLGym.PolicyGradient.policies.normal.MultivarNormalLinearParametrization,
    'multivar_normal_with_mlp_parametrization': pyRDDLGym.PolicyGradient.policies.normal.MultivarNormalMLPParametrization,
}

optimizer_lookup_table = {
    'adabelief': optax.adabelief,
    'adafactor': optax.adafactor,
    'adagrad': optax.adagrad,
    'adam': optax.adam,
    'adamw': optax.adamw,
    'adamax': optax.adamax,
    'adamaxw': optax.adamaxw,
    'amsgrad': optax.amsgrad,
    'fromage': optax.fromage,
    'lamb': optax.lamb,
    'lars': optax.lars,
    'lion': optax.lion,
    'noisy_sgd': optax.noisy_sgd,
    'novograd': optax.novograd,
    'optimistic_gradient_descent': optax.optimistic_gradient_descent,
    'dpsgd': optax.dpsgd,
    'radam': optax.radam,
    'rmsprop': optax.rmsprop,
    'sgd': optax.sgd,
    'sgd_with_momentum': optax.sgd,
    'sm3': optax.sm3,
    'yogi': optax.yogi,
}

algorithm_lookup_table = {
    'reinforce': pyRDDLGym.PolicyGradient.algorithms.reinforce.reinforce,
    'impsmp': pyRDDLGym.PolicyGradient.algorithms.impsmp.impsmp,
    'impsmp_per_parameter': pyRDDLGym.PolicyGradient.algorithms.impsmp_per_parameter.impsmp_per_parameter,
    'impsmp_per_parameter_signed': pyRDDLGym.PolicyGradient.algorithms.impsmp_per_parameter_signed.impsmp_per_parameter_signed,
}
