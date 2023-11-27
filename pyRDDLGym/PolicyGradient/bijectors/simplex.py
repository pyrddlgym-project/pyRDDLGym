import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

class SimplexBijector(tfp.bijectors.IteratedSigmoidCentered):
    # Wraps the IteratedSigmoidCentered bijector, composing
    # it with a projection onto the first N (out of N+1)
    # coordinates (so that the dimension of the codomain
    # is equal to the dimension of the domain)
    def __init__(self, action_dim, max_rate):
        super().__init__()
        self.action_dim = action_dim
        self.max_rate = max_rate

        self._inverse_jac = jax.jacrev(self._inverse)

    def _forward(self, x):
        y = super()._forward(x[..., jnp.newaxis])
        return y[..., 0] * self.max_rate

    def _inverse(self, y):
        y = (y/self.max_rate)[..., jnp.newaxis]
        s = jnp.sum(y, axis=-1)[..., jnp.newaxis]
        return super()._inverse(
            jnp.concatenate((y,s), axis=-1))[..., 0]

    def _inverse_det_jacobian(self, y):
        y = jnp.squeeze(y)
        return jnp.abs(jnp.linalg.det(self._inverse_jac(y)))

    def _inverse_log_det_jacobian(self, y):
        return jnp.log(self._inverse_det_jacobian(y))
