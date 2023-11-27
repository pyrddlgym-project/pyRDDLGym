import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

class Identity(tfp.bijectors.Identity):
    # Simply adds the _inverse_det_jacobian method to the
    # parent bijector class
    def __init__(self, action_dim):
        super().__init__()
        self.action_dim = action_dim

        self._inverse_jac = jax.jacrev(self._inverse)

    def _inverse_det_jacobian(self, y):
        return jnp.abs(jnp.linalg.det(self._inverse_jac(y)))
