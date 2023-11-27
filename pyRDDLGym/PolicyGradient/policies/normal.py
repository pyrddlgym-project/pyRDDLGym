import jax
import optax
import jax.numpy as jnp
import numpy as np
import haiku as hk
from tensorflow_probability.substrates import jax as tfp


class MultivarNormalHKParametrization:
    """The parametrized policy pi_theta.

    Currently, the policy is assumed to be parameterized as a multivariate
    normal policy N(0, Sigma) with 0 mean and a diagonal covariance matrix Sigma.
    Any parametrization constraints are enforced using a smooth bijection (actually
    diffeomorphism).

    The dm-haiku library stores the parameters theta structured as a dictionary,
    with the weights of each layer and the biases of each layer getting
    a separate dictionary key. To work with theta, it is often necessary
    to make use of the jax.tree_util module.
    """
    def __init__(self, key, action_dim, bijector, pi):
        self.action_dim = action_dim
        self.one_hot_inputs = jnp.eye(action_dim)
        self.bijector = bijector

        self.pi = hk.transform(pi)
        self.theta = self.pi.init(key, self.one_hot_inputs)
        self.n_params = sum(leaf.flatten().shape[0] for leaf in jax.tree_util.tree_leaves(self.theta))

    def apply(self, key, theta):
        return self.pi.apply(theta, key, self.one_hot_inputs)

    def pdf(self, key, theta, actions):
        mean, cov = self.apply(key, theta)
        unconstrained_actions = self.bijector.inverse(actions)
        normal_pdf = jax.scipy.stats.multivariate_normal.pdf(
            unconstrained_actions,
            mean=mean,
            cov=cov)
        density_correction = jnp.apply_along_axis(self.bijector._inverse_det_jacobian, axis=-1, arr=actions)
        return normal_pdf * density_correction

    def sample(self, key, theta, n):
        mean, cov = self.apply(key, theta)
        action_sample = jax.random.multivariate_normal(
            key, mean, cov,
            shape=n)
        action_sample = self.bijector.forward(action_sample)
        return action_sample



class MultivarNormalLinearParametrization(MultivarNormalHKParametrization):
    def __init__(self, key, action_dim, bijector):
        def pi(input):
            linear = hk.Linear(2, with_bias=False)
            output = linear(input)
            mean, cov = output.T
            return mean, jnp.diag(jax.nn.softplus(cov))

        super().__init__(
            key=key,
            action_dim=action_dim,
            bijector=bijector,
            pi=pi)

class MultivarNormalMLPParametrization(MultivarNormalHKParametrization):
    def __init__(self, key, action_dim, bijector):
        def pi(input):
            mlp = hk.Sequential([
                hk.Linear(32), jax.nn.relu,
                hk.Linear(32), jax.nn.relu,
                hk.Linear(2)
            ])
            output = mlp(input)
            mean, cov = output.T
            return mean, jnp.diag(jax.nn.softplus(cov))

        super().__init__(
            key=key,
            action_dim=action_dim,
            bijector=bijector,
            pi=pi)
