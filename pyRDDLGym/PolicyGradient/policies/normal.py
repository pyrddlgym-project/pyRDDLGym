import jax
import optax
import jax.numpy as jnp
import numpy as np
import haiku as hk
from tensorflow_probability.substrates import jax as tfp


class MultivarNormalHKParametrization:
    """The parametrized policy pi_theta.

    Currently, the policy is assumed to be parameterized as a multivariate
    normal policy N(0, Sigma) with 0 mean and a diagonal covariance matrix
    Sigma. Any parametrization constraints are enforced using a smooth
    bijection (actually diffeomorphism).

    The dm-haiku library stores the parameters theta structured as a
    dictionary, with the weights of each layer and the biases of each layer
    getting a separate dictionary key. To work with theta, it is often
    necessary to make use of the jax.tree_util module.
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

    def diagonal_of_jacobian(self, key, theta, a):
        """The following computation of the diagonal of the Jacobian matrix
        uses JAX primitives, and works with any policy parametrization, but
        computes the entire Jacobian before taking the diagonal. Therefore,
        the computation time scales rather poorly with increasing dimension.
        There is apparently no natural way in JAX of computing the diagonal
        only without also computing all of the off-diagonal terms.

        See also:
            https://stackoverflow.com/questions/70956578/jacobian-diagonal-computation-in-jax
        """
        dpi = jax.jacrev(self.pdf, argnums=1)(key, theta, a)
        dpi = jax.tree_util.tree_map(lambda x: jnp.diagonal(x, axis1=0, axis2=3), dpi)
        dpi = jax.tree_util.tree_map(lambda x: jnp.diagonal(x, axis1=0, axis2=2), dpi)
        dpi = jax.tree_util.tree_map(lambda x: x[0], dpi)
        return dpi

    def analytic_diagonal_of_jacobian(self, key, theta, a):
        """When it is possible to compute the diagonal of the Jacobian terms
        analytically (as it is for the linear parametrization of a normal distribution,
        for example), substituting the analytic computation in place of the general
        auto-differentiation as in the diagonal_of_jacobian method can significantly
        improve scaling of computation time with respect to dimension"""
        raise NotImplementedError

    def clip_theta(self):
        raise NotImplementedError



class MultivarNormalLinearParametrization(MultivarNormalHKParametrization):
    def __init__(self, key, action_dim, bijector, cov_lower_cap):
        def pi(input):
            linear = hk.Linear(2, with_bias=False)
            output = linear(input)
            mean, cov = output.T
            cov = jax.nn.softplus(cov)
            cov = jnp.diag(cov)
            return mean, cov

        super().__init__(
            key=key,
            action_dim=action_dim,
            bijector=bijector,
            pi=pi)

        # record the lower cap value for the covariance terms, correcting for softplus
        if cov_lower_cap > 0.0:
            self.cov_lower_cap = np.log(np.exp(cov_lower_cap) - 1.0)
        else:
            self.cov_lower_cap = -np.inf

    def analytic_diagonal_of_jacobian(self, key, theta, a):
        """The following computes the diagonal of the Jacobian analytically.
        It valid ONLY when the policy is parametrized by a normal distribution
        with parameters

            mu_i
            sigma_i^2 = softplus(u_i)

        In this case, it is possible to compute the partials in closed form,
        and avoid computing the off-diagonal terms in the Jacobian.

        The scaling of computation time with increasing dimension seems much
        improved.
        """
        pi_val = self.pdf(key, theta, a)[..., 0]

        theta = theta['linear']['w']
        mu = theta[:, 0]
        u = theta[:, 1]
        sigsq = jax.nn.softplus(u)

        # the softplus correction comes from the chain rule
        softplus_correction = 1 - (1/(1 + jnp.exp(u)))

        mu_mult = (jnp.diag(a[:,0,0,:]) - mu) / sigsq
        N = (jnp.diag(a[:,1,0,:]) - mu)
        sigsq_mult = 0.5 * softplus_correction * (((N * N) / sigsq) - 1) / sigsq

        partials = jnp.stack([mu_mult, sigsq_mult], axis=1) * pi_val
        return partials


    def clip_theta(self):
        """Clips the covariance parameters of the policy below to the configured
        value, accounting for the softplus transform"""
        return jax.tree_util.tree_map(
            lambda term: term.at[:,1].set(jnp.maximum(term[:,1], self.cov_lower_cap)),
            self.theta)


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
            cov = jax.nn.softplus(cov)
            cov = jnp.maximum(cov_lower_cap, cov)
            cov = jnp.diag(cov)
            return mean, cov

        super().__init__(
            key=key,
            action_dim=action_dim,
            bijector=bijector,
            pi=pi)


if __name__ == '__main__':

    def linear_param_test_analytic_derivative_matches_autograd_derivative(action_dim, test_sample_size, cpu):
        """Test that the analytic derivative and the autograd derivative yield consistent results"""
        import pyRDDLGym.PolicyGradient.bijectors
        from jax.config import config as jconfig
        if cpu:
            jconfig.update('jax_platform_name', 'cpu')

        bij_identity = pyRDDLGym.PolicyGradient.bijectors.identity.Identity(action_dim)
        key = jax.random.PRNGKey(42)
        test_policy = MultivarNormalLinearParametrization(
            key=key,
            action_dim=action_dim,
            bijector=bij_identity,
            cov_lower_cap=0.0)

        key, *subkeys = jax.random.split(key, num=4)
        a = test_policy.sample(subkeys[0], test_policy.theta, (test_sample_size, action_dim, 2, 1))

        autograd_dpi = jax.vmap(test_policy.diagonal_of_jacobian, (None, None, 0), 0)(subkeys[1], test_policy.theta, a)['linear']['w']
        analytic_dpi = jax.vmap(test_policy.analytic_diagonal_of_jacobian, (None, None, 0), 0)(subkeys[2], test_policy.theta, a)

        print('[linear_param_test_analytic_derivative_matches_autograd_derivative] First three samples of diag(Jacobian) computed using autograd:')
        print(autograd_dpi[:3])
        print('[linear_param_test_analytic_derivative_matches_autograd_derivative] First three samples of diag(Jacobian) computed using analytic formula:')
        print(analytic_dpi[:3])

        test_result = jnp.all(jnp.isclose(autograd_dpi, analytic_dpi))
        print(f'[linear_param_test_analytic_derivative_matches_autograd_derivative] All samples, all coordinates close: {test_result}')
        return test_result

    assert linear_param_test_analytic_derivative_matches_autograd_derivative(action_dim=8, test_sample_size=1000, cpu=True)
