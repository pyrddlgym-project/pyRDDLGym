# Verifying that a linear combination of two right-shift maps can be
# differentiated by JAX

import jax
import jax.numpy as jnp

jnp.set_printoptions(linewidth=9999)

# Finding the Jacobian of the right-shift map
base_vector = jnp.array([1., 1., 1., 1., 1.,
                         0., 0., 0., 0., 0.])

def right_shift(x, n):
    pad = jnp.zeros(n)
    return jnp.concatenate((pad, x[:-n]))

right_shift_single = lambda x: right_shift(x, 1)

J = jax.jacfwd(right_shift_single)

pt = jnp.ones(10)
print(J(pt))


# Jacobian of a smoothed sum of two shifts
def smoothed_shift(x, alpha):
    n = jnp.floor(alpha).astype(int)
    f = alpha - n
    return (1 - f) * right_shift(x, n) + f * right_shift(x, n+1)


shift_setter = lambda alpha: smoothed_shift(base_vector, alpha)

J = jax.jacfwd(shift_setter)


test_array = jnp.arange(start=1., stop=5., step=0.1)
print(test_array)
print(list(map(J, test_array)))



# grad of sum of smoothed sum of two shifts
def smshsum(x, alpha):
    return jnp.sum(smoothed_shift(x, alpha))

f = lambda alpha: smshsum(base_vector, alpha)

grad_f = jax.grad(f)

print("Gradient of f")
print(list(map(f, test_array)))
