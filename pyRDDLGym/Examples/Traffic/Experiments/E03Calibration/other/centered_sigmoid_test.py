import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
import matplotlib.pyplot as plt

bij = tfp.bijectors.IteratedSigmoidCentered()

x = jnp.arange(-5.0, stop=5.1, step=0.2)
r = jnp.dstack(jnp.meshgrid(x,x)).reshape(-1,2)
im = jnp.apply_along_axis(bij.forward, axis=1, arr=r)


fig, ax = plt.subplots()
ax.scatter(im[:,0], im[:,1])
ax.plot([0,1],[0,0], color='black')
ax.plot([0,0],[0,1], color='black')
ax.plot([0,1],[1,0], color='black')
plt.show()
