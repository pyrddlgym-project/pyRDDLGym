import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
import matplotlib.pyplot as plt

bij_isc = tfp.bijectors.IteratedSigmoidCentered()
bij_sc = tfp.bijectors.SoftmaxCentered()

x = jnp.arange(-5.0, stop=5.1, step=0.2)
r = jnp.dstack(jnp.meshgrid(x,x)).reshape(-1,2)
im_isc = jnp.apply_along_axis(bij_isc.forward, axis=1, arr=r)
im_sc = jnp.apply_along_axis(bij_sc.forward, axis=1, arr=r)


fig, ax = plt.subplots()
ax.scatter(im_isc[:,0], im_isc[:,1], label='IteratedSigmoidCentered')
ax.scatter(im_sc[:,0], im_sc[:,1], label='SoftmaxCentered')
ax.plot([0,1],[0,0], color='black')
ax.plot([0,0],[0,1], color='black')
ax.plot([0,1],[1,0], color='black')
ax.legend()
plt.tight_layout()
plt.show()
