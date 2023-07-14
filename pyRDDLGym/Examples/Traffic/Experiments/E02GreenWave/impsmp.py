import jax
import optax
import jax.numpy as jnp
import numpy as np

from sys import argv
from time import perf_counter as timer

from pyRDDLGym import ExampleManager
from pyRDDLGym import RDDLEnv
from pyRDDLGym.Core.Jax.JaxRDDLLogic import FuzzyLogic
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxRDDLCompilerWithGrad


from utils import warmup
from utils import offset_plan

from jax.config import config as jconfig
jconfig.update('jax_debug_nans', True)
jconfig.update('jax_platform_name', 'cpu')

key = jax.random.PRNGKey(3452)

# specify the model
EnvInfo = ExampleManager.GetEnvInfo('traffic2phase')
myEnv = RDDLEnv.RDDLEnv(domain=EnvInfo.get_domain(),
                        instance='instances/instance01.rddl')
model = myEnv.model
N = myEnv.numConcurrentActions

init_state_subs, t_warmup = warmup(myEnv, EnvInfo)
t_plan = myEnv.horizon
rollout_horizon = t_plan - t_warmup

compiler = JaxRDDLCompilerWithGrad(
    rddl=model,
    use64bit=True,
    logic=FuzzyLogic())
compiler.compile()

# define policy fn
all_red_stretch = [1.0, 0.0, 0.0, 0.0]
full_cycle = all_red_stretch + [1.0] + [0.0]*59 + all_red_stretch + [1.0] + [0.0]*23
BASE_PHASING = jnp.array([1.0] + full_cycle*3, dtype=jnp.float64)

def non_jittable_right_shift(x, n):
    """ Shifts a numpy array to the right by n items,
        padding the shifted slots with zeros

        This version of the function is simpler, but
        cannot be jitted, because the presence of an array
        slice with a dynamic index
    """
    pad = jnp.zeros(n)
    return jnp.concatenate((pad, x[:-n]))

def right_shift(x, n):
    """ Shifts a numpy array to the right by n items,
        padding the shifted slots with zeros
    """
    z = jnp.zeros(x.shape[0])
    rolled = jnp.roll(x, n)
    mask = jnp.arange(x.shape[0]) < n
    return jnp.where(mask, z, rolled)

def smoothed_shift(x, alpha):
    """ Differentiably interpolates the right-shift mapping to floating-point
        arguments """
    n = jnp.floor(alpha).astype(int)
    q = alpha - n
    return (1 - q)*right_shift(x, n) + q*right_shift(x, n+1)

def policy_fn(key, policy_params, hyperparams, step, states):
    """ Given floating-point offsets given as policy_params=(alpha0, alpha1, ..., alphaN),
        produces the corresponding phasings for the traffic lights 0, 1, ..., N+1.
        Returns the actions at sim.-step 'step'"""
    tau0 = BASE_PHASING
    tau1 = smoothed_shift(BASE_PHASING, policy_params[0])
    tau2 = smoothed_shift(BASE_PHASING, policy_params[1])
    tau3 = smoothed_shift(BASE_PHASING, policy_params[2])
    tau4 = smoothed_shift(BASE_PHASING, policy_params[3])
    return {'advance':
        jnp.array((tau0[step], tau1[step], tau2[step], tau3[step], tau4[step]))
    }

# obtain rollout sampler
n_batch = 1
sampler = compiler.compile_rollouts(policy=policy_fn,
                                    n_steps=rollout_horizon,
                                    n_batch=n_batch)

init_train = {}
for (name, value) in init_state_subs.items():
    value = np.asarray(value)[np.newaxis, ...]
    train_value = np.repeat(value, repeats=n_batch, axis=0)
    train_value = train_value.astype(compiler.REAL)
    init_train[name] = train_value
for (state, next_state) in model.next_state.items():
    init_train[next_state] = init_train[state]

with jax.disable_jit(disable=False):
    t0 = timer()
    print(sampler(
        key,
        policy_params=(1.01, 1.51, 2.01, 2.51),
        hyperparams=None,
        subs=init_train,
        model_params=compiler.model_params
    ))
    t1 = timer()
    print('Took', t1-t0, 'seconds')

print(compiler)
