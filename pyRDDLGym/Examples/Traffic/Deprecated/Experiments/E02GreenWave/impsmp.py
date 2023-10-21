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
jnp.set_printoptions(linewidth=9999)

key = jax.random.PRNGKey(3452)

# specify the model
EnvInfo = ExampleManager.GetEnvInfo('traffic2phase')
myEnv = RDDLEnv.RDDLEnv(domain=EnvInfo.get_domain(),
                        instance='instances/instance01.rddl')
model = myEnv.model
N = myEnv.numConcurrentActions

init_state_subs, t_warmup = warmup(myEnv, EnvInfo)
#init_state_subs, t_warmup = myEnv.sampler.subs, 0
t_plan = myEnv.horizon
rollout_horizon = t_plan - t_warmup

compiler = JaxRDDLCompilerWithGrad(
    rddl=model,
    use64bit=True,
    logic=FuzzyLogic(weight=15))
compiler.compile()


# define policy fn
all_red_stretch = [1.0, 0.0, 0.0, 0.0]
full_cycle = all_red_stretch + [1.0] + [0.0]*59 + all_red_stretch + [1.0] + [0.0]*9
BASE_PHASING = jnp.array(full_cycle*10, dtype=jnp.float64)

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

def policy_fn(key, policy_params, hyperparams, step, states):
    """ Given floating-point offsets given as policy_params=(alpha0, alpha1, ..., alphaN),
        produces the corresponding phasings for the traffic lights 0, 1, ..., N+1.
        Returns the actions at sim.-step 'step'

        Uses the integer right shift operator
    """
    shifts = jnp.cumsum(policy_params).astype(int)
    tau0 = BASE_PHASING
    tau1 = right_shift(BASE_PHASING, shifts[0])
    tau2 = right_shift(BASE_PHASING, shifts[1])
    tau3 = right_shift(BASE_PHASING, shifts[2])
    tau4 = right_shift(BASE_PHASING, shifts[3])
    return {'advance':
        jnp.array((tau0[step], tau1[step], tau2[step], tau3[step], tau4[step]))
    }

def smoothed_shift(x, alpha):
    """ Differentiably interpolates the right-shift mapping to floating-point
        arguments """
    n = jnp.floor(alpha).astype(int)
    q = alpha - n
    return (1 - q)*right_shift(x, n) + q*right_shift(x, n+1)

def smoothed_policy_fn(key, policy_params, hyperparams, step, states):
    """ Given floating-point offsets given as policy_params=(alpha0, alpha1, ..., alphaN),
        produces the corresponding phasings for the traffic lights 0, 1, ..., N+1.
        Returns the actions at sim.-step 'step'

        Uses the smoothed right shift operator
    """
    shifts = jnp.cumsum(policy_params)
    tau0 = BASE_PHASING
    tau1 = smoothed_shift(BASE_PHASING, shifts[0])
    tau2 = smoothed_shift(BASE_PHASING, shifts[1])
    tau3 = smoothed_shift(BASE_PHASING, shifts[2])
    tau4 = smoothed_shift(BASE_PHASING, shifts[3])
    return {'advance':
        jnp.array((tau0[step], tau1[step], tau2[step], tau3[step], tau4[step]))
    }

# obtain rollout sampler
n_rollouts = 1
sampler = compiler.compile_rollouts(policy=policy_fn,
                                    n_steps=rollout_horizon,
                                    n_batch=n_rollouts)

smoothed_sampler = compiler.compile_rollouts(policy=smoothed_policy_fn,
                                             n_steps=rollout_horizon,
                                             n_batch=n_rollouts)

# initial state
init_train = {}
for (name, value) in init_state_subs.items():
    value = jnp.asarray(value)[jnp.newaxis, ...]
    train_value = jnp.repeat(value, repeats=n_rollouts, axis=0)
    train_value = train_value.astype(compiler.REAL)
    init_train[name] = train_value
for (state, next_state) in model.next_state.items():
    init_train[next_state] = init_train[state]


def rollout_reward(policy_params):
    rollouts = sampler(
        key,
        policy_params,
        hyperparams=None,
        subs=init_train,
        model_params=compiler.model_params)
    return jnp.sum(rollouts['reward'], axis=1)

def smoothed_rollout_reward(policy_params):
    rollouts = smoothed_sampler(
        key,
        policy_params,
        hyperparams=None,
        subs=init_train,
        model_params=compiler.model_params)
    return jnp.sum(rollouts['reward'], axis=1)


with jax.disable_jit(disable=False):
    t0 = timer()

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    rng = jnp.arange(4.0, stop=16.0, step=0.1)
    rewards = np.zeros(rng.shape[0])
    policy_params_ = jnp.broadcast_to(rng[..., None], shape=(rng.shape[0], 4))
    policy_params2 = np.full((rng.shape[0],4), 9.)

    for idx, policy_params in enumerate(policy_params_):
        policy_params2[idx,0] = policy_params[0]
        pp = jnp.asarray(policy_params2[idx,:])
        rollouts = smoothed_sampler(
            key,
            policy_params=pp,
            hyperparams=None,
            subs=init_train,
            model_params=compiler.model_params
        )
        print(rollouts['reward'].shape)
        R = jnp.sum(rollouts['reward'], axis=1)
        print(pp, R)
        rewards[idx] = R

    ax.plot(rng, -rewards)
    ax.grid(visible=True)
    plt.savefig('img/rewards_plot.png')


    print(smoothed_rollout_reward(jnp.array([8.5, 8.5, 8.5, 8.5])))
    r = []
    r.append(rollout_reward(jnp.array([8., 8., 8., 8.])))
    r.append(rollout_reward(jnp.array([9., 8., 8., 8.])))
    r.append(rollout_reward(jnp.array([8., 9., 8., 8.])))
    r.append(rollout_reward(jnp.array([9., 9., 8., 8.])))
    r.append(rollout_reward(jnp.array([8., 8., 9., 8.])))
    r.append(rollout_reward(jnp.array([9., 8., 9., 8.])))
    r.append(rollout_reward(jnp.array([8., 9., 9., 8.])))
    r.append(rollout_reward(jnp.array([9., 9., 9., 8.])))
    r.append(rollout_reward(jnp.array([8., 8., 8., 9.])))
    r.append(rollout_reward(jnp.array([9., 8., 8., 9.])))
    r.append(rollout_reward(jnp.array([8., 9., 8., 9.])))
    r.append(rollout_reward(jnp.array([9., 9., 8., 9.])))
    r.append(rollout_reward(jnp.array([8., 8., 9., 9.])))
    r.append(rollout_reward(jnp.array([9., 8., 9., 9.])))
    r.append(rollout_reward(jnp.array([8., 9., 9., 9.])))
    r.append(rollout_reward(jnp.array([9., 9., 9., 9.])))
    print(r)
    print('mean=', jnp.mean(jnp.asarray(r)))

    #print(rollout_reward((1., 1.5, 2., 2.5)))
    #print(rollout_reward((1.5, 6.5, 10., 13.5)))

    #dR = jax.grad(rollout_reward)
    #print(dR)

    t1 = timer()
    print('Took', t1-t0, 'seconds')

print(compiler)
