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

#init_state_subs, t_warmup = warmup(myEnv, EnvInfo)
init_state_subs, t_warmup = myEnv.sampler.subs, 0
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
FIXED_TIME_PLAN = jnp.broadcast_to(BASE_PHASING[..., None], shape=(BASE_PHASING.shape[0], 5))


def policy_fn(key, policy_params, hyperparams, step, states):
    """ Each traffic light is assumed to follow a fixed-time plan """
    return {'advance': FIXED_TIME_PLAN[step]}

# obtain rollout sampler
n_rollouts = 1
sampler = compiler.compile_rollouts(policy=policy_fn,
                                    n_steps=rollout_horizon,
                                    n_batch=n_rollouts)


# initial state
train_subs = {}
for (name, value) in init_state_subs.items():
    value = jnp.asarray(value)[jnp.newaxis, ...]
    train_value = jnp.repeat(value, repeats=n_rollouts, axis=0)
    train_value = train_value.astype(compiler.REAL)
    train_subs[name] = train_value
for (state, next_state) in model.next_state.items():
    train_subs[next_state] = train_subs[state]

with jax.disable_jit(disable=False):
    t0 = timer()

    train_subs['SOURCE-ARRIVAL-RATE'] = train_subs['SOURCE-ARRIVAL-RATE'].at[(0,1)].set(0.3)
    rollouts = sampler(
            key,
            policy_params=None,
            hyperparams=None,
            subs=train_subs,
            model_params=compiler.model_params)
    GROUND_TRUTH_LOOPS = rollouts['pvar']['flow-into-link'][0,:,:]
    GROUND_TRUTH_RATES = jnp.copy(train_subs['SOURCE-ARRIVAL-RATE'])
    vrates = jnp.copy(train_subs['SOURCE-ARRIVAL-RATE'])

    def rollout(rates, subs, sampler, GROUND_TRUTH_LOOPS):
        subs['SOURCE-ARRIVAL-RATE'] = rates
        rollouts = sampler(
            key,
            policy_params=None,
            hyperparams=None,
            subs=subs,
            model_params=compiler.model_params)
        delta = rollouts['pvar']['flow-into-link'][0,:,:] - GROUND_TRUTH_LOOPS
        return jnp.sum(delta*delta)

    droll = jax.grad(rollout, argnums=0)

    for r in np.arange(0., stop=0.5, step=0.025):
        vrates = vrates.at[(0,1)].set(r)
        sample = rollout(vrates, train_subs, sampler, GROUND_TRUTH_LOOPS)
        dsample = droll(vrates, train_subs, sampler, GROUND_TRUTH_LOOPS)
        print(r, sample, dsample)
        print('===')

    t1 = timer()
    print('Took', t1-t0, 'seconds')

print(compiler)
