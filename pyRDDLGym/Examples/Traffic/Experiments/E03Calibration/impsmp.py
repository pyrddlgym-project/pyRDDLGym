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
init_train = {}
for (name, value) in init_state_subs.items():
    value = jnp.asarray(value)[jnp.newaxis, ...]
    train_value = jnp.repeat(value, repeats=n_rollouts, axis=0)
    train_value = train_value.astype(compiler.REAL)
    init_train[name] = train_value
for (state, next_state) in model.next_state.items():
    init_train[next_state] = init_train[state]

def rollout(base_subs, new_rates):
    new_subs = {}
    for (item, value) in base_subs.items():
        new_subs[item] = jnp.copy(value)
    new_subs['SOURCE-ARRIVAL-RATE'] = new_rates
    return sampler(
        key,
        policy_params=None,
        hyperparams=None,
        subs=new_subs,
        model_params=compiler.model_params)

with jax.disable_jit(disable=False):
    t0 = timer()


    #print(rollout_reward())

    #print(init_train.keys())
    #print(init_train['SOURCE-ARRIVAL-RATE'])
    #init_train['SOURCE-ARRIVAL-RATE'] = init_train['SOURCE-ARRIVAL-RATE'].at[(0,-1)].set(0.7)
    #print(rollout_reward())


    init_train['SOURCE-ARRIVAL-RATE'] = init_train['SOURCE-ARRIVAL-RATE'].at[(0,1)].set(0.3)
    rollouts = rollout(init_train, init_train['SOURCE-ARRIVAL-RATE'])
    #print(rollouts.keys())
    #print(type(rollouts['pvar']))
    #print(rollouts['pvar'].keys())
    #print(rollouts['pvar']['flow-into-link'].shape)
    #print(rollouts['pvar']['flow-on-link'].shape)

    GROUND_TRUTH = rollouts['pvar']['flow-into-link'][0,:,:]
    print(GROUND_TRUTH)

    def test(new_rates):
        rollouts = rollout(init_train, new_rates)
        delta = rollouts['pvar']['flow-into-link'][0,:,:] - GROUND_TRUTH
        return jnp.sum(delta*delta)

    dt = jax.grad(test)


    for r in np.arange(0., stop=0.5, step=0.025):
        mutate_rates = jnp.copy(init_train['SOURCE-ARRIVAL-RATE'])
        mutate_rates = mutate_rates.at[(0,1)].set(r)
        print(r, test(mutate_rates), dt(mutate_rates))
        print('===')

    t1 = timer()
    print('Took', t1-t0, 'seconds')

print(compiler)
