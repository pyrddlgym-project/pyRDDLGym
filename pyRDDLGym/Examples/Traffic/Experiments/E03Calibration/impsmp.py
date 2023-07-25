import jax
import optax
import jax.numpy as jnp
import haiku as hk
import numpy as np
from tensorflow_probability.substrates import jax as tfp

from sys import argv
from time import perf_counter as timer
import matplotlib.pyplot as plt
import json

from pyRDDLGym import ExampleManager
from pyRDDLGym import RDDLEnv
from pyRDDLGym.Core.Jax.JaxRDDLLogic import FuzzyLogic
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxRDDLCompilerWithGrad


from utils import warmup
from utils import offset_plan

from jax.config import config as jconfig
jconfig.update('jax_debug_nans', True)
jconfig.update('jax_platform_name', 'gpu')
jconfig.update('jax_enable_x64', False)
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
n_rollouts = 64
n_batches = 2
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

    base_rates = jnp.full(shape=(n_rollouts,), fill_value=0.3)
    train_subs['SOURCE-ARRIVAL-RATE'] = train_subs['SOURCE-ARRIVAL-RATE'].at[:,1].set(base_rates)
    rollouts = sampler(
            key,
            policy_params=None,
            hyperparams=None,
            subs=train_subs,
            model_params=compiler.model_params)
#    GROUND_TRUTH_LOOPS = rollouts['pvar']['flow-into-link']
    GROUND_TRUTH_LOOPS = rollouts['pvar']['flow-into-link'][:,:,:]
    GROUND_TRUTH_RATES = base_rates
    vrates = jnp.copy(train_subs['SOURCE-ARRIVAL-RATE'])

    dim_batch = n_rollouts
    dim_input = 1

    def policy_stoch_params(input):
        mlp = hk.Sequential([
            hk.Linear(32), jax.nn.relu,
            hk.Linear(32), jax.nn.relu,
            hk.Linear(2)
        ])
        mean, stdev = jnp.exp(mlp(input))/10
        return mean, stdev

    policy = hk.transform(policy_stoch_params)
    key, subkey = jax.random.split(key)
    input = jnp.ones([dim_input])
    theta = policy.init(subkey, input)

    def policy_stoch_pdf(theta, rng, a):
        mean, stdev = policy.apply(theta, rng, jnp.array([1]))
        return jax.scipy.stats.norm.pdf(a, loc=mean, scale=stdev)

    def policy_stoch_pdf_scalar(theta, rng, a):
        mean, stdev = policy.apply(theta, rng, jnp.array([1]))
        return jax.scipy.stats.norm.pdf(a, loc=mean, scale=stdev)[0]

    def policy_stoch_sample(theta, rng):
        mean, stdev = policy.apply(theta, rng, jnp.array([1]))
        normed_sample = jax.random.normal(rng, shape=(dim_batch,))
        actions = mean + stdev*normed_sample
        return actions

    def reward_stoch(rng, subs, actions):
        subs_copy = {}
        for name, val in subs.items():
            subs_copy[name] = jnp.copy(val)
        subs_copy['SOURCE-ARRIVAL-RATE'] = subs_copy['SOURCE-ARRIVAL-RATE'].at[:,1].set(actions)
        rollouts = sampler(
            rng,
            policy_params=None,
            hyperparams=None,
            subs=subs_copy,
            model_params=compiler.model_params)
#        return rollouts['pvar']['flow-into-link'] - GROUND_TRUTH_LOOPS
        return rollouts['pvar']['flow-into-link'][:,:,:] - GROUND_TRUTH_LOOPS

    def reward_determ(rng, subs):
        mlp = hk.Sequential([
            hk.Linear(32), jax.nn.relu,
            hk.Linear(32), jax.nn.relu,
            hk.Linear(1)
        ])
        input = jnp.ones([dim_input])
        actions = jnp.exp(mlp(input))/10

        subs_copy = {}
        for k, v in subs.items():
            subs_copy[k] = jnp.copy(subs[k])
        subs_copy['SOURCE-ARRIVAL-RATE'] = subs_copy['SOURCE-ARRIVAL-RATE'].at[:,1].set(actions)

        rollouts = sampler(
            rng,
            policy_params=None,
            hyperparams=None,
            subs=subs_copy,
            model_params=compiler.model_params)
        return rollouts['pvar']['flow-into-link'] - GROUND_TRUTH_LOOPS

    def reward_sgd():
        delta = reward_stoch(rng, subs, actions)
        return jnp.mean(jnp.sum(delta*delta, axis=(1,2)))

    def reward_reinforce(rng, subs, actions):
        delta = reward_stoch(rng, subs, actions)
        return jnp.sum(delta*delta, axis=(1,2))

    def reward_reinforce_scalar(rng, subs, actions):
        delta = reward_stoch(rng, subs, actions)
        return jnp.sum(delta*delta, axis=(1,2))[0]

    dJ_sgd = jax.grad(reward_sgd, argnums=2)

    reward_sgd = hk.transform(reward_sgd)

    def scalar_mult(alpha, v): return alpha * v

    weighting_map = jax.jit(jax.vmap(
        scalar_mult, in_axes=0, out_axes=0))

    def sgd_update_rule(theta, update):
        return theta - 1e-5 * update #REINFORCE
        return theta - 1e-2 * update #ImpSimp

    def sgd(key, n_iters, theta, input):
        X = np.arange(n_iters)
        Y = np.zeros(n_iters)

        for idx, x in enumerate(X):
            key, subkey0, subkey1 = jax.random.split(key, num=3)
            actions = policy.apply(theta, subkey0, input)
            Y[idx] = reward_sgd(subkey1, train_subs, actions)
            dJ = dJ_sgd(subkey1, train_subs, actions)
            theta = jax.tree_util.tree_map(sgd_update_rule, theta, dJ)
            print(idx, Y[idx])
        return X, Y

    #def HMC(a, lmb, eps, theta, key, n_iters=1000):
    #    dpi = lambda a: jax.jacrev(policy_stoch_pdf, argnums=0)(theta, key, a)
    #    dpi_concrete = dpi(a)
    #    r = lambda a: reward_reinforce(key, train_subs, a)
    #    update_dir = jax.tree_util.tree_map(
    #        lambda a: jnp.log(jnp.abs(r(a) * dpi(a)))
    #    print(rho)
    #    print(rho(a))
    #    exit()
    #    for _ in range(n_iters):
    #        update_dir = jax.tree_util.tree_map(
                

    def reinforce(key, n_iters, theta, train_subs, input):
        X, Y, Ymean, Ystdev = np.arange(n_iters), np.zeros(n_iters), np.zeros(n_iters), np.zeros(n_iters)

        for idx in range(n_iters):
            updates = jax.tree_util.tree_map(lambda _: jnp.zeros(1), theta)
            for _ in range(n_batches):
                key, *subkeys = jax.random.split(key, num=3)
                mean, stdev = policy.apply(theta, subkeys[0], jnp.array([1]))
                actions = policy_stoch_sample(theta, subkeys[0])
                pi = policy_stoch_pdf(theta, subkeys[0], actions)
                dpi = jax.jacrev(policy_stoch_pdf, argnums=0)(theta, subkeys[0], actions)
                rewards = reward_reinforce(subkeys[1], train_subs, actions)
                factors = rewards / (pi + 1e-6)
                w = lambda x: jnp.mean(weighting_map(factors, x), axis=0)

                u = jax.tree_util.tree_map(w, dpi)
                updates = jax.tree_util.tree_map(lambda x, y: x+(y/n_batches), updates, u)
            theta = jax.tree_util.tree_map(sgd_update_rule, theta, updates)

            print(mean, stdev)
            Y[idx] = jnp.mean(rewards)
            Ymean[idx] = mean
            Ystdev[idx] = stdev
            print(idx, rewards, Y[idx])

        return X, Y, Ymean, Ystdev

    log_clip_val = jnp.asarray([1e-6], dtype=compiler.REAL)

    @jax.jit
    def unnormalized_log_rho(key, theta, train_subs, a):
        dpi = jax.jacrev(policy_stoch_pdf, argnums=0)(theta, key, a)
        #dpi_norm = jax.tree_util.tree_reduce(lambda x,y: x + jnp.sum(y*y), dpi, initializer=jnp.zeros(1))
        #dpi_norm = jnp.sqrt(dpi_norm)
        dpi_norm = jax.tree_util.tree_reduce(lambda x, y: jnp.maximum(x, jnp.max(jnp.abs(y))), dpi, initializer=jnp.asarray([-jnp.inf]))
        rewards = reward_reinforce(key, train_subs, a)
        density = jnp.abs(rewards) * dpi_norm
        return jnp.log(jnp.maximum(density, log_clip_val))

    impsmp_num_iters = int(n_batches)
    impsmp_num_burnin_steps = int(impsmp_num_iters/8)

    def impsmp(key, n_iters, theta, train_subs, input):
        X, Y, Ymean, Ystdev = np.arange(n_iters), np.zeros(n_iters), np.zeros(n_iters), np.zeros(n_iters)

        for idx in range(n_iters):
            updates = jax.tree_util.tree_map(lambda _: jnp.zeros(1), theta)
            key, *subkeys = jax.random.split(key, num=3)
            mean, stdev = policy.apply(theta, subkeys[0], jnp.array([1]))
            #actions0 = policy_stoch_sample(theta, subkeys[0])

            log_density = lambda a: unnormalized_log_rho(subkeys[0], theta, train_subs, a)

            adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
                tfp.mcmc.HamiltonianMonteCarlo(
                    target_log_prob_fn=log_density,
                    num_leapfrog_steps=3,
                    step_size=0.1),
                num_adaptation_steps=int(impsmp_num_burnin_steps * 0.8))

            actions, is_accepted = tfp.mcmc.sample_chain(
                seed=subkeys[1],
                num_results=impsmp_num_iters,
                num_burnin_steps=impsmp_num_burnin_steps,
                #current_state=actions0[0],
                current_state=jnp.asarray([0.1]),
                kernel=adaptive_hmc,
                trace_fn=lambda _, pkr: pkr.inner_results.is_accepted)

            #print(actions)
            #print(jnp.mean(actions))
            #print(is_accepted)
            #print(jnp.mean(is_accepted))

            for a in actions:
                #pi = policy_stoch_pdf(theta, subkeys[0], a)
                dpi = jax.jacrev(policy_stoch_pdf, argnums=0)(theta, subkeys[0], a)
                rho = jnp.exp(log_density(a))
                rewards = reward_reinforce(subkeys[1], train_subs, a)
                factors = rewards / (rho + 1e-8)
                w = lambda x: jnp.mean(weighting_map(factors, x), axis=0)

                u = jax.tree_util.tree_map(w, dpi)
                updates = jax.tree_util.tree_map(lambda x, y: x+(y/n_batches), updates, u)

            theta = jax.tree_util.tree_map(sgd_update_rule, theta, updates)

            print(mean, stdev)
            Y[idx] = jnp.mean(rewards)
            Ymean[idx] = mean
            Ystdev[idx] = stdev
            print(idx, rewards, Y[idx])

        return X, Y, Ymean, Ystdev

    key, subkey = jax.random.split(key)
    X, Y, Ymean, Ystdev = reinforce(key=subkey, n_iters=100, theta=theta, train_subs=train_subs, input=input)
    #X, Y, Ymean, Ystdev = impsmp(key=subkey, n_iters=100, theta=theta, train_subs=train_subs, input=input)

    with open(f'img/reinforce_b{n_rollouts*n_batches}.json', 'w') as file:
        json.dump({
            'X': X.tolist(),
            'Y': Y.tolist(),
            'Ymean': Ymean.tolist(),
            'Ystdev': Ystdev.tolist()}, file)
    exit()


    def rollout(rate, subs, sampler, GROUND_TRUTH_LOOPS):
        subs_copy = {}
        for k, v in subs.items():
            subs_copy[k] = jnp.copy(subs[k])
        subs_copy['SOURCE-ARRIVAL-RATE'] = subs_copy['SOURCE-ARRIVAL-RATE'].at[(0,1)].set(rate)

        rollouts = sampler(
            key,
            policy_params=None,
            hyperparams=None,
            subs=subs_copy,
            model_params=compiler.model_params)
        delta = rollouts['pvar']['flow-into-link'] - GROUND_TRUTH_LOOPS
        return jnp.sum(delta*delta)

    drollout = jax.grad(rollout, argnums=0)


    fig, ax = plt.subplots()
    X = np.arange(0., stop=0.5, step=0.025)
    Y = np.zeros(shape=X.shape)
    Y2 = np.zeros(shape=X.shape)
    for idx, r in enumerate(np.arange(0., stop=0.5, step=0.025)):
        sample = rollout(r, train_subs, sampler, GROUND_TRUTH_LOOPS)
        dsample = drollout(r, train_subs, sampler, GROUND_TRUTH_LOOPS)
        print(r, sample, dsample)
        Y[idx] = sample
        Y2[idx] = dsample
        print('===')

    ax.plot(X, Y)
    #ax.plot(X, Y2)
    plt.savefig('img/convex.png')

    t1 = timer()
    print('Took', t1-t0, 'seconds')

print(compiler)
