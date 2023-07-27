import jax
import optax
import jax.numpy as jnp
import haiku as hk
import numpy as np
from tensorflow_probability.substrates import jax as tfp

from sys import argv
from time import perf_counter as timer
import matplotlib.pyplot as plt
import seaborn as sns
import json

from pyRDDLGym import ExampleManager
from pyRDDLGym import RDDLEnv
from pyRDDLGym.Core.Jax.JaxRDDLLogic import FuzzyLogic
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxRDDLCompilerWithGrad


from utils import warmup
from utils import offset_plan

from jax.config import config as jconfig
jconfig.update('jax_debug_nans', True)
jconfig.update('jax_platform_name', 'cpu')
jconfig.update('jax_enable_x64', False)
jnp.set_printoptions(
    linewidth=9999,
    formatter={'float': lambda x: "{0:0.3f}".format(x)})

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
n_rollouts = 128
n_batches = 2
sampler = compiler.compile_rollouts(policy=policy_fn,
                                    n_steps=rollout_horizon,
                                    n_batch=n_rollouts)
sampler_hmc = compiler.compile_rollouts(policy=policy_fn,
                                        n_steps=rollout_horizon,
                                        n_batch=1)


# initial state
train_subs = {}
subs_hmc = {}
for (name, value) in init_state_subs.items():
    value = jnp.asarray(value)[jnp.newaxis, ...]
    train_value = jnp.repeat(value, repeats=n_rollouts, axis=0)
    train_value = train_value.astype(compiler.REAL)
    train_subs[name] = train_value

    hmc_value = jnp.repeat(value, repeats=1, axis=0)
    hmc_value = hmc_value.astype(compiler.REAL)
    subs_hmc[name] = hmc_value
for (state, next_state) in model.next_state.items():
    train_subs[next_state] = train_subs[state]
    subs_hmc[next_state] = subs_hmc[state]

with jax.disable_jit(disable=False):
    t0 = timer()

    base_rates = jnp.full(shape=(n_rollouts,), fill_value=0.3)
    GROUND_TRUTH_RATES = base_rates
    train_subs['SOURCE-ARRIVAL-RATE'] = train_subs['SOURCE-ARRIVAL-RATE'].at[:,1].set(base_rates)
    rollouts = sampler(
            key,
            policy_params=None,
            hyperparams=None,
            subs=train_subs,
            model_params=compiler.model_params)
    GROUND_TRUTH_LOOPS = rollouts['pvar']['flow-into-link'][:,:,:]

    subs_hmc['SOURCE-ARRIVAL-RATE'] = subs_hmc['SOURCE-ARRIVAL-RATE'].at[:,1].set(0.3)
    rollout_hmc = sampler_hmc(
            key,
            policy_params=None,
            hyperparams=None,
            subs=subs_hmc,
            model_params=compiler.model_params)
    GROUND_TRUTH_LOOPS_HMC = rollout_hmc['pvar']['flow-into-link'][:,:,:]
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
        return rollouts['pvar']['flow-into-link'][:,:,:] - GROUND_TRUTH_LOOPS

    def reward_stoch_hmc(rng, subs, action):
        subs_copy = {}
        for name, val in subs.items():
            subs_copy[name] = jnp.copy(val)
        subs_copy['SOURCE-ARRIVAL-RATE'] = subs_copy['SOURCE-ARRIVAL-RATE'].at[:,1].set(action)
        rollout = sampler_hmc(
            rng,
            policy_params=None,
            hyperparams=None,
            subs=subs_copy,
            model_params=compiler.model_params)
        return rollout['pvar']['flow-into-link'][:,:,:] - GROUND_TRUTH_LOOPS_HMC

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

    def reward_hmc(rng, subs, action):
        delta = reward_stoch_hmc(rng, subs, action)
        return jnp.sum(delta*delta, axis=(1,2))

    dJ_sgd = jax.grad(reward_sgd, argnums=2)

    reward_sgd = hk.transform(reward_sgd)

    def scalar_mult(alpha, v): return alpha * v

    weighting_map = jax.jit(jax.vmap(
        scalar_mult, in_axes=0, out_axes=0))

    @jax.jit
    def collect_covar_data(theta, C, iter):
        ip0 = 0
        for leaf in jax.tree_util.tree_leaves(theta):
            leaf = leaf.flatten()
            ip1 = ip0 + leaf.shape[0]
            C = C.at[ip0:ip1, iter].set(leaf)
            ip0 = ip1
        return C

    def sgd_update_rule(theta, update):
        return theta - 1e-5 * update #REINFORCE
        #return theta - 1e-2 * update #ImpSampling

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


    def reinforce(key, n_iters, theta, train_subs, input):

        # initialize stats collection
        X, Y = np.arange(n_iters), np.zeros(shape=(n_iters,3))
        n_params = sum(leaf.flatten().shape[0] for leaf in jax.tree_util.tree_leaves(theta))
        C = jnp.zeros(shape=(n_params, n_iters))

        # initialize optimizer
        optimizer = optax.sgd(learning_rate=1e-5)
        #optimizer = optax.rmsprop(learning_rate=1e-3)
        opt_state = optimizer.init(theta)

        # run
        for idx in range(n_iters):
            grads = jax.tree_util.tree_map(lambda _: jnp.zeros(1), theta)
            for _ in range(n_batches):
                key, *subkeys = jax.random.split(key, num=3)
                mean, stdev = policy.apply(theta, subkeys[0], jnp.array([1]))
                actions = policy_stoch_sample(theta, subkeys[0])
                pi = policy_stoch_pdf(theta, subkeys[0], actions)
                dpi = jax.jacrev(policy_stoch_pdf, argnums=0)(theta, subkeys[0], actions)
                rewards = reward_reinforce(subkeys[1], train_subs, actions)
                factors = rewards / (pi + 1e-6)
                w = lambda x: jnp.mean(weighting_map(factors, x), axis=0)

                accumulant = jax.tree_util.tree_map(w, dpi)
                grads = jax.tree_util.tree_map(lambda x, y: x+(y/n_batches), grads, accumulant)

            updates, opt_state = optimizer.update(grads, opt_state)
            theta = optax.apply_updates(theta, updates)
            #theta = jax.tree_util.tree_map(sgd_update_rule, theta, updates)

            Y[idx,0] = jnp.mean(rewards)
            Y[idx,1] = mean
            Y[idx,2] = stdev
            print(idx, rewards, Y[idx])

            C = collect_covar_data(theta, C, idx)

        return X, Y, C

    clip_val = jnp.asarray([1e-6], dtype=compiler.REAL)

    @jax.jit
    def unnormalized_log_rho(key, theta, subs, a):
        dpi = jax.jacrev(policy_stoch_pdf, argnums=0)(theta, key, a)
        #dpi_norm = jax.tree_util.tree_reduce(lambda x,y: x + jnp.sum(y*y), dpi, initializer=jnp.zeros(1))
        #dpi_norm = jnp.sqrt(dpi_norm)
        dpi_norm = jax.tree_util.tree_reduce(lambda x, y: jnp.maximum(x, jnp.max(jnp.abs(y))), dpi, initializer=jnp.asarray([-jnp.inf]))
        rewards = reward_hmc(key, subs, a)
        density = jnp.abs(rewards) * dpi_norm
        return jnp.log(jnp.maximum(density, clip_val))

    batch_unnormalized_log_rho = jax.jit(jax.vmap(
        unnormalized_log_rho, (None, None, None, 0), 0))

    impsmp_num_iters = int(n_rollouts * n_batches)
    impsmp_num_burnin_steps = int(impsmp_num_iters/8)

    def impsmp(key, n_iters, theta, train_subs, subs_hmc, input, est_Z=False):

        # initialize stats collection
        X, Y = np.arange(n_iters), np.zeros(shape=(n_iters,3))
        n_params = sum(leaf.flatten().shape[0] for leaf in jax.tree_util.tree_leaves(theta))
        C = jnp.zeros(shape=(n_params, n_iters))

        # initialize optimizer
        hmc_initializer = jnp.asarray([0.1])
        #optimizer = optax.rmsprop(learning_rate=1e-3)
        optimizer = optax.sgd(learning_rate=1e-2)
        opt_state = optimizer.init(theta)

        for idx in range(n_iters):
            key, *subkeys = jax.random.split(key, num=4)
            mean, stdev = policy.apply(theta, subkeys[0], jnp.array([1]))

            log_density = lambda a: unnormalized_log_rho(subkeys[0], theta, subs_hmc, a)

            adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
                tfp.mcmc.HamiltonianMonteCarlo(
                    target_log_prob_fn=log_density,
                    num_leapfrog_steps=3,
                    step_size=0.1),
                num_adaptation_steps=int(impsmp_num_burnin_steps * 0.8))

            samples, is_accepted = tfp.mcmc.sample_chain(
                seed=subkeys[1],
                num_results=impsmp_num_iters,
                num_burnin_steps=impsmp_num_burnin_steps,
                current_state=hmc_initializer,
                kernel=adaptive_hmc,
                trace_fn=lambda _, pkr: pkr.inner_results.is_accepted)
            samples = jnp.squeeze(samples)

            grads = jax.tree_util.tree_map(lambda _: jnp.zeros(1), theta)
            for si in range(n_batches):
                actions = samples[si*n_rollouts:(si+1)*n_rollouts]
                dpi = jax.jacrev(policy_stoch_pdf, argnums=0)(theta, subkeys[0], actions)
                rho = jnp.squeeze(jnp.exp(batch_unnormalized_log_rho(subkeys[0], theta, subs_hmc, actions)))
                rewards = reward_reinforce(subkeys[0], train_subs, actions)
                factors = rewards / (rho + 1e-8)

                if est_Z:
                    pi = policy_stoch_pdf(theta, subkeys[0], actions)
                    Zinv = jnp.sum(pi/rho)
                    Zinv = jnp.maximum(Zinv, clip_val)
                    factors = factors / Zinv

                w = lambda x: jnp.mean(weighting_map(factors, x), axis=0)

                accumulant = jax.tree_util.tree_map(w, dpi)
                grads = jax.tree_util.tree_map(lambda x, y: x+(y/n_batches), grads, accumulant)

            updates, opt_state = optimizer.update(grads, opt_state)
            theta = optax.apply_updates(theta, updates)
            #theta = jax.tree_util.tree_map(sgd_update_rule, theta, updates)

            # Initialize the next chain at a random element of the current chain
            hmc_intializer = jax.random.choice(subkeys[2], samples)

            Y[idx,0] = jnp.mean(rewards)
            Y[idx,1] = mean
            Y[idx,2] = stdev
            print(idx, rewards, Y[idx])

            C = collect_covar_data(theta, C, idx)

        return X, Y, C

    key, subkey = jax.random.split(key)
    id = f'impsmp_b{n_rollouts*n_batches}'
#    X, Y, C = reinforce(key=subkey, n_iters=100, theta=theta, train_subs=train_subs, input=input)
    X, Y, C = impsmp(key=subkey, n_iters=100, theta=theta, train_subs=train_subs, subs_hmc=subs_hmc, input=input, est_Z=False)

    Covar = jnp.cov(C)
    sns.heatmap(Covar)
    plt.savefig(f'img/{id}_covar.png')

    with open(f'img/{id}.json', 'w') as file:
#    with open(f'img/impsmp_b{n_rollouts*n_batches}_rmsprop.json', 'w') as file:
        json.dump({
            'X': X.tolist(),
            'Y': Y.T.tolist(),
            'C': C.tolist(),
            'Covar': C.tolist()}, file)

    t1 = timer()
    print('Took', t1-t0, 'seconds')

    exit()
