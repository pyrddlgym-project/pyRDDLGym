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


from jax.config import config as jconfig
jconfig.update('jax_debug_nans', True)
jconfig.update('jax_platform_name', 'gpu')
jconfig.update('jax_enable_x64', False)
jnp.set_printoptions(
    linewidth=9999,
    formatter={'float': lambda x: "{0:0.3f}".format(x)})

key = jax.random.PRNGKey(3264)


# specify the model
EnvInfo = ExampleManager.GetEnvInfo('traffic4phase')
myEnv = RDDLEnv.RDDLEnv(domain=EnvInfo.get_domain(),
#                        instance='instances/instance01.rddl')
                        instance='instances/instance_2x2.rddl')

model = myEnv.model
N = myEnv.numConcurrentActions

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
full_cycle = all_red_stretch + [1.0] + [0.0]*59 + all_red_stretch + [1.0] + [0.0]*59
BASE_PHASING = jnp.array(full_cycle*10, dtype=jnp.float64)
FIXED_TIME_PLAN = jnp.broadcast_to(BASE_PHASING[..., None], shape=(BASE_PHASING.shape[0], N))


def policy_fn(key, policy_params, hyperparams, step, states):
    """ Each traffic light is assumed to follow a fixed-time plan """
    return {'advance': FIXED_TIME_PLAN[step]}

# compile batch and sequential rollout samplers
n_rollouts = 128
n_batches = 2
sampler = compiler.compile_rollouts(policy=policy_fn,
                                    n_steps=rollout_horizon,
                                    n_batch=n_rollouts)
sampler_hmc = compiler.compile_rollouts(policy=policy_fn,
                                        n_steps=rollout_horizon,
                                        n_batch=1)


# set up initial states
subs_train = {}
subs_hmc = {}
for (name, value) in init_state_subs.items():
    value = jnp.array(value)[jnp.newaxis, ...]
    train_value = jnp.repeat(value, repeats=n_rollouts, axis=0)
    train_value = train_value.astype(compiler.REAL)
    subs_train[name] = train_value

    hmc_value = jnp.repeat(value, repeats=1, axis=0)
    hmc_value = hmc_value.astype(compiler.REAL)
    subs_hmc[name] = hmc_value
for (state, next_state) in model.next_state.items():
    subs_train[next_state] = subs_train[state]
    subs_hmc[next_state] = subs_hmc[state]


# ground truth rates
source_indices = range(16,24)
true_source_rates = jnp.array([0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.3])
num_nonzero_sources = 7
source_rates = true_source_rates[:num_nonzero_sources]
s0, s1, s2 = source_indices[0], source_indices[num_nonzero_sources], source_indices[-1]

one_hot_inputs = jnp.eye(num_nonzero_sources)


with jax.disable_jit(disable=False):
    t0 = timer()

    # Prepare the ground truth inflow rates
    nulled_rates = jnp.zeros(shape=(n_rollouts, s2-s1))
    base_rates = jnp.broadcast_to(source_rates[None,...], shape=(n_rollouts,s1-s0))

    # Prepare the ground truth for batched rollouts
    subs_train['SOURCE-ARRIVAL-RATE'] = subs_train['SOURCE-ARRIVAL-RATE'].at[:,s0:s1].set(base_rates)
    subs_train['SOURCE-ARRIVAL-RATE'] = subs_train['SOURCE-ARRIVAL-RATE'].at[:,s1:s2].set(nulled_rates)
    rollouts = sampler(
            key,
            policy_params=None,
            hyperparams=None,
            subs=subs_train,
            model_params=compiler.model_params)
    GROUND_TRUTH_LOOPS = rollouts['pvar']['flow-into-link'][:,:,:]

    # Prepare the ground truth for sequential rollouts
    subs_hmc['SOURCE-ARRIVAL-RATE'] = subs_hmc['SOURCE-ARRIVAL-RATE'].at[:,s0:s1].set(source_rates)
    subs_hmc['SOURCE-ARRIVAL-RATE'] = subs_hmc['SOURCE-ARRIVAL-RATE'].at[:,s1:s2].set(0.)
    rollout_hmc = sampler_hmc(
            key,
            policy_params=None,
            hyperparams=None,
            subs=subs_hmc,
            model_params=compiler.model_params)
    GROUND_TRUTH_LOOPS_HMC = rollout_hmc['pvar']['flow-into-link'][:,:,:]
    vrates = jnp.copy(subs_train['SOURCE-ARRIVAL-RATE'])



    def policy_stoch_params(input):
        # Currently assumes a multivariate normal policy with
        # a diagonal covariance matrix
        mlp = hk.Sequential([
            hk.Linear(32), jax.nn.relu,
            hk.Linear(32), jax.nn.relu,
            hk.Linear(2)
        ])
        output = jnp.exp(mlp(input))/10
        mean, cov = (jnp.squeeze(x) for x in jnp.split(output, 2, axis=1))
        return mean, jnp.diag(cov)

    policy = hk.transform(policy_stoch_params)
    key, subkey = jax.random.split(key)
    theta = policy.init(subkey, one_hot_inputs)

    def print_shapes(x):
        print(x.shape)

    def policy_stoch_pdf(theta, rng, actions):
        mean, cov = policy.apply(theta, rng, one_hot_inputs)
        return jax.scipy.stats.multivariate_normal.pdf(actions, mean=mean, cov=cov)

    def policy_stoch_sample(theta, rng):
        mean, cov = policy.apply(theta, rng, one_hot_inputs)
        action_sample = jax.random.multivariate_normal(
            rng, mean, cov,
            shape=(n_rollouts,))
        print(action_sample.shape)
        return action_sample

    def reward_stoch(rng, subs, actions):
        # FIX: Figure out how to rewrite this to avoid the copy
        #      operation (while not having side-effects)
        subs_copy = {}
        for name, val in subs.items():
            subs_copy[name] = jnp.copy(val)
        subs_copy['SOURCE-ARRIVAL-RATE'] = subs_copy['SOURCE-ARRIVAL-RATE'].at[:,s0:s1].set(actions)
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
        subs_copy['SOURCE-ARRIVAL-RATE'] = subs_copy['SOURCE-ARRIVAL-RATE'].at[:,s0:s1].set(action)
        rollout = sampler_hmc(
            rng,
            policy_params=None,
            hyperparams=None,
            subs=subs_copy,
            model_params=compiler.model_params)
        return rollout['pvar']['flow-into-link'][:,:,:] - GROUND_TRUTH_LOOPS_HMC

    def reward_reinforce(rng, subs, actions):
        delta = reward_stoch(rng, subs, actions)
        return jnp.sum(delta*delta, axis=(1,2))

    def reward_hmc(rng, subs, actions):
        delta = reward_stoch_hmc(rng, subs, actions)
        return jnp.sum(delta*delta, axis=(1,2))

    @jax.jit
    def collect_covar_data(theta, C, iter):
        ip0 = 0
        for leaf in jax.tree_util.tree_leaves(theta):
            leaf = leaf.flatten()
            ip1 = ip0 + leaf.shape[0]
            C = C.at[ip0:ip1, iter].set(leaf)
            ip0 = ip1
        return C


    def reinforce(key, n_iters, theta, subs_train, input):
        # initialize stats collection
        X, Y = np.arange(n_iters), np.zeros(shape=(n_iters,3,num_nonzero_sources))
        n_params = sum(leaf.flatten().shape[0] for leaf in jax.tree_util.tree_leaves(theta))
        C = jnp.zeros(shape=(n_params, n_iters))

        # initialize optimizer
        #optimizer = optax.sgd(learning_rate=1e-5)
        optimizer = optax.rmsprop(learning_rate=1e-3)
        opt_state = optimizer.init(theta)

        # run
        for idx in range(n_iters):
            grads = jax.tree_util.tree_map(lambda _: jnp.zeros(1), theta)
            cmlt_rewards = 0
            for _ in range(n_batches):
                key, *subkeys = jax.random.split(key, num=3)
                means, stdevs = policy.apply(theta, subkeys[0], one_hot_inputs)
                actions = policy_stoch_sample(theta, subkeys[0])
                pi = policy_stoch_pdf(theta, subkeys[0], actions)
                dpi = jax.jacrev(policy_stoch_pdf, argnums=0)(theta, subkeys[0], actions)
                rewards = reward_reinforce(subkeys[0], subs_train, actions)
                factors = rewards / (pi + 1e-6)

                w = lambda x: jnp.mean(weighting_map(factors, x), axis=(0,1))

                accumulant = jax.tree_util.tree_map(w, dpi)
                grads = jax.tree_util.tree_map(lambda x, y: x+(y/n_batches), grads, accumulant)
                cmlt_rewards += jnp.mean(rewards)/n_batches

            updates, opt_state = optimizer.update(grads, opt_state)
            theta = optax.apply_updates(theta, updates)

            Y[idx,0] = cmlt_rewards
            Y[idx,1] = jnp.squeeze(means)
            Y[idx,2] = jnp.squeeze(stdevs)
            print(idx, cmlt_rewards, Y[idx])

            C = collect_covar_data(theta, C, idx)

        return X, Y, C


    # === REINFORCE + Importance sampling ====
    clip_val = jnp.array([1e-6], dtype=compiler.REAL)

    @jax.jit
    def scalar_mult(alpha, v):
        n, m = len(v.shape), len(alpha.shape)
        alpha = jnp.expand_dims(alpha, tuple(range(m,n)))
        return alpha * v

    weighting_map = jax.jit(jax.vmap(
        scalar_mult, in_axes=1, out_axes=0))

    @jax.jit
    def scalar_mult_impsmp(alpha, v): return alpha * v

    weighting_map_impsmp = jax.jit(jax.vmap(
        scalar_mult_impsmp, in_axes=0, out_axes=0))

    @jax.jit
    def unnormalized_log_rho(key, theta, subs, a):
        dpi = jax.jacrev(policy_stoch_pdf, argnums=0)(theta, key, a)
        #dpi_norm = jax.tree_util.tree_reduce(lambda x,y: x + jnp.sum(y*y), dpi, initializer=jnp.zeros(1))
        #dpi_norm = jnp.sqrt(dpi_norm)
        dpi_norm = jax.tree_util.tree_reduce(lambda x, y: jnp.maximum(x, jnp.max(jnp.abs(y))), dpi, initializer=jnp.array([-jnp.inf]))
        rewards = reward_hmc(key, subs, a)
        density = jnp.abs(rewards) * dpi_norm
        return jnp.log(jnp.maximum(density, clip_val))[0]

    batch_unnormalized_log_rho = jax.jit(jax.vmap(
        unnormalized_log_rho, (None, None, None, 0), 0))

    impsmp_num_iters = int(n_rollouts * n_batches)
    impsmp_num_burnin_steps = int(impsmp_num_iters/8)

    def impsmp(key, n_iters, theta, subs_train, subs_hmc, input, est_Z=False):

        # initialize stats collection
        X, Y = np.arange(n_iters), np.zeros(shape=(n_iters, 3, num_nonzero_sources))
        n_params = sum(leaf.flatten().shape[0] for leaf in jax.tree_util.tree_leaves(theta))
        C = jnp.zeros(shape=(n_params, n_iters))

        # hmc initial points
        key, subkey = jax.random.split(key, num=2)
        hmc_initializer = jax.random.uniform(
            subkey,
            shape=(num_nonzero_sources,),
            minval=0.05, maxval=0.35)

        # initialize optimizer
        optimizer = optax.rmsprop(learning_rate=1e-3)
        #optimizer = optax.sgd(learning_rate=1e-2)
        opt_state = optimizer.init(theta)

        # run
        for idx in range(n_iters):
            key, *subkeys = jax.random.split(key, num=4)
            mean, cov = policy.apply(theta, subkeys[0], one_hot_inputs)

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

            grads = jax.tree_util.tree_map(lambda _: jnp.zeros(1), theta)
            cmlt_rewards = 0.

            for si in range(n_batches):
                actions = samples[si*n_rollouts:(si+1)*n_rollouts]
                dpi = jax.jacrev(policy_stoch_pdf, argnums=0)(theta, subkeys[0], actions)
                rho = jnp.exp(batch_unnormalized_log_rho(subkeys[0], theta, subs_hmc, actions))
                rewards = reward_reinforce(subkeys[0], subs_train, actions)
                factors = rewards / (rho + 1e-8)

                if est_Z:
                    pi = policy_stoch_pdf(theta, subkeys[0], actions)
                    Zinv = jnp.sum(pi/rho)
                    Zinv = jnp.maximum(Zinv, clip_val)
                    factors = factors / Zinv

                w = lambda x: jnp.mean(weighting_map_impsmp(factors, x), axis=0)

                accumulant = jax.tree_util.tree_map(w, dpi)
                grads = jax.tree_util.tree_map(lambda x, y: x+(y/n_batches), grads, accumulant)
                cmlt_rewards += jnp.mean(rewards)/n_batches

            updates, opt_state = optimizer.update(grads, opt_state)
            theta = optax.apply_updates(theta, updates)

            # Initialize the next chain at a random point of the current chain
            hmc_intializer = jax.random.choice(subkeys[2], samples)

            Y[idx,0] = cmlt_rewards
            Y[idx,1] = mean
            Y[idx,2] = jnp.diag(cov)
            print(idx, rewards, Y[idx])

            C = collect_covar_data(theta, C, idx)

        return X, Y, C

    key, subkey = jax.random.split(key)
    method = 'impsmp'
    if method == 'impsmp': method_fn = impsmp
    elif method == 'reinforce': method_fn = reinforce
    else: raise KeyError

    id = f'{method}_2x2_b{n_rollouts*n_batches}_nS{num_nonzero_sources}'

    t1 = timer()
    X, Y, C = method_fn(key=subkey, n_iters=100, theta=theta, subs_train=subs_train, subs_hmc=subs_hmc, input=input, est_Z=False)
    t2 = timer()

    Covar = jnp.cov(C)
    sns.heatmap(Covar)
    plt.savefig(f'img/{id}_covar.png')

    with open(f'img/{id}.json', 'w') as file:
        json.dump({
            'num_nonzero_sources': num_nonzero_sources,
            'X': X.tolist(),
            'Y': Y.T.tolist(),
            'C': C.tolist(),
            'Covar': Covar.tolist()}, file)

    print('Setup took', t1-t0, 'seconds')
    print('Optimization took', t2-t1, 'seconds')
