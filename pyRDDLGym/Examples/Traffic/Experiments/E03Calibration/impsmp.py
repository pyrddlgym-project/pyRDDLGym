import jax
import optax
import jax.numpy as jnp
import numpy as np
import haiku as hk
from tensorflow_probability.substrates import jax as tfp

import functools
from time import perf_counter as timer
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import json

from pyRDDLGym import ExampleManager
from pyRDDLGym import RDDLEnv
from pyRDDLGym.Core.Jax.JaxRDDLLogic import FuzzyLogic
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxRDDLCompilerWithGrad

from conv_od_to_flows_and_turns import prepare_shortest_path_assignment
from conv_od_to_flows_and_turns import convert_od_to_flows_and_turn_props_jax

t0 = timer()

from jax.config import config as jconfig
use64bit = True # tfp.bijectors.SoftClip seems to have a bug with float64
jconfig.update('jax_debug_nans', True)
jconfig.update('jax_platform_name', 'cpu')
jconfig.update('jax_enable_x64', use64bit)
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
    use64bit=use64bit,
    logic=FuzzyLogic(weight=15))
compiler.compile()


# Define policy fn
# Every traffic light is assumed to follow a fixed-time plan
all_red_stretch = [1.0, 0.0, 0.0, 0.0]
protected_left_stretch = [1.0] + [0.0]*19
through_stretch = [1.0] + [0.0]*59
full_cycle = (all_red_stretch + protected_left_stretch + all_red_stretch + through_stretch)*2
BASE_PHASING = jnp.array(full_cycle*10, dtype=compiler.REAL)
FIXED_TIME_PLAN = jnp.broadcast_to(
    BASE_PHASING[..., jnp.newaxis],
    shape=(BASE_PHASING.shape[0], N))

def policy_fn(key, policy_params, hyperparams, step, states):
    return {'advance': FIXED_TIME_PLAN[step]}


# The batched and sequential rollout samplers have different
# shapes of the argument and return arrays. (The batched versions
# have an additional dimension for the batch.) Because jit-compilation
# requires static array shapes, the two sampler types are compiled
# separately

# number of shards per batch (e.g. for splitting the computation on a GPU)
n_shards = 8
# number of rollouts per shard
n_rollouts = 128
batch_size = n_rollouts * n_shards

sampler = compiler.compile_rollouts(policy=policy_fn,
                                    n_steps=rollout_horizon,
                                    n_batch=n_rollouts)
sampler_hmc = compiler.compile_rollouts(policy=policy_fn,
                                        n_steps=rollout_horizon,
                                        n_batch=1)


# Set up initial states
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


srcs, sinks, transfer_matrix = prepare_shortest_path_assignment(myEnv)
transfer_matrix = jnp.array(transfer_matrix)[jnp.newaxis, ...]

# Set up the bijector
class SimplexBijector(tfp.bijectors.IteratedSigmoidCentered):
#class SimplexBijector(tfp.bijectors.SoftmaxCentered):
    # Wraps the IteratedSigmoidCentered bijector, adding
    # a projection onto the first N (out of N+1) coordinates
    def __init__(self, n_srcs, n_sinks, max_rate):
        super().__init__()
        self.n_srcs = n_srcs
        self.n_sinks = n_sinks
        self.action_dim = n_srcs * n_sinks
        self.max_rate = max_rate

        self._inverse_jac = jax.jacrev(self._inverse)

    def _forward(self, x):
        x = x.reshape(-1, self.n_srcs, self.n_sinks)
        y = super()._forward(x)
        return (y[..., :-1] * self.max_rate).reshape(-1, self.action_dim)

    def _inverse(self, y):
        y = (y/self.max_rate).reshape(-1, self.n_srcs, self.n_sinks)
        s = jnp.sum(y, axis=-1)[..., jnp.newaxis]
        return super()._inverse(
            jnp.concatenate((y,s), axis=-1)).reshape(-1, self.action_dim)

    def _inverse_det_jacobian(self, y):
        y = jnp.squeeze(y)
        return jnp.abs(jnp.linalg.det(self._inverse_jac(y)))

    def _inverse_log_det_jacobian(self, y):
        return jnp.log(self._inverse_det_jacobian(y))

bijector_obj = SimplexBijector(
    n_srcs=len(srcs),
    n_sinks=len(sinks),
    max_rate=0.4)


# Set up ground truth inflow rates
s0, s1 = 16, 24 # The range of source link indices among all link indices
#ground_truth_od = np.zeros(shape=(len(srcs),len(sinks)))
#ground_truth_od[0, 4] = 0.3
ground_truth_od = np.random.normal(loc=0., scale=3., size=(len(srcs), len(sinks)))
ground_truth_od = jnp.array(ground_truth_od)[jnp.newaxis, ...]
ground_truth_od = bijector_obj.forward(ground_truth_od)

action_dim = len(srcs)*len(sinks)
one_hot_inputs = jnp.eye(action_dim)

ground_truth_inflows, ground_truth_turn_props = convert_od_to_flows_and_turn_props_jax(
    ground_truth_od, transfer_matrix)

batch_ground_inflows = jnp.broadcast_to(
    ground_truth_inflows,
    shape=(n_rollouts, s1-s0))
batch_ground_turn_props = jnp.broadcast_to(
    ground_truth_turn_props,
    shape=(n_rollouts, 24, 24))

subs_train['SOURCE-ARRIVAL-RATE'] = subs_train['SOURCE-ARRIVAL-RATE'].at[:,s0:s1].set(batch_ground_inflows)
subs_train['BETA'] = batch_ground_turn_props
rollouts = sampler(
        key,
        policy_params=None,
        hyperparams=None,
        subs=subs_train,
        model_params=compiler.model_params)
GROUND_TRUTH_LOOPS = rollouts['pvar']['flow-into-link'][:,:,:]

# Prepare the ground truth for sequential rollouts
subs_hmc['SOURCE-ARRIVAL-RATE'] = subs_hmc['SOURCE-ARRIVAL-RATE'].at[:,s0:s1].set(ground_truth_inflows)
subs_hmc['BETA'] = ground_truth_turn_props

rollout_hmc = sampler_hmc(
        key,
        policy_params=None,
        hyperparams=None,
        subs=subs_hmc,
        model_params=compiler.model_params)
GROUND_TRUTH_LOOPS_HMC = rollout_hmc['pvar']['flow-into-link'][:,:,:]


def parametrized_policy(input):
    # Currently assumes a multivariate normal policy with
    # a diagonal covariance matrix

    mlp = hk.Sequential([
        hk.Linear(32), jax.nn.relu,
        hk.Linear(32), jax.nn.relu,
        hk.Linear(2)
    ])
    output = mlp(input)
    mean, cov = (jnp.squeeze(x) for x in jnp.split(output, 2, axis=1))
    return mean, jnp.diag(jax.nn.softplus(cov))

policy = hk.transform(parametrized_policy)
policy_apply = jax.jit(policy.apply)
key, subkey = jax.random.split(key)
theta = policy.init(subkey, one_hot_inputs)

def policy_pdf(theta, rng, actions):
    mean, cov = policy_apply(theta, rng, one_hot_inputs)
    unconstrained_actions = bijector_obj.inverse(actions)
    normal_pdf = jax.scipy.stats.multivariate_normal.pdf(unconstrained_actions, mean=mean, cov=cov)
    density_correction = jnp.squeeze(jnp.apply_along_axis(
        bijector_obj._inverse_det_jacobian, axis=1, arr=actions))
    return normal_pdf * density_correction

def policy_sample(theta, rng):
    mean, cov = policy_apply(theta, rng, one_hot_inputs)
    action_sample = jax.random.multivariate_normal(
        rng, mean, cov,
        shape=(n_rollouts,))
    action_sample = bijector_obj.forward(action_sample)
    return action_sample

def reward_batched(rng, subs, actions):
    od = actions.reshape(n_rollouts, len(srcs), len(sinks))
    inflows, turn_props = convert_od_to_flows_and_turn_props_jax(od, transfer_matrix)
    subs['SOURCE-ARRIVAL-RATE'] = subs['SOURCE-ARRIVAL-RATE'].at[:,s0:s1].set(inflows)
    subs['BETA'] = turn_props
    rollouts = sampler(
        rng,
        policy_params=None,
        hyperparams=None,
        subs=subs,
        model_params=compiler.model_params)
    delta = rollouts['pvar']['flow-into-link'][:,:,:] - GROUND_TRUTH_LOOPS
    return jnp.sum(delta*delta, axis=(1,2))

def reward_seqtl(rng, subs, action):
    od = action.reshape(1, len(srcs), len(sinks))
    inflows, turn_props = convert_od_to_flows_and_turn_props_jax(od, transfer_matrix)
    subs['SOURCE-ARRIVAL-RATE'] = subs['SOURCE-ARRIVAL-RATE'].at[:,s0:s1].set(inflows)
    subs['BETA'] = turn_props
    rollout = sampler_hmc(
        rng,
        policy_params=None,
        hyperparams=None,
        subs=subs,
        model_params=compiler.model_params)
    delta = rollout['pvar']['flow-into-link'][:,:,:] - GROUND_TRUTH_LOOPS_HMC
    return jnp.sum(delta*delta, axis=(1,2))

clip_val = jnp.array([1e-3], dtype=compiler.REAL)

def scalar_mult(alpha, v):
    return alpha * v

weighting_map = jax.vmap(
    scalar_mult, in_axes=0, out_axes=0)

@jax.jit
def collect_covar_data(theta, C, iter):
    ip0 = 0
    for leaf in jax.tree_util.tree_leaves(theta):
        leaf = leaf.flatten()
        ip1 = ip0 + leaf.shape[0]
        C = C.at[ip0:ip1, iter].set(leaf)
        ip0 = ip1
    return C

def parse_optimizer(config):
    if config['optimizer'] == 'rmsprop':
        optimizer = optax.rmsprop(learning_rate=config['lr'], momentum=config['momentum'])
    elif config['optimizer'] == 'adam':
        optimizer = optax.adam(learning_rate=config['lr'])
    elif config['optimizer'] == 'sgd':
        optimizer = optax.sgd(learning_rate=config['lr'], momentum=config['momentum'])
    else:
        raise ValueError
    return optimizer

# === REINFORCE ===
reinforce_config = {
    'optimizer': 'rmsprop',
    'lr': 1e-3,
    'momentum': 0.1,
}

@jax.jit
def reinforce_inner_loop(key, theta, subs):
    grads = jax.tree_util.tree_map(lambda _: jnp.zeros(1), theta)

    batch_rewards = 0.
    batch_actions = jnp.empty(shape=(batch_size, action_dim))

    for si in range(n_shards):
        key, subkey = jax.random.split(key)
        actions = policy_sample(theta, subkey)
        pi = policy_pdf(theta, subkey, actions)
        dpi = jax.jacrev(policy_pdf, argnums=0)(theta, subkey, actions)
        rewards = reward_batched(subkey, subs, actions)
        factors = rewards / (pi + 1e-6)

        w = lambda x: jnp.mean(weighting_map(factors, x), axis=0)
        accumulant = jax.tree_util.tree_map(w, dpi)
        grads = jax.tree_util.tree_map(lambda x, y: x+(y/n_shards), grads, accumulant)

        batch_rewards += jnp.mean(rewards)/n_shards
        batch_actions = batch_actions.at[si*n_rollouts:(si+1)*n_rollouts].set(actions)

    return key, grads, batch_rewards, batch_actions

def reinforce(key, n_iters, theta, subs, config):
    # initialize stats collection
    X, Y = np.arange(n_iters), np.zeros(shape=(n_iters,5,action_dim))
    n_params = sum(leaf.flatten().shape[0] for leaf in jax.tree_util.tree_leaves(theta))
    C = jnp.zeros(shape=(n_params, n_iters))

    # initialize optimizer
    optimizer = parse_optimizer(config)
    opt_state = optimizer.init(theta)

    # run
    for idx in range(n_iters):
        subt0 = timer()

        key, subkey = jax.random.split(key)
        mean, cov = policy_apply(theta, subkey, one_hot_inputs)

        key, grads, batch_rewards, batch_actions = reinforce_inner_loop(key, theta, subs)
        updates, opt_state = optimizer.update(grads, opt_state)
        theta = optax.apply_updates(theta, updates)

        C = collect_covar_data(theta, C, idx)

        Y[idx,0] = batch_rewards
        Y[idx,1] = mean
        Y[idx,2] = jnp.diag(cov)
        Y[idx,3] = jnp.squeeze(jnp.mean(batch_actions, axis=0))
        Y[idx,4] = jnp.squeeze(jnp.std(batch_actions, axis=0))
        subt1 = timer()
        print(f'Iter {idx} :: REINFORCE :: Runtime={subt1-subt0}s')
        print(f'Untransformed parametrized policy [Mean, Diag(Cov)] = \n{Y[idx,1:3]}')
        print(f'Transformed action sample statistics [[Means], [StDevs]] = \n{Y[idx,3:5]}')
        print(f'Eval. reward={Y[idx,0,0]}\n')

    return X, Y, C


# === REINFORCE with Importance Sampling ====
impsmp_config = {
    'hmc_num_iters': int(batch_size),
    'hmc_step_size': 0.5,
    'optimizer': 'rmsprop',
    'lr': 1e-3,
    'momentum': 0.1,
    'est_Z': False,
}
impsmp_config['hmc_num_burnin_steps'] = int(impsmp_config['hmc_num_iters']/8)

@jax.jit
def unnormalized_log_rho(key, theta, subs, a):
    dpi = jax.jacrev(policy_pdf, argnums=0)(theta, key, a)
    #dpi_norm = jax.tree_util.tree_reduce(lambda x,y: x + jnp.sum((y/100)**2), dpi, initializer=jnp.zeros(1))
    #dpi_norm = 100*jnp.sqrt(dpi_norm)
    dpi_norm = jax.tree_util.tree_reduce(lambda x, y: jnp.maximum(x, jnp.max(jnp.abs(y))), dpi, initializer=jnp.array([-jnp.inf]))
    rewards = reward_seqtl(key, subs, a)
    density = jnp.abs(rewards) * dpi_norm
    return jnp.log(density)[0]
    #return jnp.log(jnp.maximum(density, clip_val))[0]

batch_unnormalized_log_rho = jax.jit(jax.vmap(
    unnormalized_log_rho, (None, None, None, 0), 0))

@functools.partial(jax.jit, static_argnums=(5,))
def impsmp_inner_loop(key, theta, subs_train, subs_hmc, samples, est_Z=False):
    key, *subkeys = jax.random.split(key, num=6)

    grads = jax.tree_util.tree_map(lambda _: jnp.zeros(1), theta)
    hmc_sample_reward_mean = 0.
    if est_Z: Zinv = 0.

    for si in range(n_shards):
        actions = samples[si*n_rollouts:(si+1)*n_rollouts]
        dpi = jax.jacrev(policy_pdf, argnums=0)(theta, subkeys[2], actions)
        rewards = reward_batched(subkeys[2], subs_train, actions)
        rho = jnp.exp(batch_unnormalized_log_rho(subkeys[3], theta, subs_hmc, actions[:,jnp.newaxis,:]))
        factors = rewards / (rho + 1e-8)

        if est_Z:
            pi = policy_pdf(theta, subkeys[2], actions)
            Zinv += jnp.sum(pi/rho)

        w = lambda x: jnp.mean(weighting_map(factors, x), axis=0)
        accumulant = jax.tree_util.tree_map(w, dpi)
        grads = jax.tree_util.tree_map(lambda x, y: x+(y/n_shards), grads, accumulant)
        hmc_sample_reward_mean += jnp.mean(rewards)/n_shards

    if est_Z:
        grads = jax.tree_util.tree_map(lambda x: x / jnp.maximum(Z_inv, clip_val), grads)

    return key, grads, hmc_sample_reward_mean

def impsmp(key, n_iters, theta, subs_train, subs_hmc, config):
    # initialize stats collection
    X, Y = np.arange(n_iters), np.zeros(shape=(n_iters, 5, action_dim))
    n_params = sum(leaf.flatten().shape[0] for leaf in jax.tree_util.tree_leaves(theta))
    C = jnp.zeros(shape=(n_params, n_iters))

    # initialize hmc
    key, subkey = jax.random.split(key)
    hmc_initializer = jax.random.uniform(
        subkey,
        shape=(1, action_dim),
        minval=0., maxval=0.05)

    # initialize optimizer
    optimizer = parse_optimizer(config)
    opt_state = optimizer.init(theta)

    # initialize unconstraining bijector
    unconstraining_bijector = [
        bijector_obj
    ]

    # run
    for idx in range(n_iters):
        subt0 = timer()
        key, *subkeys = jax.random.split(key, num=6)

        log_density = functools.partial(
            unnormalized_log_rho, subkeys[1], theta, subs_hmc)

        adaptive_hmc_kernel = tfp.mcmc.TransformedTransitionKernel(
            inner_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
                inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
                    target_log_prob_fn=log_density,
                    num_leapfrog_steps=3,
                    step_size=config['hmc_step_size']),
                num_adaptation_steps=int(config['hmc_num_burnin_steps'] * 0.8)),
            bijector=unconstraining_bijector)

        samples, is_accepted = tfp.mcmc.sample_chain(
            seed=subkeys[2],
            num_results=config['hmc_num_iters'],
            num_burnin_steps=config['hmc_num_burnin_steps'],
            current_state=hmc_initializer,
            kernel=adaptive_hmc_kernel,
            trace_fn=lambda _, pkr: pkr.inner_results.inner_results.is_accepted)

        samples = jnp.squeeze(samples)

        key, grads, hmc_sample_reward_mean = impsmp_inner_loop(
            key, theta, subs_train, subs_hmc, samples, est_Z=config['est_Z'])

        updates, opt_state = optimizer.update(grads, opt_state)
        theta = optax.apply_updates(theta, updates)

        # Initialize the next chain at a random point of the current chain
        hmc_intializer = jax.random.choice(subkeys[3], samples)

        mean, cov = policy_apply(theta, subkeys[4], one_hot_inputs)
        eval_actions = policy_sample(theta, subkeys[4])
        eval_rewards = jnp.mean(reward_batched(subkeys[4], subs_train, eval_actions))

        Y[idx,0,0] = eval_rewards
        Y[idx,0,1] = hmc_sample_reward_mean
        Y[idx,1] = mean
        Y[idx,2] = jnp.diag(cov)
        Y[idx,3] = jnp.mean(eval_actions, axis=0)
        Y[idx,4] = jnp.std(eval_actions, axis=0)
        subt1 = timer()
        print(f'Iter {idx} :: Importance Sampling :: Runtime={subt1-subt0}s :: HMC acceptance rate={jnp.mean(is_accepted)*100:.2f}%')
        print(f'Untransformed parametrized policy [Mean, Diag(Cov)] = \n{Y[idx,1:3]}')
        print(f'Transformed action sample statistics [Mean, StDev] = \n{Y[idx,3:5]}')
        print(f'HMC sample reward={Y[idx,0,1]} :: Eval reward={Y[idx,0,0]}\n')

        C = collect_covar_data(theta, C, idx)

    return X, Y, C

# === Debugging utils ===
def print_shapes(x):
    print(x.shape)

def eval_single_rollout(a):
    a = jnp.asarray(a)
    rewards = reward_seqtl(key, subs_hmc, a)
    return rewards


method = 'reinforce'
n_iters = 100
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

with jax.disable_jit(disable=False):
    key, subkey = jax.random.split(key)
    t1 = timer()
    if method == 'impsmp':
        method_config = impsmp_config
        X, Y, C = impsmp(key=subkey, n_iters=n_iters, theta=theta, subs_train=subs_train, subs_hmc=subs_hmc, config=method_config)
    elif method == 'reinforce':
        method_config = reinforce_config
        X, Y, C = reinforce(key=subkey, n_iters=n_iters, theta=theta, subs=subs_train, config=method_config)
    else: raise KeyError
    t2 = timer()

id = f'{method}_2x2_b{batch_size}_nS{action_dim}_opt{method_config["optimizer"]}_iters{n_iters}_od'

thetacov = jnp.cov(C)
sns.heatmap(thetacov)
plt.savefig(f'img/{timestamp}_{id}_covar.png')

with open(f'img/{timestamp}_{id}.json', 'w') as file:
    savedata = {
        'action_dim': action_dim,
        'X': X.tolist(),
        'Y': Y.T.tolist(),}
    savedata.update(method_config)
    json.dump(savedata, file)

print('Setup took', t1-t0, 'seconds')
print('Optimization took', t2-t1, 'seconds')
