import argparse
from time import perf_counter as timer
from scipy.stats import bernoulli

from pyRDDLGym import RDDLEnv
from pyRDDLGym import ExampleManager
from pyRDDLGym.Policies.Agents import RandomAgent

from learned_plans.two_cycle import plan as JAX2CyclePlan

parser = argparse.ArgumentParser(description='Evaluating benchmarks for a single-intersection traffic scenario')
parser.add_argument('agent', type=str, choices=[
    'webster',
    'random',
    'min',
    'max',
    'jax2cycle'
])
parser.add_argument('-r', '--render', action='store_true', help='Whether to render (visualize) the rollout')
parser.add_argument('-w', '--warmup', type=int, default=-1)
args = parser.parse_args()


EnvInfo = ExampleManager.GetEnvInfo('traffic')

t0 = timer()
myEnv = RDDLEnv.RDDLEnv(
    domain=EnvInfo.get_domain(),
    instance='webster_exp_equal_split.rddl')
t1 = timer()
print(f"Env. generated. Took {t1-t0}s")

# set up renderer
if args.render:
    myEnv.set_visualizer(EnvInfo.get_visualizer())


# set up agent
webster_change_ts = {32, 36, 69, 73, 106, 110, 143, 147}
webster_CT = 148
def get_webster_action(step):
    return {'advance___i0': step%webster_CT in webster_change_ts}

random_agent = RandomAgent(action_space=myEnv.action_space,
                           num_actions=myEnv.numConcurrentActions)
flips = bernoulli.rvs(0.5, size=myEnv.horizon)
def get_random_action(step):
    return {'advance___i0': flips[step]}
    return random_agent.sample_action()

min_change_ts = {5, 9, 15, 19, 25, 29, 35, 39}
min_CT = 40
def get_min_action(step):
    return {'advance___i0': step%min_CT in min_change_ts}

max_change_ts = {59, 63, 123, 127, 187, 191, 251, 255}
max_CT = 256
def get_max_action(step):
    return {'advance___i0': step%min_CT in max_change_ts}

def get_jax_2_cycle_action(step):
    return {'advance___i0': JAX2CyclePlan[step]}

if args.agent == 'random':
    get_action = get_random_action
elif args.agent == 'webster':
    get_action = get_webster_action
elif args.agent == 'min':
    get_action = get_min_action
elif args.agent == 'max':
    get_action = get_max_action
elif args.agent == 'jax2cycle':
    get_action = get_jax_2_cycle_action


# run evaluation
total_reward = 0
state = myEnv.reset()

t0 = timer()
# pass the warmup period
if args.warmup == -1:
    args.warmup = webster_CT*2

for step in range(args.warmup):
    action = get_webster_action(step)
    next_state, reward, done, info = myEnv.step(action)
    state = next_state
    if done:
        print('[eval_webster.py] Warning: Environment finished during warmup')
        exit()

# evaluate
for step in range(myEnv.horizon-args.warmup):
    if args.render: myEnv.render()
    action = get_action(step)
    next_state, reward, done, info = myEnv.step(action)
    total_reward += reward
    state = next_state
    if done:
        break

t1 = timer()
print(f"Episode total reward {total_reward}, time={t1-t0}s")
myEnv.close()
