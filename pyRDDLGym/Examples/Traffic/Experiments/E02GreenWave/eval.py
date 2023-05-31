from sys import argv
from time import perf_counter as timer
from pyRDDLGym import RDDLEnv
from pyRDDLGym import ExampleManager
from pyRDDLGym.Policies.Agents import RandomAgent

from tsd import TimeSpaceDiagram
from warmup import warmup

try:
    agent_id = argv[1]
except IndexError as e:
    print('argv: agent_id')
    raise e

EnvInfo = ExampleManager.GetEnvInfo('traffic2phase')

myEnv = RDDLEnv.RDDLEnv(
    domain=EnvInfo.get_domain(),
    instance='instances/instance01.rddl')

# warmup
_, num_warmup_steps = warmup(myEnv, EnvInfo)

# set up renderer
myEnv.set_visualizer(EnvInfo.get_visualizer())

# set up agent
random_agent = RandomAgent(action_space=myEnv.action_space,
                           num_actions=myEnv.numConcurrentActions)
N = myEnv.numConcurrentActions
def sample_random_agent_action(t):
    return random_agent.sample_action(t)
def sample_min_agent_action(t):
    return {f'advance___i{n}': True for n in range(N)}
def sample_max_agent_action(t):
    return {f'advance___i{n}': False for n in range(N)}
def sample_offset_max_agent_action(t):
    base_plan = [False]*8*(N-1) + [True]*5 + [False]*60 + ([True]*32 + [False]*60)*5
    return {f'advance___i{n}': base_plan[t+8*(N-n-1)] for n in range(N)}

if agent_id == 'random':
    sample_agent_action = sample_random_agent_action
elif agent_id == 'min':
    sample_agent_action = sample_min_agent_action
elif agent_id == 'max':
    sample_agent_action = sample_max_agent_action
elif agent_id == 'offset_max':
    sample_agent_action = sample_offset_max_agent_action
else:
    raise ValueError(f'GreenWave eval: Unrecognized agent id {agent_id}')


# run evaluation
tsd = TimeSpaceDiagram(
    env=myEnv,
    num_warmup_steps=num_warmup_steps,
    agent_id=agent_id)
total_reward = 0

t0 = timer()
t_step = 0
for step in range(myEnv.horizon):
    #myEnv.render()
    action = sample_agent_action(step)
    state, reward, done, info = myEnv.step(action)
    tsd.step(state)
    total_reward += reward
    if done:
        break

t1 = timer()
print(f"Episode total reward {total_reward}, time={t1-t0}s")
myEnv.close()

tsd.plot(cmltrew=total_reward)
