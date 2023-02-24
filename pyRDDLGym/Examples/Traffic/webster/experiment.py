from pyRDDLGym import RDDLEnv
from pyRDDLGym import ExampleManager
from pyRDDLGym.Policies.Agents import RandomAgent

from time import perf_counter as timer

ENV = 'traffic'

EnvInfo = ExampleManager.GetEnvInfo(ENV)

t0 = timer()
myEnv = RDDLEnv.RDDLEnv(domain=EnvInfo.get_domain(), instance='instance2.rddl')
t1 = timer()
print(f"Env. generated. Took {t1-t0}s")

myEnv.set_visualizer(EnvInfo.get_visualizer())

agent = RandomAgent(action_space=myEnv.action_space,
                    num_actions=myEnv.numConcurrentActions)

total_reward = 0
state = myEnv.reset()

t0 = timer()
for step in range(myEnv.horizon):
    myEnv.render()
    action = agent.sample_action()
    next_state, reward, info, done = myEnv.step(action)
    total_reward += reward
    state = next_state
    if done:
        break

t1 = timer()
print(f"Episode total reward {total_reward}, time={t1-t0}s")
myEnv.close()
