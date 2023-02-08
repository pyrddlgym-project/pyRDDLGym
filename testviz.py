from pyRDDLGym import RDDLEnv
from pyRDDLGym import ExampleManager
from pyRDDLGym.Policies.Agents import RandomAgent

import time

ExampleManager.ListExamples()

# ENV = 'Reservoir discrete'
# ENV = 'Reservoir continuous'
ENV = 'HVAC'
# ENV = 'MarsRover'

# get the environment info
EnvInfo = ExampleManager.GetEnvInfo(ENV)
# access to the domain file
EnvInfo.get_domain()
#list all available instances for that domain
EnvInfo.list_instances()
# access to instance 0  
EnvInfo.get_instance(0)
# obtain the dedicated visualizer object of the domain if exists
EnvInfo.get_visualizer()


# set up the environment class, choose instance 0 because every example has at least one example instance
myEnv = RDDLEnv.RDDLEnv(domain=EnvInfo.get_domain(), instance=EnvInfo.get_instance(1))
# # set up the environment visualizer
myEnv.set_visualizer(EnvInfo.get_visualizer())

agent = RandomAgent(action_space=myEnv.action_space, num_actions=myEnv.numConcurrentActions)


total_reward = 0
state = myEnv.reset()
for step in range(myEnv.horizon):
    myEnv.render()
    action = agent.sample_action()
    next_state, reward, done, info = myEnv.step(action)
    total_reward += reward
    print()
    print('step       = {}'.format(step))
    print('state      = {}'.format(state))
    print('action     = {}'.format(action))
    print('next state = {}'.format(next_state))
    print('reward     = {}'.format(reward))
    state = next_state
    if done:
        break
    # time.sleep(10)

print("episode ended with reward {}".format(total_reward))
myEnv.close()