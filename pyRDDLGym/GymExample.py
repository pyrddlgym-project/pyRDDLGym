from pyRDDLGym.Core.Env import RDDLEnv as RDDLEnv
from pyRDDLGym.Policies.Agents import RandomAgent
from pyRDDLGym.Examples.ExampleManager import ExampleManager

# ENV = 'Power generation'
ENV = 'MarsRover'
# ENV = 'UAV continuous'
# ENV = 'UAV discrete'
# ENV = 'UAV mixed'
# ENV = 'Wildfire'
# ENV = 'Mountaincar'
# ENV = 'Cartpole'
# ENV = 'Elevators'
# ENV = 'Recsim'
# ENV = 'RaceCar'

def main():
    # get the environment info
    EnvInfo = ExampleManager.GetEnvInfo(ENV)
    # set up the environment class, choose instance 0 because every example has at least one example instance
    myEnv = RDDLEnv.RDDLEnv(domain=EnvInfo.get_domain(), instance=EnvInfo.get_instance(0))
    # set up the environment visualizer
    myEnv.set_visualizer(EnvInfo.get_visualizer())

    # set up an example aget
    agent = RandomAgent(action_space=myEnv.action_space, num_actions=myEnv.NumConcurrentActions)

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
    print("episode ended with reward {}".format(total_reward))





if __name__ == "__main__":
    main()