# import os
from time import time

from pyRDDLGym import RDDLEnv
from pyRDDLGym import ExampleManager
from pyRDDLGym.Policies.Agents import RandomAgent
from pyRDDLGym.Visualizer.TextViz import TextVisualizer

# ENV = 'PowerGeneration'
# ENV = 'MarsRover'
# ENV = 'UAV continuous'
# ENV = 'UAV discrete'
# ENV = 'UAV mixed'
ENV = 'Wildfire'
# ENV = 'MountainCar'
# ENV = 'CartPole continuous'
# ENV = 'CartPole discrete'
# ENV = 'Elevators'
# ENV = 'RecSim'
# ENV = 'RaceCar'


def main():
    # get the environment info
    EnvInfo = ExampleManager.GetEnvInfo(ENV)
    # set up the environment class, choose instance 0 because every example has at least one example instance
    myEnv = RDDLEnv.RDDLEnv(domain=EnvInfo.get_domain(), 
                            instance=EnvInfo.get_instance(0),
                            enforce_action_constraints=True)
    
    # set up the environment visualizer
    # frames_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Visualizer', 'Frames')
    myEnv.set_visualizer(TextVisualizer)#EnvInfo.get_visualizer())
                        # movie_gen=MovieGenerator(frames_path, ENV, 200), movie_per_episode=True)
    
    # set up an example aget
    agent = RandomAgent(action_space=myEnv.action_space, num_actions=myEnv.NumConcurrentActions, seed=0)

    # Start recording time
    stime = time()

    for episode in range(1):
        total_reward = 0
        state = myEnv.reset()
        for step in range(myEnv.horizon):
            # myEnv.render()
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
        print("episode {} ended with reward {}".format(episode, total_reward))
    
    myEnv.close()

    etime = time()
    print(f"Time taken for {ENV} environment for {myEnv.horizon} time steps: {etime-stime} seconds")

if __name__ == "__main__":
    main()
