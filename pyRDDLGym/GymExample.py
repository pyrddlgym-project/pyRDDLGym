import sys

from pyRDDLGym import RDDLEnv
from pyRDDLGym import ExampleManager
from pyRDDLGym.Policies.Agents import RandomAgent
# from pyRDDLGym.Visualizer.MovieGenerator import MovieGenerator

def main(env, inst):
    print(f'preparing to launch instance {inst} of domain {env}...')
    
    # get the environment info
    # ExampleManager.RebuildExamples()
    EnvInfo = ExampleManager.GetEnvInfo(env)
    
    # set up the environment class, choose instance 0 because every example has at least one example instance
    myEnv = RDDLEnv.RDDLEnv(domain=EnvInfo.get_domain(), 
                            instance=EnvInfo.get_instance(inst),
                            enforce_action_constraints=False,
                            debug=False)
    
    # set up the environment visualizer
    # frames_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Visualizer', 'Frames')
    myEnv.set_visualizer(EnvInfo.get_visualizer())
                        # movie_gen=MovieGenerator(frames_path, ENV, 200), movie_per_episode=True)
    
    # set up an example aget
    agent = RandomAgent(action_space=myEnv.action_space, 
                        num_actions=myEnv.numConcurrentActions)

    for episode in range(1):
        total_reward = 0
        state = myEnv.reset()
        for step in range(myEnv.horizon):
            myEnv.render()
            action = agent.sample_action()
            next_state, reward, done, info = myEnv.step(action)
            total_reward += reward
            print()
            print(f'step       = {step}')
            print(f'state      = {state}')
            print(f'action     = {action}')
            print(f'next state = {next_state}')
            print(f'reward     = {reward}')
            state = next_state
            if done:
                break
        print(f'episode {episode} ended with reward {total_reward}')
    
    myEnv.close()


if __name__ == "__main__":
    args = sys.argv
    if len(args) < 3:
        env, inst = 'HVAC', '0'
    else:
        env, inst = args[1:3]
    main(env, inst)
