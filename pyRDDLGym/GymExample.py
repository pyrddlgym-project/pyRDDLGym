import sys

from pyRDDLGym import ExampleManager
from pyRDDLGym import RDDLEnv
from pyRDDLGym.Core.Policies.Agents import RandomAgent


def main(domain, instance, episodes=1):
    
    # get the environment info
    EnvInfo = ExampleManager.GetEnvInfo(domain)
    
    # set up the environment
    env = RDDLEnv.RDDLEnv(domain=EnvInfo.get_domain(),
                          instance=EnvInfo.get_instance(instance))
    env.seed(42)
    
    # set up the environment visualizer
    env.set_visualizer(EnvInfo.get_visualizer())
    
    # set up an example agent
    agent = RandomAgent(action_space=env.action_space,
                        num_actions=env.numConcurrentActions,
                        seed=42)
    
    # main evaluation loop
    # can do the following with one command
    # agent.evaluate(myEnv, episodes=episodes, verbose=True, render=True)
    for episode in range(episodes):
        total_reward = 0
        state = env.reset()
        for step in range(env.horizon):
            env.render()
            action = agent.sample_action(state)
            next_state, reward, done, info = env.step(action)
            print(f'step       = {step}\n'
                  f'state      = {state}\n'
                  f'action     = {action}\n'
                  f'next state = {next_state}\n'
                  f'reward     = {reward}\n')
            total_reward += reward
            state = next_state
            if done:
                break
        print(f'episode {episode} ended with return {total_reward}')
       
    env.close()


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 2:
        domain, instance, episodes = 'HVAC', '0', 1
    elif len(args) < 3:
        domain, instance, episodes = args[0], args[1], 1
    else:
        domain, instance, episodes = args[0], args[1], int(args[2])
    main(domain, instance, episodes)
