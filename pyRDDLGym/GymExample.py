import sys

from pyRDDLGym import ExampleManager
from pyRDDLGym import RDDLEnv
from pyRDDLGym.Core.Policies.Agents import RandomAgent


def main(domain, instance, method_name=None, episodes=1):
    
    # get the environment info
    EnvInfo = ExampleManager.GetEnvInfo(domain)
    
    # set up the environment class
    myEnv = RDDLEnv.RDDLEnv(domain=EnvInfo.get_domain(),
                            instance=EnvInfo.get_instance(instance),
                            enforce_action_constraints=False,
                            log=method_name is not None,
                            simlogname=method_name)
    myEnv.seed(42)
    
    # set up the environment visualizer
    myEnv.set_visualizer(EnvInfo.get_visualizer())
    
    # set up an example agent
    agent = RandomAgent(action_space=myEnv.action_space,
                        num_actions=myEnv.numConcurrentActions,
                        seed=42)
    
    # main evaluation loop
    # can do the following with one command
    # agent.evaluate(myEnv, episodes=episodes, verbose=True, render=True)
    for episode in range(episodes):
        total_reward = 0
        state = myEnv.reset()
        for step in range(myEnv.horizon):
            myEnv.render()
            action = agent.sample_action(state)
            next_state, reward, done, info = myEnv.step(action)
            total_reward += reward
            print(f'step       = {step}\n'
                  f'state      = {state}\n'
                  f'action     = {action}\n'
                  f'next state = {next_state}\n'
                  f'reward     = {reward}\n')
            state = next_state
            if done:
                break
        print(f'episode {episode} ended with return {total_reward}')
        
    myEnv.close()


if __name__ == "__main__":
    args = sys.argv
    method_name = None
    episodes = 1
    if len(args) < 3:
        domain, instance = 'HVAC', '0'
    elif len(args) < 4:
        domain, instance = args[1:3]
    elif len(args) < 5:
        domain, instance, method_name = args[1:4]
    else:
        domain, instance, method_name, episodes = args[1:5]
        episodes = int(episodes)
    main(domain, instance, method_name, episodes)
