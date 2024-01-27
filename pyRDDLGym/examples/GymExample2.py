'''In this example, a random policy is constructed and its performance is
evaluated on a specified domain. 

The syntax for running this example is:

    python GymExample2.py <domain> <instance> [<episodes>] [<seed>]
    
where:
    <domain> is the name of a domain located in the /Examples directory
    <instance> is the instance number
    <episodes> is a positive integer for the number of episodes to simulate
    (defaults to 1)
    <seed> is a positive integer RNG key (defaults to 42)
'''
import sys

from pyRDDLGym import ExampleManager
from pyRDDLGym import RDDLEnv
from pyRDDLGym.Core.Policies.Agents import RandomAgent


def main(domain, instance, episodes=1, seed=42):
    
    # set up the environment
    info = ExampleManager.GetEnvInfo(domain)
    env = RDDLEnv.RDDLEnv.build(info, instance)
    env.seed(seed)
    
    # set up an example agent
    agent = RandomAgent(action_space=env.action_space,
                        num_actions=env.max_allowed_actions,
                        seed=seed)
    
    # main evaluation loop, same as the following line:
    # agent.evaluate(env, episodes=episodes, verbose=True, render=True)
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
    
    # important when logging to save all traces
    env.close()


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 2:
        print('python GymExample2.py <domain> <instance> [<episodes>] [<seed>]')
        exit(1)
    kwargs = {'domain': args[0], 'instance': args[1]}
    if len(args) >= 3: kwargs['episodes'] = int(args[2])
    if len(args) >= 4: kwargs['seed'] = int(args[3])
    main(**kwargs)
