'''In this example, a random policy is constructed and its performance is
evaluated on a specified domain. 

The syntax for running this example is:

    python run_gym2.py <domain> <instance> [<episodes>] [<seed>]
    
where:
    <domain> is the name of a domain located in the /Examples directory
    <instance> is the instance number
    <episodes> is a positive integer for the number of episodes to simulate
    (defaults to 1)
    <seed> is a positive integer RNG key (defaults to 42)
'''
import sys

import pyRDDLGym
from pyRDDLGym.core.policy import RandomAgent


def main(domain, instance, episodes=1, seed=42):
    
    # set up the environment
    env = pyRDDLGym.make(domain, instance, enforce_action_constraints=True)
    env.seed(seed)
    
    # set up an example agent
    agent = RandomAgent(action_space=env.action_space,
                        num_actions=env.max_allowed_actions,
                        seed=seed)
    
    # main evaluation loop, same as the following line:
    # agent.evaluate(env, episodes=episodes, verbose=True, render=True)
    for episode in range(episodes):
        total_reward = 0
        state, _ = env.reset()
        for step in range(env.horizon):
            env.render()
            action = agent.sample_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
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
        print('python run_gym2.py <domain> <instance> [<episodes>] [<seed>]')
        exit(1)
    kwargs = {'domain': args[0], 'instance': args[1]}
    if len(args) >= 3: kwargs['episodes'] = int(args[2])
    if len(args) >= 4: kwargs['seed'] = int(args[3])
    main(**kwargs)
