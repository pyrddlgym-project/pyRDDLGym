'''In this example, the PPO algorithm is used to solve a given domain.
    
The syntax for running this example is:

    python RLExample.py <domain> <instance> <steps>
    
where:
    <domain> is the name of a domain located in the /Examples directory
    <instance> is the instance number
    <steps> is the number of trials to simulate for training
'''
import sys
from stable_baselines3 import PPO

from pyRDDLGym.Core.Env.RDDLEnv import RDDLEnv
from pyRDDLGym.Examples.ExampleManager import ExampleManager

    
def main(domain, instance, steps):
    
    # set up the environment
    info = ExampleManager.GetEnvInfo(domain)    
    env = RDDLEnv.build(info, instance, enforce_action_constraints=True, 
                        new_gym_api=True, compact_action_space=True)
    env.reset()
    
    # train the agent
    model = PPO('MultiInputPolicy', env, verbose=1)
    model.learn(total_timesteps=steps)
    
    # evaluate
    total_reward = 0
    state, _ = env.reset()
    for step in range(env.horizon):
        env.render()
        action, _ = model.predict(state)
        next_state, reward, done, *_ = env.step(action)
        print(f'step       = {step}\n'
              f'state      = {state}\n'
              f'action     = {action}\n'
              f'next state = {next_state}\n'
              f'reward     = {reward}\n')
        total_reward += reward
        state = next_state
        if done:
            break
    print(f'episode ended with return {total_reward}')
    
    env.close()
        
        
if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 3:
        print('python RLExample.py <domain> <instance> <steps>')
        exit(1)
    domain, instance, steps = args[:3]
    main(domain, instance, steps)
