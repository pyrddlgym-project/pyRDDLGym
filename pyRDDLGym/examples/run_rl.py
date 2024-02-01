'''In this example, the PPO algorithm is used to solve a given domain.
    
The syntax for running this example is:

    python run_rl.py <domain> <instance> <method> [<steps>] [<learning_rate>]
    
where:
    <domain> is the name of a domain located in the /Examples directory
    <instance> is the instance number
    <method> is the algorithm to train (e.g. PPO, DQN etc.)
    <steps> is the number of trials to simulate for training
    <learning_rate> is the learning rate to use to train the agent
'''
import sys
from stable_baselines3 import *

import pyRDDLGym
from pyRDDLGym.baselines.stable_baselines.env import StableBaselinesRDDLEnv
from pyRDDLGym.baselines.stable_baselines.agent import StableBaselinesAgent

METHODS = {'a2c': A2C, 'ddpg': DDPG, 'dqn': DQN, 'ppo': PPO, 'sac': SAC, 
           'td3': TD3}

def main(domain, instance, method, steps=200000, learning_rate=None):
    
    # set up the environment
    env = pyRDDLGym.make(domain, instance, 
                         base_class=StableBaselinesRDDLEnv,
                         enforce_action_constraints=True)
    
    # train the PPO agent
    kwargs = {'verbose': 1}
    if learning_rate is not None:
        kwargs['learning_rate'] = learning_rate
    model = METHODS[method]('MultiInputPolicy', env, **kwargs)    
    model.learn(total_timesteps=steps)
    
    # wrap the agent in a RDDL policy and evaluate
    ppo_agent = StableBaselinesAgent(model)
    ppo_agent.evaluate(env, episodes=1, verbose=True, render=True)
    
    env.close()
        
        
if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 3:
        print('python run_rl.py <domain> <instance> <method> [<steps>] [<learning_rate>]')
        exit(1)
    if args[2] not in METHODS:
        print(f'<method> in {set(METHODS.keys())}')
        exit(1)
    kwargs = {'domain': args[0], 'instance': args[1], 'method': args[2]}
    if len(args) >= 4: kwargs['steps'] = int(args[3])
    if len(args) >= 5: kwargs['learning_rate'] = float(args[4])
    main(**kwargs)
