'''In this example, the PPO algorithm is used to solve a given domain.
    
The syntax for running this example is:

    python run_rl.py <domain> <instance> <steps>
    
where:
    <domain> is the name of a domain located in the /Examples directory
    <instance> is the instance number
    <steps> is the number of trials to simulate for training
'''
import sys
from stable_baselines3 import PPO

import pyRDDLGym
from pyRDDLGym.baselines.stable_baselines.env import RDDLStableBaselinesEnv
from pyRDDLGym.core.policy import BaseAgent

    
def main(domain, instance, steps):
    
    # set up the environment
    env = pyRDDLGym.make(domain, instance, 
                         enforce_action_constraints=True,
                         base_class=RDDLStableBaselinesEnv)
    
    # train the PPO agent
    model = PPO('MultiInputPolicy', env, verbose=1)
    model.learn(total_timesteps=steps)
    
    # container to hold trained PPO agent
    class PPOAgent(BaseAgent):
        use_tensor_obs = True
        
        def sample_action(self, state):
            return model.predict(state)[0]
        
    # evaluate
    PPOAgent().evaluate(env, episodes=1, verbose=True, render=True)
    env.close()
        
        
if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 3:
        print('python run_rl.py <domain> <instance> <steps>')
        exit(1)
    domain, instance, steps = args[:3]
    steps = int(steps)
    main(domain, instance, steps)
