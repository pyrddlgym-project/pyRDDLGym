import sys

from pyRDDLGym import ExampleManager
from pyRDDLGym import RDDLEnv
from pyRDDLGym.Core.Policies.Agents import RandomAgent

def main(env, inst, method_name=None, episodes=1):
    print(f'preparing to launch instance {inst} of domain {env}...')
    
    # get the environment info
    EnvInfo = ExampleManager.GetEnvInfo(env)
    
    # set up the environment class
    log = False if method_name is None else True
    myEnv = RDDLEnv.RDDLEnv(domain=EnvInfo.get_domain(), 
                            instance=EnvInfo.get_instance(inst),
                            enforce_action_constraints=False,
                            log=log,
                            simlogname=method_name)
    myEnv.seed(42)
    
    # set up the environment visualizer
    myEnv.set_visualizer(EnvInfo.get_visualizer())
    
    # set up an example agent
    agent = RandomAgent(action_space=myEnv.action_space, 
                        num_actions=myEnv.numConcurrentActions,
                        seed=42)
    
    # main simulation loop
    for episode in range(episodes):
        total_reward = 0
        state = myEnv.reset()
        for step in range(myEnv.horizon):
            myEnv.render()
            action = agent.sample_action()
            next_state, reward, done, _ = myEnv.step(action)
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
    method_name = None
    episodes = 1
    if len(args) < 3:
        env, inst = 'HVAC', '0'
    elif len(args) < 4:
        env, inst = args[1:3]
    elif len(args) < 5:
        env, inst, method_name = args[1:4]
    else:
        env, inst, method_name, episodes = args[1:5]
        try:
            episodes = int(episodes)
        except:
            raise ValueError('episodes argument must be an integer, received:' + episodes)
    main(env, inst, method_name, episodes)
