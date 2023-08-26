import sys

from pyRDDLGym import ExampleManager
from pyRDDLGym import RDDLEnv
from pyRDDLGym.Core.Policies.Agents import RandomAgent


def main(domain, instance, method_name=None, episodes=1):
    print(f'preparing to launch instance {instance} of domain {domain}...')
    
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
    
    # main simulation loop
    agent.evaluate(myEnv, episodes=episodes, verbose=True)
    
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
        try:
            episodes = int(episodes)
        except:
            raise ValueError('episodes argument must be an integer, received:' + episodes)
    main(domain, instance, method_name, episodes)
