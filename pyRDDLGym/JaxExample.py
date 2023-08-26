import os
import sys

from pyRDDLGym.Core.Env.RDDLEnv import RDDLEnv
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import load_config, \
    JaxRDDLBackpropPlanner, JaxOfflineController, JaxOnlineController
from pyRDDLGym.Examples.ExampleManager import ExampleManager

    
def main(domain, instance, method):
    
    # create the environment
    EnvInfo = ExampleManager.GetEnvInfo(domain)    
    env = RDDLEnv(domain=EnvInfo.get_domain(),
                  instance=EnvInfo.get_instance(instance),
                  enforce_action_constraints=True)
    env.set_visualizer(EnvInfo.get_visualizer())
    
    # load the config file with planner settings
    abs_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(abs_path, 'JaxPlanConfigs', f'{domain}_{method}.cfg') 
    planner_args, _, train_args = load_config(config_path)
    
    # create the planning algorithm
    planner = JaxRDDLBackpropPlanner(rddl=env.model, **planner_args)
    
    # create the controller agent    
    if method == 'replan':
        controller = JaxOnlineController(planner, **train_args)
    else:
        controller = JaxOfflineController(planner, **train_args)
    
    # evaluate the agent
    controller.evaluate(env, ground_state=False, verbose=True, render=True)
        
        
if __name__ == "__main__":
    domain, instance, method = 'Wildfire', 0, 'drp'
    if len(sys.argv) >= 4:
        domain, instance, method = sys.argv[1:4]
        
    main(domain, instance, method)
    
