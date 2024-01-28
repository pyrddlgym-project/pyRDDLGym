'''In this example, the user has the choice to run the Jax planner with three
different options:
    
    1. slp runs the straight-line planner offline, which trains an open-loop plan
    2. drp runs the deep reactive policy, which trains a policy network
    3. replan runs the straight-line planner online, at every decision epoch
    
The syntax for running this example is:

    python run_jax.py <domain> <instance> <method>
    
where:
    <domain> is the name of a domain located in the /Examples directory
    <instance> is the instance number
    <method> is either slp, drp, or replan
'''
import os
import sys

import pyRDDLGym
from pyRDDLGym.baselines.jaxplan.planner import (
    load_config, JaxRDDLBackpropPlanner, JaxOfflineController, JaxOnlineController
)

    
def main(domain, instance, method):
    
    # set up the environment
    env = pyRDDLGym.make(domain, instance, vectorized=True, enforce_action_constraints=True)
    
    # load the config file with planner settings
    abs_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(abs_path, 'jax_configs', f'{domain}_{method}.cfg') 
    planner_args, _, train_args = load_config(config_path)
    
    # create the planning algorithm
    planner = JaxRDDLBackpropPlanner(rddl=env.model, **planner_args)
    
    # create the controller   
    if method == 'replan':
        controller = JaxOnlineController(planner, **train_args)
    else:
        controller = JaxOfflineController(planner, **train_args)
    
    controller.evaluate(env, verbose=True, render=True)
    
    env.close()
        
        
if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 3:
        print('python run_jax.py <domain> <instance> <method>')
        exit(1)
    if args[2] not in ['drp', 'slp', 'replan']:
        print('<method> in [drp, slp, replan]')
        exit(1)
    domain, instance, method = args[:3]
    main(domain, instance, method)
    