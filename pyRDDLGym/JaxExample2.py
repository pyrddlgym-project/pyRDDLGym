'''In this example, the straight-line Jax planner is run for a fixed amount of
time (60 seconds). However, every 10 seconds, the plan is visualized.
    
    1. slp runs the straight-line planner offline, which trains an open-loop plan
    2. drp runs the deep reactive policy, which trains a policy network
    3. replan runs the straight-line planner online, at every decision epoch
    
The syntax for running this example is:

    python JaxExample2.py <domain> <instance>
    
where:
    <domain> is the name of a domain located in the /Examples directory
    <instance> is the instance number
'''
import os
import sys

from pyRDDLGym.Core.Env.RDDLEnv import RDDLEnv
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import load_config
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxRDDLBackpropPlanner
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxOfflineController
from pyRDDLGym.Examples.ExampleManager import ExampleManager

    
def main(domain, instance):
    
    # create the environment
    EnvInfo = ExampleManager.GetEnvInfo(domain)    
    env = RDDLEnv(domain=EnvInfo.get_domain(),
                  instance=EnvInfo.get_instance(instance),
                  enforce_action_constraints=True)
    env.set_visualizer(EnvInfo.get_visualizer())
    
    # load the config file with planner settings
    abs_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(abs_path, 'JaxPlanConfigs', f'{domain}_slp.cfg') 
    planner_args, _, train_args = load_config(config_path)
    
    # create the planning algorithm and controller
    planner = JaxRDDLBackpropPlanner(rddl=env.model, **planner_args)
    controller = JaxOfflineController(planner, params={}, **train_args)
    
    # expand budget in config
    train_args['train_seconds'] = 60
    
    # train for 10 seconds, evaluate, then repeat
    eval_period = 10
    time_last_eval = 0
    for callback in planner.optimize_generator(**train_args):
        if callback['elapsed_time'] - time_last_eval > eval_period:
            controller.params = callback['best_params']
            controller.evaluate(env, verbose=False, render=True)
            time_last_eval = callback['elapsed_time']
            
    env.close()
        
if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 2:
        print('python JaxExample2.py <domain> <instance>')
        exit(0)
    domain, instance = args[:2]
    main(domain, instance)
    
