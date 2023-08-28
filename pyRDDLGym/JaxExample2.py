import os
import sys
import time

from pyRDDLGym.Core.Env.RDDLEnv import RDDLEnv
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import load_config
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxRDDLBackpropPlanner
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxOfflineController
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
    
    # create the planning algorithm and controller
    planner = JaxRDDLBackpropPlanner(rddl=env.model, **planner_args)
    controller = JaxOfflineController(planner, params={}, **train_args)
    
    # expand budget in config
    train_args['train_seconds'] = 60
    
    # we will train for 10 seconds, then evaluate, then repeat
    eval_period = 10
    time_last_eval = 0
    for callback in planner.optimize_generator(**train_args):
        if callback['elapsed_time'] - time_last_eval > eval_period:
            controller.params = callback['best_params']
            controller.evaluate(env, ground_state=False, verbose=False, render=True)
            time_last_eval = callback['elapsed_time']

        
if __name__ == "__main__":
    domain, instance = 'Wildfire', 0
    if len(sys.argv) == 2:
        domain = sys.argv[1]
    elif len(sys.argv) >= 3:
        domain, instance = sys.argv[1:3] 
    main(domain, instance, 'slp')
    
