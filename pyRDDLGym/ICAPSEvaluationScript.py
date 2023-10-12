import json
import os
import sys

from pyRDDLGym.Core.Env.RDDLEnv import RDDLEnv
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import load_config
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxRDDLBackpropPlanner
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxOnlineController

from rddlrepository.Manager.RDDLRepoManager import RDDLRepoManager as RDDLRepoManager

 
def main_jaxplan(domain, instance):
    
    # create the environment
    manager = RDDLRepoManager(rebuild=True)
    EnvInfo = manager.get_problem(domain)    
    env = RDDLEnv(domain=EnvInfo.get_domain(),
                  instance=EnvInfo.get_instance(instance),
                  enforce_action_constraints=True,
                  log=True, simlogname=f'JaxPlan_{domain}_{instance}.log')
    
    # load the config file with planner settings
    abs_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(abs_path, 'ICAPSConfigs', f'{domain}.cfg') 
    planner_args, _, train_args = load_config(config_path)
    
    # create the planning algorithm
    planner = JaxRDDLBackpropPlanner(rddl=env.model, **planner_args)
    policy = JaxOnlineController(planner, **train_args)
    
    # evaluation
    result = policy.evaluate(env, verbose=True, episodes=10, ground_state=False)
    
    # dump all history to files
    with open(f'JaxPlan_{domain}_{instance}.json', 'w') as fp:
        json.dump(result, fp, sort_keys=True, indent=4)        
    env.close()


if __name__ == '__main__':
    args = sys.argv[1:]
    domain, instance = args[0], args[1]
    main_jaxplan(domain, instance)
    