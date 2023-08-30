import sys

from pyRDDLGym.Core.Env.RDDLEnv import RDDLEnv
from pyRDDLGym.Core.Gurobi.GurobiRDDLPlanner import GurobiStraightLinePlan
from pyRDDLGym.Core.Gurobi.GurobiRDDLPlanner import GurobiOnlineController
from pyRDDLGym.Examples.ExampleManager import ExampleManager


def main(domain, instance, horizon):
    
    # create the environment
    EnvInfo = ExampleManager.GetEnvInfo(domain)    
    env = RDDLEnv(domain=EnvInfo.get_domain(),
                  instance=EnvInfo.get_instance(instance),
                  enforce_action_constraints=True)
    env.set_visualizer(EnvInfo.get_visualizer())
    
    # create the controller agent
    controller = GurobiOnlineController(
        rddl=env.model,
        plan=GurobiStraightLinePlan(),
        rollout_horizon=horizon,
        model_params={'NonConvex': 2, 'OutputFlag': 1})
    
    # evaluate the agent
    controller.evaluate(env, verbose=True, render=True)

            
if __name__ == "__main__":
    args = sys.argv
    domain, instance, horizon = 'Wildfire', 0, 5
    if len(args) == 2:
        domain = args[1]
    elif len(args) == 3:
        domain, instance = args[1:3]
    elif len(args) >= 4:
        domain, instance, horizon = args[1:4]
        horizon = int(horizon)
    
    main(domain, instance, horizon)
    
