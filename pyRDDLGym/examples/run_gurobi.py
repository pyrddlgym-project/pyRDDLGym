'''This example runs the Gurobi planner. 

The syntax is:

    python run_gurobi.py <domain> <instance> <horizon>
    
where:
    <domain> is the name of a domain located in the /Examples directory
    <instance> is the instance number
    <horizon> is a positive integer representing the lookahead horizon
'''
import sys

import pyRDDLGym
from pyRDDLGym.baselines.gurobiplan.planner import (
    GurobiStraightLinePlan, GurobiOnlineController
)


def main(domain, instance, horizon):
    
    # create the environment
    env = pyRDDLGym.make(domain, instance, enforce_action_constraints=True)
    
    # create the controller
    controller = GurobiOnlineController(rddl=env.model,
                                        plan=GurobiStraightLinePlan(),
                                        rollout_horizon=horizon,
                                        model_params={'NonConvex': 2, 'OutputFlag': 1})
    controller.evaluate(env, verbose=True, render=True)
    
    env.close()

            
if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 3:
        print('python run_gurobi.py <domain> <instance> <horizon>')
        exit(1)
    domain, instance, horizon = args[:3]
    horizon = int(horizon)
    main(domain, instance, horizon)
    
