import sys

from pyRDDLGym.Core.Gurobi.GurobiRDDLPlan import GurobiRDDLStraightLinePlan
from pyRDDLGym.Core.Env.RDDLEnv import RDDLEnv
from pyRDDLGym.Examples.ExampleManager import ExampleManager
from pyRDDLGym.Core.Gurobi.GurobiRDDLBilevelOptimizer import GurobiRDDLBilevelOptimizer


def slp_replan(domain, inst, trials):
    EnvInfo = ExampleManager.GetEnvInfo(domain)    
    model = RDDLEnv(domain=EnvInfo.get_domain(),
                    instance=EnvInfo.get_instance(inst)).model
    policy = GurobiRDDLStraightLinePlan()
    planner = GurobiRDDLBilevelOptimizer(
        model, policy, state_bounds={'prevProd': (0, 10), 'temperature': (-30, 30)},
        rollout_horizon=30)
    planner.solve(10)
    
            
if __name__ == "__main__":
    if len(sys.argv) < 4:
        dom, inst, trials = 'PowerGen discrete', 0, 1
    else:
        dom, inst, trials = sys.argv[1:4]
        trials = int(trials)
    slp_replan(dom, inst, trials)
    
