import sys

from pyRDDLGym.Core.Gurobi.GurobiRDDLPlan import GurobiLinearPolicy
from pyRDDLGym.Core.Env.RDDLEnv import RDDLEnv
from pyRDDLGym.Examples.ExampleManager import ExampleManager
from pyRDDLGym.Core.Gurobi.GurobiRDDLBilevelOptimizer import GurobiRDDLBilevelOptimizer


def slp_replan(domain, inst, trials):
    EnvInfo = ExampleManager.GetEnvInfo(domain)    
    model = RDDLEnv(domain=EnvInfo.get_domain(),
                    instance=EnvInfo.get_instance(inst)).model
    
    def feature(action, states):
        obj = action.split('__')[-1]
        for state, value in states.items():
            if state.endswith(obj):
                return [value]
        return None
    
    policy = GurobiLinearPolicy(feature)
    planner = GurobiRDDLBilevelOptimizer(
        model, policy, state_bounds={'rlevel': (0, 100)},
        rollout_horizon=10)
    planner.solve(10)
    
            
if __name__ == "__main__":
    if len(sys.argv) < 4:
        dom, inst, trials = 'Reservoir continuous', 0, 1
    else:
        dom, inst, trials = sys.argv[1:4]
        trials = int(trials)
    slp_replan(dom, inst, trials)
    
