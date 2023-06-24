import sys

from pyRDDLGym.Core.Gurobi.GurobiRDDLPlan import GurobiLinearPolicy
from pyRDDLGym.Core.Env.RDDLEnv import RDDLEnv
from pyRDDLGym.Examples.ExampleManager import ExampleManager
from pyRDDLGym.Core.Gurobi.GurobiRDDLBilevelOptimizer import GurobiRDDLBilevelOptimizer
from pyRDDLGym.Core.Simulator.RDDLSimulator import RDDLSimulator
from pyRDDLGym.Core.Grounder.RDDLGrounder import RDDLGrounder


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
        model, policy,
        state_bounds={'rlevel': (0, 100)},
        rollout_horizon=10,
        model_params={'NonConvex': 2, 'OutputFlag': 1, 'MIPGap': 0.1})
    planner.solve(10)
    
    rddl = RDDLGrounder(model._AST).Ground()
    world = RDDLSimulator(rddl)
    state, _ = world.reset()
    total_reward = 0.0
    for step in range(10):
        actions = policy.evaluate(planner.compiler, 
                                  planner.params, 
                                  step, world.subs)
        next_state, reward, done = world.step(actions)
        total_reward += reward 
        print()
        print(f'step       = {step}')
        print(f'state      = {state}')
        print(f'action     = {actions}')
        print(f'next state = {next_state}')
        print(f'reward     = {reward}')
        state = next_state
        if done: 
            break
    print(f'episode ended with reward {total_reward}')

            
if __name__ == "__main__":
    if len(sys.argv) < 4:
        dom, inst, trials = 'Reservoir continuous', 0, 1
    else:
        dom, inst, trials = sys.argv[1:4]
        trials = int(trials)
    slp_replan(dom, inst, trials)
    
