import numpy as np
import sys

from pyRDDLGym.Core.Gurobi.GurobiRDDLCompiler import GurobiRDDLCompiler
from pyRDDLGym.Core.Gurobi.GurobiRDDLPlan import GurobiPiecewisePolicy
from pyRDDLGym.Core.Env.RDDLEnv import RDDLEnv
from pyRDDLGym.Examples.ExampleManager import ExampleManager
from pyRDDLGym.Core.Simulator.RDDLSimulator import RDDLSimulator


def slp_replan(domain, inst, trials):
    EnvInfo = ExampleManager.GetEnvInfo(domain)    
    model = RDDLEnv(domain=EnvInfo.get_domain(), 
                    instance=EnvInfo.get_instance(inst)).model
    
    plan = GurobiPiecewisePolicy(
        action_bounds={'release___t1': (0, 100),
                       'release___t2': (0, 200),
                       'release___t3': (0, 400)},
        state_bounds={'rlevel___t1': (0, 100),
                      'rlevel___t2': (0, 200),
                      'rlevel___t3': (0, 400)},
        dependencies_constr={'release___t1': ['rlevel___t1'],
                             'release___t2': ['rlevel___t2'],
                             'release___t3': ['rlevel___t3']},
        dependencies_values={}
    )
    
    planner = GurobiRDDLCompiler(model, plan, rollout_horizon=5,
                                 model_params={'NonConvex': 2, 'OutputFlag': 1, 'MIPGap': 0.0})
    
    world = RDDLSimulator(planner.rddl)
    rewards = np.zeros((planner.rddl.horizon, trials))
    for trial in range(trials):
        print('\n' + '*' * 30 + '\n' + f'starting trial {trial + 1}\n' + '*' * 30)
        total_reward = 0
        state, _ = world.reset()
        for step in range(planner.rddl.horizon):
            sol = planner.solve(world.subs)
            actions = sol[0]
            next_state, reward, done = world.step(actions)
            total_reward += reward 
            rewards[step, trial] = reward
            
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
    
    print('\nfinal policy:\n')
    print(plan.to_string(planner, planner.policy_params))
            
if __name__ == "__main__":
    if len(sys.argv) < 4:
        dom, inst, trials = 'Reservoir linear', 1, 1
    else:
        dom, inst, trials = sys.argv[1:4]
        trials = int(trials)
    slp_replan(dom, inst, trials)
    
