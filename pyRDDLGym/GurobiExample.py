import numpy as np
import sys

from pyRDDLGym.Core.Gurobi.GurobiRDDLCompiler import GurobiRDDLCompiler
from pyRDDLGym.Core.Gurobi.GurobiRDDLPlan import GurobiFactoredPWSCPolicy
from pyRDDLGym.Core.Env.RDDLEnv import RDDLEnv
from pyRDDLGym.Examples.ExampleManager import ExampleManager
from pyRDDLGym.Core.Simulator.RDDLSimulator import RDDLSimulator


def slp_replan(domain, inst, trials):
    EnvInfo = ExampleManager.GetEnvInfo(domain)    
    model = RDDLEnv(domain=EnvInfo.get_domain(), 
                    instance=EnvInfo.get_instance(inst)).model
    MAX_ORDER = model.nonfluents['MAX-ORDER']
     
    plan = GurobiFactoredPWSCPolicy(
        action_bounds={'order___i1': (0, MAX_ORDER),
                       'order___i2': (0, MAX_ORDER),
                       'order___i3': (0, MAX_ORDER),
                       'order___i4': (0, MAX_ORDER),
                       'order___i5': (0, MAX_ORDER)},
        state_bounds={'stock___i1': (-50, 50),
                      'stock___i2': (-50, 50),
                      'stock___i3': (-50, 50),
                      'stock___i4': (-50, 50),
                      'stock___i5': (-50, 50)}
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

            
if __name__ == "__main__":
    if len(sys.argv) < 4:
        dom, inst, trials = 'Inventory randomized', 1, 1
    else:
        dom, inst, trials = sys.argv[1:4]
        trials = int(trials)
    slp_replan(dom, inst, trials)
    
