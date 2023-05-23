import numpy as np
import sys

from pyRDDLGym.Core.Gurobi.GurobiRDDLCompiler import GurobiRDDLCompiler
from pyRDDLGym.Core.Env.RDDLEnv import RDDLEnv
from pyRDDLGym.Examples.ExampleManager import ExampleManager
from pyRDDLGym.Core.Simulator.RDDLSimulator import RDDLSimulator


def slp_replan(domain, inst, trials):
    EnvInfo = ExampleManager.GetEnvInfo(domain)    
    model = RDDLEnv(domain=EnvInfo.get_domain(), 
                    instance=EnvInfo.get_instance(inst)).model
    planner = GurobiRDDLCompiler(model, rollout_horizon=5, verbose=False)
    
    world = RDDLSimulator(planner.rddl)
    rewards = np.zeros((planner.rddl.horizon, trials))
    for trial in range(trials):
        print('\n' + '*' * 30 + '\n' + f'starting trial {trial + 1}\n' + '*' * 30)
        total_reward = 0
        state, _ = world.reset()
        for step in range(planner.rddl.horizon):
            sol, succ = planner.solve(world.subs)
            if not succ:
                print(f'failed to find a feasible solution, exiting...')
                return
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
        dom, inst, trials = 'CartPole continuous', 0, 1
    else:
        dom, inst, trials = sys.argv[1:4]
        trials = int(trials)
    slp_replan(dom, inst, trials)
    
