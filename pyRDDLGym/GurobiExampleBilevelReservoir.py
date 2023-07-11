from pyRDDLGym.Examples.ExampleManager import ExampleManager
from pyRDDLGym.Core.Env.RDDLEnv import RDDLEnv
from pyRDDLGym.Core.Grounder.RDDLGrounder import RDDLGrounder
from pyRDDLGym.Core.Gurobi.GurobiRDDLBilevelOptimizer import GurobiRDDLBilevelOptimizer
from pyRDDLGym.Core.Gurobi.GurobiRDDLPlan import GurobiFactoredPWSCPolicy
from pyRDDLGym.Core.Simulator.RDDLSimulator import RDDLSimulator


def evaluate(world, policy, planner, n_steps, n_episodes):
    avg_reward = 0.
    for _ in range(n_episodes):
        world.reset()
        total_reward = 0.0
        for t in range(n_steps):
            if policy is None:
                actions = {}
            else:
                actions = policy.evaluate(planner.compiler, planner.params, t, world.subs)
            _, reward, done = world.step(actions)
            total_reward += reward 
            if done: 
                break
        avg_reward += total_reward / n_episodes
    return avg_reward

            
def gurobi_solve(domain, inst, horizon):
    EnvInfo = ExampleManager.GetEnvInfo(domain)    
    model = RDDLEnv(domain=EnvInfo.get_domain(),
                    instance=EnvInfo.get_instance(inst)).model
    state_bounds = {'rlevel___t1': (0, 100),
                    'rlevel___t2': (0, 200),
                    'rlevel___t3': (0, 400),
                    'rlevel___t4': (0, 500)}
    action_bounds = {'release___t1': (0, 100),
                     'release___t2': (0, 200),
                     'release___t3': (0, 400),
                     'release___t4': (0, 500)}
    
    policy = GurobiFactoredPWSCPolicy(
        action_bounds=action_bounds,
        state_bounds=state_bounds,
    )
    planner = GurobiRDDLBilevelOptimizer(
        model, policy,
        state_bounds=state_bounds,
        rollout_horizon=horizon,
        use_cc=True,
        model_params={'PreSparsify': 1, 'Presolve': 2, 'OutputFlag': 1, 'MIPGap': 0.0})
    
    rddl = RDDLGrounder(model._AST).Ground()
    world = RDDLSimulator(rddl)    
    reward_hist = []
    
    avg_reward = evaluate(world, None, planner, horizon, 500)
    print(f'\naverage reward achieved: {avg_reward}\n')
    reward_hist.append(avg_reward)
        
    for callback in planner.solve(10, float('nan')): 
        avg_reward = evaluate(world, policy, planner, horizon, 500)
        print(f'\naverage reward achieved: {avg_reward}\n')
        reward_hist.append(avg_reward)
    
    import matplotlib.pyplot as plt
    plt.plot(callback['error_hist'])
    plt.savefig(f'{domain}_{inst}_{horizon}_error.pdf')    
    plt.clf()
    plt.plot(reward_hist)
    plt.savefig(f'{domain}_{inst}_{horizon}_rewards.pdf')

            
if __name__ == "__main__":
    gurobi_solve('Reservoir linear', 1, 10)
    
