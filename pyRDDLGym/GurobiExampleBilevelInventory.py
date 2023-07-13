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
            subs = world.subs
            if policy is None:
                actions = {}
            else:
                actions = policy.evaluate(planner.compiler, planner.params, t, subs)
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
                    
    MAX_ORDER = model.nonfluents['MAX-ITEMS']
    action_bounds = {'order___i1': (0, MAX_ORDER),
                     'order___i2': (0, MAX_ORDER),
                     'order___i3': (0, MAX_ORDER),
                     'order___i4': (0, MAX_ORDER),
                     'order___i5': (0, MAX_ORDER)}                   
    state_bounds = {'stock___i1': (0, MAX_ORDER),
                    'stock___i2': (0, MAX_ORDER),
                    'stock___i3': (0, MAX_ORDER),
                    'stock___i4': (0, MAX_ORDER),
                    'stock___i5': (0, MAX_ORDER)}                   
    state_init_bounds = {'stock___i1': (0, 0),
                         'stock___i2': (0, 0),
                         'stock___i3': (0, 0),
                         'stock___i4': (0, 0),
                         'stock___i5': (0, 0)}
    
    policy = GurobiFactoredPWSCPolicy(
        action_bounds=action_bounds,
        state_bounds=state_bounds,
        upper_bound=True
    )
    planner = GurobiRDDLBilevelOptimizer(
        model, policy,
        state_bounds=state_init_bounds,
        rollout_horizon=horizon,
        use_cc=True,
        model_params={'Presolve': 2, 'OutputFlag': 1, 'MIPGap': 0.01})
    
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
        print('\nfinal policy:\n')
        print(callback['policy_string'])
    
    import matplotlib.pyplot as plt
    plt.plot(callback['error_hist'])
    plt.savefig(f'{domain}_{inst}_{horizon}_error.pdf')    
    plt.clf()
    plt.plot(reward_hist)
    plt.savefig(f'{domain}_{inst}_{horizon}_rewards.pdf')

            
if __name__ == "__main__":
    gurobi_solve('Inventory continuous', 1, 10)
    
