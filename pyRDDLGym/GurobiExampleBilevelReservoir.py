from pyRDDLGym.Core.Gurobi.GurobiRDDLPlan import GurobiRDDLStraightLinePlan
from pyRDDLGym.Core.Env.RDDLEnv import RDDLEnv
from pyRDDLGym.Examples.ExampleManager import ExampleManager
from pyRDDLGym.Core.Gurobi.GurobiRDDLBilevelOptimizer import GurobiRDDLBilevelOptimizer
from pyRDDLGym.Core.Simulator.RDDLSimulator import RDDLSimulator
from pyRDDLGym.Core.Grounder.RDDLGrounder import RDDLGrounder


def evaluate(world, policy, planner, n_steps, n_episodes):
    avg_reward = 0.
    for _ in range(n_episodes):
        world.reset()
        total_reward = 0.0
        for t in range(n_steps):
            actions = policy.evaluate(planner.compiler, planner.params, t, world.subs)
            _, reward, done = world.step(actions)
            total_reward += reward 
            if done: 
                break
        avg_reward += total_reward / n_episodes
    return avg_reward

            
def slp_replan(domain, inst):
    EnvInfo = ExampleManager.GetEnvInfo(domain)    
    model = RDDLEnv(domain=EnvInfo.get_domain(),
                    instance=EnvInfo.get_instance(inst)).model
    
    policy = GurobiRDDLStraightLinePlan(
        action_bounds={'release___t1': (0, 100),
                       'release___t2': (0, 200),
                       'release___t3': (0, 400)}
    )
    planner = GurobiRDDLBilevelOptimizer(
        model, policy,
        state_bounds={'rlevel___t1': (0, 100),
                      'rlevel___t2': (0, 200),
                      'rlevel___t3': (0, 400)},
        rollout_horizon=40,
        use_cc=True,
        model_params={'OutputFlag': 1, 'MIPGap': 0.0})
    
    rddl = RDDLGrounder(model._AST).Ground()
    world = RDDLSimulator(rddl)    
    reward_hist = []
    
    for callback in planner.solve(10, float('nan')): 
        avg_reward = evaluate(world, policy, planner, 40, 500)
        print(f'\naverage reward achieved: {avg_reward}\n')
        reward_hist.append(avg_reward)
    
    import matplotlib.pyplot as plt
    plt.plot(callback['error_hist'])
    plt.savefig('error.pdf')    
    plt.clf()
    plt.plot(reward_hist)
    plt.savefig('rewards.pdf')

            
if __name__ == "__main__":
    slp_replan('Reservoir linear', 0)
    
