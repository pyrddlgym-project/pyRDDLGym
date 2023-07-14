from pyRDDLGym.Examples.ExampleManager import ExampleManager
from pyRDDLGym.Core.Env.RDDLEnv import RDDLEnv
from pyRDDLGym.Core.Grounder.RDDLGrounder import RDDLGrounder
from pyRDDLGym.Core.Gurobi.GurobiRDDLBilevelOptimizer import GurobiRDDLBilevelOptimizer
from pyRDDLGym.Core.Gurobi.GurobiRDDLPlan import GurobiPWSCPolicy
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


def pretty(d, indent=0) -> str:
    res = ''
    if isinstance(d, (list, tuple)):
        for (i, v) in enumerate(d):
            res += '\n' + '\t' * indent + str(i) + ':' + pretty(v, indent + 1)
    elif isinstance(d, dict):
        for (k, v) in d.items():
            res += '\n' + '\t' * indent + str(k) + ':' + pretty(v, indent + 1)
    else:
        res += '\n' + '\t' * indent + str(d)
    return res

            
def gurobi_solve(domain, inst, horizon, num_cases):
    EnvInfo = ExampleManager.GetEnvInfo(domain)    
    model = RDDLEnv(domain=EnvInfo.get_domain(),
                    instance=EnvInfo.get_instance(inst)).model
                    
    state_bounds = {'rlevel___t1': (0, 100),
                    'rlevel___t2': (0, 200),
                    'rlevel___t3': (0, 400),
                    'rlevel___t4': (0, 500)}
    state_init_bounds = state_bounds
    action_bounds = {'release___t1': (0, 100),
                     'release___t2': (0, 200),
                     'release___t3': (0, 400),
                     'release___t4': (0, 500)}
    
    policy = GurobiPWSCPolicy(
        action_bounds=action_bounds,
        state_bounds=state_bounds,
        upper_bound=True,
        num_cases=num_cases
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
    
    log = ''
    for callback in planner.solve(10, float('nan')): 
        avg_reward = evaluate(world, policy, planner, horizon, 500)
        print(f'\naverage reward achieved: {avg_reward}\n')
        reward_hist.append(avg_reward)    
        print('\nfinal policy:\n' + callback['policy_string'])
        log += pretty(callback) + '\n\n'
    
    log += 'reward history:\n'
    for reward in reward_hist:
        log += str(reward) + '\n'    
    
    with open(f'{domain}_{inst}_{horizon}_{num_cases}.log', 'w') as file:
        file.write(log)

            
if __name__ == "__main__":
    gurobi_solve('Reservoir linear', 1, 10, 1)
    
