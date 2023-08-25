import os
import sys

from pyRDDLGym.Core.Env.RDDLEnv import RDDLEnv
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import load_config, JaxRDDLBackpropPlanner
from pyRDDLGym.Examples.ExampleManager import ExampleManager

    
def run_straightline(config_path, env):
    
    # optimize policy
    planner_args, _, train_args = load_config(config_path)
    planner = JaxRDDLBackpropPlanner(rddl=env.model, **planner_args)
    params = planner.optimize(**train_args)
    
    # evaluate
    total_reward = 0
    state = env.reset()
    for step in range(env.horizon):
        action = planner.get_action(0, params, step, env.sampler.subs)        
        next_state, reward, done, _ = env.step(action)
        total_reward += reward             
        print(f'step       = {step}\n'
              f'state      = {state}\n'
              f'action     = {action}\n'
              f'next state = {next_state}\n'
              f'reward     = {reward}\n')
        state = next_state
        if done:
            break
    return total_reward
        

def run_replan(config_path, env):
    
    # initialize policy
    planner_args, _, train_args = load_config(config_path)
    planner = JaxRDDLBackpropPlanner(rddl=env.model, **planner_args)
    
    # optimize and evaluate
    guess = None
    total_reward = 0
    state = env.reset() 
    for step in range(env.horizon):
        subs = env.sampler.subs
        params = planner.optimize(subs=subs, guess=guess, **train_args)
        guess = planner.plan.guess_next_epoch(params)         
        action = planner.get_action(0, params, 0, subs)           
        next_state, reward, done, _ = env.step(action)
        total_reward += reward 
            
        print(f'step       = {step}\n'
              f'state      = {state}\n'
              f'action     = {action}\n'
              f'next state = {next_state}\n'
              f'reward     = {reward}\n')
        state = next_state
        if done:
            break
    return total_reward

        
if __name__ == "__main__":
    dom, inst, method = 'Wildfire', 0, 'slp'
    if len(sys.argv) >= 4:
        dom, inst, method = sys.argv[1:4]
    
    EnvInfo = ExampleManager.GetEnvInfo(dom)    
    env = RDDLEnv(domain=EnvInfo.get_domain(),
                  instance=EnvInfo.get_instance(inst),
                  enforce_action_constraints=True)
    
    abs_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(abs_path, 'JaxPlanConfigs', f'{dom}_{method}.cfg') 
    
    if method == 'replan':
        reward = run_replan(config_path, env)
    else:
        reward = run_straightline(config_path, env)
    print(f'episode ended with reward {reward}')
    
