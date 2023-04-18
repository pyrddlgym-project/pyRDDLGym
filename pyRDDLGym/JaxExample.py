import jax
import numpy as np
import sys
import time

from pyRDDLGym.Planner import JaxConfigManager
from pyRDDLGym.Core.Compiler.RDDLDecompiler import RDDLDecompiler
# from pyRDDLGym.Core.Jax.JaxRDDLModelError import JaxRDDLModelError


def print_parameterized_exprs(planner):
    model_params = planner.compiled.model_params
    print(f'model_params = {model_params}')
    ids = planner.compiled.get_ids_of_parameterized_expressions()
    for _id in ids:
        expr = planner.compiled.traced.lookup(_id)
        print(f'\nid = {_id}:\n' + RDDLDecompiler().decompile_expr(expr))
    
    
def slp_train(planner, budget, **train_args):
    step = train_args['step']
    del train_args['step']
    print('\n' + 'training plan:')
    for i, callback in enumerate(planner.optimize(**train_args, step=1)):
        if i == 0:
            elapsed = 0
            starttime = time.time()
        else:
            elapsed = time.time() - starttime
        
        if i % step == 0:
            print('[{:.4f} s] step={} train_return={:.6f} test_return={:.6f} best_return={:.6f}'.format(
                elapsed,
                str(callback['iteration']).rjust(4),
                callback['train_return'],
                callback['test_return'],
                callback['best_return']))
        if elapsed >= budget:
            print('ran out of time!')
            break
    params = callback['best_params']
    
    # key = jax.random.PRNGKey(42)
    # error = JaxRDDLModelError(planner.rddl, planner.test_policy, 
    #                           batch_size=64, logic=planner.logic)
    # error.summarize(key, params)
    # error.sensitivity(key, params)
    return params


def slp_no_replan(env, trials, timeout, timeout_ps, save):
    myEnv, planner, train_args, (dom, inst) = JaxConfigManager.get(f'{env}.cfg')
    key = train_args['key']    
    
    rewards = np.zeros((myEnv.horizon, trials))
    for trial in range(trials):
        print('\n' + '*' * 30 + '\n' + f'starting trial {trial + 1}\n' + '*' * 30)
        train_args['key'] = key
        params = slp_train(planner, timeout, **train_args)
        
        total_reward = 0
        state = myEnv.reset()
        for step in range(myEnv.horizon):
            subs = myEnv.sampler.subs
            key, subkey = jax.random.split(key)
            action = planner.get_action(subkey, params, step, subs)
            next_state, reward, done, _ = myEnv.step(action)
            total_reward += reward 
            rewards[step, trial] = reward
            
            print()
            print('step       = {}'.format(step))
            print('state      = {}'.format(state))
            print('action     = {}'.format(action))
            print('next state = {}'.format(next_state))
            print('reward     = {}'.format(reward))
            state = next_state
            if done:
                break
        print(f'episode ended with reward {total_reward}')
        
    myEnv.close()
    if save:
        np.savetxt(f'{dom}_{inst}_slp.csv', rewards, delimiter=',')

    
def slp_replan(env, trials, timeout, timeout_ps, save):
    myEnv, planner, train_args, (dom, inst) = JaxConfigManager.get(f'{env}.cfg')
    key = train_args['key']
    
    rewards = np.zeros((myEnv.horizon, trials))
    for trial in range(trials):
        print('\n' + '*' * 30 + '\n' + f'starting trial {trial + 1}\n' + '*' * 30)
        total_reward = 0
        state = myEnv.reset() 
        starttime = time.time()
        train_args['guess'] = None
        for step in range(myEnv.horizon):
            currtime = time.time()
            elapsed = currtime - starttime
            
            if elapsed < timeout:
                subs = myEnv.sampler.subs
                params = slp_train(planner,
                                   budget=min(timeout - elapsed, timeout_ps),
                                   subs=subs,
                                   **train_args)
                key, subkey = jax.random.split(key)
                action = planner.get_action(subkey, params, 0, subs)
                train_args['guess'] = planner.plan.guess_next_epoch(params)
            else:
                print('ran out of time!')
                action = {}
            
            next_state, reward, done, _ = myEnv.step(action)
            total_reward += reward 
            rewards[step, trial] = reward
            
            print()
            print(f'elapsed    = {elapsed} s')
            print(f'step       = {step}')
            print(f'state      = {state}')
            print(f'action     = {action}')
            print(f'next state = {next_state}')
            print(f'reward     = {reward}')
            state = next_state
            if done: 
                break
        print(f'episode ended with reward {total_reward}')
        
    myEnv.close()
    if save:
        np.savetxt(f'{dom}_{inst}_mpc.csv', rewards, delimiter=',')

    
def main(env, replan, trials, timeout, timeout_ps, save):
    if replan:
        slp_replan(env, trials, timeout, timeout_ps, save)
    else: 
        slp_no_replan(env, trials, timeout, timeout_ps, save)
    
        
if __name__ == "__main__":
    if len(sys.argv) < 6:
        env, trials, timeout, timeout_ps, save = 'Wildfire', 1, 60 * 2, 1, False
    else:
        env, trials, timeout, timeout_ps, save = sys.argv[1:6]
        trials = int(trials)
        timeout = int(timeout)
        timeout_ps = int(timeout_ps)
        save = save == 'True' or save == True
    replan = env.endswith('replan')
    main(env, replan, trials, timeout, timeout_ps, save) 
    
