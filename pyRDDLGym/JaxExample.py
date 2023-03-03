import jax
import sys

from pyRDDLGym.Planner import JaxConfigManager
from pyRDDLGym.Core.Compiler.RDDLDecompiler import RDDLDecompiler


def print_parameterized_exprs(planner):
    model_params = planner.compiled.model_params
    print(f'model_params = {model_params}')
    ids = [int(key.split('_')[-1]) for key in model_params]
    for _id in ids:
        expr = planner.compiled.traced.lookup(_id)
        print(f'\nid = {_id}:\n' + RDDLDecompiler().decompile_expr(expr))
    
    
def slp_train(planner, **train_args):
    # print_parameterized_exprs(planner)
    print('\n' + 'training plan:')
    for callback in planner.optimize(**train_args):
        print('step={} train_return={:.6f} test_return={:.6f}'.format(
              str(callback['iteration']).rjust(4),
              callback['train_return'],
              callback['test_return']))
    params = callback['params']
    return params


def slp_no_replan(env):
    myEnv, planner, train_args = JaxConfigManager.get(f'{env}.cfg')
    params = slp_train(planner, **train_args)
    
    key = train_args['key']    
    total_reward = 0
    state = myEnv.reset()
    for step in range(myEnv.horizon):
        subs = myEnv.sampler.subs
        key, subkey = jax.random.split(key)
        action = planner.get_action(subkey, params, step, subs)
        next_state, reward, done, _ = myEnv.step(action)
        total_reward += reward 
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

    
def slp_replan(env):
    myEnv, planner, train_args = JaxConfigManager.get(f'{env}.cfg')
    
    key = train_args['key']
    total_reward = 0
    state = myEnv.reset()
    for step in range(myEnv.horizon):
        subs = myEnv.sampler.subs
        params = slp_train(planner, subs=subs, **train_args)
        key, subkey = jax.random.split(key)
        action = planner.get_action(subkey, params, 0, subs)
        next_state, reward, done, _ = myEnv.step(action)
        total_reward += reward 
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

    
def main(env, replan):
    if replan:
        slp_replan(env)
    else: 
        slp_no_replan(env)
    
        
if __name__ == "__main__":
    if len(sys.argv) < 2:
        args = [sys.argv[0]] + ['HVAC']
    else:
        args = sys.argv
    env = args[1]
    replan = env.endswith('replan')
    main(env, replan)
    
