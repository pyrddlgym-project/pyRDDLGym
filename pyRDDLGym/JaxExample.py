from pyRDDLGym.Planner import JaxConfigManager

# ENV = 'PowerGeneration'
# ENV = 'MarsRover'
ENV = 'UAV continuous'
# ENV = 'UAV discrete'
# ENV = 'UAV mixed'
# ENV = 'RecSim'
# ENV = 'PowerGeneration'
# ENV = 'CartPole discrete'
# ENV = 'Elevators'
# ENV = 'Reservoir'
# ENV = 'RecSim'
  

def slp_no_replan():
    myEnv, planner, train_args = JaxConfigManager.get(f'Planner/{ENV}.cfg')
    
    for callback in planner.optimize(**train_args):
        print('step={} train_return={:.6f} test_return={:.6f}'.format(
              str(callback['iteration']).rjust(4),
              callback['train_return'],
              callback['test_return']))
    plan = planner.get_plan(callback['params'])
    
    total_reward = 0
    state = myEnv.reset()
    for step in range(myEnv.horizon):
        myEnv.render()
        action = plan[step]
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

    
def slp_replan():
    myEnv, planner, train_args = JaxConfigManager.get(f'Planner/{ENV} replan.cfg')
    
    total_reward = 0
    state = myEnv.reset()
    for step in range(myEnv.horizon):
        myEnv.render()
        * _, callback = planner.optimize(**train_args, init_subs=myEnv.sampler.subs)
        action = planner.get_plan(callback['params'])[0]
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

    
def main(): 
    slp_no_replan()
    
        
if __name__ == "__main__":
    main()
