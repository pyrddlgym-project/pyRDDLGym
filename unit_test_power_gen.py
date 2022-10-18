from Env import RDDLEnv as RDDLEnv
from Policies.Agents import RandomAgent
import numpy as np
import random
import math

# NOTE: set TEMP_VARIANCE = 0 in RDDL to reproduce result
FOLDER = 'Competition/Power_gen/'


def power_gen_update(state, action):
    prod_p1 = max(min(action['curProd_p1'], 10), 0)
    prod_p2 = max(min(action['curProd_p2'], 10), 0)
    prod_p3 = max(min(action['curProd_p3'], 10), 0)
    next_prevProd_p1 = prod_p1
    next_prevProd_p2 = prod_p2
    next_prevProd_p3 = prod_p3
    next_temperature = state['temperature']
    
    demand = 2 + 0.01 * (state['temperature'] - 11.7) ** 2
    fulfilledDemand = min(demand, prod_p1 + prod_p2 + prod_p3)
    
    reward = -(prod_p1 * 5 + prod_p2 * 5 + prod_p3 * 5) + (fulfilledDemand * 8) \
             -(1000. if (demand > fulfilledDemand) else 0.) \
             +(abs(prod_p1 - state['prevProd_p1']) * 1 + abs(prod_p2 - state['prevProd_p2']) * 1 \
               + abs(prod_p3 - state['prevProd_p3']) * 1);
    
    next_derived = {'prod_p1' : prod_p1,
                    'prod_p2' : prod_p2,
                    'prod_p3' : prod_p3,
                    'demand' : demand}
    
    next_interm = {'fulfilledDemand' : fulfilledDemand}
    
    next_state = {'prevProd_p1' : next_prevProd_p1,
                  'prevProd_p2' : next_prevProd_p2,
                  'prevProd_p3' : next_prevProd_p3,
                  'temperature' : next_temperature}
    
    return reward, next_state, next_derived, next_interm

def main():
    myEnv = RDDLEnv.RDDLEnv(domain=FOLDER + 'domain.rddl', instance=FOLDER + 'instance0.rddl', is_grounded=False)
    agent = RandomAgent(action_space=myEnv.action_space, num_actions=myEnv.NumConcurrentActions)
    myEnv.model.nonfluents['TEMP_VARIANCE'] = 0.
    
    from pprint import pprint
    pprint(vars(myEnv.model))
    print(myEnv.model.reward)
    
    total_reward = 0
    state = myEnv.reset()
    test_state = state
    for step in range(myEnv.model.horizon):
        myEnv.render()
        action = agent.sample_action()

        next_state, reward, done, info = myEnv.step(action)
        total_reward += reward
        print("\nstep = {}".format(step))
        print('reward = {}'.format(reward))
        print('state = {}'.format(state))
        print('action = {}'.format(action))
        print('next_state = {}'.format(next_state))
        print('derived = {}'.format(myEnv.model.derived))
        print('interm = {}'.format(myEnv.model.interm))
        state = next_state
        
        test_reward, next_test_state, test_derived, test_interm = power_gen_update(
            test_state, action)
        print('reward = {}'.format(test_reward))
        print('next_state = {}'.format(next_test_state))
        print('derived = {}'.format(test_derived))
        print('interm = {}'.format(test_interm))
        assert test_reward == reward
        assert next_test_state == next_state
        assert test_derived == myEnv.model.derived
        assert test_interm == myEnv.model.interm
        test_state = next_test_state
        
    print("episode ended with reward {}".format(total_reward))


if __name__ == "__main__":
    main()
