from Env import RDDLEnv as RDDLEnv
from Policies.Agents import RandomAgent
import numpy as np
import random
import math

FOLDER = 'Competition/Mars_rover/'


def mars_rover_update(state, action):
    power_d1 = math.sqrt(action['power_x_d1'] ** 2 + action['power_y_d1'] ** 2)
    if power_d1 >= 0.05:
        u_x_d1 = 0.05 * action['power_x_d1'] / power_d1
    else:
        u_x_d1 = action['power_x_d1']
    if power_d1 >= 0.05:
        u_y_d1 = 0.05 * action['power_y_d1'] / power_d1
    else:
        u_y_d1 = action['power_y_d1']
        
    next_derived = {'power_d1': power_d1,
                    'u_x_d1': u_x_d1,
                    'u_y_d1': u_y_d1}
    
    new_vel_x_d1 = state['vel_x_d1'] + 0.1 * u_x_d1
    new_vel_y_d1 = state['vel_y_d1'] + 0.1 * u_y_d1
    new_pos_x_d1 = state['pos_x_d1'] + 0.1 * state['vel_x_d1']
    new_pos_y_d1 = state['pos_y_d1'] + 0.1 * state['vel_y_d1']
    
    is_within1 = math.sqrt((state['pos_x_d1'] - 5) ** 2 + (state['pos_y_d1'] - 5) ** 2) < 1
    is_within2 = math.sqrt((state['pos_x_d1'] - -5) ** 2 + (state['pos_y_d1'] - -5) ** 2) < 1
    new_mineral_harvested_m1 = state['mineral_harvested_m1'] or (
        not state['mineral_harvested_m1'] and is_within1 and action['harvest_d1'])
    new_mineral_harvested_m2 = state['mineral_harvested_m2'] or (
        not state['mineral_harvested_m2'] and is_within2 and action['harvest_d1'])

    next_state = {'vel_x_d1': new_vel_x_d1,
                  'vel_y_d1': new_vel_y_d1,
                  'pos_x_d1': new_pos_x_d1,
                  'pos_y_d1': new_pos_y_d1,
                  'mineral_harvested_m1': new_mineral_harvested_m1,
                  'mineral_harvested_m2': new_mineral_harvested_m2}
    
    reward = -(u_x_d1 **2 + u_y_d1 ** 2) + (1 if is_within1 else 0) + (1 if is_within2 else 0) - 1. * action['harvest_d1']
    return reward, next_state, next_derived


def main():
    myEnv = RDDLEnv.RDDLEnv(domain=FOLDER + 'domain.rddl', instance=FOLDER + 'insta0.rddl', is_grounded=False)
    agent = RandomAgent(action_space=myEnv.action_space, num_actions=myEnv.NumConcurrentActions)
    
    from pprint import pprint
    pprint(vars(myEnv.model))
    
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
        state = next_state
        
        test_reward, next_test_state, test_derived = mars_rover_update(
            test_state, action)
        assert test_reward == reward
        assert next_test_state == next_state
        assert test_derived == myEnv.model.derived
        test_state = next_test_state
        
    print("episode ended with reward {}".format(total_reward))


if __name__ == "__main__":
    main()
