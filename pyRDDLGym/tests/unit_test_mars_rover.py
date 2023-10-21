from pyRDDLGym.Core.Env import RDDLEnv as RDDLEnv
from pyRDDLGym.Core.Policies.Agents import RandomAgent
import math

FOLDER = 'pyRDDLGym/Examples/MarsRover/'


def mars_rover_update(state, action):
    power_d1 = math.sqrt(action['power-x_d1'] ** 2 + action['power-y_d1'] ** 2)
    power_d2 = math.sqrt(action['power-x_d2'] ** 2 + action['power-y_d2'] ** 2)
    if power_d1 >= 0.05:
        u_x_d1 = 0.05 * action['power-x_d1'] / power_d1
    else:
        u_x_d1 = action['power-x_d1']
    if power_d2 >= 0.05:
        u_x_d2 = 0.05 * action['power-x_d2'] / power_d2
    else:
        u_x_d2 = action['power-x_d2']
    if power_d1 >= 0.05:
        u_y_d1 = 0.05 * action['power-y_d1'] / power_d1
    else:
        u_y_d1 = action['power-y_d1']
    if power_d2 >= 0.05:
        u_y_d2 = 0.05 * action['power-y_d2'] / power_d2
    else:
        u_y_d2 = action['power-y_d2']

    next_interm = {'power_d1': power_d1,
                   'power_d2': power_d2,
                   'u-x_d1': u_x_d1,
                   'u-y_d1': u_y_d1,
                   'u-x_d2': u_x_d2,
                   'u-y_d2': u_y_d2}

    new_vel_x_d1 = state['vel-x_d1'] + 0.1 * u_x_d1
    new_vel_y_d1 = state['vel-y_d1'] + 0.1 * u_y_d1
    new_vel_x_d2 = state['vel-x_d2'] + 0.1 * u_x_d2
    new_vel_y_d2 = state['vel-y_d2'] + 0.1 * u_y_d2
    new_pos_x_d1 = state['pos-x_d1'] + 0.1 * state['vel-x_d1']
    new_pos_y_d1 = state['pos-y_d1'] + 0.1 * state['vel-y_d1']
    new_pos_x_d2 = state['pos-x_d2'] + 0.1 * state['vel-x_d2']
    new_pos_y_d2 = state['pos-y_d2'] + 0.1 * state['vel-y_d2']

    print(action)
    new_mineral_harvested_m1 = state['mineral-harvested_m1'] or (
        (math.sqrt((state['pos-x_d1'] - 5) ** 2 +
         (state['pos-y_d1'] - 5) ** 2) < 6 and action['harvest_d1'])
        or (math.sqrt((state['pos-x_d2'] - 5) ** 2 + (state['pos-y_d2'] - 5) ** 2) < 6 and action['harvest_d2'])
    )
    new_mineral_harvested_m2 = state['mineral-harvested_m2'] or (
        (math.sqrt((state['pos-x_d1'] - (-8)) ** 2 +
         (state['pos-y_d1'] - (-8)) ** 2) < 8 and action['harvest_d1'])
        or (math.sqrt((state['pos-x_d2'] - (-8)) ** 2 + (state['pos-y_d2'] - (-8)) ** 2) < 8 and action['harvest_d2'])
    )

    next_state = {'vel-x_d1': new_vel_x_d1,
                  'vel-y_d1': new_vel_y_d1,
                  'pos-x_d1': new_pos_x_d1,
                  'pos-y_d1': new_pos_y_d1,
                  'vel-x_d2': new_vel_x_d2,
                  'vel-y_d2': new_vel_y_d2,
                  'pos-x_d2': new_pos_x_d2,
                  'pos-y_d2': new_pos_y_d2,
                  'mineral-harvested_m1': new_mineral_harvested_m1,
                  'mineral-harvested_m2': new_mineral_harvested_m2}

    mineral_m1_cond = (
        8 if (
            (math.sqrt((state['pos-x_d1'] - 5) ** 2 +
             (state['pos-y_d1'] - 5) ** 2) < 6 and action['harvest_d1'])
            or (math.sqrt((state['pos-x_d2'] - 5) ** 2 + (state['pos-y_d2'] - 5) ** 2) < 6 and action['harvest_d2'])
        )
        else 0
    )

    mineral_m2_cond = (
        5 if (
            (math.sqrt((state['pos-x_d1'] - -8) ** 2 +
             (state['pos-y_d1'] - -8) ** 2) < 8 and action['harvest_d1'])
            or (math.sqrt((state['pos-x_d2'] - -8) ** 2 + (state['pos-y_d2'] - -8) ** 2) < 8 and action['harvest_d2'])
        )
        else 0
    )

    reward = -(u_x_d1 ** 2 + u_y_d1 ** 2) - (u_x_d2 ** 2 + u_y_d2 ** 2) + \
        mineral_m1_cond + mineral_m2_cond \
        - 1. * action['harvest_d1'] - 1. * action['harvest_d2']
    return reward, next_state, next_interm


def main():
    myEnv = RDDLEnv.RDDLEnv(domain=FOLDER + 'domain.rddl',
                            instance=FOLDER + 'instance0.rddl')
    agent = RandomAgent(action_space=myEnv.action_space,
                        num_actions=myEnv.numConcurrentActions)

    from pprint import pprint
    pprint(vars(myEnv.model))
    # print(myEnv.model.reward)

    print(myEnv.model.cpfs['mineral-harvested_m1\''])
    print(myEnv.model.cpfs['mineral-harvested_m2\''])
    total_reward = 0
    state = myEnv.reset()
    test_state = state
    print(state)
    for step in range(myEnv.model.horizon):
        myEnv.render()
        action = agent.sample_action()
        action['harvest_d1'] = bool(action['harvest_d1'])
        action['harvest_d2'] = bool(action['harvest_d2'])
        next_state, reward, done, info = myEnv.step(action)
        total_reward += reward
        print("\nstep = {}".format(step))
        print('reward = {}'.format(reward))
        print('state = {}'.format(state))
        print('action = {}'.format(action))
        print('next_state = {}'.format(next_state))
        print('interm = {}'.format(myEnv.model.interm))
        state = next_state

        test_reward, next_test_state, test_interm = mars_rover_update(
            test_state, action)
        assert test_reward == reward
        assert next_test_state == next_state
        assert test_interm == myEnv.model.interm, str(
            test_interm) + '\n' + str(myEnv.model.interm)
        test_state = next_test_state

    print("episode ended with reward {}".format(total_reward))


if __name__ == "__main__":
    main()
