from Env import RDDLEnv as RDDLEnv
import numpy as np
import random

PROBLEM = 'RDDL/Thiagos_HVAC_grounded.rddl'

def sample_actions(action_space, num_actions):
    s = action_space.sample()
    action = {}
    selected_actions = random.sample(list(s), num_actions)
    for sample in selected_actions:
            action[sample] = s[sample][0].item()
    return action

def main():
    steps = 30
    myEnv = RDDLEnv.RDDLEnv(PROBLEM)
    action_space = myEnv.action_space
    max_actions = myEnv.NumConcurrentActions

    total_reward = 0
    state = myEnv.reset()
    for step in range(steps):
        action = sample_actions(action_space, max_actions)

        next_state, reward, done, info = myEnv.step(action)
        total_reward += reward
        print("step {}: reward: {}".format(step, reward))
        print("state_i:", state, "-> state_f:", next_state)
    print("episode ended with reward {}".format(total_reward))







if __name__ == "__main__":
    main()