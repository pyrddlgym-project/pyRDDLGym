from Env import RDDLEnv as RDDLEnv
import numpy as np

PROBLEM = 'RDDL/Thiagos_HVAC_grounded.rddl'

def sample_actions(action_space):
    s = action_space.sample()
    action = {}
    for sample in s:
        if (np.random.choice(2, 1)[0]):
            action[sample] = s[sample][0]
    return action

def main():
    steps = 30
    myEnv = RDDLEnv.RDDLEnv(PROBLEM)
    action_space = myEnv.action_space
    # max_actions = myEnv.NumConcurrentActions

    total_reward = 0
    state = myEnv.reset()
    for step in range(steps):
        action = sample_actions(action_space)

        next_state, reward, done, info = myEnv.step(action)
        total_reward += reward
        print("step {}: reward: {}".format(step, reward))
        print("state_i:", state, "-> state_f:", next_state)
    print("episode ended with reward {}".format(total_reward))







if __name__ == "__main__":
    main()