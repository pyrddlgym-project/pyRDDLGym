from Env import RDDLEnv as RDDLEnv
from Policies.Agents import RandomAgent
import numpy as np
import random

PROBLEM = 'RDDL/Thiagos_HVAC_grounded.rddl'
PROBLEM = 'RDDL/Thiagos_Mars_Rover.rddl'
PROBLEM = 'RDDL/Thiagos_HVAC.rddl'

def main():
    steps = 30
    myEnv = RDDLEnv.RDDLEnv(PROBLEM, is_grounded=False)
    agent = RandomAgent(action_space=myEnv.action_space, num_actions=myEnv.NumConcurrentActions)

    total_reward = 0
    state = myEnv.reset()
    for step in range(steps):
        action = agent.sample_action()

        next_state, reward, done, info = myEnv.step(action)
        total_reward += reward
        print("step {}: reward: {}".format(step, reward))
        print("state_i:", state, "-> state_f:", next_state)
    print("episode ended with reward {}".format(total_reward))







if __name__ == "__main__":
    main()