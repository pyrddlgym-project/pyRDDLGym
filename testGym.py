from Env import RDDLEnv as RDDLEnv
from Policies.Agents import RandomAgent


# PROBLEM = 'RDDL/Thiagos_HVAC_grounded.rddl'
# PROBLEM = 'RDDL/Thiagos_Mars_Rover.rddl'
# PROBLEM = 'RDDL/Thiagos_HVAC.rddl'
FOLDER = 'Competition/Power_gen/'
# FOLDER = 'Competition/Mars_rover/'
# FOLDER = 'Competition/drone_con/'
# FOLDER = 'Competition/MountainCar/'
# FOLDER = 'Competition/Cartpole/'
# FOLDER = 'Competition/Recsim/'

def main():
    myEnv = RDDLEnv.RDDLEnv(domain=FOLDER+'domain.rddl', instance=FOLDER+'instance0.rddl', is_grounded=False)
    agent = RandomAgent(action_space=myEnv.action_space, num_actions=myEnv.NumConcurrentActions)

    total_reward = 0
    state = myEnv.reset()
    for step in range(myEnv.horizon):
        myEnv.render()
        action = agent.sample_action()
        next_state, reward, done, info = myEnv.step(action)
        total_reward += reward
        print("step {}: reward: {}".format(step, reward))
        print("state_i:", state, "-> state_f:", next_state)
        state = next_state
    print("episode ended with reward {}".format(total_reward))





if __name__ == "__main__":
    main()