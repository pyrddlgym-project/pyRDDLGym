from pyRDDLGym.Core.Env import RDDLEnv as RDDLEnv
from pyRDDLGym.Core.Policies.Agents import RandomAgent

FOLDER = 'pyRDDLGym/Examples/MountainCar/'


def main():
    myEnv = RDDLEnv.RDDLEnv(domain=FOLDER + 'domain.rddl',
                            instance=FOLDER + 'instance0.rddl')
    agent = RandomAgent(action_space=myEnv.action_space,
                        num_actions=myEnv.numConcurrentActions)

    from pprint import pprint
    pprint(vars(myEnv.model))

    total_reward = 0
    state = myEnv.reset()
    for step in range(myEnv.model.horizon):
        myEnv.render()
        action = agent.sample_action()

        next_state, reward, done, info = myEnv.step(action)
        total_reward += reward
        # print("step {}: reward: {}".format(step, reward))
        # print("state_i:", state, "-> state_f:", next_state)
        print("\nstep = {}".format(step))
        print('reward = {}'.format(reward))
        print('state = {}'.format(state))
        print('action = {}'.format(action))
        print('next_state = {}'.format(next_state))
        print('derived = {}'.format(myEnv.model.derived))
        print('interm = {}'.format(myEnv.model.interm))
        state = next_state
    print("episode ended with reward {}".format(total_reward))


if __name__ == "__main__":
    main()
