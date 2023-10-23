import os

from pyRDDLGym import RDDLEnv
from pyRDDLGym.Core.Env.RDDLEnvSeeder import RDDLEnvSeeder
from pyRDDLGym.Core.Policies.Agents import NoOpAgent as RDDLAgent


def Test_instance(dom_path, inst_path, episodes=1):
    myEnv = RDDLEnv.RDDLEnv(domain=dom_path,
                            instance=inst_path,
                            enforce_action_constraints=False,
                            debug=False,
                            seeds=[42, 43])
    agent = RDDLAgent(action_space=myEnv.action_space,
                      num_actions=myEnv.numConcurrentActions)

    for episode in range(episodes):
        total_reward = 0
        state = myEnv.reset()
        # seeder = RDDLEnvSeeder(seed_list=[42, 43])
        # state = myEnv.reset(seed=seeder.Next())
        for step in range(myEnv.horizon):
            myEnv.render()
            action = agent.sample_action()
            next_state, reward, done, info = myEnv.step(action)
            total_reward += reward
            print()
            print(f'step       = {step}')
            print(f'state      = {state}')
            print(f'action     = {action}')
            print(f'next state = {next_state}')
            print(f'reward     = {reward}')
            state = next_state
            if done:
                break
        print(f'episode ended with reward {total_reward}')
    myEnv.close()


def main(path):
    print(path)

    dom_path = os.path.join(path, 'domain.rddl')
    inst_path = os.path.join(path, 'instance.rddl')
    print(dom_path)
    print(inst_path)

    Test_instance(dom_path, inst_path, 3)


if __name__ == "__main__":

    path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(path, 'SeedTest')

    main(path)
