from sys import argv
import json
from time import perf_counter as timer
from pyRDDLGym import RDDLEnv
from pyRDDLGym import ExampleManager
from pyRDDLGym.Policies.Agents import RandomAgent

from tsd import TimeSpaceDiagram
from utils import warmup
from utils import offset_plan

def eval(env, num_warmup_steps, agent_id, sample_agent_action, render=False, filepath=None):
    # run evaluation
    tsd = TimeSpaceDiagram(
        env=env,
        num_warmup_steps=num_warmup_steps,
        agent_id=agent_id)
    total_reward = 0

    t0 = timer()
    t_step = 0
    for step in range(env.horizon):
        if render:
            env.render()
        action = sample_agent_action(step)
        state, reward, done, info = env.step(action)
        tsd.step(state)
        total_reward += reward
        if done:
            break

    t1 = timer()
    print(f"Episode total reward {total_reward}, time={t1-t0}s")
    env.close()

    tsd.plot(cmltrew=total_reward,
             filepath=filepath)

if __name__ == '__main__':
    try:
        agent_id = argv[1]
    except IndexError as e:
        print('argv: agent_id')
        raise e

    with open('actions/actions_dump_20230713_151433.json', 'r') as file:
        learned_plan = json.load(file)['a'][0]

    EnvInfo = ExampleManager.GetEnvInfo('traffic2phase')

    myEnv = RDDLEnv.RDDLEnv(
        domain=EnvInfo.get_domain(),
        instance='instances/instance01.rddl')

    # warmup
    _, num_warmup_steps = warmup(myEnv, EnvInfo)

    # set up renderer
    myEnv.set_visualizer(EnvInfo.get_visualizer())

    # set up agent
    random_agent = RandomAgent(action_space=myEnv.action_space,
                               num_actions=myEnv.numConcurrentActions)
    N = myEnv.numConcurrentActions
    def sample_random_agent_action(t):
        return random_agent.sample_action(t)
    def sample_min_agent_action(t):
        return {f'advance___i{n}': True for n in range(N)}
    def sample_max_agent_action(t):
        return {f'advance___i{n}': False for n in range(N)}
    def sample_offset_max_agent_action(t):
        #base_plan = offset_plan(N) # [False]*8*(N-1) + [True]*5 + [False]*60 + ([True]*32 + [False]*60)*5
        #t_offset = 8
        #base_plan = [False]*t_offset*(N-1) + [True]*5 + [False]*60 + ([True]*18 + [False]*60)*5
        #return {f'advance___i{n}': base_plan[t+t_offset*(N-n-1)] for n in range(N)}

        t_offsets = [0, 4, 10, 19, 29]
        base_plan = [False]*t_offsets[-1] + [True]*5 + [False]*60 + ([True]*18 + [False]*60)*5
        return {f'advance___i{n}': base_plan[t+t_offsets[n]] for n in range(N)}

    def sample_trained_planner_action(t):
        return {f'advance___i{n}': learned_plan[t][n] for n in range(N)}

    if agent_id == 'random':
        sample_agent_action = sample_random_agent_action
    elif agent_id == 'min':
        sample_agent_action = sample_min_agent_action
    elif agent_id == 'max':
        sample_agent_action = sample_max_agent_action
    elif agent_id == 'offset_max':
        sample_agent_action = sample_offset_max_agent_action
    elif agent_id == 'trained_planner':
        sample_agent_action = sample_trained_planner_action
    else:
        raise ValueError(f'GreenWave eval: Unrecognized agent id {agent_id}')

    eval(env=myEnv,
         num_warmup_steps=num_warmup_steps,
         agent_id=agent_id,
         sample_agent_action=sample_agent_action)
