from math import ceil

def warmup(env, EnvInfo, render=False):
    N = env.numConcurrentActions
    min = env.non_fluents['PHASE-MIN___i0']

    if render:
        env.set_visualizer(EnvInfo.get_visualizer())

    state = env.reset()
    num_warmup_steps = 0
    while True:
        if render: env.render()
        action = {f'advance___i{n}': True for n in range(N)}
        state, reward, done, info = env.step(action)
        num_warmup_steps += 1
        if state['signal___i0'] == 3 and state['signal-t___i0'] >= min:
            break
        if done:
            raise RuntimeError('[warmup.py] Environment terminated during warmup')

    return env.sampler.subs, num_warmup_steps

def offset_plan(N):
    return [False]*8*(N-1) + [True]*5 + [False]*60 + ([True]*32 + [False]*60)*5


if __name__ == '__main__':
    import pickle

    from pyRDDLGym import ExampleManager
    from pyRDDLGym import RDDLEnv

    # specify the model
    EnvInfo = ExampleManager.GetEnvInfo('traffic2phase')
    myEnv = RDDLEnv.RDDLEnv(domain=EnvInfo.get_domain(), instance='instances/instance01.rddl')

    subs, _ = warmup(myEnv, EnvInfo, render=True)

    with open('subs.pckl', 'wb') as file:
        pickle.dump(subs, file)
