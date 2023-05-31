import os

from pyRDDLGym import RDDLEnv
from pyRDDLGym import ExampleManager
from pyRDDLGym.Policies.Agents import RandomAgent

ENVS = ['HVAC', 'RaceCar', 'UAV_continuous', 'MarsRover', 'PowerGen_continuous', 'MountainCar', 'RecSim', 'Reservoir_continuous']

INST_NAMES = ['1e']     # example instance generation
HVAC_PARAMS = [
    {'num_heaters': 1, 'num_zones': 1, 'density': 1.0, 'temp-zone-range-init': (0., 15.),
    'temp-heater-range-init': (0., 10.), 'TEMP-ZONE-MIN': 22.0, 'TEMP-ZONE-MAX': 25.0,
    'p-switch-number': 0, 'p-switch-prob': 0.02, 'horizon': 100,'discount': 1.0}
]
MarsRover_PARAMS = [
    {'num_minerals': 1, 'num_rovers': 1, 'location_bounds': (-10., 10.),
    'area_bounds': (3., 6.), 'value_bounds': (0., 20.),
    'horizon': 100, 'discount': 1.0 }
]
MountainCar_PARAMS = [
    {'terrain_xleft': -1.2, 'terrain_widths': [0.7, 1.1], 'terrain_heights': [0.9, 0.9],
     'num_points': 100, 'pos': -0.6, 'vel': 0.01, 'goal-min': 0.5, 'horizon': 200, 'discount': 1.0}
]
PowerGen_continuous_PARAMS = [
    { 'num_gas': 1, 'num_nuclear': 0, 'num_solar': 0,
    'demand_scale': 1.0, 'temp_variance': 5.0, 'temp_range': (-30.0, 40.0),
    'horizon': 100, 'discount': 1.0 }
]
RaceCar_PARAMS = [
    {'num_blocks': 0, 'min_block_size': 0.0, 'max_block_size': 0.0, 'scale': 1.0,
    'goal_radius': 0.06, 'horizon': 200, 'discount': 1.0}
]
RecSim_PARAMS = [
    {'provider_dispersion': 10.0, 'provider_fan_out': 1, 'num_provider_clusters': 1, 'num_users': 100,
    'docs_per_cluster': 2, 'user_stddev': 20.0, 'horizon': 100, 'discount': 1.0 }
]
Reservoir_continuous_PARAMS = [
    {'num_reservoirs': 1, 'max_edges': 1,
    'top_range': (100., 200.),'target_range': (0.4, 0.7), 'rain_var': 5.,
    'horizon': 100, 'discount': 1. }
]
UAV_continuous_PARAMS = [
    {'num_aircraft': 1, 'num_control': 1, 'variance': 1.0,
    'xrange': (-50., 50.), 'yrange': (-50., 50.), 'zrange': (0., 100.),
    'horizon': 100, 'discount': 1.0 }
]

INST_PARAMS = {
    'HVAC' : HVAC_PARAMS,
    'RaceCar' : RaceCar_PARAMS,
    'UAV_continuous' : UAV_continuous_PARAMS,
    'MarsRover' : MarsRover_PARAMS,
    'PowerGen_continuous' : PowerGen_continuous_PARAMS,
    'MountainCar' : MountainCar_PARAMS,
    'RecSim' : RecSim_PARAMS,
    'Reservoir_continuous' : Reservoir_continuous_PARAMS
}


def Test_instance(EnvInfo, inst_path):
    myEnv = RDDLEnv.RDDLEnv(domain=EnvInfo.get_domain(),
                            instance=inst_path,
                            enforce_action_constraints=False,
                            debug=False)
    myEnv.set_visualizer(EnvInfo.get_visualizer())
    agent = RandomAgent(action_space=myEnv.action_space,
                        num_actions=myEnv.numConcurrentActions)

    total_reward = 0
    state = myEnv.reset()
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


def main(env, params, name):
    print(env, params, name)
    inst_path = None

    path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(path, 'Instances', env)


    EnvInfo = ExampleManager.GetEnvInfo(env)
    inst_path = EnvInfo.generate_instance(name, params, path)

    if not inst_path:
        inst_path = os.path.join(path, 'instance'+name+'.rddl')
    Test_instance(EnvInfo, inst_path)






if __name__ == "__main__":
    inst_index = 4
    env = ENVS[7]
    inst_name = INST_NAMES[inst_index]
    inst_params = INST_PARAMS[env][inst_index]

    main(env, inst_params, inst_name)
