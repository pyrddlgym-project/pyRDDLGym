import os

from pyRDDLGym import RDDLEnv
from pyRDDLGym import ExampleManager
from pyRDDLGym.Core.Policies.Agents import RandomAgent

ENVS = ['HVAC', 'RaceCar', 'UAV_continuous', 'MarsRover', 'PowerGen_continuous', 'MountainCar', 'RecSim', 'Reservoir_continuous']

###################
# Competition run instances
###################
INST_NAMES = ['0c', '3c', '5c']
HVAC_PARAMS = [
    # difficulty is controlled by the number of zones + heaters (e.g. scale)
    # and the switching of occupancy affects more zones (this should make it
    # difficult for determinization methods)
    {'num_heaters': 3, 'num_zones': 2, 'density': 0.2,
     'temp-zone-range-init': (0., 15.), 'temp-heater-range-init': (0., 10.),
     'TEMP-ZONE-MIN': 22.0, 'TEMP-ZONE-MAX': 25.0,
     'p-switch-number': 0, 'p-switch-prob': 0.02,
     'horizon': 100, 'discount': 1.0},
    {'num_heaters': 5, 'num_zones': 5, 'density': 0.2,
     'temp-zone-range-init': (0., 15.), 'temp-heater-range-init': (0., 10.),
     'TEMP-ZONE-MIN': 22.0, 'TEMP-ZONE-MAX': 25.0,
     'p-switch-number': 2, 'p-switch-prob': 0.02,
     'horizon': 100, 'discount': 1.0},
    {'num_heaters': 15, 'num_zones': 20, 'density': 0.2,
     'temp-zone-range-init': (0., 15.), 'temp-heater-range-init': (0., 10.),
     'TEMP-ZONE-MIN': 22.0, 'TEMP-ZONE-MAX': 25.0,
     'p-switch-number': 15, 'p-switch-prob': 0.02,
     'horizon': 100, 'discount': 1.0}

]
MarsRover_PARAMS = [
    # the difficulty here is modulated by increasing number of minerals and rovers.
    # some problems have too many rovers, and some less rovers than minerals.
    # the agent is forced to delay some small immediate reward for taking
    # some additional time steps and fuel to accept a much larger reward in future.
    {'num_minerals': 1, 'num_rovers': 2, 'location_bounds': (-10., 10.),
     'area_bounds': (1., 5.), 'value_bounds': (0., 20.),
     'horizon': 100, 'discount': 1.0},
    {'num_minerals': 7, 'num_rovers': 6, 'location_bounds': (-10., 10.),
     'area_bounds': (1., 3.), 'value_bounds': (0., 20.),
     'horizon': 100, 'discount': 1.0},
    {'num_minerals': 20, 'num_rovers': 5, 'location_bounds': (-10., 10.),
     'area_bounds': (1., 2.), 'value_bounds': (0., 20.),
     'horizon': 100, 'discount': 1.0}
]
MountainCar_PARAMS = [
    {'terrain_xleft': -1.2, 'terrain_widths': [0.7, 1.1], 'terrain_heights': [0.9, 0.9],
     'num_points': 100, 'pos': -0.6, 'vel': 0.01, 'goal-min': 0.5, 'horizon': 200, 'discount': 1.0},
    {'terrain_xleft': -1.2, 'terrain_widths': [0.7, 2.3, 2.1, 1.1], 'terrain_heights': [0.9, 0.8, 0.7, 1.0],
     'num_points': 100, 'pos': -0.6, 'vel': 0.01, 'goal-min': 4.9, 'horizon': 200, 'discount': 1.0},
    {'terrain_xleft': -1.2, 'terrain_widths': [0.7, 2.3, 2.2, 2.1, 1.1], 'terrain_heights': [0.5, 0.8, 0.9, 1.0, 1.0],
     'num_points': 100, 'pos': -0.6, 'vel': 0.01, 'goal-min': 7.1, 'horizon': 200, 'discount': 1.0}
]
PowerGen_continuous_PARAMS = [
    # difficulty is controlled by number and diversity of plant types, many of
    # which progressively require unit commitment
    {'num_gas': 2, 'num_nuclear': 0, 'num_solar': 0,
     'demand_scale': 1.0, 'temp_variance': 5.0, 'temp_range': (-30.0, 40.0),
     'horizon': 100, 'discount': 1.0},
    {'num_gas': 2, 'num_nuclear': 2, 'num_solar': 1,
     'demand_scale': 3.0, 'temp_variance': 7.0, 'temp_range': (-30.0, 40.0),
     'horizon': 100, 'discount': 1.0},
    {'num_gas': 3, 'num_nuclear': 4, 'num_solar': 4,
     'demand_scale': 6.0, 'temp_variance': 9.0, 'temp_range': (-30.0, 40.0),
     'horizon': 100, 'discount': 1.0}
]
RaceCar_PARAMS = [
    # difficulty is controlled by the number of obstacles, as well as the radius
    # of the goal region, which is shrinking
    {'num_blocks': 1, 'min_block_size': 0.05, 'max_block_size': 0.5, 'scale': 1.0,
     'goal_radius': 0.06, 'horizon': 150, 'discount': 1.0},
    {'num_blocks': 3, 'min_block_size': 0.05, 'max_block_size': 0.4, 'scale': 1.5,
     'goal_radius': 0.04, 'horizon': 150, 'discount': 1.0},
    {'num_blocks': 5, 'min_block_size': 0.05, 'max_block_size': 0.3, 'scale': 2.0,
     'goal_radius': 0.02, 'horizon': 150, 'discount': 1.0}
]
RecSim_PARAMS = [
    # Difficulty is controlled by the number users and number of providers (providers = num_provider_clusters*provider_fan_out) and their dispersity
    # and providers.
    {'provider_dispersion': 10.0, 'provider_fan_out': 2, 'num_provider_clusters': 3, 'num_users': 10,
     'docs_per_cluster': 5, 'user_stddev': 20.0, 'horizon': 100, 'discount': 1.0 },
    {'provider_dispersion': 62.0, 'provider_fan_out': 5, 'num_provider_clusters': 20, 'num_users': 90,
     'docs_per_cluster': 5, 'user_stddev': 20.0, 'horizon': 100, 'discount': 1.0 },
    {'provider_dispersion': 62.0, 'provider_fan_out': 10, 'num_provider_clusters': 100, 'num_users': 800,
     'docs_per_cluster': 5,  'user_stddev': 20.0, 'horizon': 100, 'discount': 1.0 }
]
Reservoir_continuous_PARAMS = [
    # the number of reservoirs, amount of interdependence between them,
    # the rain variance and the narrowing of the target water level ranges all
    # contribute to difficulty
    {'num_reservoirs': 2, 'max_edges': 1,
     'top_range': (100., 200.), 'target_range': (0.4, 0.7), 'rain_var': 5.,
     'horizon': 100, 'discount': 1.0},
    {'num_reservoirs': 10, 'max_edges': 5,
     'top_range': (100., 400.), 'target_range': (0.2, 0.5), 'rain_var': 20.,
     'horizon': 100, 'discount': 1.0},
    {'num_reservoirs': 30, 'max_edges': 10,
     'top_range': (100., 600.), 'target_range': (0.1, 0.2), 'rain_var': 50.,
     'horizon': 100, 'discount': 1.0}
]
UAV_continuous_PARAMS = [
    # the difficulty of this problem is the noise of uncontrollable aircraft,
    # so the difficulty is controlled by number of such craft (relative to controlled),
    # the total number of craft (scale) and the variance of the uncontrolled state
    {'num_aircraft': 1, 'num_control': 1, 'variance': 1.0,
     'xrange': (-50., 50.), 'yrange': (-50., 50.), 'zrange': (0., 100.),
     'horizon': 100, 'discount': 1.0},
    {'num_aircraft': 3, 'num_control': 2, 'variance': 4.0,
     'xrange': (-50., 50.), 'yrange': (-50., 50.), 'zrange': (0., 100.),
     'horizon': 100, 'discount': 1.0},
    {'num_aircraft': 40, 'num_control': 10, 'variance': 10.0,
     'xrange': (-50., 50.), 'yrange': (-50., 50.), 'zrange': (0., 100.),
     'horizon': 100, 'discount': 1.0}
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
    inst_index = 0
    env = ENVS[5]
    inst_name = INST_NAMES[inst_index]
    inst_params = INST_PARAMS[env][inst_index]

    main(env, inst_params, inst_name)
