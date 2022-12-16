# from pyRDDLGym import ExampleManager
#
# from pyRDDLGym.Core.Parser import parser as parser
# from pyRDDLGym.Core.Parser.RDDLReader import RDDLReader
# from pyRDDLGym.Core.Simulator.LiftedRDDLSimulator import LiftedRDDLSimulatorWConstraints
#
# from pyRDDLGym.Core.Simulator.LiftedRDDLModel import LiftedRDDLModel
#
# #ENV = 'PowerGeneration'
# # ENV = 'MarsRover'
# #ENV = 'UAV continuous'
# # ENV = 'UAV discrete'
# #ENV = 'UAV mixed'
# # ENV = 'Wildfire'
# # ENV = 'MountainCar'
# # ENV = 'CartPole continuous'
# # ENV = 'CartPole discrete'
# # ENV = 'Elevators'
# ENV = 'Wildfire'
# #ENV = 'RecSim'
#
# import numpy as np
# np.seterr(all='raise')
#
# def main():
#     EnvInfo = ExampleManager.GetEnvInfo(ENV)
#     domain = EnvInfo.get_domain()
#     instance = EnvInfo.get_instance(0)
#
#     rddltxt = RDDLReader(domain, instance).rddltxt
#     rddlparser = parser.RDDLParser()
#     rddlparser.build()
#     rddl = rddlparser.parse(rddltxt)
#
#     model = LiftedRDDLModel(rddl)
#
#     from pprint import pprint
#     pprint(vars(model))
#     #
#
#     sim = LiftedRDDLSimulatorWConstraints(model, debug=True)
#     for episode in range(1):
#         total_reward = 0
#         state, done = sim.reset()
#         for step in range(100):
#             action = {'put-out_x1_y1': True}
#             next_state, reward, done = sim.step(action)
#             total_reward += reward
#
#             print()
#             print('step       = {}'.format(step))
#             print('state      = {}'.format(state))
#             print('action     = {}'.format(action))
#             print('next state = {}'.format(next_state))
#             print('reward     = {}'.format(reward))
#
#             state = next_state
#             if done:
#                 break
#         print('episode {} ended with reward {}'.format(episode, total_reward))
#
#
# if __name__ == "__main__":
#     main()
