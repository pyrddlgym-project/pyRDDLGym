from Env import RDDLEnv as RDDLEnv
from Policies.Agents import RandomAgent
import numpy as np
import random

import sys
from Parser import RDDLReader as RDDLReader
from Parser import parser as parser
import Grounder.RDDLGrounder as RDDLGrounder


from Visualizer.TextViz import TextVisualizer

PROBLEM = 'RDDL/Thiagos_Mars_Rover.rddl'
# PROBLEM = 'RDDL/Thiagos_HVAC.rddl'
# PROBLEM = 'RDDL/ThiagosReservoir.rddl'
# PROBLEM = 'RDDL/Thiagos_HVAC_grounded.rddl'

def main():

    # MyReader = RDDLReader.RDDLReader(PROBLEM)
    # # MyReader = RDDLReader.RDDLReader('RDDL/power_unit_commitment_domain.rddl',
    # #                                  'RDDL/power_unit_commitment_instance.rddl')
    # domain = MyReader.rddltxt

    # MyLexer = parser.RDDLlex()
    # MyLexer.build()
    # MyLexer.input(domain)

    # MyRDDLParser = parser.RDDLParser(None, False)
    # MyRDDLParser.build()

    # rddl_ast = MyRDDLParser.parse(domain)

    # grounder = RDDLGrounder.RDDLGrounder(rddl_ast)
    # model = grounder.Ground()

    myEnv = RDDLEnv.RDDLEnv(PROBLEM)
    state = myEnv.reset()

    img = myEnv.render(state)
    img.save('text2.png')


    # print('here')
    # print(myEnv.model._states)

    # print(model._states)
    # print(state)
    # sys.exit()

    # visualizer = TextVisualizer(model)

    # img = visualizer.render(model._states)
    # img.save('text.png')




    # steps = 30
    # myEnv = RDDLEnv.RDDLEnv(PROBLEM)
    # agent = RandomAgent(action_space=myEnv.action_space, num_actions=myEnv.NumConcurrentActions)

    # total_reward = 0
    # state = myEnv.reset()

    # print(state)
    # sys.exit()
    


    # for step in range(steps):
    #     action = agent.sample_action_rover()
    #     next_state, reward, done, info = myEnv.step(action)
    #     total_reward += reward
    #     print("step {}: reward: {}".format(step, reward))
    #     print("state_i:", state, "-> state_f:", next_state)
    # print("episode ended with reward {}".format(total_reward))







if __name__ == "__main__":
    main()