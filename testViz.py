from Env import RDDLEnv as RDDLEnv
from Policies.Agents import RandomAgent
import numpy as np
import random

import sys
from Parser import RDDLReader as RDDLReader
from Parser import parser as parser
import Grounder.RDDLGrounder as RDDLGrounder


from Visualizer.TextViz import TextVisualizer
from Visualizer.MarsRoverViz import MarsRoverVisualizer

PROBLEM = 'RDDL/Thiagos_Mars_Rover.rddl'
# PROBLEM = 'RDDL/Thiagos_HVAC.rddl'
# PROBLEM = 'RDDL/ThiagosReservoir.rddl'
# PROBLEM = 'RDDL/Thiagos_HVAC_grounded.rddl'

def main():

    MyReader = RDDLReader.RDDLReader(PROBLEM)
    # MyReader = RDDLReader.RDDLReader('RDDL/power_unit_commitment_domain.rddl',
    #                                  'RDDL/power_unit_commitment_instance.rddl')
    domain = MyReader.rddltxt

    MyLexer = parser.RDDLlex()
    MyLexer.build()
    MyLexer.input(domain)

    MyRDDLParser = parser.RDDLParser(None, False)
    MyRDDLParser.build()

    rddl_ast = MyRDDLParser.parse(domain)

    grounder = RDDLGrounder.RDDLGrounder(rddl_ast)
    model = grounder.Ground()

    print(model._states)
    print(model._nonfluents)

    # visualizer = MarsRoverVisualizer(model)
    visualizer = TextVisualizer(model,display=True)

    nonfluents = {'MAX_TIME': 12.0, 'MOVE_VARIANCE_MULT': 0.1, 'PICT_XPOS_p1': 1.0, 
        'PICT_YPOS_p1': -1.0, 'PICT_VALUE_p1': 5.0, 'PICT_ERROR_ALLOW_p1': 2, 
        'PICT_XPOS_p2': 10.0, 'PICT_YPOS_p2': 5.0, 'PICT_VALUE_p2': 10.0, 
        'PICT_ERROR_ALLOW_p2': 0.2, 'PICT_XPOS_p3': 2.0, 'PICT_YPOS_p3': -15.0, 
        'PICT_VALUE_p3': 7.0, 'PICT_ERROR_ALLOW_p3': 1.5}

    visualizer._nonfluents = nonfluents

    state0 = {'xPos': 0.0, 'yPos': 0.0, 'time': 0.0, 
        'picTaken_p1': False, 'picTaken_p2': False, 'picTaken_p3': False}
    state1 = {'xPos': 0.0, 'yPos': 0.0, 'time': 0.0, 
        'picTaken_p1': True, 'picTaken_p2': False, 'picTaken_p3': False}
    state2 = {'xPos': 11.0, 'yPos': 8.0, 'time': 0.0, 
        'picTaken_p1': True, 'picTaken_p2': False, 'picTaken_p3': False}
    state3 = {'xPos': 11.0, 'yPos': 8.0, 'time': 0.0, 
        'picTaken_p1': True, 'picTaken_p2': True, 'picTaken_p3': False}
    state4 = {'xPos': 3.0, 'yPos': -15.0, 'time': 0.0, 
        'picTaken_p1': True, 'picTaken_p2': True, 'picTaken_p3': False}
    state5 = {'xPos': 3.0, 'yPos': -15.0, 'time': 0.0, 
        'picTaken_p1': True, 'picTaken_p2': True, 'picTaken_p3': True}

    states_list = [state0, state1, state2, state3, state4, state5]
    states_buffer = []
    img_list = []

    for i in range(len(states_list) - 1):
    #     states_buffer += visualizer.gen_inter_state(states_list[i], states_list[i+1], 10)
        img_list.append(visualizer.render(states_list[i]))

    for i in range(len(img_list)):
        img = img_list[i]
        img.save('./img_folder/rover_'+str(i)+'.png')

    for i in range(len(img_list)):
        img = img_list[i]
        img.save('./img_folder/text_'+str(i)+'.png')        
        

    # visualizer.display_img(img_list[0])

    # visualizer.animate_buffer(states_buffer)

    # print('xxxxxxxx')
    
    # for i in states_buffer:
    #     visualizer.render(i)


    # visualizer.render(state5)
    # visualizer.gen_inter_state(state1, state2, 10)

    # myEnv = RDDLEnv.RDDLEnv(PROBLEM)
    # state = myEnv.reset()

    # img = myEnv.render(state)
    # img.save('text2.png')


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