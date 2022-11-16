import jax
jax.config.update('jax_log_compiles', True)
import numpy as np

from pyRDDLGym import ExampleManager

from pyRDDLGym.Core.Env.RDDLEnv import RDDLEnv
from pyRDDLGym.Core.Jax.JaxRDDLSimulator import JaxRDDLSimulator
from pyRDDLGym.Core.Jax.JaxRDDLStraightlinePlanner import JaxRDDLStraightlinePlanner
from pyRDDLGym.Core.Parser import parser as parser
from pyRDDLGym.Core.Parser.RDDLReader import RDDLReader


# ENV = 'PowerGeneration'
# ENV = 'MarsRover'
# ENV = 'UAV continuous'
# ENV = 'UAV discrete'
# ENV = 'UAV mixed'
# ENV = 'MountainCar'
ENV = 'Reservoir'
# ENV = 'CartPole discrete'
# ENV = 'Elevators'
# ENV = 'RaceCar'
# ENV = 'RecSim'

DO_PLAN = False


def main():
    
    # get the environment info
    EnvInfo = ExampleManager.GetEnvInfo(ENV)
    domain = RDDLReader(EnvInfo.get_domain(), EnvInfo.get_instance(0)).rddltxt
    
    MyRDDLParser = parser.RDDLParser(None, False)
    MyRDDLParser.build()
    ast = MyRDDLParser.parse(domain)   
    
    # parse RDDL file 
    key = jax.random.PRNGKey(42)
    if DO_PLAN:
        
        planner = JaxRDDLStraightlinePlanner(ast, key, 20, 2048)
        print('step loss')
        for step, _, loss in planner.optimize(100):
            print('{} {}'.format(str(step).rjust(4), loss))
    
    else:
        
        sim = JaxRDDLSimulator(ast, key=key)
        
        total_reward = 0
        state, done = sim.reset() 
        for step in range(10):
            action = {'flow': np.array([.1, .2, .3])}
            next_state, reward, done = sim.step(action)
            
            print()
            print('step       = {}'.format(step))
            print('state      = {}'.format(state))
            print('action     = {}'.format(action))
            print('next state = {}'.format(next_state))
            print('reward     = {}'.format(reward))
            state = next_state
            total_reward += reward
            if done:
                break
        print("episode ended with reward {}".format(total_reward))

    
if __name__ == "__main__":
    main()
