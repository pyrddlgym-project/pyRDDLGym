import jax
import numpy as np

jax.config.update('jax_log_compiles', True)
jax.config.update('jax_enable_x64', True)

from pyRDDLGym import ExampleManager
from pyRDDLGym.Core.Jax.JaxRDDLSimulator import JaxRDDLSimulator
from pyRDDLGym.Core.Parser import parser as parser
from pyRDDLGym.Core.Parser.RDDLReader import RDDLReader

# ENV = 'PowerGeneration'
# ENV = 'MarsRover'
# ENV = 'UAV continuous'
# ENV = 'UAV discrete'
# ENV = 'UAV mixed'
#ENV = 'Wildfire'
# ENV = 'MountainCar'
ENV = 'CartPole continuous'
# ENV = 'CartPole discrete'
# ENV = 'Elevators'
# ENV = 'RaceCar'
# ENV = 'RecSim'


def main():
    # get the environment info
    EnvInfo = ExampleManager.GetEnvInfo(ENV)
    domain = RDDLReader(EnvInfo.get_domain(), EnvInfo.get_instance(0)).rddltxt
    MyRDDLParser = parser.RDDLParser(None, False)
    MyRDDLParser.build()
    
    # parse RDDL file
    ast = MyRDDLParser.parse(domain)        
    sim = JaxRDDLSimulator(ast, key=jax.random.PRNGKey(42))
    total_reward = 0
    state, done = sim.reset() 
    for step in range(100):
        action = {'force': np.random.uniform(0, 10)}
        next_state, reward, done = sim.step(action)
        print()
        print('data = {}'.format(sim.subs))
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
