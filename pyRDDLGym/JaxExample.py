import jax
jax.config.update('jax_log_compiles', True)
import numpy as np
import optax  

from pyRDDLGym import ExampleManager
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxRDDLBackpropPlanner
from pyRDDLGym.Core.Jax.JaxRDDLCompiler import JaxRDDLCompiler
from pyRDDLGym.Core.Jax.JaxRDDLSimulator import JaxRDDLSimulator
from pyRDDLGym.Core.Parser import parser as parser
from pyRDDLGym.Core.Parser.RDDLReader import RDDLReader


#ENV = 'PowerGeneration'
#ENV = 'MarsRover'
ENV = 'UAV continuous'
# ENV = 'UAV discrete'
# ENV = 'UAV mixed'
# ENV = 'MountainCar'
#ENV = 'PowerGeneration'
# ENV = 'CartPole discrete'
#ENV = 'Elevators'
ENV = 'Reservoir'
# ENV = 'RecSim'

DO_PLAN = True


def main():
    
    # get the environment info
    EnvInfo = ExampleManager.GetEnvInfo(ENV)
    domain = RDDLReader(EnvInfo.get_domain(), EnvInfo.get_instance(0)).rddltxt
    
    MyRDDLParser = parser.RDDLParser(None, False)
    MyRDDLParser.build()
    ast = MyRDDLParser.parse(domain)   
    
    # parse RDDL file 
    key = jax.random.PRNGKey(np.random.randint(0, 2 ** 31))
    if DO_PLAN:
        
        planner = JaxRDDLBackpropPlanner(ast, key, 20, 64, 
                                         optimizer=optax.adam(0.5),
                                         initializer=jax.nn.initializers.normal())
        for callback in planner.optimize(500):
            print('step={} loss={:.4f} best_loss={:.4f} err={}'.format(
                str(callback['step']).rjust(4), callback['loss'], 
                callback['best_loss'], callback['errors']))
        print(callback['best_plan'])
        
    else:
        
        compiler = JaxRDDLCompiler(ast)
        sim = JaxRDDLSimulator(compiler, key)
        
        total_reward = 0
        state, done = sim.reset() 
        for step in range(100):
            action = {}
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
