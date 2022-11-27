import jax
import numpy as np
import optax  

from pyRDDLGym import ExampleManager
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxRDDLBackpropPlanner
from pyRDDLGym.Core.Jax.JaxRDDLSimulator import JaxRDDLSimulator
from pyRDDLGym.Core.Parser import parser as parser
from pyRDDLGym.Core.Parser.RDDLReader import RDDLReader

# ENV = 'PowerGeneration'
# ENV = 'MarsRover'
ENV = 'UAV continuous'
# ENV = 'UAV discrete'
# ENV = 'UAV mixed'
ENV = 'Wildfire'
# ENV = 'PowerGeneration'
# ENV = 'CartPole discrete'
# ENV = 'Elevators'
# ENV = 'Reservoir'
# ENV = 'RecSim'

DO_PLAN = True
  
    
def jax_simulate(ast, key, plan={}):
    sim = JaxRDDLSimulator(ast, key)
    
    for _ in range(1):
        total_reward = 0
        state, done = sim.reset() 
        for step in range(ast.instance.horizon):        
            action = {name: value[step] for name, value in plan.items()}
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


def main():
    
    EnvInfo = ExampleManager.GetEnvInfo(ENV)
    domain = EnvInfo.get_domain()
    instance = EnvInfo.get_instance(0)
    # viz = EnvInfo.get_visualizer()
    
    rddltxt = RDDLReader(domain, instance).rddltxt
    rddlparser = parser.RDDLParser()
    rddlparser.build()
    ast = rddlparser.parse(rddltxt)
    
    key = jax.random.PRNGKey(np.random.randint(0, 2 ** 31))
    
    if DO_PLAN:
        
        planner = JaxRDDLBackpropPlanner(
            ast, key, 64, 100,
            optimizer=optax.rmsprop(0.3),
            initializer=jax.nn.initializers.normal(),
            action_bounds={'flow' : (0, 1000)})
        
        for callback in planner.optimize(500):
            print('step={} loss={:.6f} test_loss={:.6f} best_loss={:.6f}'.format(
                str(callback['step']).rjust(4),
                callback['train_loss'],
                callback['test_loss'],
                callback['best_loss']))
            
        plan = callback['best_plan']
        plan = {name: np.asarray(value) for name, value in plan.items()}
        
    else:
        
        plan = {}
        
    jax_simulate(ast, key, plan)

    
if __name__ == "__main__":
    main()
