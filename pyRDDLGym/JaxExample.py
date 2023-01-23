import jax
import numpy as np
import optax  

from pyRDDLGym import ExampleManager
from pyRDDLGym import RDDLEnv
from pyRDDLGym.Core.Compiler.RDDLLiftedModel import RDDLLiftedModel
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxRDDLBackpropPlanner
from pyRDDLGym.Core.Parser import parser as parser
from pyRDDLGym.Core.Parser.RDDLReader import RDDLReader

# ENV = 'PowerGeneration'
# ENV = 'MarsRover'
ENV = 'UAV continuous'
# ENV = 'UAV discrete'
# ENV = 'UAV mixed'
ENV = 'MountainCar'
# ENV = 'PowerGeneration'
# ENV = 'CartPole discrete'
# ENV = 'Elevators'
# ENV = 'Reservoir'
# ENV = 'RecSim'
  

def rddl_simulate(plan):
    EnvInfo = ExampleManager.GetEnvInfo(ENV)
    myEnv = RDDLEnv.RDDLEnv(domain=EnvInfo.get_domain(),
                            instance=EnvInfo.get_instance(0),
                            enforce_action_constraints=True)
    myEnv.set_visualizer(EnvInfo.get_visualizer())
    
    total_reward = 0
    state = myEnv.reset()
    for step in range(myEnv.horizon):
        myEnv.render()
        action = plan[step]
        next_state, reward, done, _ = myEnv.step(action)
        total_reward += reward        
        print()
        print('step       = {}'.format(step))
        print('state      = {}'.format(state))
        print('action     = {}'.format(action))
        print('next state = {}'.format(next_state))
        print('reward     = {}'.format(reward))
        state = next_state
        if done:
            break
        
    print(f'episode ended with reward {total_reward}')
    myEnv.close()


def main():
    
    EnvInfo = ExampleManager.GetEnvInfo(ENV)
    domain = EnvInfo.get_domain()
    instance = EnvInfo.get_instance(0)
    
    rddltxt = RDDLReader(domain, instance).rddltxt
    rddlparser = parser.RDDLParser()
    rddlparser.build()
    ast = rddlparser.parse(rddltxt)
    
    model = RDDLLiftedModel(ast)
    key = jax.random.PRNGKey(np.random.randint(0, 2 ** 31))
    
    planner = JaxRDDLBackpropPlanner(
        model, key, 32, 32,
        optimizer=optax.rmsprop(0.01),
        initializer=jax.nn.initializers.normal(),
        action_bounds={'action': (0.0, 2.0)})
        
    for callback in planner.optimize(4000, 10):
        print('step={} train_return={:.6f} test_return={:.6f}'.format(
            str(callback['iteration']).rjust(4),
            callback['train_return'],
            callback['test_return']))
        
    plan, _ = planner.get_plan(callback['params'], key)
    rddl_simulate(plan)
        
if __name__ == "__main__":
    main()
