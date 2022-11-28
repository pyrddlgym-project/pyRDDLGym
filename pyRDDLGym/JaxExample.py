import jax
import numpy as np
import optax  

from pyRDDLGym import ExampleManager
from pyRDDLGym import RDDLEnv
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxRDDLBackpropPlanner
from pyRDDLGym.Core.Jax.JaxRDDLSimulator import JaxRDDLSimulator
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

DO_PLAN = True
  

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
        action = plan(step)
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


def jax_simulate(ast, key, plan):
    sim = JaxRDDLSimulator(ast, key)
    
    total_reward = 0
    state, done = sim.reset() 
    for step in range(ast.instance.horizon): 
        action = plan(step)
        next_state, reward, done = sim.step(action)
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

   
def main():
    
    EnvInfo = ExampleManager.GetEnvInfo(ENV)
    domain = EnvInfo.get_domain()
    instance = EnvInfo.get_instance(0)
    
    rddltxt = RDDLReader(domain, instance).rddltxt
    rddlparser = parser.RDDLParser()
    rddlparser.build()
    ast = rddlparser.parse(rddltxt)
    
    key = jax.random.PRNGKey(np.random.randint(0, 2 ** 31))
    
    if DO_PLAN:
        
        planner = JaxRDDLBackpropPlanner(
            ast, key, 64, 10,
            optimizer=optax.rmsprop(0.1),
            initializer=jax.nn.initializers.normal(),
            action_bounds={'action': (0.0, 2.0)})
        
        for callback in planner.optimize(500):
            print('step={} loss={:.6f} test_loss={:.6f} best_loss={:.6f}'.format(
                str(callback['iteration']).rjust(4),
                callback['train_loss'],
                callback['test_loss'],
                callback['best_loss']))
        
        def plan(step):
            action, _ = planner.test_policy(
                None, step, callback['best_params'], key)
            action = jax.tree_map(np.asarray, action)
            action = {'action': action['action'].item()}
            print(action)
            return action
        
        rddl_simulate(plan)
        
    else:
        
        def plan(step):
            return {}

        jax_simulate(ast, key, plan)

    
if __name__ == "__main__":
    main()
