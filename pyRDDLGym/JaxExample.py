import jax
import numpy as np
import optax  

from pyRDDLGym import ExampleManager
from pyRDDLGym import RDDLEnv
from pyRDDLGym.Core.Compiler.RDDLLiftedModel import RDDLLiftedModel
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxRDDLBackpropPlanner
from pyRDDLGym.Core.Jax.JaxRDDLSimulator import JaxRDDLSimulator
from pyRDDLGym.Core.Parser import parser as parser
from pyRDDLGym.Core.Parser.RDDLReader import RDDLReader
from pyRDDLGym.Policies.Agents import RandomAgent

# ENV = 'PowerGeneration'
# ENV = 'MarsRover'
ENV = 'UAV continuous'
# ENV = 'UAV discrete'
# ENV = 'UAV mixed'
ENV = 'WildlifePreserve'
# ENV = 'PowerGeneration'
# ENV = 'CartPole discrete'
# ENV = 'Elevators'
# ENV = 'Reservoir'
# ENV = 'RecSim'

DO_PLAN = False
  

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


def jax_simulate(model, key, plan):
    sim = JaxRDDLSimulator(model, key)
    
    total_reward = 0
    state, done = sim.reset() 
    for step in range(model.horizon): 
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
    
    model = RDDLLiftedModel(ast)
    key = jax.random.PRNGKey(np.random.randint(0, 2 ** 31))
    
    if DO_PLAN:
        
        planner = JaxRDDLBackpropPlanner(
            model, key, 32, 32,
            optimizer=optax.rmsprop(0.02),
            initializer=jax.nn.initializers.normal(),
            action_bounds={'action': (0.0, 2.0)})
        
        for callback in planner.optimize(4000, 10):
            print('step={} loss={:.6f} test_loss={:.6f}'.format(
                str(callback['iteration']).rjust(4),
                callback['train_loss'],
                callback['test_loss']))
        
        plan, _ = planner.get_plan(callback['params'], key)
        rddl_simulate(plan)
        
    else:
        
        myEnv = RDDLEnv.RDDLEnv(domain=domain,
                                instance=instance,
                                enforce_action_constraints=True)
        agent = RandomAgent(action_space=myEnv.action_space, 
                            num_actions=myEnv.numConcurrentActions)
        
        def plan(step):
            return {k: (bool(v) if v == 0 or v == 1 else v) 
                    for k, v in agent.sample_action().items()}

        jax_simulate(model, key, plan)

    
if __name__ == "__main__":
    main()
