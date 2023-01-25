import jax
import numpy as np
import optax  

from pyRDDLGym import ExampleManager
from pyRDDLGym import RDDLEnv
from pyRDDLGym.Core.Compiler.RDDLLiftedModel import RDDLLiftedModel
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxRDDLBackpropPlanner
from pyRDDLGym.Core.Parser import parser as parser
from pyRDDLGym.Core.Parser.RDDLReader import RDDLReader
from pyRDDLGym.Core.Jax.JaxRDDLSimulator import JaxRDDLSimulator

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
    myEnv = RDDLEnv.RDDLEnv(domain=EnvInfo.get_domain(),
                            instance=EnvInfo.get_instance(0),
                            enforce_action_constraints=True)
    model = myEnv.model
    
    planner = JaxRDDLBackpropPlanner(
        model, 
        key=jax.random.PRNGKey(42), 
        batch_size_train=32, 
        batch_size_test=32,
        rollout_horizon=5,
        optimizer=optax.rmsprop(0.01),
        action_bounds={'action': (0.0, 2.0)})
    
    total_reward = 0
    state = myEnv.reset()
    for step in range(myEnv.horizon):
        myEnv.render()
        *_, callback = planner.optimize(500, 10, init_subs=myEnv.sampler.subs)
        action = planner.get_plan(callback['params'])[0]
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
    # rddl_simulate(plan)
        
if __name__ == "__main__":
    main()
