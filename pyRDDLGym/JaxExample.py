from pyRDDLGym import RDDLEnv
from pyRDDLGym import ExampleManager
from pyRDDLGym.Core.Jax.JaxRDDLCompiler import JaxRDDLCompiler 
from pyRDDLGym.Core.Jax.JaxRDDLSimulator import JaxRDDLSimulator 
from pyRDDLGym.Policies.Agents import RandomAgent

import jax

# ENV = 'PowerGeneration'
# ENV = 'MarsRover'
# ENV = 'UAV continuous'
# ENV = 'UAV discrete'
# ENV = 'UAV mixed'
# ENV = 'Wildfire'
# ENV = 'MountainCar'
ENV = 'CartPole continuous'
# ENV = 'CartPole discrete'
# ENV = 'Elevators'
# ENV = 'RaceCar'


def main():
    # get the environment info
    EnvInfo = ExampleManager.GetEnvInfo(ENV)
    # set up the environment class, choose instance 0 because every example has at least one example instance
    myEnv = RDDLEnv.RDDLEnv(domain=EnvInfo.get_domain(), instance=EnvInfo.get_instance(0))
    # set up the environment visualizer
    myEnv.set_visualizer(EnvInfo.get_visualizer())
    
    # set up an example aget
    agent = RandomAgent(action_space=myEnv.action_space, num_actions=myEnv.NumConcurrentActions)

    # set up jax compilation
    model = myEnv.model
    jax_rddl = JaxRDDLCompiler(model).compile()
    jax_sim = JaxRDDLSimulator(jax_rddl, jax.random.PRNGKey(42))
    
    total_reward = 0
    state = jax_sim.reset()
    for step in range(myEnv.horizon):
        action = agent.sample_action()
        next_state, reward, done = jax_sim.step(action)
        total_reward += reward
        print()
        print('step       = {}'.format(step))
        print('state      = {}'.format(state))
        print('action     = {}'.format(action))
        print('next state = {}'.format(next_state))
        print('reward     = {}'.format(reward))
        jax_sim.check_state_invariants()
        jax_sim.check_action_preconditions()
        state = next_state
        if done:
            break
    print("episode ended with reward {}".format(total_reward))
    myEnv.close()
    
    
if __name__ == "__main__":
    main()
