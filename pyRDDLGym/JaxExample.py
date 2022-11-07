from pyRDDLGym import RDDLEnv
from pyRDDLGym import ExampleManager
from pyRDDLGym.Core.Jax.JaxRDDLCompiler import JaxRDDLCompiler 
from pyRDDLGym.Core.Jax.JaxRDDLSimulator import JaxRDDLSimulator 
from pyRDDLGym.Policies.Agents import RandomAgent
from pyRDDLGym.Core.Parser import parser as parser

import jax
import jax.numpy as jnp
import numpy as np
from pyRDDLGym.Core.Parser.RDDLReader import RDDLReader
jax.config.update('jax_log_compiles', True)
jax.config.update('jax_enable_x64', True)

# ENV = 'PowerGeneration'
# ENV = 'MarsRover'
# ENV = 'UAV continuous'
# ENV = 'UAV discrete'
# ENV = 'UAV mixed'
ENV = 'Wildfire'
# ENV = 'MountainCar'
# ENV = 'CartPole continuous'
# ENV = 'CartPole discrete'
# ENV = 'Elevators'
#ENV = 'RaceCar'
#ENV = 'RecSim'


def main():
    # get the environment info
    EnvInfo = ExampleManager.GetEnvInfo(ENV)
    domain = RDDLReader(EnvInfo.get_domain(), EnvInfo.get_instance(0)).rddltxt
    MyRDDLParser = parser.RDDLParser(None, False)
    MyRDDLParser.build()
    
    # parse RDDL file
    rddl_ast = MyRDDLParser.parse(domain)
    
    # set up the environment class, choose instance 0 because every example has at least one example instance
    # myEnv = RDDLEnv.RDDLEnv(domain=EnvInfo.get_domain(), instance=EnvInfo.get_instance(0))
    # set up the environment visualizer
    # myEnv.set_visualizer(EnvInfo.get_visualizer())
                
    # set up an example aget
    # agent = RandomAgent(action_space=myEnv.action_space, num_actions=myEnv.NumConcurrentActions)
    
    from pprint import pprint
    #pprint(vars(rddl_ast))
    
    jaxc = JaxRDDLCompiler(rddl_ast)
    jaxc.compile()
    x = jaxc.init_values
    
    key = jax.random.PRNGKey(42)
    for k, v in x.items():
        print('{} -> {}'.format(k, str(v)))
    for _ in range(10):
        x['put-out'] = np.random.uniform(size=(3,3)) < 0.7
        x['cut-out'] = np.random.uniform(size=(3,3)) < 0.7
        
        for k, v in x.items():
            x[k] = np.array(v, dtype=float)
        # print(x['put-out'])
        
        #x['fx'] = np.random.uniform(low=-1., high=1.)
        #x['fy'] = np.random.uniform(low=-1., high=1.)
        x0 = x
        x, key = jaxc._sample_cpfs(x, key)
        reward = jaxc._sample_reward(x0, key)
        print(reward)
        print(jax.grad(jaxc.jit_reward, has_aux=True)(x, key))
        #print(x['out-of-fuel'])
        # print(x['burning'])
        #state = (x['x'], x['y'], x['violation'])
        #print(state)
        
    #for n, v in x.items():
    #    print('{}: {}'.format(n, str(v)))
    
    # set up jax compilation
    # print('begin tracing')
    
    # model = myEnv.model
    # jax_rddl = JaxRDDLCompiler(model).compile()
    # jax_sim = JaxRDDLSimulator(jax_rddl, jax.random.PRNGKey(42))
    #
    # total_reward = 0
    # state = jax_sim.reset()
    # for step in range(200): 
    #     action = agent.sample_action()
    #     next_state, reward, done = jax_sim.step(action)
    #     total_reward += reward
    #     print()
    #     print('step       = {}'.format(step))
    #     print('state      = {}'.format(state))
    #     print('action     = {}'.format(action))
    #     print('next state = {}'.format(next_state))
    #     print('reward     = {}'.format(reward))
    #     jax_sim.check_state_invariants()
    #     jax_sim.check_action_preconditions()
    #     state = next_state
    #     if done:
    #         break
    # print("episode ended with reward {}".format(total_reward))
    # myEnv.close()
    
    
if __name__ == "__main__":
    main()
