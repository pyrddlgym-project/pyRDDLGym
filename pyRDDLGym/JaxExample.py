import jax
from pyRDDLGym.Core.Env.RDDLEnv import RDDLEnv

jax.config.update('jax_log_compiles', True)

from pyRDDLGym import ExampleManager
from pyRDDLGym.Core.Jax.JaxRDDLSimulator import JaxRDDLSimulator
from pyRDDLGym.Core.Jax.JaxRDDLStraightlinePlanner import JaxRDDLStraightlinePlanner

from pyRDDLGym.Core.Parser import parser as parser
from pyRDDLGym.Core.Parser.RDDLReader import RDDLReader

# ENV = 'PowerGeneration'
# ENV = 'MarsRover'
#ENV = 'UAV continuous'
# ENV = 'UAV discrete'
# ENV = 'UAV mixed'

#ENV = 'MountainCar'
ENV = 'Reservoir'
# ENV = 'CartPole discrete'
#ENV = 'Elevators'
#ENV = 'RaceCar'
# ENV = 'RecSim'


def main():
    # get the environment info
    EnvInfo = ExampleManager.GetEnvInfo(ENV)
    domain = RDDLReader(EnvInfo.get_domain(), EnvInfo.get_instance(0)).rddltxt
    
    MyRDDLParser = parser.RDDLParser(None, False)
    MyRDDLParser.build()
    
    # parse RDDL file
    ast = MyRDDLParser.parse(domain)    
    #sim = JaxRDDLSimulator(ast, key=jax.random.PRNGKey(42))
    #print(jax.make_jaxpr(sim.cpfs['out-of-fuel\''])(sim.subs, sim.key))
    
    #
    # myEnv = RDDLEnv(domain=EnvInfo.get_domain(), instance=EnvInfo.get_instance(0))
    # agent = RandomAgent(action_space=myEnv.action_space, num_actions=myEnv.NumConcurrentActions)
    #
    # total_reward = 0
    # state, done = sim.reset() 
    # for step in range(100):
    #     action = agent.sample_action()
    #     #action = {'put-out': np.random.uniform(size=(3,3)) < .1,
    #     #          'cut-out': np.random.uniform(size=(3,3)) < .1}
    #     next_state, reward, done = sim.step(action)
    #     print()
    #     print('step       = {}'.format(step))
    #     print('state      = {}'.format(state))
    #     print('action     = {}'.format(action))
    #     print('next state = {}'.format(next_state))
    #     print('reward     = {}'.format(reward))
    #     state = next_state
    #     total_reward += reward
    #     if done:
    #         break
    # print("episode ended with reward {}".format(total_reward))
    
    planner = JaxRDDLStraightlinePlanner(ast, jax.random.PRNGKey(42), 10, 2048)
    print('step loss')
    for step, plan, loss in planner.optimize(100):
        print('{} {}'.format(str(step).rjust(4), loss))

if __name__ == "__main__":
    main()
