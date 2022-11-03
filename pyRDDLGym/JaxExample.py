from pyRDDLGym import RDDLEnv
from pyRDDLGym import ExampleManager
from pyRDDLGym.Core.Jax.JaxCompiler import JaxCompiler
import jax
from jax import make_jaxpr

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
    
    model = myEnv.model
    from pprint import pprint
    pprint(vars(model))
    
    compiler = JaxCompiler(model)
    cpfs, reward, xi = compiler.compile()
    
    x = {'pos' : 3.,
         'vel' : 2.,
         'TIME-STEP' : 0.1,
         'jax:xi-1' : 1}
    print(cpfs['pos\''](x))


if __name__ == "__main__":
    main()