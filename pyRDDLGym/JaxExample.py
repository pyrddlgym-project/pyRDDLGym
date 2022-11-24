import jax
jax.config.update('jax_log_compiles', True)
import numpy as np
import optax  

from pyRDDLGym import ExampleManager
from pyRDDLGym import RDDLEnv
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxRDDLBackpropPlanner
from pyRDDLGym.Core.Jax.JaxRDDLCompiler import JaxRDDLCompiler
from pyRDDLGym.Core.Jax.JaxRDDLSimulator import JaxRDDLSimulator
from pyRDDLGym.Core.Parser import parser as parser
from pyRDDLGym.Core.Parser.RDDLReader import RDDLReader

# ENV = 'PowerGeneration'
# ENV = 'MarsRover'
ENV = 'UAV continuous'
# ENV = 'UAV discrete'
# ENV = 'UAV mixed'
ENV = 'Reservoir'
# ENV = 'PowerGeneration'
# ENV = 'CartPole discrete'
# ENV = 'Elevators'
# ENV = 'Reservoir'
# ENV = 'RecSim'

DO_PLAN = True


def rddl_simulate(domain, instance, viz, plan, ground_fn):
    myEnv = RDDLEnv.RDDLEnv(domain=domain, instance=instance)
   # myEnv.set_visualizer(viz)

    total_reward = 0
    state = myEnv.reset()
    for step in range(myEnv.horizon):
        myEnv.render()
        action = {name: value[step] for name, value in plan.items()}
        action = ground_fn(action)
        next_state, reward, done, info = myEnv.step(action)
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
    print("episode ended with reward {}".format(total_reward))
    myEnv.close()
    
    
def jax_simulate(ast, key, plan={}):
    sim = JaxRDDLSimulator(ast, key)
    
    total_reward = 0
    state, done = sim.reset() 
    for step in range(sim.compiled.horizon):
        
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
    viz = EnvInfo.get_visualizer()
    
    rddltxt = RDDLReader(domain, instance).rddltxt
    rddlparser = parser.RDDLParser()
    rddlparser.build()
    ast = rddlparser.parse(rddltxt)
    
    key = jax.random.PRNGKey(np.random.randint(0, 2 ** 31))
    
    if DO_PLAN:
        
        planner = JaxRDDLBackpropPlanner(
            ast, key, 64, 100,
            optimizer=optax.rmsprop(0.2),
            initializer=jax.nn.initializers.normal(),
            action_bounds={'flow' : (0, 1000)})
        
        for callback in planner.optimize(300):
            print('step={} loss={:.6f} test_loss={:.6f} best_loss={:.6f}'.format(
                str(callback['step']).rjust(4),
                callback['train_loss'],
                callback['test_loss'],
                callback['best_loss']))
            
        plan = callback['best_plan']
        plan = {name: np.asarray(value) for name, value in plan.items()}
        
        ground_fn = planner.test_compiled.ground_action_fluents
        rddl_simulate(domain, instance, viz, plan, ground_fn)        
        
    else:
        jax_simulate(ast, key, {})

    
if __name__ == "__main__":
    main()
