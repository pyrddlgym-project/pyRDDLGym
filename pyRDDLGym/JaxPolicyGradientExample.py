import jax
import numpy as np
import optax

from pyRDDLGym.Core.Env.RDDLEnv import RDDLEnv
from pyRDDLGym.Examples.ExampleManager import ExampleManager
from pyRDDLGym.Core.Jax.JaxRDDLLogic import FuzzyLogic
from pyRDDLGym.Core.Jax.JaxRDDLPolicyGradient import JaxStraightLinePlan
from pyRDDLGym.Core.Jax.JaxRDDLPolicyGradient import JaxRDDLPolicyGradient
        
if __name__ == "__main__":
    
    EnvInfo = ExampleManager.GetEnvInfo('Wildfire')
        
    myEnv = RDDLEnv(
        domain=EnvInfo.get_domain(),
        instance=EnvInfo.get_instance(0),
        enforce_action_constraints=True)
    trainer = JaxRDDLPolicyGradient(
        myEnv.model,
        policy=JaxStraightLinePlan(),
        batch_size=16,
        logic=FuzzyLogic(weight=500.),
        optimizer=optax.adam(0.1))
    traj = []
    for callback in trainer.optimize(
        key=jax.random.PRNGKey(np.random.randint(0, 2 ** 16 - 1)),
        epochs=1000, step=1):
        print('avg={:.4f}, best={:.4f}'.format(
            callback['avg_return'], callback['best_return']))
        traj.append(callback['avg_return'])
    
    import matplotlib.pyplot as plt
    plt.plot(traj)
    plt.savefig('trajectory.pdf')
    params = callback['best_params']
    
    key = jax.random.PRNGKey(42)
    avg_reward = 0.
    for _ in range(50):
        total_reward = 0.0
        myEnv.reset()
        for step in range(myEnv.horizon):
            subs = myEnv.sampler.subs
            key, subkey = jax.random.split(key)
            action = trainer.get_action(subkey, params, step, subs)
            next_state, reward, done, _ = myEnv.step(action)
            total_reward += reward
            if done:
                break
        avg_reward += total_reward / 50.0
        print(f'episode ended with reward {total_reward}')
    print(f'average reward {avg_reward}')
