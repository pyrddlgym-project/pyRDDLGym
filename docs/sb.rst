Baselines: Reinforcement Learning with Stable Baselines
===============

While it is not the intended use case of RDDL (being model-free), pyRDDLGym also provides convenience
wrappers that can work with (deep) RL implementations, notably stable-baselines3.

Simplifying the Action Space
-------------------

In order to run stable-baselines3 with a pyRDDLGym env, we first need to observe that the default ``gym.spaces.Dict`` action space
representation in pyRDDLGym environments is not directly compatible with stable-baselines. For example, the DQN implementation
in stable-baselines3 only accepts environments in which the action space is a ``Discrete`` object.

Thus, pyRDDLGym's ``RDDLEnv`` offers a convenient field ``compact_action_space`` 
that when set to True, perform simplification automatically when compiling the environment 
to maximize compatibility with RL implementations such as stable-baselines.

.. code-block:: python

    info = ExampleManager.GetEnvInfo(domain)
    env = RDDLEnv.RDDLEnv.build(info, instance, compact_action_space=True)

To illustrate, for the built-in MarsRover example, 
without the ``compact_action_space`` flag set, the action space would be represented as

.. code-block:: python

    Dict(
        'power-x___d1': Box(-0.1, 0.1, (1,), float32), 
        'power-x___d2': Box(-0.1, 0.1, (1,), float32), 
        'power-y___d1': Box(-0.1, 0.1, (1,), float32), 
        'power-y___d2': Box(-0.1, 0.1, (1,), float32), 
        'harvest___d1': Discrete(2), 'harvest___d2': Discrete(2)
    )

However, with the ``compact_action_space`` flag set, the action space would be simplified to

.. code-block:: python

    Dict(
        'discrete': MultiDiscrete([2 2]), 
        'continuous': Box(-0.1, 0.1, (4,), float32)
    )

where the discrete and continuous action variable components will be automatically aggregated.
Actions provided to the environment must therefore follow this form, i.e. must be a dictionary
with the discrete field is assigned a (2,) array of integer type, and the continuous field is assigned
a (4,) array of float type.

.. note::
   When ``compact_action_space`` is set to True, the ``vectorized`` option is required 
   and is automatically set to True (see Tensor Representation section in Getting Started section). 

.. warning::
   The action simplification rules apply ``max-nondef-actions`` only to boolean actions, 
   and assume this value is either 1 or greater than or equal to the total number of boolean actions.
   Any other scenario is currently not supported in pyRDDLGym and will raise an exception.
   
Running Stable Baselines
-------------------

By fixing the action space as described above, most pyRDDLGym environments can be used directly
in stable-baselines3 RL implementations without considerable modification to the workflow.
For example, the following code example trains a Deep Q-Network (DQN) on a domain and 
instance of one's choosing (assuming the action space simplifies to a ``Discrete``):

.. code-block:: python

    from pyRDDLGym import ExampleManager, RDDLEnv
    from stable_baselines3 import PPO
    
    # build the environment
    info = ExampleManager.GetEnvInfo(domain)
    env = RDDLEnv.RDDLEnv.build(info, instance, new_gym_api=True, compact_action_space=True)
    
    # train a DQN agent
    model = PPO('MultiInputPolicy', env, ...)
    model.learn(total_timesteps=int(2e5))

Note that, at the time of this writing, the ``new_gym_api`` flag must be set since the 
stable-baselines implementation assumes the newest version of gym.

Below is a complete worked example for solving CartPole using PPO:

.. code-block:: python
    
	from stable_baselines3 import PPO
    from pyRDDLGym.Core.Env.RDDLEnv import RDDLEnv
    from pyRDDLGym.Examples.ExampleManager import ExampleManager

    # set up the environment
    info = ExampleManager.GetEnvInfo('CartPole_discrete')    
    env = RDDLEnv.build(info, 0, new_gym_api=True, compact_action_space=True)
    
    # train the agent
    model = PPO('MultiInputPolicy', env, verbose=1)
    model.learn(total_timesteps=40000)
    
    # evaluate
    total_reward = 0
    state, _ = env.reset()
    for step in range(env.horizon):
        env.render()
        action, _ = model.predict(state)
        state, reward, done, *_ = env.step(action)
        total_reward += reward
        if done:
            break
    print(f'episode ended with return {total_reward}')
    env.close()


Limitations
-------------------

We cite several limitations of using stable-baselines in pyRDDLGym:

* The required action space in the stable-baselines agent implementation must be compatible with the action space produced by pyRDDLGym
* Only special types of constraints on boolean actions are supported (as described above).
