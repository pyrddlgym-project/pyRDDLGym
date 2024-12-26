.. _rl:

pyRDDLGym-rl: Reinforcement Learning with stable-baselines3 and rllib.
===============

pyRDDLGym-rl provides wrappers for deep reinforcement learning algorithms (i.e. Stable Baselines 3 and RLlib) to work with pyRDDLGym.


Installing
-----------------

This package requires either ``stable-baselines3>=2.2.1`` or ``ray[rllib]>=2.9.2``.
You can install pyRDDLGym-rl and all of its requirements via pip:

.. code-block:: shell

    pip install stable-baselines3  # need this to run stable baselines agents
    pip install -U "ray[rllib]"  # or this to run rllib agents
    pip install rddlrepository pyRDDLGym-rl

To install the latest pre-release version via git:

.. code-block:: shell

    pip install git+https://github.com/pyrddlgym-project/pyRDDLGym-rl.git


Running the Basic Stable Baselines 3 Example
-------------------

To run the stable-baselines3 example, navigate to the install directory of pyRDDLGym-rl, and type:

.. code-block:: shell

    python -m pyRDDLGym_rl.examples.run_stable_baselines <domain> <instance> <method> <steps> <learning_rate>

where:

* ``<domain>`` is the name of the domain in rddlrepository, or a path pointing to a domain file
* ``<instance>`` is the name of the instance in rddlrepository, or a path pointing to an instance file
* ``<method>`` is the RL algorithm to use [a2c, ddpg, dqn, ppo, sac, td3]
* ``<steps>`` is the (optional) number of samples to generate from the environment for training
* ``<learning_rate>`` is the (optional) learning rate to specify for the algorithm.

Running the Basic RLlib Example
-------------------

To run the RLlib example, from the install directory of pyRDDLGym-rl, type:

.. code-block:: shell

    python -m pyRDDLGym_rl.examples.run_rllib <domain> <instance> <method> <iters>
    
where:

* ``<domain>`` is the name of the domain in rddlrepository, or a path pointing to a domain file
* ``<instance>`` is the name of the instance in rddlrepository, or a path pointing to an instance file
* ``<method>`` is the RL algorithm to use [dqn, ppo, sac]
* ``<iters>`` is the (optional) number of iterations of training.


Running Stable Baselines 3 from the Python API
-------------------

The following example sets up the Stable Baselines 3 PPO algorithm to work with pyRDDLGym:

.. code-block:: python
	
    from stable_baselines3 import *	
	
    import pyRDDLGym
    from pyRDDLGym_rl.core.agent import StableBaselinesAgent
    from pyRDDLGym_rl.core.env import SimplifiedActionRDDLEnv
   
    # create the environment
    env = pyRDDLGym.make("domain", "instance", base_class=SimplifiedActionRDDLEnv)
    
    # train the PPO agent (pass additional arguments, such as learning rate, here)
    agent = PPO('MultiInputPolicy', env, verbose=1)    
    agent.learn(total_timesteps=steps)
    
    # wrap the agent in a RDDL policy and evaluate
    ppo_agent = StableBaselinesAgent(agent)
    ppo_agent.evaluate(env, episodes=1, verbose=True, render=True)
    
    env.close()


.. raw:: html 

   <a href="notebooks/training_ppo_policy_using_stable_baselines3.html"> 
       <img src="_static/notebook_icon.png" alt="Jupyter Notebook" style="width:64px;height:64px;margin-right:5px;margin-top:5px;margin-bottom:5px;">
       Related example: Training a PPO policy with Stable Baselines 3.
   </a>
   
   
Running RLlib from the Python API
-------------------

The following example sets up the RLlib PPO algorithm to work with pyRDDLGym:

.. code-block:: python
	
    from ray.tune.registry import register_env
    from ray.rllib.algorithms.ppo import PPOConfig
    
    import pyRDDLGym
    from pyRDDLGym_rl.core.agent import RLLibAgent
    from pyRDDLGym_rl.core.env import SimplifiedActionRDDLEnv
        
    # set up the environment
    def env_creator(cfg):
        return pyRDDLGym.make(cfg['domain'], cfg['instance'], base_class=SimplifiedActionRDDLEnv)    
    register_env('RLLibEnv', env_creator)
	
	# create agent
    config = {'domain': "domain", 'instance': "instance"}
    agent = PPOConfig().environment('RLLibEnv', cfg=config).build()
    
    # train agent
    for _ in range(iters):
        print(algo.train()['episode_reward_mean'])
    
    # wrap the agent in a RDDL policy and evaluate
    ppo_agent = RLLibAgent(agent)
    ppo_agent.evaluate(env_creator(config), episodes=1, verbose=True, render=True)
	
    env.close()


.. raw:: html 

   <a href="notebooks/training_ppo_policy_using_rllib.html"> 
       <img src="_static/notebook_icon.png" alt="Jupyter Notebook" style="width:64px;height:64px;margin-right:5px;margin-top:5px;margin-bottom:5px;">
       Related example: Training a PPO policy with rllib.
   </a>
   
   
The Environment Wrapper
-------------------

You can use the environment wrapper with your own RL implementations, or a package that is not currently supported by us:

.. code-block:: python

    import pyRDDLGym
    from pyRDDLGym_rl.core.env import SimplifiedActionRDDLEnv
    env = pyRDDLGym.make("domain", "instance", base_class=SimplifiedActionRDDLEnv)

The goal of this wrapper is to simplify the action space as much as possible.
To illustrate, the action space of the MarsRover domain is defined as:

.. code-block:: python

    Dict(
        'power-x___d1': Box(-0.1, 0.1, (1,), float32), 
        'power-x___d2': Box(-0.1, 0.1, (1,), float32), 
        'power-y___d1': Box(-0.1, 0.1, (1,), float32), 
        'power-y___d2': Box(-0.1, 0.1, (1,), float32), 
        'harvest___d1': Discrete(2), 'harvest___d2': Discrete(2)
    )

However, the action space of the wrapper simplifies to

.. code-block:: python

    Dict(
        'discrete': MultiDiscrete([2 2]), 
        'continuous': Box(-0.1, 0.1, (4,), float32)
    )

where the discrete and continuous action variable components have been aggregated.
Actions provided to the environment must therefore follow this form, i.e. must be a dictionary
with the discrete field is assigned a (2,) array of integer type, and the continuous field is assigned
a (4,) array of float type.

.. note::
   The ``vectorized`` option is required by the wrapper and is automatically set to True. 

.. warning::
   The action simplification rules apply ``max-nondef-actions`` only to boolean actions, 
   and assume this value is either 1 or greater than or equal to the total number of boolean actions.
   Any other scenario is currently not supported in pyRDDLGym-rl and will raise an exception.
   
Limitations
-------------------

We cite several limitations of pyRDDLGym-rl:

* The required action space in the stable-baselines/RLlib agent implementation must be compatible with the action space produced by pyRDDLGym (e.g. DQN only handles Discrete spaces)
* Only special types of constraints on boolean actions are supported (as described above).
