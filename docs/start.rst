Getting Started
===============

Initializing Environments
-------------------

Initializing environments in pyRDDLGym is done by instantiating a ``RDDEnv`` object:

.. code-block:: python

    from pyRDDLGym import RDDLEnv
    env = RDDLEnv.RDDLEnv(domain="domain.rddl", instance='instance.rddl')

where ``domain.rddl`` and ``instance.rddl`` are paths pointing to RDDL files of your choosing.

.. note::
   ``RDDLEnv`` objects are instances of ``gym.Env``, and can thus be used in 
   most workflows where gym environments are required.

Built-in Environments
-----------------

pyRDDLGym is shipped with a number of varied domains and accompanying instances, whose RDDL files
are included as part of the distribution. These environments can be accessed through the
``ExampleManager``.

To list all examples currently packaged with pyRDDLGym:

.. code-block:: python

    from pyRDDLGym import ExampleManager
    ExampleManager.ListExamples()

To request more info about a domain:

.. code-block:: python

    info = ExampleManager.GetEnvInfo(<domain-name>)

To then list all instances of a domain:

.. code-block:: python

    info.list_instances()

To then set up an OpenAI gym environment with the default visualizer:

.. code-block:: python

    env = RDDLEnv.RDDLEnv(domain=info.get_domain(), instance=info.get_instance(0))
    env.set_visualizer(info.get_visualizer())

Alternatively:

.. code-block:: python

    env = RDDLEnv.RDDLEnv.build(info, instance, **other_kwargs)

Policies
----------------------------

A policy interacts with an environment by providing actions or controls in each state.
We provide two simple policies as part of pyRDDLGym:

- **NoOpAgent** - returns the default action values specified in the RDDL domain.
- **RandomAgent** - samples a random action according to the env.action_space and the maximum number of allowed concurrent actions as specified in the RDDL file.

To initialize a random policy for example:

.. code-block:: python

    from Policies.Agents import RandomAgent
    agent = RandomAgent(action_space=env.action_space, num_actions=env.max_allowed_actions)

.. note::
   Random policies only respect box constraints, due to limitations on OpenAI gym spaces.
   To handle complex constraints, it is recommended to implement a custom ``Agent``. 
   
Interacting with an Environment
----------------------------

Now lets see what a complete agent-environment loop looks like in pyRDDLGym.
The example below will run the ``MarsRover`` environment for a single episode or trial.
If the ``env.render()`` function is used, we will also see a window pop up rendering the environment:

.. code-block:: python

    from pyRDDLGym import RDDLEnv
    from pyRDDLGym import ExampleManager
    from pyRDDLGym.Core.Policies.Agents import RandomAgent

    # set up the Mars Rover problem instance 0
    info = ExampleManager.GetEnvInfo('MarsRover')
    env = RDDLEnv.RDDLEnv.build(info, 0)
    
    # set up a random policy
    agent = RandomAgent(action_space=env.action_space, num_actions=env.max_allowed_actions)
    
    # perform a roll-out from the initial state
    # until either termination or the horizon is reached
    total_reward = 0
    state = env.reset()
    for _ in range(env.horizon):
          env.render()
          action = agent.sample_action(state)
          next_state, reward, done, _ = env.step(action)
          total_reward += reward
          state = next_state
          if done:
                break
    env.close()

The convenience ``evaluate`` function replaces the above interaction ``for`` loop with a single command:

.. code-block:: python
	
   total_reward = agent.evaluate(env, episodes=1, render=True)['mean']
  
The ``evaluate`` returns a dictionary of summary statistics about the 
returns collected on different episodes, such as mean, median, standard deviation, etc.

Spaces
------

The state and action spaces of pyRDDLGym are standard ``gym.spaces``, 
accessible through the standard API: ``env.state_space`` and ``env.action_space``.
State/action spaces are of type ``gym.spaces.Dict``, where each key-value pair 
consists of the fluent name and its current value.

Thus, RDDL types are converted to ``gym.spaces`` with the appropriate bounds as 
specified in the RDDL ``action-preconditions`` and ``state-invariants`` fields. 
The conversion is as following:

- real -> Box with bounds as specified in action-preconditions, or with np.inf and symmetric bounds
- int -> Discrete with bounds as specified in action-preconditions, or with np.inf and symmetric bounds
- bool -> Discrete(2)

.. note::
   When passing an action dictionary to a ``RDDLEnv``, for example to the ``step()`` function,
   any missing key-value pairs in the dictionary will be assigned the default (or no-op) values
   as specified in the RDDL domain description.

Visualization
-------------

pyRDDLGym visualization is just like regular Gym, which can be done by calling ``env.render()``.
Every domain has a default visualizer assigned to it, which is either a graphical 
``ChartVisualizer`` that plots the state trajectory over time, or a custom domain-dependent implementation.

Assigning a visualizer for an environment can be done by calling 
``env.set_visualizer(viz)`` with ``viz`` as the desired visualization object.
For example, to assign the ``ChartVisualizer``:

.. code-block:: python

    from pyRDDLGym.Visualizer.ChartViz import ChartVisualizer
    env.set_visualizer(ChartVisualizer)

To assign the ``TextVisualizer``, which produces a textual representation of the 
state similar to the standard console output:

.. code-block:: python

    from pyRDDLGym.Visualizer.TextViz import TextVisualizer
    env.set_visualizer(TextVisualizer)

Recording Movies
--------------------------

A ``MovieGenerator`` class is provided to capture videos of the environment interaction over time:

.. code-block:: python
    
    from pyRDDLGym import RDDLEnv
    from pyRDDLGym import ExampleManager
    from pyRDDLGym.Visualizer.MovieGenerator import MovieGenerator

    # set up the environment
    info = ExampleManager.GetEnvInfo('MarsRover')
    env = RDDLEnv.RDDLEnv.build(info, 0)
	
    # set up the movie generator
    movie_gen = MovieGenerator('myFilePath', 'myEnvName', max_frames=1000)
    
    # set up the environment visualizer, passing a movie generator to capture frames
    env.set_visualizer(info.get_visualizer(), movie_gen=movie_gen)

Upon calling ``env.close()``, the images captured will be combined into video format and saved to the desired path.
Any temporary files created to capture individual frames during interaction will be deleted from disk.

.. note::
   Videos will not be saved until the environment is closed with ``env.close()``. However, still frames will be recorded
   to disk while the interaction with the environment is taking place, in order to 
   accommodate long interactions that could use too much memory.
   Therefore, it is important to not delete these images while the recording is 
   taking place, which will be deleted automatically once recording is complete.

Logging Simulation Data
--------------------------

The record of all past interactions with an environment can be logged to a machine
readable file, such as a CSV. To write to a CSV file:

.. code-block:: python
	
	env = RDDLEnv.RDDLEnv.build(info, instance, log_path='path/to/file.csv')
                            
Upon interacting with the environment, a log file is created at the specified path
which can be later parsed using standard CSV packages. Similar to movie generation,
logging will continue until ``env.close()`` is called.

Writing Custom Domains
--------------------------

Writing new user defined domains is as easy as writing a few lines of text in a mathematical fashion!
It is only required to specify the lifted constants, variables (all are referred as fluents in RDDL),
behavior/dynamic of the problem and generating an instance with the actual objects and initial state in RDDL - and pyRDDLGym will do the rest.
The syntax for building RDDL domains is described here: :ref:`rddl-description`.

