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

pyRDDLGym ships with the RDDL files for many interesting domains and accompanying instances.
These domains and instances can be accessed through the ``ExampleManager``. 

For example, to list all domains currently packaged with pyRDDLGym:

.. code-block:: python

    from pyRDDLGym import ExampleManager
    ExampleManager.ListExamples()

To request more info about a domain:

.. code-block:: python

    info = ExampleManager.GetEnvInfo("domain-name")

To then list all instances of a domain:

.. code-block:: python

    info.list_instances()

To then set up a Gym environment with the default visualizer:

.. code-block:: python

    env = RDDLEnv.RDDLEnv(domain=info.get_domain(), instance=info.get_instance(0))
    env.set_visualizer(info.get_visualizer())

Alternatively:

.. code-block:: python

    env = RDDLEnv.RDDLEnv.build(info, instance, **other_kwargs)

Policies
----------------------------

A policy interacts with an environment by providing actions or controls in each state.
pyRDDLGym provides two simple policies 
(in addition to Baselines discussed in future sections, many of which are also policies):

- **NoOpAgent** - returns the default action values specified in the RDDL domain.
- **RandomAgent** - samples a random action according to the env.action_space and the maximum number of allowed concurrent actions as specified in the RDDL file.

For example, to initialize a random policy:

.. code-block:: python

    from Policies.Agents import RandomAgent
    agent = RandomAgent(action_space=env.action_space, num_actions=env.max_allowed_actions)

All policies must implement a ``sample_action`` function for sampling an
action in each state:

.. code-block:: python

    action = agent.sample_action(state)
 
.. note::
   Random policies respect only box constraints, due to limitations in Gym.
   To handle arbitrary nonlinear constraints, implement a custom ``Agent``
   with its own ``sample_action`` function.
   
Interacting with an Environment
----------------------------

Now lets see what a complete agent-environment loop looks like in pyRDDLGym.
The example below will run the ``MarsRover`` environment for a single episode/trial.
The ``env.render()`` function displays a pop-up window rendering the current state to the screen:

.. code-block:: python

    from pyRDDLGym import RDDLEnv
    from pyRDDLGym import ExampleManager
    from pyRDDLGym.Core.Policies.Agents import RandomAgent

    # set up the Mars Rover instance 0
    info = ExampleManager.GetEnvInfo('MarsRover')
    env = RDDLEnv.RDDLEnv.build(info, 0)
    
    # set up a random policy
    agent = RandomAgent(action_space=env.action_space, num_actions=env.max_allowed_actions)
    
    # perform a roll-out from the initial state
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

Alternatively, the ``evaluate()`` bypasses the need to write out the ``for`` loop above:

.. code-block:: python
	
   total_reward = agent.evaluate(env, episodes=1, render=True)['mean']
  
The ``agent.evaluate()`` call returns a dictionary of summary statistics about the 
total rewards collected across episodes, such as mean, median, standard deviation, etc.

Spaces
------

The state and action spaces of a ``RDDLEnv`` are standard ``gym.spaces`` and are
accessible via ``env.state_space`` and ``env.action_space``, respectively.
In most cases, state and action spaces are ``gym.spaces.Dict`` objects, whose key-value pairs
are fluent names and their current values.

In order to compute meaningful bounds on RDDL variables, pyRDDLGym parses and analyzes the 
``action-preconditions`` and ``state-invariants`` expressions. If constraints are box constraints,
the conversion happens as follows:

- real -> Box with bounds as specified in action-preconditions, or with np.inf and symmetric bounds
- int -> Discrete with bounds as specified in action-preconditions, or with np.inf and symmetric bounds
- bool -> Discrete(2)

.. note::
   Any constraints that cannot be rewritten as box constraints are ignored, due to limitations of Gym.

.. note::
   When passing an action dictionary to a ``RDDLEnv``, for example to the ``step()`` function,
   any missing key-value pairs in the dictionary will be assigned the default (or no-op) values
   as specified in the RDDL domain description.

Visualization
-------------

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

All visualizers can be activated in an environment by calling ``env.render()``
on each call to ``env.step()`` or ``env.reset()``, just like regular Gym.

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
   Videos will not be saved until the environment is closed with ``env.close()``. However, frames will be recorded
   to disk continuously while the environment interaction is taking place (to save RAM), which will be used to generate the video.
   Therefore, it is important to not delete these images while the recording is 
   taking place, which will be deleted automatically once recording is complete.

Logging Simulation Data
--------------------------

A record of all past interactions with an environment can be logged to a machine
readable CSV file for later analysis:

.. code-block:: python
	
	env = RDDLEnv.RDDLEnv.build(info, instance, log_path='path/to/file.csv')
                            
Upon interacting with the environment, pyRDDLGym appends the new observations to the log file at the
specified path. Logging continues until ``env.close()`` is called.

Writing Custom Domains
--------------------------

Writing new domains is as easy as writing a few lines of text in a mathematical fashion!
It is only required to specify two ``.rddl`` files, one containing the lifted domain description,
and another containing the instance specification, and pointing the ``RDDLEnv`` initialization
to these two files as discussed at the beginning of this page.

The syntax required for building RDDL domains is described here: :ref:`rddl-description`.
