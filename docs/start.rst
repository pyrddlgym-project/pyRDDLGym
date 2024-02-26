Getting Started: Basics
===============

Initializing Built-In Environments
-------------------

To initialize a built-in environment from rddlrepository 
(you must have it installed, i.e. ``pip install rddlrepository``):

.. code-block:: python

    import pyRDDLGym
    env = pyRDDLGym.make("Cartpole_Continuous_gym", "0", **other_kwargs)

where "Cartpole_Continuous_gym" is the name of the domain and "0" is the name of the instance.
``**other_kwargs`` is an optional set of keyword arguments that can be passed to the environment.

Initializing Environments from RDDL Files
-------------------

To initialize an environment from RDDL description files stored on the file system:

.. code-block:: python

    import pyRDDLGym
    env = pyRDDLGym.make("/path/to/domain.rddl", "/path/to/instance.rddl")

where both arguments must be valid file paths to domain and instance RDDL description files.

.. note::
   ``make()`` returns an object of type ``RDDLEnv``, which is also a ``gymnasium.Env``, and can thus be used in 
   most workflows where gym or gymnasium environments are required.

Writing new domains and instances is as easy as writing a few lines of text in a mathematical fashion!
The complete and up-to-date syntax of the RDDL language is described :ref:`here <rddl-description>`.

Policies
----------------------------

A policy interacts with an environment by providing actions or controls in each state.
pyRDDLGym provides two simple policies 
(in addition to Baselines discussed in future sections, many of which are also policies):

- **NoOpAgent** - returns the default action values specified in the RDDL domain.
- **RandomAgent** - samples a random action according to the env.action_space and the maximum number of allowed concurrent actions as specified in the RDDL file.

For example, to initialize a random policy:

.. code-block:: python

    from pyRDDLgym.core.policy import RandomAgent
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

Interaction with an environment is done by calling ``env.step()`` 
and ``env.reset()``, just like regular Gym/Gymnasium.

All fluent values are passed and received as Python ``dict`` objects,
whose keys are valid fluent names as defined in the RDDL domain description.

The structure of the keys for parameterized fluents deserves attention, since the keys 
need to specify not only the fluent name, but also the objects assigned to their parameters.
In pyRDDLGym, the fluent name must be followed by ``___`` (3 underscores), then the 
list of objects separated by ``__`` (2 underscores). To illustrate, for the fluent
``put-out(?x, ?y)``, the required key for objects ``(x1, y1)`` is ``put-out___x1__y1``.

.. note::
   When passing an action dictionary to a ``RDDLEnv``,
   any missing key-value pairs in the dictionary will be assigned the default (or no-op) values
   as specified in the RDDL domain description.

Now lets see what a complete agent-environment loop looks like in pyRDDLGym.
The example below will run the ``Cartpole_Continuous_gym`` environment for a single episode/trial.
The ``env.render()`` function displays a pop-up window rendering the current state to the screen:

.. code-block:: python

    import pyRDDLGym
    from pyRDDLGym.core.policy import RandomAgent

    # set up the Mars Rover instance 0
    env = pyRDDLGym.make("Cartpole_Continuous_gym", "0")
    
    # set up a random policy
    agent = RandomAgent(action_space=env.action_space, num_actions=env.max_allowed_actions)
    
    # perform a roll-out from the initial state
    total_reward = 0
    state, _ = env.reset()
    for step in range(env.horizon):
        env.render()
        action = agent.sample_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        print(f'state = {state}, action = {action}, reward = {reward}')
        total_reward += reward
        state = next_state
        if terminated or truncated:
            break
    print(f'episode ended with reward {total_reward}')

Alternatively, the ``evaluate()`` bypasses the need to write out the ``for`` loop above:

.. code-block:: python
	
   total_reward = agent.evaluate(env, episodes=1, render=True)['mean']
  
The ``agent.evaluate()`` call returns a dictionary of summary statistics about the 
total rewards collected across episodes, such as mean, median, standard deviation, etc.

Spaces
------

The state and action spaces of a ``RDDLEnv`` are standard ``gymnasium.spaces`` and are
accessible via ``env.state_space`` and ``env.action_space``, respectively.
In most cases, state and action spaces are ``gymnasium.spaces.Dict`` objects, whose key-value pairs
are fluent names and their current values.

To compute bounds on RDDL fluents, pyRDDLGym analyzes the 
``action-preconditions`` and ``state-invariants`` expressions. 
For box constraints, the conversion happens as follows:

- real -> ``Box(l, u)`` where ``(l, u)`` are the bounds on the fluent
- int -> ``Discrete(l, u)`` where ``(l, u)`` are the bounds on the fluent
- bool -> ``Discrete(2)``

.. note::
   Any constraints that cannot be rewritten as box constraints are ignored, due to limitations of Gymnasium.
   If no valid box bounds for a fluent are available, they are set to ``[-np.inf, np.inf]``

Using Built-In Visualizers
-------------

Every domain has a default visualizer assigned to it, which is either a graphical 
``ChartVisualizer`` that plots the state trajectory over time, or a custom domain-dependent implementation.

Assigning a visualizer for an environment can be done by calling 
``env.set_visualizer(viz)`` with ``viz`` as the desired visualization object (or a string identifier).

For example, to assign the ``ChartVisualizer`` or the ``HeatmapVisualizer``, 
which use line charts or heatmaps to track the state across time:

.. code-block:: python

    env.set_visualizer("chart")
    env.set_visualizer("heatmap")

To assign the ``TextVisualizer``, which produces a textual representation of the 
state similar to the standard console output:

.. code-block:: python

    env.set_visualizer("text")

Using a Custom Visualizer
-------------

To assign a custom visualizer object ``MyDomainViz`` that implements a valid ``render(state)`` method,

.. code-block:: python

    from pyRDDLGym.core.visualizer.viz import BaseViz 

    class MyDomainViz(BaseViz)
        # here goes the visualization implementation
        def render(self, state):
            ...

    env.set_visualizer(MyDomainViz)

.. warning::
   The visualizer argument in ``set_visualizer`` should not contain the customary 
   ``()`` when initializing the visualizer object, since this is done internally.
   So, instead of writing ``env.set_visualizer(MyDomainViz(**MyArgs))``, write 
   ``env.set_visualizer(MyDomainViz, viz_kwargs=MyArgs)``.

All visualizers can be activated in an environment by calling ``env.render()``
on each call to ``env.step()`` or ``env.reset()``, just like regular Gym/Gymnasium.

Recording Movies
--------------------------

A ``MovieGenerator`` class is provided to capture videos of the environment interaction over time:

.. code-block:: python
    
    from pyRDDLGym.core.visualizer.movie import MovieGenerator
    recorder = MovieGenerator("/path/to/save", "env_name")
    env.set_visualizer(VizClass, movie_gen=recorder)

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
	
    env = pyRDDLGym.make("Cartpole_Continuous_gym", "0", log_path="path/to/output.csv")
                            
Upon interacting with the environment, pyRDDLGym appends the new observations to the log file at the
specified path. Logging continues until ``env.close()`` is called.
