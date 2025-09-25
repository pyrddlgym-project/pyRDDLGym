Getting Started: Basics
===============

Initializing Environments
-------------------

Built-In Environments
^^^^^^^^^^^^^^^^^^^

To initialize a built-in environment from the `rddlrepository <https://github.com/pyrddlgym-project/rddlrepository>`_:

.. code-block:: python

    import pyRDDLGym
    env = pyRDDLGym.make("CartPole_Continuous_gym", "0")

where "CartPole_Continuous_gym" is the name of the domain and "0" is the instance.

From RDDL Files
^^^^^^^^^^^^^^^^^^^

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

Built-In Policies
^^^^^^^^^^^^^^^^^^^

pyRDDLGym provides two simple policies, which are all instances of ``pyRDDLGym.core.policy.BaseAgent``:

- ``NoOpAgent`` returns the default action values specified in the RDDL domain.
- ``RandomAgent`` samples a random action according to the ``env.action_space`` and the ``max-nondef-actions``.

For example, to initialize a random policy:

.. code-block:: python

    from pyRDDLGym.core.policy import RandomAgent
    agent = RandomAgent(action_space=env.action_space, num_actions=env.max_allowed_actions)

All policies must implement a ``sample_action()`` function for sampling an action in each state:

.. code-block:: python

    action = agent.sample_action(state)
 
.. note::
   Random policies respect only box constraints, due to limitations in Gym.
   To handle arbitrary nonlinear constraints, implement a custom ``BaseAgent``
   with its own ``sample_action()`` function.

.. raw:: html 

   <a href="notebooks/simulating_pyrddlgym_random_policy.html"> 
       <img src="_static/notebook_icon.png" alt="Jupyter Notebook" style="width:64px;height:64px;margin-right:5px;margin-top:5px;margin-bottom:5px;">
       Related example: Simulating an environment in pyRDDLGym with a built-in policy.
   </a>


Custom Policies
^^^^^^^^^^^^^^^^^^^

To implement your own custom policy, inherit from ``pyRDDLGym.core.policy.BaseAgent``:

.. code-block:: python

    from pyRDDLGym.core.policy import BaseAgent
    
    class CustomAgent(BaseAgent):
    
        def sample_action(self, state):
            # here goes the code that returns the current action
            ...     

.. raw:: html 

   <a href="notebooks/simulating_pyrddlgym_custom_policy.html"> 
       <img src="_static/notebook_icon.png" alt="Jupyter Notebook" style="width:64px;height:64px;margin-right:5px;margin-top:5px;margin-bottom:5px;">
       Related example: Simulating an environment in pyRDDLGym with a custom policy.
   </a>
   

Interacting with an Environment
----------------------------

Interaction with an environment is done by calling ``env.step(action)`` 
and ``env.reset()``, just like regular Gym/Gymnasium.

Reading and Passing Fluents
^^^^^^^^^^^^^^^^^^^

All fluent values are passed and received as Python ``dict`` objects,
whose keys are valid fluent names as defined in the RDDL domain description.

The structure of the keys for parameterized fluents deserves attention, since the keys 
need to specify not only the fluent name, but also the objects assigned to their parameters.
In pyRDDLGym, the fluent name must be followed by ``___`` (3 underscores), then the 
list of objects separated by ``__`` (2 underscores). To illustrate, for the fluent
``put-out(?x, ?y)``, the required key for objects ``(x1, y1)`` is ``put-out___x1__y1``.

Another option is to pass a dict whose keys are lifted fluent names, i.e. ``put-out``, in which
case the values must be numpy arrays (of the necessary shape and dtype).

.. note::
   When passing an action dictionary to a ``RDDLEnv``,
   any missing key-value pairs in the dictionary will be assigned the default (or no-op) values
   as specified in the RDDL domain description.

Interaction Loop
^^^^^^^^^^^^^^^^^^^

We now show what a complete agent-environment loop looks like in pyRDDLGym.
The example below will run the ``CartPole_Continuous_gym`` environment for a single episode, 
rendering the state to the screen in real time:

.. code-block:: python

    import pyRDDLGym
    from pyRDDLGym.core.policy import RandomAgent

    # set up the Mars Rover instance 0
    env = pyRDDLGym.make("CartPole_Continuous_gym", "0")
    
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

Alternatively, the ``evaluate()`` bypasses the need to write out the entire loop:

.. code-block:: python
	
    total_reward = agent.evaluate(env, episodes=1, render=True)['mean']
  
The ``agent.evaluate()`` call returns a dictionary of summary statistics about the 
total rewards collected across episodes, such as mean, median, and standard deviation.

Setting the Random Seed
^^^^^^^^^^^^^^^^^^^

In order to get reproducible results, it is necessary to set the random seed. 
This can be passed to ``env.reset()`` once at the start of the experiment:

.. code-block:: python
	
    env.reset(seed=42)

or alternatively to ``agent.evaluate()``:

.. code-block:: python
	
    agent.evaluate(env, seed=42)

Other objects that require randomness typically support setting random seeds.
For example, to set the seed of the ``RandomAgent`` instance:

.. code-block:: python

    agent = RandomAgent(action_space=env.action_space, num_actions=env.max_allowed_actions, seed=42)


Handling Simulation Errors
^^^^^^^^^^^^^^^^^^^

By default, ``evaluate()`` will not raise errors if action preconditions or state invariants are violated.
State invariant violations are stored in the ``truncated`` field returned by ``env.step()``. 
If you wish to enforce action constraints, simply initialize your environment like this:

.. code-block:: python
	
    import pyRDDLGym
    env = pyRDDLGym.make("CartPole_Continuous_gym", "0", enforce_action_constraints=True)

By default, ``evaluate()`` will not raise an exception if a numerical error occurs during an intermediate calculation,
such as divide by zero or under/overflow. If you wish to raise/catch all numerical errors, add the following
before calling ``evaluate()``:

.. code-block:: python

    import numpy as np
    np.seterror(all='raise')

More details about controlling error handling behavior can be found 
`here <https://numpy.org/doc/stable/reference/generated/numpy.seterr.html>`_.

.. warning::
   Branched error handling in operations such as ``if`` and ``switch`` 
   is incompatible with vectorized computation. To illustrate, an expression like
   ``if (pvar(?x) == 0) then default(?x) else 1.0 / pvar(?x)`` will evaluate ``1.0 / pvar(?x)`` first
   for all values of ``?x``, regardless of the branch condition, and will thus trigger an exception if ``pvar(?x) == 0``
   for some value of ``?x``. For the time being, we recommend suppressing errors as described above.


Gym ``state_space`` and ``action_space``
^^^^^^^^^^^^^^^^^^^

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
   If no valid box bounds for a fluent are available, they are set to ``(-np.inf, np.inf)``


Visualizing Environments
-------------

Built-In Visualizers
^^^^^^^^^^^^^^^^^^^

Every domain has a default visualizer assigned to it, which is either a 
``ChartVisualizer`` that plots the state trajectory as a graph, or a domain-dependent implementation.

Assigning a visualizer to an environment can be done by calling 
``env.set_visualizer(viz)`` with ``viz`` as the desired visualization object (or a string identifier).

For example, to assign the ``ChartVisualizer`` or the ``HeatmapVisualizer``, 
which use line charts or heatmaps to track the state across time, 
or the ``TextVisualizer``, which produces a textual representation of the state:

.. code-block:: python

    env.set_visualizer("chart")
    env.set_visualizer("heatmap")
    env.set_visualizer("text")
    
Calling ``env.set_visualizer(viz=None, ...)`` will not change the visualizer already assigned: this is useful
if you want to record movies using the default viz as described later.

Custom Visualizers
^^^^^^^^^^^^^^^^^^^

To assign a custom visualizer object ``MyDomainViz`` that implements a valid ``render(state)`` method,

.. code-block:: python

    from pyRDDLGym.core.visualizer.viz import BaseViz 

    class MyDomainViz(BaseViz)
        
        def render(self, state):
            # here goes the visualization implementation
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
    recorder = MovieGenerator("/folder/path/to/save/animation", "env_name", max_frames=999999)
    env.set_visualizer(viz=None, movie_gen=recorder)

Upon calling ``env.close()``, the images captured will be combined into video format and saved to the desired path.
Any temporary files created to capture individual frames during interaction will be deleted from disk.

.. note::
   Videos will not be saved until the environment is closed with ``env.close()``. However, frames will be recorded
   to disk continuously while the environment interaction is taking place (to save RAM), which will be used to generate the video.
   Therefore, it is important to not delete these images while the recording is taking place.

.. raw:: html 

   <a href="notebooks/recording_movies_in_pyrddlgym.html"> 
       <img src="_static/notebook_icon.png" alt="Jupyter Notebook" style="width:64px;height:64px;margin-right:5px;margin-top:5px;margin-bottom:5px;">
       Related example: Recording a movie of a simulation in pyRDDLGym.
   </a>
   
   
Logging Simulation Data
--------------------------

A record of all past interactions with an environment can be logged to a machine
readable CSV file for later analysis:

.. code-block:: python
	
    env = pyRDDLGym.make("CartPole_Continuous_gym", "0", log_path="/path/to/output.csv")
                            
Upon interacting with the environment, pyRDDLGym appends the new observations to the log file at the
specified path. Logging continues until ``env.close()`` is called.
