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
   ``RDDLEnv`` objects are instances of ``gym.Env``, and can thus be used in most cases where gym environments are required.

Built-in Environments
-----------------

pyRDDLGym is shipped with a number of varied domains and accompanying instances, whose RDDL files
are included as part of the distribution. These environments can be accessed through the
``ExampleManager``:

To list all examples currently packaged with pyRDDLGym:

.. code-block:: python

    from pyRDDLGym import ExampleManager
    ExampleManager.ListExamples()

To request more info about a domain:

.. code-block:: python

    info = ExampleManager.GetEnvInfo(ENV)

To list all instances of a domain:

.. code-block:: python

    info.list_instances()

To set up an OpenAI gym environment:

.. code-block:: python

    env = RDDLEnv.RDDLEnv(domain=info.get_domain(), instance=info.get_instance(0))
    env.set_visualizer(info.get_visualizer())

Or alternatively:

.. code-block:: python

    info = ExampleManager.GetEnvInfo(domain)
    env = RDDLEnv.RDDLEnv.build(info, instance, **other_kwargs)

Interacting with the Environment
----------------------------

As in all MDP applications, we define policies to interact with an environment by providing actions or controls in each state.
We provide two simple policies as part of pyRDDLGym:

- **NoOpAgent** - returns the default action values specified in the RDDL domain.
- **RandomAgent** - samples a random action according to the env.action_space and the maximum number of allowed concurrent actions as specified in the RDDL file.

To initialize a random agent for example:

.. code-block:: python

    from Policies.Agents import RandomAgent
    agent = RandomAgent(action_space=env.action_space, num_actions=env.max_allowed_actions)

Now lets see what a complete agent-environment loop looks like in pyRDDLGym.
The example below will run the ``MarsRover`` environment for the amount of time steps specified in the instance ``horizon`` field.
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

We also provide a convenience ``evaluate`` function, so it is not necessary to implement the interaction loop above explicitly:

.. code-block:: python
	
   total_reward = agent.evaluate(env, episodes=1, render=True)['mean']
  
The above call to ``evaluate`` returns a dictionary of summary statistics about the returns collected on different episodes, such as mean, median, standard deviation, etc.

New vs Old Gym API
------

The new and old Gym APIs return the transition information from ``step()`` and ``reset()`` in slightly different 
format. The new API output of ``step()`` produces a tuple of the form ``state, reward, done, fail, info``
where ``fail`` determines whether the agent is out of bounds or failed the episode. Similarly, ``reset()`` in the 
new API produces ``state, info``.

pyRDDLGym offers a convenient flag to switch between the old and the new API:

.. code-block:: python

    info = ExampleManager.GetEnvInfo(domain)
    env = RDDLEnv.RDDLEnv.build(info, instance, new_gym_api=True)

pyRDDLGym computes the auxiliary ``fail`` information by evaluating whether the state invariants
are satisfied in the current state.

Spaces
------

The state and action spaces of pyRDDLGym are standard ``gym.spaces``, accessible through the standard API: ``env.state_space`` and ``env.action_space``.
State/action spaces are of type ``gym.spaces.Dict``, where each key-value pair where the key name is the state/action and the value is the state/action current value or action to apply.

Thus, RDDL types are converted to ``gym.spaces`` with the appropriate bounds as specified in the RDDL ``action-preconditions`` and ``state-invariants`` fields. The conversion is as following:

- real -> Box with bounds as specified in action-preconditions, or with np.inf and symmetric bounds.
- int -> Discrete with bounds as specified in action-preconditions, or with np.inf and symmetric bounds.
- bool -> Discrete(2)

There is no need in pyRDDLGym to specify the values of all the existing action in the RDDL domain description, only thus the agent wishes to assign non-default values, the infrastructure will construct the full action vector as necessary with the default action values according to the RDDL description.

Tensor Representation
-------------------

Some algorithms require a tensor representation of states and/or actions. The ``RDDLEnv`` class provides a ``vectorized`` option
to work directly with the tensor representations of state and action fluents. With this option, for example, a ``bool`` action fluent
``put-out(?x, ?y)`` taking two parameters ``?x`` and ``?y`` with 3 objects each would be provided as a boolean-valued 
3-by-3 matrix (rank 2 tensor). State fluents would also be returned in an equivalent format. 

This option can be enabled simply as

.. code-block:: python

    info = ExampleManager.GetEnvInfo(domain)
    env = RDDLEnv.RDDLEnv.build(info, instance, vectorized=True)

With this option enabled, the bounds of the ``observation_space`` and ``action_space`` 
of the gym environment are instances of ``gym.spaces.Box`` with the correct shape and dtype.


Inspecting the Model
-------------------

The pyRDDLGym compiler provides a convenient API for querying a variety of properties about RDDL constructs in a domain, 
which can be accessed through the ``model`` field of a ``RDDLEnv``

.. code-block:: python
	
    info = ExampleManager.GetEnvInfo('MarsRover')
    env = RDDLEnv.RDDLEnv.build(info, 0)
    model = env.model

Below are some commonly-used fields of ``model`` that can be accessed directly.
	
.. list-table:: Commonly-used properties accessible in ``model``
   :widths: 50 60
   :header-rows: 1
   
   * - syntax
     - description
   * - ``horizon``
     - horizon as defined in the instance
   * - ``discount``
     - discount factor as defined in the instance
   * - ``max_allowed_actions``
     - ``max-nondef-actions`` as defined in the instance
   * - ``variable_types``
     - dict of pvariable types (e.g. non-fluent, ...) for each variable
   * - ``variable_ranges``
     - dict of pvariable ranges (e.g. real, ...) for each variable
   * - ``objects``
     - dict of all defined objects for each type
   * - ``nonfluents``
     - dict of initial values for each non-fluent
   * - ``states``
     - dict of initial values for each state-fluent
   * - ``actions``
     - dict of default values for each action-fluent
   * - ``interm``
     - dict of initial values for each interm-fluent
   * - ``observ``
     - dict of initial values for each observ-fluent
   * - ``cpfs``
     - dict of ``Expression`` objects for each cpf
   * - ``reward``
     - ``Expression`` object for reward function
   * - ``preconditions``
     - list of ``Expression`` objects for each action-precondition
   * - ``invariants``
     - list of ``Expression`` objects for each state-invariant

``Expression`` objects are symbolic syntax trees that describe the flow of computations
in each cpf, constraint relation, or the reward function of the RDDL domain.

The ``args()`` function of an ``Expression`` object accesses its sub-expressions, 
which can be either ``Expression`` instances or collections containing aggregation variables,
types, or other information required by the engine. Similarly, the ``etype()`` argument
provides identifying information about the expression.

Visualization
-------------

pyRDDLGym visualization is just like regular Gym, which can be done by calling ``env.render()``.
Every domain has a default visualizer assigned to it, which is either a graphical ``ChartVisualizer`` that plots the state trajectory over time, or a custom domain-dependent implementation.

Assigning a visualizer for an environment can be done by calling the environment method ``env.set_visualizer(viz)`` with ``viz`` as the desired visualization object.

.. code-block:: python

    from pyRDDLGym import RDDLEnv
    from pyRDDLGym import ExampleManager

    # set up the environment
    info = ExampleManager.GetEnvInfo('MarsRover')
    env = RDDLEnv.RDDLEnv.build(info, 0)

    # set up the environment visualizer
    env.set_visualizer(info.get_visualizer())

To override the default visualizer instance with the ``ChartVisualizer``, simply replace the last line above with

.. code-block:: python

    from pyRDDLGym.Visualizer.ChartViz import ChartVisualizer
    env.set_visualizer(ChartVisualizer)

In order to build custom visualizations (for new user defined domains), 
one can inherit the class ``Visualizer.StateViz.StateViz()`` and return in the ``visualizer.render()`` method a PIL image for the gym to render to the screen.
The environment initialization has the following general structure:

.. code-block:: python

    class MyDomainViz(StateViz)
        # here goes the visualization implementation

    env.set_visualizer(MyDomainViz)

.. warning::
   The visualizer argument in ``set_visualizer`` should not contain the customary ``()`` when initializing the visualizer object, since this is done internally.
   So, instead of writing ``env.set_visualizer(MyDomainViz(**MyArgs))``, write ``env.set_visualizer(MyDomainViz, viz_kwargs=MyArgs)``.
   
Recording Movies
--------------------------

A ``MovieGenerator`` class is provided to allow capture of videos of the environment during interaction:

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
   to disk while the interaction with the environment is taking place, in order to accommodate long interactions that could use too much memory.
   Therefore, it is important to not delete these images while the recording is taking place, and as they are deleted automatically once recording is complete.

Logging Data
--------------------------

In addition to recording images or video of agent behavior, it is also possible to log the raw simulation
data about state, action, reward etc. in a separate log file. It is also possible to log compilation information
to assist in debugging or for error reporting.

To log information about the RDDL compilation to a file:

.. code-block:: python
	
	env = RDDLEnv.RDDLEnv.build(info, instance, debug=True)

Upon executing this command, a log file is created with the name <domain name>_<instance name>.log in the installation's root directory.
Currently, the following information is written in the generated log file:

* description of pvariables as they are stored in memory (e.g., parameters, data type, data shape)
* dependency graph between CPFs
* calculated order of evaluation of CPFs
* information used by the simulator for operating on pvariables stored as arrays
* simulation bounds for state and action fluents (unbounded or non-box constraints are represented as [-inf, inf])
* for JAX compilation, also prints the JAX compiled expressions corresponding to CPFs, reward and constraint expressions.

To log simulation data to a file:

.. code-block:: python
	
	env = RDDLEnv.RDDLEnv.build(info, instance, log_path='path/to/file')
                            
Upon interacting with the environment, a log file is created in the Logs folder in pyRDDLGym.

Custom Domains
--------------------------

Writing new user defined domains is as easy as writing a few lines of text in a mathematical fashion!
It is only required to specify the lifted constants, variables (all are referred as fluents in RDDL),
behavior/dynamic of the problem and generating an instance with the actual objects and initial state in RDDL - and pyRDDLGym will do the rest.
The syntax for building RDDL domains is described here: :ref:`rddl-description`.


