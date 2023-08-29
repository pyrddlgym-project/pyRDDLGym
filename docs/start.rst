Getting Started
===============

Initializing Environments
-------------------

Initializing environments in pyRDDLGym is done by instantiating a ``RDDEnv`` object:

.. code-block:: python

    from pyRDDLGym import RDDLEnv
    myEnv = RDDLEnv.RDDLEnv(domain="domain.rddl", instance='instance.rddl')

where ``domain.rddl`` and ``instance.rddl`` are paths pointing to RDDL files of your choosing.

.. note::
   ``RDDLEnv`` objects are instances of ``gym.Env``, and can thus be used in most cases where gym environments are required.

Built-in Environments
-----------------

pyRDDLGym is shipped with a number of varied domains and accompanying instances, whose RDDL files
are included as part of the distribution. These environments can be accessed through the
``ExampleManager``:

.. code-block:: python

    from pyRDDLGym import ExampleManager
    ExampleManager.ListExamples()

The ``ListExample()`` function lists all the example environments included in pyRDDLGym.
Retrieving information about a particular domain/environment ENV is easy:

.. code-block:: python

    EnvInfo = ExampleManager.GetEnvInfo(ENV)

This information can be used to setup an OpenAI gym environment as we did above:

.. code-block:: python

    myEnv = RDDLEnv.RDDLEnv(domain=EnvInfo.get_domain(), instance=EnvInfo.get_instance(0))

Here, the argument of the method ``get_instance(<num>)`` is the ID number of the instance (0 in this case).
Listing all the available instances of the problem can be done as follows:

.. code-block:: python

    EnvInfo.list_instances()

Last, setting up the dedicated visualizer for the example is done via

.. code-block:: python

    myEnv.set_visualizer(EnvInfo.get_visualizer())

Interacting with the Environment
----------------------------

As in all MDP applications, we define policies to interact with an environment by providing actions or controls in each state.
We provide two simple policies as part of pyRDDLGym:

- **NoOpAgent** - returns the default action values specified in the RDDL domain.
- **RandomAgent** - samples a random action according to the env.action_space and the maximum number of allowed concurrent actions as specified in the RDDL file.

To initialize a random agent for example:

.. code-block:: python

    from Policies.Agents import RandomAgent
    agent = RandomAgent(action_space=myEnv.action_space, num_actions=myEnv.NumConcurrentActions)

Now lets see what a complete agent-environment loop looks like in pyRDDLGym.
The example below will run the ``MarsRover`` environment for the amount of time steps specified in the instance ``horizon`` field.
If the ``env.render()`` function is used, we will also see a window pop up rendering the environment:

.. code-block:: python

    from pyRDDLGym import RDDLEnv
    from pyRDDLGym import ExampleManager
    from pyRDDLGym.Policies.Agents import RandomAgent

    # get the environment info
    EnvInfo = ExampleManager.GetEnvInfo('MarsRover')

    # set up the environment class
    # choose instance 0 because every example has at least one example instance
    myEnv = RDDLEnv.RDDLEnv(domain=EnvInfo.get_domain(), instance=EnvInfo.get_instance(0))
    
    # set up the environment visualizer
    myEnv.set_visualizer(EnvInfo.get_visualizer())

    # set up an agent
    agent = RandomAgent(action_space=myEnv.action_space, num_actions=myEnv.NumConcurrentActions)
    
    # perform a roll-out from the initial state
    # until either termination or the horizon is reached
    total_reward = 0
    state = myEnv.reset()
    for _ in range(myEnv.horizon):
          myEnv.render()
          next_state, reward, done, info = myEnv.step(agent.sample_action())
          total_reward += reward
          state = next_state
          if done:
                break
    myEnv.close()

We also provide a convenience ``evaluate`` function for policy evaluation, so it is not necessary to implement the interaction loop explicitly:

.. code-block:: python
	
   total_reward = agent.evaluate(myEnv, episodes=1, render=True)['mean']
  
The above call to ``evaluate`` returns a dictionary of summary statistics about the returns collected on different episodes, such as mean, median, standard deviation, etc.

Spaces
------

The state and action spaces of pyRDDLGym are standard ``gym.spaces``, accessible through the standard API: ``env.state_space`` and ``env.action_space``.
State/action spaces are of type ``gym.spaces.Dict``, where each key-value pair where the key name is the state/action and the value is the state/action current value or action to apply.

Thus, RDDL types are converted to ``gym.spaces`` with the appropriate bounds as specified in the RDDL ``action-preconditions`` and ``state-invariants`` fields. The conversion is as following:

- real -> Box with bounds as specified in action-preconditions, or with np.inf and symmetric bounds.
- int -> Discrete with bounds as specified in action-preconditions, or with np.inf and symmetric bounds.
- bool -> Discrete(2)

There is no need in pyRDDLGym to specify the values of all the existing action in the RDDL domain description, only thus the agent wishes to assign non-default values, the infrastructure will construct the full action vector as necessary with the default action values according to the RDDL description.

Constants
---------

RDDL allows for the constants of the problem instead of being hard-coded, to be specified and in the non-fluent block of the instance.
Meaning every instance can have different constants, e.g., different bounds on action, different static object location, etc.

While these constants are not available through the state of the problem, it is possible to access them through gym (or directly through the RDDL description) with a dedicated API: ``env.non_fluents``.
The non_fluents property returns a dictionary whose keys are the grounded non-fluents and the values are the appropriate values.

Termination
-----------

An addition made to the RDDL language during the development of this infrastructure is the termination block.
The termination block is intended to specify terminal states in the MDP, when reached the simulation will end.
A terminal state is a valid state of the MDP (to emphasize the difference from ``state-invariants``).
An example of terminal state can be any state within the goal set for which the simulation should not continue, or a state where there are no possible actions and the simulation should end 
(e.g., hitting a wall when it is not allowed). 
When a terminal state is reached the state is returned from the environment with the ``done`` flag returned as ``True``.
The reward is handled independently by the reward function, thus if there is a specific reward for the terminal state, it should specified in the reward formula.

The termination block has the following syntax:

.. code-block:: shell

    termination {
        Terminal_condition1;
        Terminal_condition2;
        ...
    };

where ``Terminal_condition#`` are boolean-valued expressions.
The termination decision is a disjunction of all the conditions in the block (i.e. termination if at least one is true).

Visualization
-------------

pyRDDLGym visualization is just like regular Gym, which can be done by calling ``env.render()``.
Every domain has a default visualizer assigned to it, which is either a graphical ``ChartVisualizer`` that plots the state trajectory over time, or a custom domain-dependent implementation.

Assigning a visualizer for an environment can be done by calling the environment method ``env.set_visualizer(viz)`` with ``viz`` as the desired visualization object.

.. code-block:: python

    from pyRDDLGym import RDDLEnv
    from pyRDDLGym import ExampleManager

    EnvInfo = ExampleManager.GetEnvInfo('MarsRover')
    myEnv = RDDLEnv.RDDLEnv(domain=EnvInfo.get_domain(), instance=EnvInfo.get_instance(0))

    # set up the environment visualizer
    myEnv.set_visualizer(EnvInfo.get_visualizer())

In order to build custom visualizations (for new user defined domains), 
one can inherit the class ``Visualizer.StateViz.StateViz()`` and return in the ``visualizer.render()`` method a PIL image for the gym to render to the screen.
The environment initialization has the following general structure:

.. code-block:: python

    from pyRDDLGym import RDDLEnv
    from pyRDDLGym.Visualizer.StateViz import StateViz

    class MyDomainViz(StateViz)
        # here goes the visualization implementation

    myEnv = RDDLEnv.RDDLEnv(domain='myDomain.rddl', instance='myInstance.rddl')

    # set up the environment visualizer
    myEnv.set_visualizer(MyDomainViz)

.. warning::
   The visualizer argument in ``set_visualizer`` should not contain the customary ``()`` when initializing the visualizer object, since this is done internally.
   So, instead of writing ``myEnv.set_visualizer(MyDomainViz(**MyArgs))``, write ``myEnv.set_visualizer(MyDomainViz, viz_kwargs=MyArgs)``.
   
Recording Movies
--------------------------

A ``MovieGenerator`` class is provided to allow capture of videos of agent behavior:

.. code-block:: python

    from pyRDDLGym import RDDLEnv
    from pyRDDLGym.Visualizer.StateViz import StateViz
    from pyRDDLGym.Visualizer.MovieGenerator import MovieGenerator

    # load the environment
    myEnv = RDDLEnv.RDDLEnv(domain='myDomain.rddl', instance='myInstance.rddl')
	
    # set up the movie generator
    movie_gen = MovieGenerator('myFilePath', 'myEnvName', max_frames=1000)
    
    # set up the environment visualizer, passing a movie generator to capture frames
    myEnv.set_visualizer(EnvInfo.get_visualizer(), movie_gen=movie_gen)

    # interact with myEnv as usual
    ...

    # close the environment
    myEnv.close()

Upon calling ``myEnv.close()``, the images captured will be combined into video format and saved to the desired path.
Any temporary files created to capture individual frames during interaction will be deleted from disk.

Custom Domains
--------------------------

Writing new user defined domains is as easy as writing a few lines of text in a mathematical fashion!
It is only required to specify the lifted constants, variables (all are referred as fluents in RDDL),
behavior/dynamic of the problem and generating an instance with the actual objects and initial state in RDDL - and pyRDDLGym will do the rest.
The syntax for building RDDL domains is described here: :ref:`rddl-description`.


