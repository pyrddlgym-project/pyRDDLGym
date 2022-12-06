Getting Started
===============

Initializing Environments
-------------------

Initializing environments is very easy in pyRDDLGym and can be done via:

.. code-block:: python
    from pyRDDLGym import RDDLEnv
    myEnv = RDDLEnv.RDDLEnv(domain="domain.rddl", instance='instance.rddl')

where ``domain.rddl`` and ``instance.rddl`` are RDDL files of your choosing.

Built in environments
-----------------
pyRDDLGym is shipped with 12 environments designed completely in RDDL.
The RDDL files are part of the distribution and can be accessed.
In order to use the built in environments and keep the api of the RDDLEnv standard we supply an ExampleManager class:

.. code-block:: python
    from pyRDDLGym import ExampleManager
    ExampleManager.ListExamples()

The ``ListExample()`` static function lists all the example environments in pyRDDLGym Then in order to retrieve the information of a specific environment:

.. code-block:: python
    EnvInfo = ExampleManager.GetEnvInfo(ENV)

Where ENV is a string name of the desired example.
Setting up an environment at the point is just:

.. code-block:: python
    myEnv = RDDLEnv.RDDLEnv(domain=EnvInfo.get_domain(), instance=EnvInfo.get_instance(0))

Where the argument of the method ``get_instance(<num>)`` is the ID number of the instance (0 in this case).
Listing all the available instances of the problem is accessed via

.. code-block:: python
    EnvInfo.list_instances()

Last, setting up the dedicated visualizer for the example is done via

.. code-block:: python
    myEnv.set_visualizer(EnvInfo.get_visualizer())


Interacting with the Environment
----------------------------
pyRDDLGym is build on Gym as so implements the classic “agent-environment loop”. The infrastructure comes with two simple agents:

- **NoOpAgent** - which allows the environment to evolve according to the default behavior as specified in the RDDL file.
- **RandomAgent** - which sends a rendom action according to the env.action_space and the maximum number of allowed concurrent actions as specified in the RDDL file.

Using a pre existing agent, or using of of your own is as simple as:

.. code-block:: python
    from Policies.Agents import RandomAgent
    agent = RandomAgent(action_space=myEnv.action_space, num_actions=myEnv.NumConcurrentActions)

Lets see what a complete the agent-environment loop looks like in pyRDDLGym.
This example will run the example ``MarsRover``. The loop will run for the amount of time steps specified in the environment’s ``horizon`` field.
If the ``env.render()`` function will be used we will also see a window pop up rendering the environment

.. code-block:: python
    from pyRDDLGym import RDDLEnv
    from pyRDDLGym import ExampleManager
    from pyRDDLGym.Policies.Agents import RandomAgent

    # get the environment info
    EnvInfo = ExampleManager.GetEnvInfo('MarsRover')

    # set up the environment class, choose instance 0 because every example has at least one example instance
    myEnv = RDDLEnv.RDDLEnv(domain=EnvInfo.get_domain(), instance=EnvInfo.get_instance(0))
    # set up the environment visualizer
    myEnv.set_visualizer(EnvInfo.get_visualizer())

    # set up an aget
    agent = RandomAgent(action_space=myEnv.action_space, num_actions=myEnv.NumConcurrentActions)

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

Spaces
------
The state and action spaces of pyRDDLGym are standard ``gym.spaces``, and inquirable through the standard API: ``env.state_space`` and ``env.action_space``.
State/action spaces are of type ``gym.spaces.Dict``, where each key-value pair where the key name is the state/action and the value is the state/action current value or action to apply.

Thus, RDDL types are converted to ``gym.spaces`` with the appropriate bounds as specified in the RDDL ``action-preconditions`` and ``state-invariants`` fields. The conversion is as following:

- real -> Box with bounds as specified in action-preconditions, or with np.inf and symetric bounds.
- int -> Discrete with bounds as specified in action-preconditions, or with np.inf and symetric bounds.
- bool -> Discrete(2)

There is no need in pyRDDLGym to specify the values of all the existing action in the RDDL domain description, only thus the agent wishes to assign non-default values, the infrastructure will construct the full action vector as necessary with the default action values according to the RDDL description.

Note: enum types are not supported by pyRDDLGym at this stage.

Constants
---------

RDDL allows for the constants of the problem instead of being hardcoded, to be specified and in the non-fluent block of the instance.
Meaning every instance can have different constants, e.g., different bounds on action, different static object location, etc.

While these constants are not available through the state of the problem, it is possible to access them through gym (or directly through the RDDL description) with a dedicated API: ``env.non_fluents``.
The non_fluents property returns a python dictionary where the keys are the grounded non-fluents and the values are the appropriate values.

Termination
-----------

An Addition made to the RDDL language during the development of this infrastructure is the termination block.
The termination block is intended to specify terminal states in the MDP, when reached the simulation will end.
A terminal state is a valid state of the MDP (to emphasize the difference from ``state-invariants``).
An example of terminal state can be any state within the goal set for which the simulation should not continue, or a state where there are no possible actions and the simulation should end.
E.g., hitting a wall when it is not allowed. When a terminal state is reached the state is returned from the environment and the ``done`` flag is returned as True.
The reward is handled independently by the reward function, thus if there is a specific reward for the terminal state, it should specified in the reward formula.
The termination block has the following syntax:

.. code-block:: shell
    termination {
        Terminal_condition1;
        Terminal_condition2;
        ...
    };

where ``Terminal_condition#`` are boolean formulas.
The termination decision is a disjunction of all the conditions in the block (termination if at least one is True).

Visualization
-------------

pyRDDLGym visualization is just like regular Gym.
Users can visualize the current state of the simulation by calling ``env.render()``.
The standard visualizer that comes out of the box with every pyRDDLGym domain (even used defined domain will have it without explicitly doing anything) is the TextViz.
TextViz just renders an image with textual description of the states and their current values.

Replacing the built is TextViz is simple as calling the environment method ``env.set_visualizer(viz)`` with ``viz`` as the desired visualization object.

.. code-block:: python
    from pyRDDLGym import RDDLEnv
    from pyRDDLGym import ExampleManager

    EnvInfo = ExampleManager.GetEnvInfo('MarsRover')
    myEnv = RDDLEnv.RDDLEnv(domain=EnvInfo.get_domain(), instance=EnvInfo.get_instance(0))

    # set up the environment visualizer
    myEnv.set_visualizer(EnvInfo.get_visualizer())

In order to build custom visualizations (for new user defined domains),
one just need to inherit the class ``Visualizer.StateViz.StateViz()`` and return in the ``visualizer.render()`` method a PIL image for the gym to render to the screen.
The environment initialization will look something like that:

.. code-block:: python
    from pyRDDLGym import RDDLEnv
    from pyRDDLGym.Visualizer.StateViz import StateViz

    class MyDomainViz(StateViz)
    # here goes the visualization implementation


    myEnv = RDDLEnv.RDDLEnv(domain='myDomain.rddl', instance='myInstance.rddl')

    # set up the environment visualizer
    myEnv.set_visualizer(MyDomainViz)

Custom Domains
--------------------------

Writing new user defined domains is as easy as writing a few lines of text in a mathematical fashion!
All is required is to specify the lifted constants, variables (all are referred as fluents in RDDL),
behavior/dynamic of the problem and generating an instance with the actual objects and initial state in RDDL - and pyRDDLGym will do the rest.


