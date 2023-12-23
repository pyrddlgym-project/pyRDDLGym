Going into Details
===============

Inspecting the Model
-------------------

The pyRDDLGym compiler provides a convenient API for querying a variety of properties about RDDL constructs in a domain.
These can be accessed through the ``model`` field of a ``RDDLEnv``:

.. code-block:: python
	
    info = ExampleManager.GetEnvInfo('MarsRover')
    env = RDDLEnv.RDDLEnv.build(info, 0)
    model = env.model

Below are some commonly used fields of ``model`` that can be accessed directly.
	
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

The ``args()`` function of an ``Expression`` object provides its sub-expressions, 
which can be ``Expression`` instances or collections containing aggregation variables,
types, or other information required by the engine. 

Similarly, the ``etype()`` argument provides identifying information about the expression.

New vs Old API
------

The new and old Gym APIs return ``env.step()`` and ``env.reset()`` in slightly different 
format. This can cause problems when passing the environment to some third-party packages. 

For example, the new API output of ``step()`` produces a tuple ``state, reward, done, fail, info``
where ``fail`` determines if the agent is out of bounds or failed the episode. Similarly, ``reset()`` in the 
new API produces ``state, info``. The old API did not produce the ``fail`` information.

pyRDDLGym offers a convenient flag to switch between the old and the new API:

.. code-block:: python

    info = ExampleManager.GetEnvInfo(domain)
    env = RDDLEnv.RDDLEnv.build(info, instance, new_gym_api=True)

pyRDDLGym computes the auxiliary ``fail`` information by evaluating whether the state invariants
are satisfied in the current state.

Tensor Representation
-------------------

Some algorithms require a tensor representation of states and/or actions. 
The ``RDDLEnv`` class provides a ``vectorized`` option
to work directly with the tensor representations of state and action fluents. 

For example, a ``bool`` action fluent ``put-out(?x, ?y)`` taking two parameters 
``?x`` and ``?y``, with 3 objects each, would be provided as a boolean-valued 
3-by-3 matrix. State fluents also follow this format.

This option can be enabled as follows:

.. code-block:: python

    info = ExampleManager.GetEnvInfo(domain)
    env = RDDLEnv.RDDLEnv.build(info, instance, vectorized=True)

With this option enabled, the bounds of the ``observation_space`` and ``action_space`` 
of the environment are instances of ``gym.spaces.Box`` with the correct shape and dtype.

Building a Custom Visualizer
-------------

In order to build custom visualizations (for new user defined domains), 
inherit the class ``Visualizer.StateViz.StateViz()`` and override the 
``visualizer.render()`` function to produce a PIL image to render to the screen:

.. code-block:: python

    class MyDomainViz(StateViz)
        # here goes the visualization implementation

    env.set_visualizer(MyDomainViz)

.. warning::
   The visualizer argument in ``set_visualizer`` should not contain the customary 
   ``()`` when initializing the visualizer object, since this is done internally.
   So, instead of writing ``env.set_visualizer(MyDomainViz(**MyArgs))``, write 
   ``env.set_visualizer(MyDomainViz, viz_kwargs=MyArgs)``.
  
Logging Debug Data
--------------------------

To log information about the RDDL compilation to a file for debugging, error reporting
or diagnosis:

.. code-block:: python
	
	env = RDDLEnv.RDDLEnv.build(info, instance, debug=True)

A log file will be created with the name <domain name>_<instance name>.log in the installation's root directory.

Currently, the following information is logged:

* description of pvariables as they are stored in memory (e.g., parameters, data type, data shape)
* dependency graph between CPFs
* calculated order of evaluation of CPFs
* information used by the simulator for operating on pvariables stored as arrays
* simulation bounds for state and action fluents (unbounded or non-box constraints are represented as [-inf, inf])
* for JAX compilation, also prints the JAX compiled expressions corresponding to CPFs, reward and constraint expressions.
