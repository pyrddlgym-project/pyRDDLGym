Getting Started: Advanced Tools
===============

Inspecting the Model
-------------------

The pyRDDLGym compiler provides a convenient API for querying a variety of properties about RDDL constructs in a domain.
These can be accessed through the ``model`` field of a ``RDDLEnv``:

.. code-block:: python
	
    import pyRDDLGym
    env = pyRDDLGym.make("Cartpole_Continuous_gym", "0")
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
   * - ``variable_params``
     - dict of parameters and their types for each variable
   * - ``type_to_objects``
     - dict of all defined objects for each type
   * - ``non_fluents``
     - dict of initial values for each non-fluent
   * - ``state_fluents``
     - dict of initial values for each state-fluent
   * - ``action_fluents``
     - dict of default values for each action-fluent
   * - ``interm_fluents``
     - dict of initial values for each interm-fluent
   * - ``observ_fluents``
     - dict of initial values for each observ-fluent
   * - ``cpfs``
     - dict of ``Expression`` objects for each cpf
   * - ``reward``
     - ``Expression`` object for reward function
   * - ``preconditions``
     - list of ``Expression`` objects for each action-precondition
   * - ``invariants``
     - list of ``Expression`` objects for each state-invariant

``Expression`` objects are abstract syntax trees that describe the flow of computations
in each cpf, constraint relation, or the reward function of the RDDL domain:
- the ``etype()`` function provides basic information about the expression, such as its type
- the ``args()`` function provides its sub-expressions, which consists of other ``Expression`` objects, aggregation variables, or other information required by the engine.

Grounding a Domain
------

By default, pyRDDLGym simulates domains in a tensored manner directly from the (lifted) domain description. 
Specifically, parameterized variables are represented internally as numpy arrays,
whose values are propagated in a vectorized manner through mathematical expressions.

However, sometimes it is required to work with the grounded representation. For example, 
given a p-variable ``some-var(?x, ?y)`` of two parameters ``?x`` and ``?y``, and the expression
``cpf'(?x, ?y) = some-var(?x, ?y) + 1.0;``, the grounded representation is as follows:

.. code-block:: shell

    cpf___x1__y1' = some-var___x1__y1 + 1.0;
    cpf___x1__y2' = some-var___x1__y2 + 1.0;
    cpf___x2__y1' = some-var___x2__y1 + 1.0;
    cpf___x2__y2' = some-var___x2__y2 + 1.0;
    ...

where ``x1, x2...`` are the values of ``?x`` and ``y1, y2...`` are the values of ``?y``.
In other words, all p-variables are replaced by sets of non-parameterized variables (one per valid combination of objects),
and all expressions are replaced by sets of expressions whose p-variable dependencies are replaced by their non-parameterized
counterparts. In all cases, the grounded and lifted representations should produce the same numerical results, 
albeit in a slightly different format.
 
pyRDDLGym provides a convenient class for producing a grounded model from a lifted domain representation, as shown below:

.. code-block:: python
    
    from pyRDDLGym.core.grounder import RDDLGrounder
    grounded = RDDLGrounder(env.model._AST).ground()

The ``grounded`` object returned is also an environment model, 
so the properties discussed in the table at the top of the page work interchangeably with grounded and lifted models.

.. warning::
   Currently, the grounder only supports RDDL 1.0 syntax. Therefore, it does not ground
   expressions containing switch statements, matrix operations, vector distributions and Discrete, 
   nested pvariables, and parameters that are not located inside a pvariable 
   (e.g. statements like ``?x == ?y``). There is currently (limited) support for enumerated types.
   
   
Vectorized Input and Output
-------------------

Some algorithms require a tensor representation of parameterized state-fluent outputs and/or action-fluent inputs. 
The ``RDDLEnv`` class provides a ``vectorized`` option
to allow tensor representations of state and action fluents to be passed into and out of pyRDDLGym. 

For example, a ``bool`` action-fluent ``put-out(?x, ?y)`` taking two parameters 
``?x`` and ``?y``, with 3 objects each, would be provided as a boolean-valued 
3-by-3 matrix. State-fluents would be returned in a similar format from the environment.

This option can be enabled as follows:

.. code-block:: python
	
    import pyRDDLGym
    env = pyRDDLGym.make("Cartpole_Continuous_gym", "0", vectorized=True)

With this option enabled, the bounds of the ``observation_space`` and ``action_space`` 
of the environment are instances of ``gymnasium.spaces.Box`` with the correct shape and dtype.

Exception Handling
------

By default, ``evaluate()`` will not raise errors if action preconditions or state invariants are violated.
State invariant violations are stored in the ``truncated`` field returned by ``env.step()``. If you wish to enforce action
constraints, simply initialize your environment like this:

.. code-block:: python
	
    import pyRDDLGym
    env = pyRDDLGym.make("Cartpole_Continuous_gym", "0", enforce_action_constraints=True)

By default, ``evaluate()`` will not raise an exception if a numerical error occurs during an intermediate calculation,
such as divide by zero or under/overflow. This behavior can be controlled through numpy. 

For example, if you wish to raise/catch all numerical errors, you can add the following lines
before calling ``env.evaluate()``:

.. code-block:: python

    import numpy as np
    np.seterror(all='raise')

More details about controlling error handling behavior can be found 
`here <https://numpy.org/doc/stable/reference/generated/numpy.seterr.html>`_.

.. warning::
   Currently, branched error handling in operations such as ``if`` and ``switch`` 
   is incompatible with vectorized computation. To illustrate, an expression like
   ``if (pvar(?x) == 0) then default(?x) else 1.0 / pvar(?x)`` will evaluate ``1.0 / pvar(?x)`` first
   for all values of ``?x``, regardless of the branch condition, and will thus trigger an exception if ``pvar(?x) == 0``
   for some value of ``?x``. For the time being, we recommend suppressing errors as described above.

Generating Debug Logs
--------------------------

To log information about the RDDL compilation to a file for debugging, error reporting
or diagnosis:

.. code-block:: python
	
    import pyRDDLGym
    env = pyRDDLGym.make("Cartpole_Continuous_gym", "0", debug_path="\path\to\log\file")

where ``debug_path`` is the full path to the debug file minus the extension.
A log file will be created in the specified path with the ``.log`` extension.

Currently, the following information is logged:

* description of pvariables as they are stored in memory (e.g., parameters, data type, data shape)
* dependency graph between CPFs
* calculated order of evaluation of CPFs
* information used by the simulator for operating on pvariables stored as arrays
* simulation bounds for state and action fluents (unbounded or non-box constraints are represented as [-inf, inf])
* if you are using ``pyRDDLGym-jax``, the computation graphs will also be logged
* if you are using ``pyRDDLGym-rl``, the observation and action spaces information will also be logged

Running pyRDDLGym through TCP
-------------------

Some older algorithms and infrastructure built around the Java rddlsim required 
a TCP connection with a server that provides the environment interaction.
pyRDDLGym provides a ``RDDLSimServer`` class that functions in a similar way.

To create and run a server built around a specific domain or instance:

.. code-block:: python
	
    from pyRDDLGym.core.server import RDDLSimServer	
    server = RDDLSimServer("/path/to/domain.rddl", "/path/to/instance.rddl", rounds, time, port=2323)
    server.run()	
	
The ``rounds`` specifies the number of epsiodes/rounds of simulation to perform,
and ``time`` specifies the time the server connection should remain open. The optional ``port``
parameter allows multiple connections to be established in parallel at different ports. 
Finally, the ``run()`` command starts the server.


The pyRDDLGym Compiler (for Advanced Users)
-------------------

.. warning::
   This section is intended for advanced users who wish to become familiar with the 
   backend of pyRDDLGym. We can only provide limited support for backend-related 
   topics (bug reports and pull requests are always welcome), 
   and the API can change at any time making this guide obsolete at times.

At the lowest level of abstraction of pyRDDLGym, syntactic analysis is performed statically whenever possible 
for optimal performance, due to the fact that pyRDDLGym is written purely in Python
and does not (currently) provide C/C++ binaries.

The syntax analysis and compilation of RDDL description files into Python objects 
happens statically when the environment is created, which are directly referenced 
by the user-facing API (e.g. the ``RDDLEnv``, simulator, optimizers, etc.). 
These objects include outputs from several distinct stages of compilation:

.. list-table:: Compiler Components
   :widths: 60 120
   :header-rows: 1
   
   * - name 
     - description
   * - `RDDLParser <https://github.com/pyrddlgym-project/pyRDDLGym/blob/main/pyRDDLGym/core/parser/parser.py>`_
     - Parses a RDDL description file into an intermediate AST.
   * - `RDDLPlanningModel <https://github.com/pyrddlgym-project/pyRDDLGym/blob/main/pyRDDLGym/core/compiler/model.py>`_
     - Converts a parsed AST into a user-friendly model object.
   * - `RDDLValueInitializer <https://github.com/pyrddlgym-project/pyRDDLGym/blob/main/pyRDDLGym/core/compiler/initializer.py>`_
     - Compiles the initial values of all pvariables into numerical arrays.
   * - `RDDLLevelAnalysis <https://github.com/pyrddlgym-project/pyRDDLGym/blob/main/pyRDDLGym/core/compiler/levels.py>`_
     - Summarizes the dependencies between CPFs, and computes their order of evaluation.
   * - `RDDLObjectsTracer <https://github.com/pyrddlgym-project/pyRDDLGym/blob/main/pyRDDLGym/core/compiler/tracer.py>`_
     - Traces the RDDL AST to compile type information about each subexpression, and does type checking.

The following code illustrates the parsing of a domain description, 
returning a ``RDDL`` object that represents its AST representation:

.. code-block:: python
    
    from pyRDDLGym.core.parser.reader import RDDLReader	
    from pyRDDLGym.core.parser.parser import RDDLParser
    
    rddl_string = RDDLReader("\path\to\domain.rddl", "\path\to\instance.rddl").rddltxt
    parser = RDDLParser()
    parser.build()
    ast = parser.parse(rddl_string)

The AST can then be passed to a ``RDDLPlanningModel``, which compiles
the AST into a user-friendly API with accessible properties and functions 
for operating on and modifying the (lifted) domain:

.. code-block:: python

    from pyRDDLGym.core.compiler.model import RDDLLiftedModel	
    model = RDDLLiftedModel(ast)

It is now possible to extract the initial values of the pvariables by using a ``RDDLValueInitializer``,
which reads from the ``init-fluents`` block in the instance whenever possible, and
otherwise from the ``default`` values in the domain:

.. code-block:: python

    from pyRDDLGym.core.compiler.initializer import RDDLValueInitializer	
    values = RDDLValueInitializer(model).initialize()

It is also possible to compute the graph that summarizes the pvariables/CPFs on which each CPF depends,
and run a topological sort on the graph to determine the correct order of CPF evaluation:

.. code-block:: python

    from pyRDDLGym.core.compiler.levels import RDDLLevelAnalysis    
    sorter = RDDLLevelAnalysis(model)
    dependencies = sorter.build_call_graph()
    levels = sorter.compute_levels()

Finally, the code can be traced to compile static type information about each subexpression in the AST, which 
includes for example any free parameters in the scope of the subexpression and their types,
the type of the value returned by the subexpression during evaluation,
or information that is expensive to compute dynamically during simulation 
(e.g. instructions how to operate on pvariables stored as tensors):
 
.. code-block:: python
    
    from pyRDDLGym.core.compiler.tracer import RDDLObjectsTracer    
    trace_info = RDDLObjectsTracer(model, cpf_levels=levels).trace()

This creates an object of type ``RDDLTracedObjects`` which can be queried for 
compiled information about each subexpression in the AST, i.e.:

.. code-block:: python
    
    trace_info.cached_objects_in_scope(expr)   # list free parameters in scope
    trace_info.cached_object_type(expr)        # type of the value returned (None if primitive)
    trace_info.cached_is_fluent(expr)          # whether expr is fluent (returned value can change over time)
    trace_info.cached_sim_info(expr)           # low-level instructions for operating on returned value tensor

For debugging purposes, it is also possible to decompile the model representation
back into a RDDL language string:

.. code-block:: python

    from pyRDDLGym.core.debug.decompiler import RDDLDecompiler
    rddl_string = RDDLDecompiler().decompile_domain(model)

    
