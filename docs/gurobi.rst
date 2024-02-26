.. _gurobiplan:

pyRDDLGym-gurobi: Gurobi MIP Compiler and Planner
===============

In this tutorial, we discuss the compilation of RDDL into a Gurobi mixed-integer program (MIP) for computing optimal actions.
The Gurobi planner can optimize discrete state/action problems where the JAX planner could perform poorly.


Requirements
------------
This package requires Python 3.8+

* pyRDDLGym>=2.0
* gurobipy>=10.0.1


Installing via pip
-----------------

You can install pyRDDLGym-gurobi and all of its requirements via pip:

.. code-block:: shell

    pip install git+https://github.com/pyrddlgym-project/pyRDDLGym-gurobi


Running the Basic Example
-------------------

The basic example provided in pyRDDLGym-gurobi will run the Gurobi planner on a 
domain and instance of your choosing. To run this, navigate to the install directory of pyRDDLGym-gurobi, and run:

.. code-block:: shell

    python -m pyRDDLGym_gurobi.examples.run_plan <domain> <instance> <horizon>

where:

* ``<domain>`` is the domain identifier as specified in rddlrepository, or a path pointing to a valid domain.rddl file
* ``<instance>`` is the instance identifier in rddlrepository, or a path pointing to a valid instance.rddl file
* ``<horizon>`` is the lookahead horizon used by the planner.


Running from the Python API
-------------------

If you are working with the Python API, you can instantiate the environment and planner as follows:

.. code-block:: python

    import pyRDDLGym
    from pyRDDLGym_gurobi.core.planner import GurobiStraightLinePlan, GurobiOnlineController

    # Create the environment
    env = pyRDDLGym.make("domain", "instance")

    # Create the planner
    plan = GurobiStraightLinePlan()
    controller = GurobiOnlineController(rddl=env.model, plan=plan, rollout_horizon=5)

    # Run the planner
    controller.evaluate(env, episodes=1, verbose=True, render=True)
	
    env.close()
		
.. note::
   An online and offline controller type are provided in pyRDDLGym-gurobi, 
   which mirror the functionality of the JAX planner discussed in the previous section.
   Both are instances of pyRDDLGym's ``BaseAgent``, so the ``evaluate()`` 
   function can be used to streamline evaluation.

  
Passing Parameters to the Gurobi Backend
-------------------

Gurobi is by its nature highly `configurable <https://www.gurobi.com/documentation/current/refman/parameters.html>`_. 
Parameters can be passed to the Gurobi model through a ``gurobi.env`` file, or directly through the pyRDDLGym interface.

Suppose we wish to instruct Gurobi to limit each optimization to 60 seconds, 
and to print progress during optimization to console. 
To apply the first approach, create a ``gurobi.env`` file in the same
directory where your launch script is located, with the following content:

.. code-block:: shell

    TimeLimit 60
    OutputFlag 1
 
To apply the second approach, you can pass these parameters as a dictionary to the 
``model_params`` argument of the controller instance:

.. code-block:: python

    controller = GurobiOnlineController(rddl=model, plan=plan, rollout_horizon=5,
                                        model_params={'TimeLimit': 60, 'OutputFlag': 1})

Current Limitations
-------------------

We cite several limitations of the current baseline Gurobi optimizer:

* Stochastic variables introduce computational difficulties since mixed-integer problems are inherently deterministic
	* the planner currently applies determinization, where stochastic variables are substituted with their means (we hope to incorporate more sophisticated techniques from optimization to better deal with stochasticity)
* Discrete non-linear domains can require exponential computation time
	* the planner uses piecewise linear functions to approximate non-linearities, and quadratic expressions in other cases
	* if the planner does not make progress, we recommend reducing the planning horizon, simplying the RDDL description as much as possible, or tweaking the parameters of the Gurobi model.
