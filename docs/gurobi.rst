Baselines: The Gurobi MIP Planner
===============

In this tutorial, we discuss the compilation of RDDL into a Gurobi mixed-integer program (MIP) for computing optimal actions.
The Gurobi planner can optimize discrete state/action problems where the JAX planner could perform poorly.

Setting up the Gurobi Planner
-------------------

To run the Gurobi planner, you will need to install a valid academic or institutional 
`Gurobi license <https://www.gurobi.com/academia/academic-program-and-licenses/>`_, as well as the Gurobi python package
version 10.0.1 or later

.. code-block:: shell
	
    python -m pip install gurobipy==10.0.1

Running the Gurobi Planner
-------------------

The Gurobi planner is very simple to run, and comes wrapped inside both an online and an offline controller, 
following the structure of the JAX planner.

First, set up the environment as usual, extracting the underlying RDDL model:

.. code-block:: python

    from pyRDDLGym.Core.Env.RDDLEnv import RDDLEnv
    from pyRDDLGym.Examples.ExampleManager import ExampleManager

    info = ExampleManager.GetEnvInfo(domain)    
    env = RDDLEnv.build(info, instance)
    model = env.model

Next, create the plan or policy architecture that we will be optimizing. In this case, we will instantiate 
an open loop plan (a sequence of controls that will be optimized directly):

.. code-block:: python
	
    from pyRDDLGym.Core.Gurobi.GurobiRDDLPlanner import GurobiStraightLinePlan
    plan = GurobiStraightLinePlan()
   
Finally, create the controller, which is an instance of the ``BaseAgent`` class. This means we
can call ``evaluate(env)`` to directly begin optimization on the environment
 
.. code-block:: python

    from pyRDDLGym.Core.Gurobi.GurobiRDDLPlanner import GurobiOnlineController
    controller = GurobiOnlineController(rddl=model, plan=plan, rollout_horizon=5)
    controller.evaluate(env, verbose=True, render=True)
 
Putting this together, we have:

.. code-block:: python

    from pyRDDLGym.Core.Env.RDDLEnv import RDDLEnv
    from pyRDDLGym.Core.Gurobi.GurobiRDDLPlanner import GurobiStraightLinePlan
    from pyRDDLGym.Core.Gurobi.GurobiRDDLPlanner import GurobiOnlineController
    from pyRDDLGym.Examples.ExampleManager import ExampleManager
    
    # create the environment
    info = ExampleManager.GetEnvInfo(domain)    
    env = RDDLEnv.build(info, instance, enforce_action_constraints=True)
    model = env.model
    
    # create and evaluate the controller
    plan = GurobiStraightLinePlan()
    controller = GurobiOnlineController(rddl=model, plan=plan, rollout_horizon=5)
    controller.evaluate(env, verbose=True, render=True)
    
    env.close()
  
Passing Parameters to the Gurobi Backend
-------------------

Gurobi is by its nature highly `configurable <https://www.gurobi.com/documentation/current/refman/parameters.html>`_. 
Parameters can be passed to the Gurobi model through a ``gurobi.env`` file, or directly through the pyRDDLGym interface.

To understand the first approach, suppose we wish to instruct Gurobi to limit each optimization to 60 seconds, 
and to print progress during optimization to console. Create a ``gurobi.env`` file in the same
directory where the launch script is located, and with the following content:

.. code-block:: shell

    TimeLimit 60
    OutputFlag 1
 
In the second approach, you can alternatively pass these parameters as a dictionary to the 
``model_params`` argument of the controller instance:

.. code-block:: python

    controller = GurobiOnlineController(rddl=model, plan=plan, rollout_horizon=5,
                                        model_params={'TimeLimit': 60, 'OutputFlag': 1})

An online and offline controller type are provided in pyRDDLGym, which mirror the functionality of the JAX
planner discussed previously.

Current Limitations
-------------------

We cite several limitations of the current baseline JAX optimizer:

* Stochastic variables introduce computational difficulties since mixed-integer problems are inherently deterministic
	* the planner currently applies determinization, where stochastic variables are substituted with their means (we hope to incorporate more sophisticated techniques from optimization to better deal with stochasticity)
* Discrete non-linear domains can require exponential computation time.
	* the planner uses piecewise linear functions to approximate non-linearities, and quadratic expressions in other cases
	* if the planner does not make progress, we recommend reducing the planning horizon, simplying the RDDL description as much as possible, or tweaking the parameters of the Gurobi model.
