Baselines: The Gurobi MIP Planner
===============

In this tutorial, we discuss the limitations of the JAX planner and present an alternative 
framework that automatically compiles RDDL code into a Gurobi mixed-integer program (MIP)
that can be optimized to compute optimal controls.

Limitations of the JAX Planner
-------------------

While the JAX planner is highly scalable and adaptable to high dimensional problems, and its 
automatic continuous model relaxations even allow many discrete problems to be quickly optimized,
it's performance suffers when the problem structure is entirely discrete. 

To diagnose this issue, it is advisable to compare the training loss to the test loss at the time of convergence.
A low, or drastically improving, training loss with a similar test loss indicates that the continuous model relaxation
is accurate around the optimal plan or policy. On the other hand, a low training loss coupled with a high test loss 
indicates that the continuous model relaxation is poor, and it is likely that the solution obtained will be poor as well.

Setting up the Gurobi Planner
-------------------

The Gurobi planner can optimize many discrete state/action problems where the JAX planner performs poorly as indicated above.
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
can call ``evaluate(env)`` to directly begin optimization on the environment!
 
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

To understand the first approach, suppose we wish to subject each Gurobi optimization to a time limit of 60 seconds, 
as to print progress during optimization to console. You can specify this by creating a ``gurobi.env`` file in the same
directory where your launch script is located, with the following content:

.. code-block:: shell

    TimeLimit 60
    OutputFlag 1
 
To understand the second approach, you can alternatively pass these parameters as a dictionary to the 
``model_params`` argument of the controller instance:

.. code-block:: python

    controller = GurobiOnlineController(rddl=model, plan=plan, rollout_horizon=5,
                                        model_params={'TimeLimit': 60, 'OutputFlag': 1})

An online and offline controller type are provided in pyRDDLGym, which mirror the functionality of the JAX
planner discussed previously.

Current Limitations
-------------------

We cite several limitations of the current baseline JAX optimizer:

* Stochastic variables introduce computational difficulties since mixed-integer problems are inherently deterministic.
	* The Gurobi planner currently applies determinization, where stochastic variables are substituted with their means. We hope to incorporate more sophisticated techniques from optimization to better deal with stochasticity.
* Discrete non-linear domains can require exponential computation time.
	* The Gurobi planner uses piecewise linear functions to approximate non-linearities, and quadratic expressions in other cases. Non-linear mixed-integer programs cannot be solved in polynomial time, so the plananer can get stuck if the state/action dimension are high.
	* We recommend reducing the planning horizon, simplying the RDDL description as much as possible, and tweaking the parameters of the Gurobi model if the planner does not make progress on your problem.
