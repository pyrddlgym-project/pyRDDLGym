.. _gurobiplan:

pyRDDLGym-gurobi: Planning by Nonlinear Programming with GurobiPlan
===============

In this tutorial, we discuss the compilation of RDDL into a Gurobi mixed-integer program (MIP) for computing optimal actions.
We also discuss how pyRDDLGym-gurobi (or GurobiPlan) uses gurobi to do optimal control.


Installing
-----------------

Before installing GurobiPlan, you will need to obtain a valid gurobi license.
You can then install GurobiPlan and all its requirements via pip:

.. code-block:: shell

    pip install pyRDDLGym-gurobi


Running GurobiPlan
-------------------

From the Command Line
^^^^^^^^^^^^^^^^^^^

From the install directory of pyRDDLGym-gurobi run:

.. code-block:: shell

    python -m pyRDDLGym_gurobi.examples.run_plan <domain> <instance>

where:

* ``<domain>`` is the domain identifier as specified in rddlrepository, or a path pointing to a valid domain file
* ``<instance>`` is the instance identifier in rddlrepository, or a path pointing to a valid instance file.


From Python
^^^^^^^^^^^^^^^^^^^

In Python, instantiate the environment and planner as follows:

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
   An online and offline controller are provided in GurobiPlan mirroring the functionality of JaxPlan.
   Both are instances of pyRDDLGym's ``BaseAgent``, so the ``evaluate()`` function can streamline evaluation.

 
Configuring GurobiPlan
-------------------

The recommended way to manage planner settings is to write a configuration file 
with all the necessary hyper-parameters, which follows the same general format
as JaxPlan, i.e. for straight-line planning:

.. code-block:: shell

    [Gurobi]
    NonConvex=2
    OutputFlag=0

    [Optimizer]
    method='GurobiStraightLinePlan'
    method_kwargs={}
    rollout_horizon=5
    verbose=1

The configuration file contains two sections:

* the ``[Gurobi]`` section dictates `parameters <https://www.gurobi.com/documentation/current/refman/parameters.html>`_ passed to the Gurobi engine
* the ``[Optimizer]`` section contains a ``method`` argument to indicate the type of plan/policy, its hyper-parameters, and other aspects of the optimization like rollout horizon.

The full list of settings that can be specified in the ``[Optimizer]`` section of the configuration file are as follows:

.. list-table:: ``[Optimizer]``
   :widths: 40 80
   :header-rows: 1

   * - Setting
     - Description
   * - allow_synchronous_state
     - Whether state variables can depend on each other synchronously
   * - epsilon
     - Small constant for comparing equality of numbers in Gurobi
   * - float_range
     - Range of floating values in Gurobi
   * - piecewise_options
     - Parameter string to configure Gurobi nonlinear approximation
   * - rollout_horizon
     - Length of the planning horizon
   * - verbose
     - Print nothing(0)/summary(1)/detailed(2) compiler messages

The configuration file can then be parsed and passed to GurobiPlan as follows:

.. code-block:: python
    
    import os
    from pyRDDLGym_gurobi.core.planner import load_config
    
    # pass the parameters to the controller and proceed as usual
    controller_kwargs = load_config("/path/to/config/file")  
    controller = GurobiOnlineController(rddl=env.model, **controller_kwargs)
    ...

.. note::
   You can also pass Gurobi backend parameters by creating a ``gurobi.env`` file in the same
   directory where your launch script is located.

 
Limitations
-------------------

We cite several limitations of the current version of GurobiPlan:

* Stochastic variables introduce computational difficulties since mixed-integer problems are inherently deterministic
	* the planner currently applies determinization, where stochastic variables are replaced with their means
* Discrete non-linear domains can require exponential computation time
	* GurobiPlan uses piecewise linear functions to approximate non-linearities, and quadratic expressions in other cases
	* we recommend reducing the planning horizon, simplying the RDDL as much as possible, and tweaking the Gurobi specific parameters.

Citation
-------------------

If you use the code provided in this repository, please use the following bibtex for citation:

.. code-block:: bibtex

    @inproceedings{
        gimelfarb2024jaxplan,
        title={JaxPlan and GurobiPlan: Optimization Baselines for Replanning in Discrete and Mixed Discrete and Continuous Probabilistic Domains},
        author={Michael Gimelfarb and Ayal Taitler and Scott Sanner},
        booktitle={34th International Conference on Automated Planning and Scheduling},
        year={2024},
        url={https://openreview.net/forum?id=7IKtmUpLEH}
    }

    