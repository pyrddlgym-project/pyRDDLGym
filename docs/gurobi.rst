.. _gurobiplan:

pyRDDLGym-gurobi: Planning by Nonlinear Programming with GurobiPlan
===============

In this tutorial, we discuss the compilation of RDDL into a Gurobi mixed-integer program (MIP) for computing optimal actions.
We also discuss how pyRDDLGym-gurobi (or GurobiPlan as it is referred to in the literature) uses gurobi to build optimal controllers.


Installing
-----------------

Before installing GurobiPlan, you will need to obtain a valid gurobi license.
You can then install GurobiPlan and all of its requirements via pip:

.. code-block:: shell

    pip install pyRDDLGym-gurobi


Running the Basic Example
-------------------

The basic example provided in pyRDDLGym-gurobi will run the Gurobi planner on a 
domain and instance of your choosing. To run this, navigate to the install directory of pyRDDLGym-gurobi, and run:

.. code-block:: shell

    python -m pyRDDLGym_gurobi.examples.run_plan <domain> <instance>

where:

* ``<domain>`` is the domain identifier as specified in rddlrepository, or a path pointing to a valid domain.rddl file
* ``<instance>`` is the instance identifier in rddlrepository, or a path pointing to a valid instance.rddl file.


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
   An online and offline controller type are provided in GurobiPlan, 
   which mirror the functionality of the JAX planner discussed in the previous section.
   Both are instances of pyRDDLGym's ``BaseAgent``, so the ``evaluate()`` 
   function can be used to streamline evaluation.

 
Configuring GurobiPlan
-------------------

The recommended way to manage planner settings is to write a configuration file 
with all the necessary hyper-parameters, which follows the same general format
as for the JAX planner. Below is the basic structure of a configuration file for straight-line planning:

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

The configuration file can then be parsed and passed to the planner as follows:

.. code-block:: python
    
    import os
    from pyRDDLGym_gurobi.core.planner import load_config
    
    # load the config
    abs_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(abs_path, 'default.cfg') 
    controller_kwargs = load_config(config_path)  
    
    # pass the parameters to the controller and proceed as usual
    controller = GurobiOnlineController(rddl=env.model, **controller_kwargs)
    ...

.. note::
   You can also pass Gurobi backend parameters by creating a ``gurobi.env`` file in the same
   directory where your launch script is located. However, we no longer recommend this approach.


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

 
Current Limitations
-------------------

We cite several limitations of the current baseline Gurobi optimizer:

* Stochastic variables introduce computational difficulties since mixed-integer problems are inherently deterministic
	* the planner currently applies determinization, where stochastic variables are substituted with their means (we hope to incorporate more sophisticated techniques from optimization to better deal with stochasticity)
* Discrete non-linear domains can require exponential computation time
	* the planner uses piecewise linear functions to approximate non-linearities, and quadratic expressions in other cases
	* if the planner does not make progress, we recommend reducing the planning horizon, simplying the RDDL description as much as possible, or tweaking the parameters of the Gurobi model.

Citations
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

    