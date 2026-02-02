.. _jaxplan:

pyRDDLGym-jax: Gradient-Based Simulation and Planning with JaxPlan
===============

In this tutorial, we discuss how a RDDL model can be automatically compiled into a differentiable JAX simulator. 
We also show how pyRDDLGym-jax (or JaxPlan) leverages gradient-based optimization for optimal control. 

Installing
-----------------

To install the **bare-bones** version of JaxPlan with minimum installation requirements:

.. code-block:: shell

    pip install pyRDDLGym-jax

To install JaxPlan with the automatic **hyper-parameter tuning and rddlrepository**:
    
.. code-block:: shell

    pip install pyRDDLGym-jax[extra]

(Since version 1.0) To install JaxPlan with the **visualization dashboard**:

.. code-block:: shell

    pip install pyRDDLGym-jax[dashboard]

(Since version 1.0) To install JaxPlan with **all options**:

.. code-block:: shell

    pip install pyRDDLGym-jax[extra,dashboard]
    
To install the **pre-release version** via git:

.. code-block:: shell

    pip install git+https://github.com/pyrddlgym-project/pyRDDLGym-jax.git


Simulating Environments using JAX
-------------------

pyRDDLGym ordinarily simulates domains using numPy.
If you require additional structure such as gradients, or better simulation performance, 
switch to a JAX simulation backend:

.. code-block:: python
	
	import pyRDDLGym
	from pyRDDLGym_jax.core.simulator import JaxRDDLSimulator
	env = pyRDDLGym.make("CartPole_Continuous_gym", "0", backend=JaxRDDLSimulator)
	
.. note::
   All RDDL syntax (both new and old) is supported in the RDDL-to-JAX compiler. 
   In almost all cases, the JAX backend should return numerical results identical to the default backend.
   However, not all operations can support gradients (see Limitations).

.. raw:: html 

   <a href="notebooks/accelerating_simulation_with_jax.html"> 
       <img src="_static/notebook_icon.png" alt="Jupyter Notebook" style="width:64px;height:64px;margin-right:5px;margin-top:5px;margin-bottom:5px;">
       Related example: Accelerating simulation with JAX.
   </a>
   

Background on Differentiable Planning 
-------------------

Open-Loop Planning
^^^^^^^^^^^^^^^^^^^

The open-loop planning problem for a deterministic environment seeks a sequence of actions (plan)
that maximize accumulated reward over a fixed horizon

.. math::

	\max_{a_1, \dots a_T} \sum_{t=1}^{T} R(s_t, a_t), \quad s_{t + 1} = f(s_t, a_t)
	
If the state and action spaces are continuous, and f and R are differentiable, 
`gradient ascent can optimize the actions <https://proceedings.neurips.cc/paper/2017/file/98b17f068d5d9b7668e19fb8ae470841-Paper.pdf>`_.
Specifically, given learning rate :math:`\eta`, gradient ascent updates the plan
:math:`a_\tau'` at decision epoch :math:`\tau` as

.. math::
	
	a_{\tau}' = a_{\tau} + \eta \sum_{t=1}^{T} \nabla_{a_\tau} R(s_t, a_t),
	
where the gradient of the reward at all times :math:`t \geq \tau` is computed by automatic differentiation in JAX.

Closed-Loop Planning
^^^^^^^^^^^^^^^^^^^

An open-loop plan could be sub-optimal by failing to correct for deviations in the state trajectory from its anticipated course.
One solution is to "replan" periodically or at each decision epoch. 
Another solution is to compute a closed-loop `deep reactive policy network <https://ojs.aaai.org/index.php/AAAI/article/view/4744>`_ :math:`a_t \gets \pi_\theta(s_t)`.
JaxPlan supports both options.

Stochastic Reparameterization Trick
^^^^^^^^^^^^^^^^^^^

A secondary problem is that the gradients of stochastic samples are not well-defined.
JaxPlan works around this by using the reparameterization trick, 
i.e. writing :math:`s_{t+1} = \mathcal{N}(s_t, a_t^2)` as :math:`s_{t+1} = s_t + a_t * \mathcal{N}(0, 1)`, 
where the latter is amenable to backprop while the first is not.

The reparameterization trick can be generalized, assuming there exists a closed-form function f such that

.. math::

    s_{t+1} = f(s_t, a_t, \xi_t)
    
and :math:`\xi_t` are random variables drawn from some distribution independent of states and actions. 
For a detailed discussion of reparameterization in the context of planning, 
please see `this paper <https://ojs.aaai.org/index.php/AAAI/article/view/4744>`_ 
or `this paper <https://ojs.aaai.org/index.php/AAAI/article/view/21226>`_.

JaxPlan automatically reparameterizes whenever possible. 
For Bernoulli, Discrete and related distributions on finite support, it applies 
the `Gumbel-softmax trick <https://arxiv.org/pdf/1611.01144.pdf>`_. 
For other distributions without natural reparameterization 
(i.e. Poisson, Binomial), JaxPlan applies `various differentiable relaxations <https://github.com/pyrddlgym-project/pyRDDLGym-jax?tab=readme-ov-file#citing-jaxplan>`_ 
to approximate the gradients.

.. note::
   As of JaxPlan version 3.0, most discrete and continuous distributions support gradients (approximate when required). 
   The notable exception is Multinomial which does not yet support non-zero gradients.


Running JaxPlan
-------------------

.. _jax-intro:

From the Command Line
^^^^^^^^^^^^^^^^^^^

A command line app is provided to run JaxPlan on a specific problem instance:

.. code-block:: shell
    
    jaxplan plan <domain> <instance> <method> --episodes <episodes>
    
where:

* ``<domain>`` is the domain identifier in rddlrepository, or a path pointing to a valid domain file
* ``<instance>`` is the instance identifier in rddlrepository, or a path pointing to a valid instance file
* ``<method>`` is the planning method to use (i.e. drp, slp, replan) or a path to a valid config file
* ``<episodes>`` is the (optional) number of episodes to evaluate the final policy.

The ``<method>`` parameter describes the type of planning representation:

* ``slp`` is the `straight-line plan <https://proceedings.neurips.cc/paper/2017/file/98b17f068d5d9b7668e19fb8ae470841-Paper.pdf>`_
* ``drp`` is the `deep reactive policy network <https://ojs.aaai.org/index.php/AAAI/article/view/4744>`_ 
* ``replan`` uses replanning at every decision epoch
* any other argument is interpreted as a file path to a valid configuration file.

For example, the following will execute an open-loop controller to fly 4 drones:

.. code-block:: shell

    jaxplan plan Quadcopter 1 slp
   

From Python
^^^^^^^^^^^^^^^^^^^

To initialize and run an open-loop controller in Python:

.. code-block:: python

    import pyRDDLGym
    from pyRDDLGym_jax.core.planner import JaxStraightLinePlan, JaxBackpropPlanner, JaxOfflineController

    # set up the environment (note the vectorized option must be True)
    env = pyRDDLGym.make("domain", "instance", vectorized=True)

    # create the planning algorithm
    plan = JaxStraightLinePlan(**plan_args)
    planner = JaxBackpropPlanner(rddl=env.model, plan=plan, **planner_args)
    controller = JaxOfflineController(planner, **train_args)

    # evaluate the planner
    controller.evaluate(env, episodes=1, verbose=True, render=True)
    env.close()

The ``**plan_args``, ``**planner_args`` and ``**train_args`` are keyword arguments passed during initialization, 
but we strongly recommend using configuration files as discussed in the next section.

.. note::
   All controllers are instances of pyRDDLGym's ``BaseAgent`` and support the ``evaluate()`` function. 

.. raw:: html 

   <a href="notebooks/open_loop_planning_with_jaxplan.html"> 
       <img src="_static/notebook_icon.png" alt="Jupyter Notebook" style="width:64px;height:64px;margin-right:5px;margin-top:5px;margin-bottom:5px;">
       Related example: Open-loop planning with straightline plans in JaxPlan.
   </a>
   
To use periodic replanning, simply change the controller type to:

.. code-block:: python

    controller = JaxOnlineController(planner, **train_args)	
    
.. raw:: html 

   <a href="notebooks/closed_loop_replanning_with_jaxplan.html"> 
       <img src="_static/notebook_icon.png" alt="Jupyter Notebook" style="width:64px;height:64px;margin-right:5px;margin-top:5px;margin-bottom:5px;">
       Related example: Closed-loop replanning with JaxPlan.
   </a>
   
   
To use a deep reactive policy, simply change the ``plan`` type to:

.. code-block:: python

    plan = JaxDeepReactivePolicy(**plan_args)

.. raw:: html 

   <a href="notebooks/closed_loop_planning_drp_with_jaxplan.html"> 
       <img src="_static/notebook_icon.png" alt="Jupyter Notebook" style="width:64px;height:64px;margin-right:5px;margin-top:5px;margin-bottom:5px;">
       Related example: Closed-loop planning with deep reactive policies in JaxPlan.
   </a>
   
.. note::
   ``JaxStraightlinePlan`` and ``JaxDeepReactivePolicy`` are instances of the abstract class ``JaxPlan``. 
   Other policy representations could be defined by overriding this class and its abstract methods.


Configuring JaxPlan
-------------------

The recommended way to manage planner settings is to write a configuration file 
with all required hyper-parameters. 

Configuration Files
^^^^^^^^^^^^^^^^^^^

As of JaxPlan version 3.0, the configuration file contains three sections:

* ``[Compiler]`` dictates how RDDL expressions are translated to JAX
* ``[Planner]`` specifies the type of plan or policy, its hyper-parameters, optimizer, etc.
* ``[Optimize]`` specifies budget on iterations, time limit, stopping rule, etc.
   
For straight-line planning, below is an example of a working configuration file:

.. code-block:: shell

    [Compiler]
    method='DefaultJaxRDDLCompilerWithGrad'
    sigmoid_weight=20

    [Planner]
    method='JaxStraightLinePlan'
    method_kwargs={}
    optimizer='rmsprop'
    optimizer_kwargs={'learning_rate': 0.001}

    [Optimize]
    key=42
    epochs=5000
    train_seconds=30

To use a policy network with two hidden layers of size 128:

.. code-block:: shell

    [Planner]
    method='JaxDeepReactivePolicy'
    method_kwargs={'topology': [128, 128]}
  
To use replanning with a rollout horizon of 5:

.. code-block:: shell

    [Optimize]
    rollout_horizon=5

Expand the following sections to see which parameters can be set in each section (for version 3.0):

.. collapse:: Possible config parameters under [Compiler]
   
    .. list-table:: ``[Compiler]`` settings for all ``JaxRDDLCompilerWithGrad`` instances
      :widths: 40 80
      :header-rows: 1

      * - Setting
        - Description
      * - allow_synchronous_state
        - Whether next state variables allowed to depend on other next state variables
      * - cpfs_without_grad
        - Set of cpfs whose gradients are to be ignored (use STE estimator)
      * - method
        - Type of ``core.logic.JaxRDDLCompilerWithGrad`` defines translation from RDDL to JAX
      * - print_warnings
        - Whether to print compilation warnings
      * - use64bit
        - Whether to use 64 bit arithmetic
    
    .. list-table:: ``[Compiler]`` settings for ``DefaultJaxRDDLCompilerWithGrad``
      :widths: 40 80
      :header-rows: 1

      * - Setting
        - Description
      * - argmax_weight
        - Controls strength of softmax relaxation of argmax and argmin operators
      * - bernoulli_sigmoid_weight
        - Controls strength of sigmoid relaxation of Bernoulli
      * - binomial_eps
        - Underflow correction of Binomial
      * - binomial_nbins
        - Maximum bins for Binomial relaxation before switching to Normal approximation
      * - binomial_softmax_weight
        - Controls strength of softmax relaxation of Binomial
      * - discrete_eps
        - Underflow correction of Discrete 
      * - discrete_softmax_weight
        - Controls strength of softmax relaxation of Discrete
      * - floor_weight
        - Controls strength of tanh relaxation of floor and ceil operators
      * - geometric_eps
        - Underflow correction of Geometric
      * - geometric_floor_weight
        - Controls strength of tanh relaxation of floor operator in Geometric
      * - poisson_comparison_weight
        - Controls strength of exponential approximation of Poisson
      * - poisson_min_cdf
        - Controls when to use exponential or Normal approximation of Poisson
      * - poisson_nbins
        - Maximum bins for Poisson relaxation before switching to Normal approximation
      * - round_weight
        - Controls strength of tanh relaxation of round operators
      * - sigmoid_weight
        - Controls strength of sigmoid/tanh relaxation of relational operators
      * - sqrt_eps
        - Underflow correction of sqrt operators
      * - switch_weight
        - Controls strength of softmax relaxation of switch operators
      * - use_floor_ste
        - Whether to use STE for floor relaxation
      * - use_if_else_ste
        - Whether to use STE for if-then-else relaxation
      * - use_logic_ste
        - Whether to use STE for relaxation of logical operators
      * - use_round_ste
        - Whether to use STE for round relaxation
      * - use_sigmoid_ste
        - Whether to use STE for sigmoid-relaxed operators
      * - use_tanh_ste
        - Whether to use STE for tanh-relaxed operators (e.g. sign)


.. collapse:: Possible config parameters under [Planner]

    .. list-table:: ``[Planner]``
      :widths: 40 80
      :header-rows: 1

      * - Setting
        - Description
      * - action_bounds
        - Dict of (lower, upper) bound tensors for each action-fluent
      * - batch_size_test
        - Batch size for evaluation
      * - batch_size_train
        - Batch size for training
      * - clip_grad
        - Bound on gradient magnitude
      * - dashboard
        - Whether to show a dashboard with training progress
      * - ema_decay
        - Decay rate of EMA of policy parameters
      * - line_search_kwargs
        - Arguments for `zoom line search <https://optax.readthedocs.io/en/latest/api/transformations.html#optax.scale_by_zoom_linesearch>`_
      * - method
        - Type of ``core.planner.JaxPlan`` specifies the policy class
      * - method_kwargs
        - Arguments for policy constructor (see next tables for options)
      * - noise_kwargs
        - Arguments for `gradient noise <https://optax.readthedocs.io/en/latest/api/transformations.html#optax.add_noise>`_
      * - optimizer
        - Name of optimizer from `optax <https://optax.readthedocs.io/en/latest/api/optimizers.html>`_
      * - optimizer_kwargs
        - Arguments for optimizer constructor such as ``learning_rate``
      * - parallel_updates
        - Number of independent policies to optimize in parallel
      * - pgpe
        - Type of ``core.planner.PGPE`` for parameter-exploring policy gradient update
      * - pgpe_kwargs
        - Arguments for PGPE constructor (see table below for default choices)
      * - preprocessor
        - Type of ``core.planner.Preprocessor`` for input preprocessing such as normalization
      * - preprocessor_kwargs
        - Arguments for preprocessor constructor
      * - rollout_horizon
        - Rollout horizon of the computation graph
      * - use_symlog_reward
        - Whether to apply `symlog transform <https://arxiv.org/abs/2301.04104>`_ to returns
      * - utility
        - Utility function to optimize
      * - utility_kwargs
        - Arguments for utility such as hyper-parameters

    .. list-table:: Possible ``method_kwargs`` arguments for ``JaxStraightLinePlan``
      :widths: 40 80
      :header-rows: 1

      * - Setting
        - Description
      * - initializer
        - Type of ``jax.nn.initializers``
      * - initializer_kwargs
        - Arguments for initializer constructor
      * - max_constraint_iter
        - Maximum iterations of `gradient projection <https://ipc2018-probabilistic.bitbucket.io/planner-abstracts/conformant-sogbofa-ipc18.pdf>`_
      * - min_action_prob
        - Minimum bound on boolean action to avoid sigmoid saturation
      * - use_new_projection
        - Whether to use new sorting gradient projection for boolean action preconditions
      * - wrap_non_bool
        - Whether to wrap non-boolean actions with nonlinearity for box constraints
      * - wrap_sigmoid
        - Whether to wrap boolean actions with sigmoid
      * - wrap_softmax
        - Whether to wrap boolean actions with softmax to satisfy ``max-nondef-actions``

    .. list-table:: Possible ``method_kwargs`` arguments for ``JaxDeepReactivePolicy``
      :widths: 40 80
      :header-rows: 1

      * - Setting
        - Description   
      * - activation
        - Activation for hidden layers in ``jax.numpy`` or ``jax.nn`` 
      * - initializer
        - Type of ``haiku.initializers``
      * - initializer_kwargs
        - Arguments for initializer constructor
      * - normalize
        - Whether to apply layer norm to inputs
      * - normalize_per_layer
        - Whether to apply layer norm to each input individually
      * - normalizer_kwargs
        - Arguments for ``haiku.LayerNorm`` constructor
      * - topology
        - List specifying number of neurons per hidden layer
      * - wrap_non_bool
        - Whether to wrap non-boolean actions with nonlinearity for box constraints   

    .. list-table:: Possible ``pgpe_kwargs`` arguments for ``GaussianPGPE``
      :widths: 40 80
      :header-rows: 1

      * - Setting
        - Description
      * - batch_size
        - Number of parameters to sample per gradient descent step
      * - end_entropy_coeff
        - Ending entropy regularization coeffient
      * - init_sigma
        - Initial standard deviation of meta policy
      * - max_kl_update
        - Maximum bound on kl-divergence between successive updates
      * - min_reward_scale
        - Minimum scaling factor for ``scale_reward``
      * - optimizer
        - Name of optimizer from ``optax``
      * - optimizer_kwargs_mu
        - Arguments for optimizer constructor for mean such as ``learning_rate``
      * - optimizer_kwargs_sigma
        - Arguments for optimizer constructor for std such as ``learning_rate``
      * - scale_reward
        - Whether to apply reward scaling in parameter updates
      * - sigma_range
        - Clipping bounds for standard deviation of meta policy
      * - start_entropy_coeff
        - Starting entropy regularization coeffient
      * - super_symmetric
        - Whether to use super-symmetric sampling for standard deviation
      * - super_symmetric_accurate
        - Whether to use the accurate formula for super symmetric sampling in the paper


.. collapse:: Possible config parameters under [Optimize]

    .. list-table:: ``[Optimize]``
      :widths: 40 80
      :header-rows: 1

      * - Setting
        - Description
      * - epochs
        - Maximum number of iterations
      * - key
        - RNG seed for JAX
      * - model_params
        - Dict of hyper-parameter values for the model relaxation
      * - policy_hyperparams
        - Dict of hyper-parameter values for the policy
      * - print_hyperparams
        - Whether to print the planner hyper-parameters
      * - print_progress
        - Whether to show the progress bar
      * - print_summary
        - Whether to print the planner summary
      * - stopping_rule
        - Type of ``JaxPlannerStoppingRule`` for stopping the optimizer
      * - stopping_rule_kwargs
        - Arguments for stopping rule constructor
      * - test_rolling_window
        - Smoothing window for test return calculation
      * - train_seconds
        - Maximum seconds to iterate


Using Configuration Files
^^^^^^^^^^^^^^^^^^^

Configuration files can be parsed and passed to the ``plan``, ``planner`` and ``controller`` as in the basic example:

.. code-block:: python

    from pyRDDLGym_jax.core.planner import load_config
    planner_args, plan_args, train_args = load_config("/path/to/config")
    # continue to initialize plan, planner and controller
    ...


Constraints on Action-Fluents
-------------------

Boolean Action-Fluents
^^^^^^^^^^^^^^^^^^^

By default, boolean actions are wrapped using the sigmoid function:

.. math::
    
    a = \frac{1}{1 + e^{-w \theta}},

where :math:`\theta` are the trainable action parameters and :math:`w` is a 
hyper-parameter controlling the sharpness. 
At test time, the action is aliased by evaluating the expression :math:`a > 0.5`, or equivalently :math:`\theta > 0`. 
This setting can be controlled in JaxPlan by setting ``wrap_sigmoid``.

.. warning::
   If ``wrap_sigmoid = True``, then ``w`` should be specified in ``policy_hyperparams`` dictionary per boolean action fluent.
   

Box Constraints
^^^^^^^^^^^^^^^^^^^

Box constraints are useful for bounding each action fluent independently within some range.
Box constraints typically do not need to be specified manually, since they are automatically 
parsed from the ``action_preconditions`` in the RDDL domain description.

However, it is possible to override these bounds
by passing a dictionary of bounds for each action fluent into the ``action_bounds`` argument. 
The syntax for specifying optional box constraints in the config is:

.. code-block:: shell
	
    [Optimize]
    action_bounds={ <action_fluent1>: (lower1, upper1), <action_fluent2>: (lower2, upper2), ... }
   
where ``lower#`` and ``upper#`` can be any list, nested list or array.

By default, box constraints are enforced using projected gradient.
An alternative approach applies a `differentiable transformation <https://ojs.aaai.org/index.php/AAAI/article/view/4744>`_ 
to action fluents. In JaxPlan, this can be controlled by setting ``wrap_non_bool``. 

Concurrency
^^^^^^^^^^^^^^^^^^^

Concurrency constraints are of the form :math:`\sum_i a_i \leq B` where :math:`B`
is ``max-nondef-actions`` in the RDDL instance. ``JaxBackpropPlanner`` will automatically 
apply `projected gradient <https://ojs.aaai.org/index.php/ICAPS/article/view/3467>`_ 
to satisfy constraints at each optimization step (for straight-line plans only).

.. note::
   Concurrency constraints are applied to boolean actions only.
   Deep reactive policies currently support only :math:`B = 1`.


Automatically Tuning Hyper-Parameters
-------------------

JaxPlan provides a Bayesian optimization algorithm for automatically tuning hyper-parameters:

* supports multi-processing by evaluating multiple hyper-parameter settings in parallel
* leverages Bayesian optimization to search the hyper-parameter space more efficiently
* supports all types of policies that use config files.

From the Command Line
^^^^^^^^^^^^^^^^^^^

The command line app runs the automated tuning on several key hyper-parameters:

.. code-block:: shell

    jaxplan tune <domain> <instance> <method> <trials> <iters> <workers> <dashboard>
    
where:

* ``domain`` and ``instance`` specify the domain and instance names
* ``method`` is the planning method (i.e., slp, drp, replan)
* ``trials`` is the (optional) number of trials/episodes to average in evaluating each hyper-parameter setting
* ``iters`` is the (optional) maximum number of iterations/evaluations of Bayesian optimization to perform
* ``workers`` is the (optional) number of parallel evaluations to be done at each iteration, e.g. maximum total evaluations is ``trials * workers``
* ``dashboard`` is whether the optimizations are tracked and displayed in a dashboard application.

From Python
^^^^^^^^^^^^^^^^^^^

To customize the hyper-parameter tuning algorithm in detail, first create an abstract config file,
where concrete hyper-parameters to tune are replaced by keywords. To tune the sigmoid relaxation in the compiler and
the optimizer learning rate, for example:

.. code-block:: shell

    [Compiler]
    method='DefaultJaxRDDLCompilerWithGrad'
    sigmoid_weight=TUNABLE_WEIGHT

    [Planner]
    method='JaxStraightLinePlan'
    method_kwargs={}
    optimizer='rmsprop'
    optimizer_kwargs={'learning_rate': TUNABLE_LEARNING_RATE}
    ...

.. warning::
   During tuning, keywords are replaced by concrete values via simple string matching.
   Therefore, you must select keywords not appearing (as substrings) in any other parts of the config file.
   
Next, for each config variable, specify its search range and transformation to apply:

.. code-block:: python

    from pyRDDLGym_jax.core.tuning import JaxParameterTuning, Hyperparameter
    from pyRDDLGym_jax.core.planner import load_config_from_string

    # load env as usual
    ...

    # load the abstract config file with planner settings
    with open('path/to/config', 'r') as file:
        config_template = file.read() 
    
    # map parameters in the config that will be tuned
    def power_10(x):
        return 10.0 ** x
    hyperparams = [Hyperparameter("TUNABLE_WEIGHT", -1., 5., power_10),
                   Hyperparameter("TUNABLE_LEARNING_RATE", -5., 1., power_10)]
    
    # build the tuner and tune (online indicates not to use replanning)
    tuning = JaxParameterTuning(env=env, config_template=config_template, hyperparams=hyperparams,
                                online=False, eval_trials=trials, num_workers=workers, gp_iters=iters)
    tuning.tune(key=42, log_file="path/to/logfile")
    
    # parse the concrete config file with the best tuned values, and evaluate as usual
    planner_args, _, train_args = load_config_from_string(tuning.best_config)
    ...
    
JaxPlan supports tuning most numeric parameters in the config file. 
If you wish to tune the replanning mode set ``online=True``.

.. collapse:: Possible settings for ``JaxParameterTuning``

    .. list-table:: ``JaxParameterTuning`` constructor arguments
      :widths: 40 80
      :header-rows: 1

      * - Setting
        - Description
      * - acquisition
        - ``AcquisitionFunction`` object for the Gaussian process
      * - config_template
        - Config file content with abstract parameters to tune as described above
      * - env
        - The ``RDDLEnv`` instance
      * - eval_trials
        - Number of independent trials/rollouts to evaluate each hyper-parameter combination
      * - gp_init_kwargs
        - Optional keyword arguments to pass to the Gaussian process constructor
      * - gp_iters
        - Number of rounds of tuning to perform
      * - gp_params
        - Optional additional keyword arguments to pass to the Gaussian process (i.e. kernel)
      * - hyperparams
        - List of ``Hyperparameter`` objects
      * - num_workers
        - Number of parallel evaluations to perform in each round of tuning
      * - online
        - Whether to use replanning mode for tuning
      * - poll_frequency
        - How often to check for completed processes (defaults to 0.2 seconds)
      * - pool_context
        - The type of pool context for multiprocessing (defaults to "spawn")
      * - rollouts_per_trial
        - For ``online=False``, how many evaluation rollouts to perform per ``eval_trial``
      * - timeout_tuning
        - Maximum amount of time to allocate to tuning
      * - verbose
        - Whether to print intermediate results to the standard console
     
.. raw:: html 

   <a href="notebooks/tuning_hyperparameters_in_jaxplan.html"> 
       <img src="_static/notebook_icon.png" alt="Jupyter Notebook" style="width:64px;height:64px;margin-right:5px;margin-top:5px;margin-bottom:5px;">
       Related example: Tuning policy hyper-parameters in JaxPlan.
   </a>
   

VIsualizing with Dashboard
-------------------

As of version 1.0, the embedded visualization tools have been replaced with 
a plotly dashboard, offering a more comprehensive way to introspect trained policies.
To activate the dashboard for planning, simply add the following line in the config file:

.. code-block:: shell

    [Planner]
    dashboard=True


Risk-Aware Planning with Utility Optimization
-------------------

By default, JaxPlan will optimize the expected discounted sum of future reward, 
which may not be desirable for risk-sensitive applications.
JaxPlan can also optimize a subset of `non-linear utility functions <https://ojs.aaai.org/index.php/AAAI/article/view/21226>`_:

* "mean" is the risk-neutral or ordinary expected return
* "mean_std" is the standard deviation penalized return
* "mean_var" is the variance penalized return
* "mean_semidev" is the mean-semideviation risk measure
* "mean_semivar" is the mean-semivariance risk measure
* "sharpe" is the sharpe ratio
* "entropic" (or "exponential") is the entropic or exponential utility
* "var" is the value at risk
* "cvar" is the conditional value at risk.

A utility function can be specified by passing a string above to the ``utility`` argument of the planner,
and optional hyper-parameters dict to the ``utility_kwargs`` argument, i.e. for CVAR at 5 percent:

.. code-block:: shell

    [Planner]
    utility='cvar'
    utility_kwargs={'alpha': 0.05}

The utility function could also be provided explicitly as a function mapping a JAX array to a scalar, 
with additional arguments specifying the hyper-parameters of the utility function referred to by name:

.. code-block:: python

    @jax.jit
    def my_utility_function(x, aversion: float=1.0) -> float:
        return ...
    planner = JaxBackpropPlanner(..., utility=my_utility_function, utility_kwargs={'aversion': 2.0})

.. raw:: html 

   <a href="notebooks/risk_aware_planning_with_jaxplan.html"> 
       <img src="_static/notebook_icon.png" alt="Jupyter Notebook" style="width:64px;height:64px;margin-right:5px;margin-top:5px;margin-bottom:5px;">
       Related example: Risk-aware planning with RAPTOR in JaxPlan.
   </a>


Dealing with Non-Differentiability
-------------------

Model Relaxations
^^^^^^^^^^^^^^^^^^^

Many RDDL programs contain expressions that do not support derivatives.
A common technique to deal with this is to approximate non-differentiable operations using similar differentiable ones.

For instance, consider the following problem of classifying points ``(x, y)`` in 2D-space as 
+1 if they lie in the top-right or bottom-left quadrants, and -1 otherwise:

.. code-block:: python

    def classify(x, y):
        if x > 0 and y > 0 or not x > 0 and not y > 0:
            return +1
        else:
            return -1
		    
Relational expressions such as ``x > 0`` and ``y > 0``, 
and logical expressions such as ``and`` and ``or`` do not have obvious derivatives. 
To complicate matters further, the ``if`` statement depends on both ``x`` and ``y`` 
so it does not have partial derivatives with respect to ``x`` nor ``y``.

JaxPlan works around these limitations by approximating such operations with JAX expressions that support derivatives.
The ``JaxRDDLCompilerWithGrad`` describes how relaxations are performed, and it is highly configurable and inheritable. 
The type of compiler instance can be passed to a planner by specifying:

.. code-block:: shell
    
    [Compiler]
    method='MyJaxRDDLCompilerWithGradType'
    method_kwargs=...


The default ``DefaultJaxRDDLCompilerWithGrad`` implements a 
variety of differentiable relaxations from the `literature <https://github.com/pyrddlgym-project/pyRDDLGym-jax?tab=readme-ov-file#citing-jaxplan>`_ 
that have been carefully tuned for the best possible results, but they are also constantly improving with each new release.

.. list-table:: Default ``DefaultJaxRDDLCompilerWithGrad`` rules
  :widths: 60 60
  :header-rows: 1

  * - Exact RDDL Operation
    - Approximate Operation
  * - ^, &, |, ~, forall, exists, etc.
    - `Fuzzy t-norm logic <https://www.sciencedirect.com/science/article/pii/S0004370221001533>`_
  * - ==, >, <, >=, <=, sgn, etc.
    - `Tanh <https://arxiv.org/pdf/2110.05651>`_ and `Sigmoid <https://arxiv.org/pdf/2110.05651>`_
  * - argmax, argmin
    - `Softmax <https://arxiv.org/pdf/2110.05651>`_
  * - floor, div, mod, etc.
    - `SoftFloor <https://www.tensorflow.org/probability/api_docs/python/tfp/substrates/jax/bijectors/Softfloor>`_
  * - round
    - `SoftRound <https://arxiv.org/pdf/2006.09952>`_
  * - if-then-else
    - `Linear <https://arxiv.org/pdf/2110.05651>`_
  * - switch
    - `Softmax <https://arxiv.org/pdf/2110.05651>`_
  * - Bernoulli, Discrete
    - `Gumbel-Softmax <https://arxiv.org/pdf/1611.01144>`_ or `Sigmoid <https://arxiv.org/pdf/2110.05651>`_
  * - Geometric
    - `SoftFloor <https://www.tensorflow.org/probability/api_docs/python/tfp/substrates/jax/bijectors/Softfloor>`_
  * - Binomial
    - `Gumbel-Softmax <https://arxiv.org/pdf/1611.01144>`_ for small population, Normal for large population
  * - Poisson
    - `rsample <https://arxiv.org/pdf/2405.14473>`_ for small rate, Normal for large rate

Some relaxations naturally introduce hyper-parameters to control the quality of the approximation.
These hyper-parameters can be retrieved and modified programmatically as follows:

.. code-block:: python

    model_params = planner.compiled.model_params
    model_params[key] = ...
    planner.optimize(..., model_params=model_params)


Parameter-Exploring Policy Gradient
^^^^^^^^^^^^^^^^^^^

Since version 2.0, JaxPlan runs a parallel instance of
`parameter-exploring policy gradients (PGPE) <https://link.springer.com/chapter/10.1007/978-3-319-09903-3_13>`_.
In some cases, this allows JaxPlan to continue making progress when the model relaxations are poor 
or the gradient descent optimizer fails to make progress. 

It is enabled by default, but can be configured in the config file as follows:

.. code-block:: shell

    [Planner]
    pgpe='GaussianPGPE'
    pgpe_kwargs={'optimizer_kwargs_mu': {'learning_rate': 0.01}, 'optimizer_kwargs_sigma': {'learning_rate': 0.01}}

   
Third-Party Optimizers
^^^^^^^^^^^^^^^^^^^

Gradient-free methods such as global optimization could work when gradients are uninformative.
As of version 0.3, it is possible to export the optimization problem
to be solved by another optimizer such as scipy:

.. code-block:: python
    
    loss_fn, grad_fn, guess, unravel_fn = planner.as_optimization_problem()

The loss function ``loss_fn`` and gradient map ``grad_fn`` express policy parameters as 1D numpy arrays,
so they can be used as inputs for other packages that do not make use of JAX. The 
``unravel_fn`` allows the 1D array to be mapped back to a JAX pytree.

.. raw:: html 

   <a href="notebooks/building_optimization_problem_with_jaxplan.html"> 
       <img src="_static/notebook_icon.png" alt="Jupyter Notebook" style="width:64px;height:64px;margin-right:5px;margin-top:5px;margin-bottom:5px;">
       Related example: Building an optimization problem for third-party optimizers.
   </a>
   

Limitations
-------------------

We cite several limitations of the current version of JaxPlan:

* Not all operations have natural differentiable relaxations or are supported by the compiler:

  * nested fluents such as ``fluent1(fluent2(?p))``
  * Multinomial sampling

* Some relaxations can accumulate high error:

  * particularly problematic for long rollout horizon, so we recommend reducing or tuning it
  * model relaxations and hyper-parameters can be tuned for optimal results

* Some relaxations can not be mathematically consistent with one another:

  * dichotomy of equality, e.g. a == b, a > b and a < b do not necessarily "sum" to one, but in most cases should be close
	* it is recommended to override operations in the compiler if this is a concern

* Termination conditions and complex (i.e. nonlinear) state or action constraints are not included in the optimization:

  * constraints can be logged in the optimizer callback and used during optimization (e.g. to build lagrangians)

* Optimizer can fail to make progress when the problem is largely discrete:

  * to diagnose, monitor and compare the training loss and the test loss over time

The goal of JaxPlan is to provide a standard planning baseline that can be easily built upon.
We also welcome any suggestions or modifications about how to improve the robustness of JaxPlan 
on a broader subset of RDDL.


Citation
-------------------

If you use the code provided by JaxPlan, please use the following bibtex for citation:

.. code-block:: bibtex

    @inproceedings{
        gimelfarb2024jaxplan,
        title={JaxPlan and GurobiPlan: Optimization Baselines for Replanning in Discrete and Mixed Discrete and Continuous Probabilistic Domains},
        author={Michael Gimelfarb and Ayal Taitler and Scott Sanner},
        booktitle={34th International Conference on Automated Planning and Scheduling},
        year={2024},
        url={https://openreview.net/forum?id=7IKtmUpLEH}
    }

