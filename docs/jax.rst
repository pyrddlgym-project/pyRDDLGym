.. _jaxplan:

pyRDDLGym-jax: Gradient-Based Simulation and Planning with JaxPlan
===============

In this tutorial, we discuss how a RDDL model can be automatically compiled into a differentiable JAX simulator. 
We also show how pyRDDLGym-jax (or JaxPlan) leverages gradient-based optimization for optimal control. 

Installing
-----------------

To install the bare-bones version of JaxPlan with minimum installation requirements:

.. code-block:: shell

    pip install pyRDDLGym-jax

To install JaxPlan with the automatic hyper-parameter tuning and rddlrepository:
    
.. code-block:: shell

    pip install pyRDDLGym-jax[extra]

(Since version 1.0) To install JaxPlan with the visualization dashboard:

.. code-block:: shell

    pip install pyRDDLGym-jax[dashboard]

(Since version 1.0) To install JaxPlan with all options:

.. code-block:: shell

    pip install pyRDDLGym-jax[extra,dashboard]
    
To install the latest pre-release version via git:

.. code-block:: shell

    pip install git+https://github.com/pyrddlgym-project/pyRDDLGym-jax.git


Simulating using JAX
-------------------

pyRDDLGym ordinarily simulates domains using numPy.
If you require additional structure such as gradients, or better simulation performance, 
the environment can use a JAX simulation backend instead:

.. code-block:: python
	
	import pyRDDLGym
	from pyRDDLGym_jax.core.simulator import JaxRDDLSimulator
	env = pyRDDLGym.make("CartPole_Continuous_gym", "0", backend=JaxRDDLSimulator)
	
.. note::
   All RDDL syntax (both new and old) is supported in the RDDL-to-JAX compiler. 
   In almost all cases, the JAX backend should return numerical results identical to the default backend.
   However, not all operations currently support gradients (see the Limitations section).

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

Reparameterization Trick for Stochastic Problems
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
   As of JaxPlan version 2.2 (JAX version 0.4.25), most discrete and continuous distributions 
   support gradients for the most common use cases. The notable exceptions are Binomial 
   (supported for small counts only), NegativeBinomial and Multinomial.


Running JaxPlan
-------------------

From the Command Line
^^^^^^^^^^^^^^^^^^^

A command line app is provided to run JaxPlan on a specific problem instance:

.. code-block:: shell
    
    jaxplan plan <domain> <instance> <method> --episodes <episodes>
    
where:

* ``<domain>`` is the domain identifier in rddlrepository, or a path pointing to a valid domain file
* ``<instance>`` is the instance identifier in rddlrepository, or a path pointing to a valid instance file
* ``<method>`` is the planning method to use (i.e. drp, slp, replan) or a path to a valid .cfg file
* ``<episodes>`` is the (optional) number of episodes to evaluate the final policy.

The ``<method>`` parameter describes the type of planning representation:

* ``slp`` is the `straight-line plan <https://proceedings.neurips.cc/paper/2017/file/98b17f068d5d9b7668e19fb8ae470841-Paper.pdf>`_
* ``drp`` is the `deep reactive policy network <https://ojs.aaai.org/index.php/AAAI/article/view/4744>`_ 
* ``replan`` uses replanning at every decision epoch
* any other argument is interpreted as a file path to a valid configuration file.

For example, the following will train an open-loop controller to fly 4 drones:

.. code-block:: shell

    jaxplan plan Quadcopter 1 slp
   

From Python
-------------------
^^^^^^^^^^^^^^^^^^^

.. _jax-intro:

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
   All controllers are instances of pyRDDLGym's ``BaseAgent`` and support the ``evaluate()`` function. 



Configuring JaxPlan
-------------------

The recommended way to manage planner settings is to write a configuration (.cfg) file 
with all required hyper-parameters, i.e. for straight-line planning:

.. code-block:: shell

    [Model]
    logic='FuzzyLogic'
    comparison_kwargs={'weight': 20}
    rounding_kwargs={'weight': 20}
    control_kwargs={'weight': 20}

    [Optimizer]
    method='JaxStraightLinePlan'
    method_kwargs={}
    optimizer='rmsprop'
    optimizer_kwargs={'learning_rate': 0.001}
    batch_size_train=1
    batch_size_test=1

    [Training]
    key=42
    epochs=5000
    train_seconds=30

The configuration file contains three sections:

* ``[Model]`` dictates how non-differentiable expressions are handled (discussed later)
* ``[Optimizer]`` requires a ``method`` argument to indicate the type of plan/policy, its hyper-parameters, optimizer, etc.
* ``[Training]`` indicates budget on iterations, time, verbosity, etc.
   
To use a policy network with two hidden layers of size 128:

.. code-block:: shell

    [Optimizer]
    method='JaxDeepReactivePolicy'
    method_kwargs={'topology': [128, 128]}
  
To use replanning with a straight-line plan and a lookahead horizon of 5:

.. code-block:: shell

    [Optimizer]
    method='JaxStraightlinePlan'
    rollout_horizon=5
  
Configuration files can be parsed and passed to the planner as follows:

.. code-block:: python

    from pyRDDLGym_jax.core.planner import load_config
    planner_args, plan_args, train_args = load_config("/path/to/config.cfg")
    
    # continue as described in the previous section
    plan = ...
    planner = ...
    controller = ...

.. collapse:: Possible settings for ``[Model]`` section
   
    .. list-table:: ``[Model]``
      :widths: 40 80
      :header-rows: 1

      * - Setting
        - Description
      * - logic
        - Type of ``core.logic.FuzzyLogic``, how expressions are relaxed
      * - logic_kwargs
        - kwargs to pass to logic object constructor
      * - complement
        - Type of ``core.logic.Complement``, how logical complement is relaxed
      * - complement_kwargs
        - kwargs to pass to complement object constructor
      * - comparison
        - Type of ``core.logic.SigmoidComparison``, how comparisons are relaxed
      * - comparison_kwargs
        - kwargs to pass to comparison object constructor
      * - control
        - Type of ``core.logic.ControlFlow``, how comparisons are relaxed
      * - control_kwargs
        - kwargs to pass to control flow object constructor
      * - rounding
        - Type of ``core.logic.Rounding``, how to round float to int values
      * - rounding_kwargs
        - kwargs to pass to rounding object constructor
      * - sampling
        - Type of ``core.logic.RandomSampling``, how to sample discrete distributions
      * - sampling_kwargs
        - kwargs to pass to sampling object constructor (see table below for default options)
      * - tnorm
        - Type of ``core.logic.TNorm``, how logical expressions are relaxed
      * - tnorm_kwargs
        - kwargs to pass to tnorm object constructor


.. collapse:: Possible settings for ``sampling_kwargs`` in ``[Model]`` section for ``SoftRandomSampling``

    .. list-table:: ``sampling_kwargs`` in ``[Model]`` for ``SoftRandomSampling``
      :widths: 40 80
      :header-rows: 1

      * - Setting
        - Description
      * - bernoulli_gumbel_softmax
        - Whether to use Gumbel-Softmax for Bernoulli relaxation
      * - binomial_max_bins
        - Maximum bins for Binomial relaxation
      * - poisson_exp_sampling
        - Whether to use `exponential sampling <https://arxiv.org/abs/2405.14473>`_ for Poisson relaxation
      * - poisson_max_bins
        - Maximum bins for Poisson relaxation
      * - poisson_min_cdf
        - Required cdf within truncated region to use Poisson relaxation


.. collapse:: Possible settings for ``[Optimizer]`` section

    .. list-table:: ``[Optimizer]``
      :widths: 40 80
      :header-rows: 1

      * - Setting
        - Description
      * - action_bounds
        - Dictionary of (lower, upper) bounds on each action-fluent
      * - batch_size_test
        - Batch size for evaluation
      * - batch_size_train
        - Batch size for training
      * - clip_grad
        - Clip gradients to within a given magnitude
      * - compile_non_fluent_exact
        - Whether model relaxations are skipped for non-fluent expressions
      * - cpfs_without_grad
        - A set of CPFs that do not allow gradients to flow through them
      * - line_search_kwargs
        - Arguments for optional `zoom line search <https://optax.readthedocs.io/en/latest/api/transformations.html#optax.scale_by_zoom_linesearch>`_
      * - method
        - Type of ``core.planner.JaxPlan``, specifies the policy class
      * - method_kwargs
        - kwargs to pass to policy constructor (see next two tables for options)
      * - noise_kwargs
        - Arguments for optional `gradient noise <https://optax.readthedocs.io/en/latest/api/transformations.html#optax.add_noise>`_: ``noise_grad_eta``, ``noise_grad_gamma`` and ``seed``
      * - optimizer
        - Name of optimizer from `optax <https://optax.readthedocs.io/en/latest/api/optimizers.html>`_ to use
      * - optimizer_kwargs
        - kwargs to pass to optimizer constructor, i.e. ``learning_rate``
      * - parallel_updates
        - Number of independent policies to initialize and update in parallel
      * - pgpe
        - Optional type of ``core.planner.PGPE`` for `parallel policy gradient update <https://link.springer.com/chapter/10.1007/978-3-319-09903-3_13>`_
      * - pgpe_kwargs
        - kwargs to pass to PGPE constructor (for ``GaussianPGPE`` see table below)
      * - preprocessor
        - Optional type of ``core.planner.Preprocessor`` for preprocessing fluent tensors (i.e. normalization, etc.)
      * - preprocessor_kwargs
        - kwargs to pass to preprocessor constructor
      * - print_warnings
        - Whether to print compilation warnings to console (errors will still be printed)
      * - rollout_horizon
        - Rollout horizon of the computation graph
      * - use64bit
        - Whether to use 64 bit precision
      * - use_symlog_reward
        - Whether to apply the `symlog transform <https://arxiv.org/abs/2301.04104>`_ to the returns
      * - utility
        - Optional utility function to optimize
      * - utility_kwargs
        - kwargs to pass hyper-parameters to utility


.. collapse:: Possible settings for ``method_kwargs`` in ``[Optimizer]`` section for ``JaxStraightLinePlan``

    .. list-table:: ``method_kwargs`` in ``[Optimizer]`` for ``JaxStraightLinePlan``
      :widths: 40 80
      :header-rows: 1

      * - Setting
        - Description
      * - initializer
        - Type of ``jax.nn.initializers``, specifies parameter initialization
      * - initializer_kwargs
        - kwargs to pass to the initializer
      * - max_constraint_iter
        - Maximum iterations of `gradient projection <https://ipc2018-probabilistic.bitbucket.io/planner-abstracts/conformant-sogbofa-ipc18.pdf>`_ for boolean action preconditions
      * - min_action_prob
        - Minimum probability of boolean action to avoid sigmoid saturation
      * - use_new_projection
        - Whether to use new sorting gradient projection for boolean action preconditions
      * - wrap_non_bool
        - Whether to wrap non-boolean actions with nonlinearity for box constraints
      * - wrap_sigmoid
        - Whether to wrap boolean actions with sigmoid
      * - wrap_softmax
        - Whether to wrap with softmax to satisfy boolean action preconditions


.. collapse:: Possible settings for ``method_kwargs`` in ``[Optimizer]`` section for ``JaxDeepReactivePolicy``

    .. list-table:: ``method_kwargs`` in ``[Optimizer]`` for ``JaxDeepReactivePolicy``
      :widths: 40 80
      :header-rows: 1

      * - Setting
        - Description   
      * - activation
        - Name of activation for hidden layers, from ``jax.numpy`` or ``jax.nn`` 
      * - initializer
        - Type of ``haiku.initializers``, specifies parameter initialization
      * - initializer_kwargs
        - kwargs to pass to the initializer
      * - normalize
        - Whether to apply `layer norm to inputs <https://ojs.aaai.org/index.php/AAAI/article/view/4744>`_
      * - normalize_per_layer
        - Whether to apply layer norm to each input individually
      * - normalizer_kwargs
        - kwargs to pass to ``haiku.LayerNorm`` constructor for layer norm
      * - topology
        - List specifying number of neurons per hidden layer
      * - wrap_non_bool
        - Whether to wrap non-boolean actions with nonlinearity for box constraints   


.. collapse:: Possible settings for ``GaussianPGPE`` policy gradient

    .. list-table:: ``GaussianPGPE`` Policy Gradient Fallback
      :widths: 40 80
      :header-rows: 1

      * - Setting
        - Description
      * - batch_size
        - Number of parameters to sample per gradient descent step
      * - end_entropy_coeff
        - Ending entropy regularization coeffient
      * - init_sigma
        - Initial standard deviation
      * - max_kl_update
        - Maximum bound on kl-divergence between successive updates
      * - min_reward_scale
        - Minimum reward scaling factor if ``scale_reward = True``
      * - optimizer
        - Name of optimizer from optax to use
      * - optimizer_kwargs_mu
        - kwargs to pass to optimizer constructor for mean, i.e. ``learning_rate``
      * - optimizer_kwargs_sigma
        - kwargs to pass to optimizer constructor for std, i.e. ``learning_rate``
      * - scale_reward
        - Whether to apply reward scaling during parameter updates
      * - sigma_range
        - Clipping bounds for standard deviation
      * - start_entropy_coeff
        - Starting entropy regularization coeffient
      * - super_symmetric
        - Whether to use super-symmetric sampling for standard deviation
      * - super_symmetric_accurate
        - Whether to use the accurate formula for super symmetric sampling


.. collapse:: Possible settings for ``[Training]`` section

    .. list-table:: ``[Training]``
      :widths: 40 80
      :header-rows: 1

      * - Setting
        - Description
      * - dashboard
        - Whether to display training results in a dashboard
      * - epochs
        - Maximum number of iterations of gradient descent   
      * - key
        - An integer to seed the RNG with for reproducibility
      * - model_params
        - Dictionary of hyper-parameter values to pass to the model relaxation
      * - policy_hyperparams
        - Dictionary of hyper-parameter values to pass to the policy
      * - print_progress
        - Whether to print the progress bar from the planner to console
      * - print_summary
        - Whether to print summary information from the planner to console
      * - restart_epochs
        - Number of consecutive epochs without progress to restart optimizer
      * - stopping_rule
        - A stopping criterion for the optimizer, subclass of ``JaxPlannerStoppingRule``
      * - stopping_rule_kwargs
        - kwargs to pass to stopping rule constructor
      * - test_rolling_window
        - Smoothing window over which to calculate test return
      * - train_seconds
        - Maximum seconds to train for


Boolean Actions
-------------------

Constraints on Action-Fluents
-------------------

Supporting Boolean Action-Fluents
^^^^^^^^^^^^^^^^^^^

By default, boolean actions are wrapped using the sigmoid function:

.. math::
    
    a = \frac{1}{1 + e^{-w \theta}},

where :math:`\theta` denotes the trainable action parameters, and :math:`w` denotes a 
hyper-parameter that controls the sharpness of the approximation.

.. warning::
   If the sigmoid wrapping is used, then the weights ``w`` should be specified in 
   ``policy_hyperparams`` for each boolean action fluent (as a dictionary) when interfacing with the planner.
   
At test time, the action is aliased by evaluating the expression 
:math:`a > 0.5`, or equivalently :math:`\theta > 0`. 
The sigmoid wrapper can be controlled by setting ``wrap_sigmoid``.

Box Constraints
^^^^^^^^^^^^^^^^^^^

Box constraints are useful for bounding each action fluent independently within some range.
Box constraints typically do not need to be specified manually, since they are automatically 
parsed from the ``action_preconditions`` in the RDDL domain description.

However, if the user wishes, it is possible to override these bounds
by passing a dictionary of bounds for each action fluent into the ``action_bounds`` argument. 
The syntax for specifying optional box constraints in the config file is:

.. code-block:: shell
	
    [Optimizer]
    action_bounds={ <action_name1>: (lower1, upper1), <action_name2>: (lower2, upper2), ... }
   
where ``lower#`` and ``upper#`` can be any list, nested list or array.

By default, the box constraints on actions are enforced using the projected gradient method.
An alternative approach is to map the actions to the box via a 
`differentiable transformation <https://ojs.aaai.org/index.php/AAAI/article/view/4744>`_.
In JaxPlan, this can be enabled by setting ``wrap_non_bool = True``. 

Concurrency
^^^^^^^^^^^^^^^^^^^

Concurrency constraints are of the form :math:`\sum_i a_i \leq B` where :math:`B`
is ``max-nondef-actions`` in the RDDL instance. ``JaxBackpropPlanner`` will automatically 
apply `projected gradient <https://ojs.aaai.org/index.php/ICAPS/article/view/3467>`_ 
to satisfy constraints at each optimization step (for straight-line plans only).

.. note::
   Concurrency constraints on action-fluents are applied to boolean actions only.
   Deep reactive policies only support :math:`B = 1`.


Reward Normalization
-------------------

Some domains yield rewards that vary significantly in magnitude between time steps, 
making optimization difficult without some kind of normalization.
JaxPlan can apply an optional `symlog transform <https://arxiv.org/pdf/2301.04104v1.pdf>`_ to the sampled returns

.. math::
    
    \mathrm{symlog}(x) = \mathrm{sign}(x) * \ln(|x| + 1)

which compresses the magnitudes of large positive or negative outcomes.
This can be controlled by ``use_symlog_reward``.


Utility Optimization
-------------------

By default, JaxPlan will optimize the expected sum of future reward, 
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

.. code-block:: python

    [Optimizer]
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


Automatically Tuning Hyper-Parameters
-------------------

JaxPlan provides a Bayesian optimization algorithm for automatically tuning key hyper-parameters of the planner. It:

* supports multi-processing by evaluating multiple hyper-parameter settings in parallel
* leverages Bayesian optimization to search the hyper-parameter space more efficiently
* supports all types of policies that use config files.

From the Command Line
^^^^^^^^^^^^^^^^^^^

The command line app runs the automated tuning on the most important hyper-parameters:

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

To customize the hyper-parameter tuning algorithm in detail, first specify a config file template
where concrete hyper-parameter to tune are replaced by keywords, i.e.:

.. code-block:: shell

    [Model]
    logic='FuzzyLogic'
    comparison_kwargs={'weight': MODEL_WEIGHT_TUNE}
    rounding_kwargs={'weight': MODEL_WEIGHT_TUNE}
    control_kwargs={'weight': MODEL_WEIGHT_TUNE}

    [Optimizer]
    method='JaxStraightLinePlan'
    method_kwargs={}
    optimizer='rmsprop'
    optimizer_kwargs={'learning_rate': LEARNING_RATE_TUNE}
    ...

.. warning::
   Keywords defined above will be replaced during tuning with concrete values using a simple string replacement.
   This means you must select keywords that are not already used (nor appear as substrings) in other parts of the config file.
   
Next, for each config variable, specify its search range and transformation to apply:

.. code-block:: python

    import pyRDDLGym
    from pyRDDLGym_jax.core.tuning import JaxParameterTuning, Hyperparameter
    from pyRDDLGym_jax.core.planner import load_config_from_string
    
    # set up the environment   
    env = pyRDDLGym.make(domain, instance, vectorized=True)
    
    # load the abstract config file with planner settings
    with open('path/to/config.cfg', 'r') as file:
        config_template = file.read() 
    
    # map parameters in the config that will be tuned
    def power_10(x):
        return 10.0 ** x
    hyperparams = [Hyperparameter("MODEL_WEIGHT_TUNE", -1., 5., power_10),
                   Hyperparameter("LEARNING_RATE_TUNE", -5., 1., power_10)]
    
    # build the tuner and tune (online indicates not to use replanning)
    tuning = JaxParameterTuning(env=env,
                                config_template=config_template, hyperparams=hyperparams,
                                online=False, eval_trials=trials, num_workers=workers, gp_iters=iters)
    tuning.tune(key=42, log_file="path/to/logfile.log")
    
    # parse the concrete config file with the best tuned values, and evaluate as usual
    planner_args, _, train_args = load_config_from_string(tuning.best_config)
    ...
    
JaxPlan supports tuning most numeric parameters that can be specified in the config file.
If you wish to tune a replanning algorithm set ``online=True``.

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
   

JaxPlan Dashboard
-------------------

As of JaxPlan version 1.0, the embedded visualization tools have been replaced with 
a plotly dashboard, which offers a much more comprehensive and efficient way to introspect trained policies.
To activate the dashboard for planning, simply add the following line in the config file:

.. code-block:: shell

    [Training]
    dashboard=True


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
The ``FuzzyLogic`` describes how relaxations are performed, and it is highly configurable. 
It can be passed to a planner through the config file, or directly as follows:

.. code-block:: python
    
    from pyRDDLGym.core.logic import FuzzyLogic
    planner = JaxBackpropPlanner(model, ..., logic=FuzzyLogic())

By default, ``FuzzyLogic`` uses `t-norm fuzzy logics <https://en.wikipedia.org/wiki/T-norm_fuzzy_logics#Motivation>`_
to approximate the logical operations, and a 
`variety of differentiable relaxations from the literature <https://github.com/pyrddlgym-project/pyRDDLGym-jax?tab=readme-ov-file#citing-jaxplan>`_ 
to support other operations automatically.

Some operations introduce model hyper-parameters to control the quality of the approximation.
These hyper-parameters be retrieved and modified at any time as follows:

.. code-block:: python

    model_params = planner.compiled.model_params
    model_params[key] = ...
    planner.optimize(..., model_params=model_params)

It is possible to control these rules by subclassing ``FuzzyLogic``, or by 
modifying ``tnorm``, ``complement`` or other constructor arguments in the config.

.. collapse:: Default rules for ``FuzzyLogic``

    .. list-table:: Default Differentiable Mathematical Operations
      :widths: 60 60
      :header-rows: 1

      * - Exact RDDL Operation
        - Approximate Operation
      * - :math:`a \text{ ^ } b`
        - :math:`a * b`
      * - :math:`\sim a`
        - :math:`1 - a`
      * - forall_{?p : type} x(?p)
        - :math:`\prod_{?p} x(?p)`
      * - if (c) then a else b
        - :math:`c * a + (1 - c) * b` `[1] <https://arxiv.org/pdf/2110.05651>`_
      * - :math:`a == b`
        - :math:`1 - \tanh^2(w * (a - b))` `[1] <https://arxiv.org/pdf/2110.05651>`_
      * - :math:`a > b`, :math:`a >= b`
        - :math:`\mathrm{sigmoid}(w * (a - b))` `[1] <https://arxiv.org/pdf/2110.05651>`_
      * - argmax_{?p : type} x(?p)
        - Softmax `[1] <https://arxiv.org/pdf/2110.05651>`_
      * - sgn(a)
        - :math:`\tanh(w * a)`
      * - floor(a)
        - SoftFloor `[2] <https://www.tensorflow.org/probability/api_docs/python/tfp/substrates/jax/bijectors/Softfloor>`_
      * - round(a)
        - See `[3] <https://arxiv.org/pdf/2006.09952>`_
      * - Bernoulli(p)
        - Gumbel-Softmax `[4] <https://arxiv.org/pdf/1611.01144>`_
      * - Discrete(type, {cases ...} )
        - Gumbel-Softmax `[4] <https://arxiv.org/pdf/1611.01144>`_


Other Techniques
^^^^^^^^^^^^^^^^^^^

Since version 2.0, JaxPlan runs a parallel instance of
`parameter-exploring policy gradients (PGPE) <https://link.springer.com/chapter/10.1007/978-3-319-09903-3_13>`_.
In some cases, this allows JaxPlan to continue making progress when the model relaxations are poor. 
It can be configured as follows:

.. code-block:: shell

    [Optimizer]
    pgpe='GaussianPGPE'
    pgpe_kwargs=...

   
Manual Gradient Calculation
-------------------

As of version 0.3, it is possible to export the optimization problem in JaxPlan
to be solved by another optimizer (e.g., scipy):

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

* Not all operations have natural differentiable relaxations. Currently, the following are not supported:
	* nested fluents such as ``fluent1(fluent2(?p))``
* Some relaxations can accumulate high error
	* this is particularly problematic for long horizon, so we recommend reducing or tuning the rollout horizon for best results
  * the model relaxations and their hyper-parameters should also be tweaked for optimal results
* Some relaxations may not be mathematically consistent with one another:
	* no guarantees are provided about dichotomy of equality, e.g. a == b, a > b and a < b do not necessarily "sum" to one, but in many cases should be close
	* if this is a concern, it is recommended to override some operations in ``FuzzyLogic``
* Termination conditions and state/action constraints are not considered in the optimization
	* constraints are logged in the optimizer callback and can be used to define loss functions that take the constraints into account
* The optimizer can fail to make progress when the structure of the problem is largely discrete:
	* to diagnose this, monitor the training loss and the test loss over time
	* a low, or drastically improving, training loss with a similar test loss indicates that the continuous model relaxation is likely accurate around the optimum
	* on the other hand, a low training loss and a high test loss indicates that the continuous model relaxation is poor.

The goal of JaxPlan is to provide a standard planning baseline that can be easily built upon.
We also welcome any suggestions or modifications about how to improve the robustness of JaxPlan 
on a broader subset of RDDL.


Citations
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

