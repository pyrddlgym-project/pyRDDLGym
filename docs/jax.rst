.. _jaxplan:

pyRDDLGym-jax: JAX Compiler and Planner
===============

In this tutorial, we discuss how a RDDL model can be compiled into a differentiable simulator using JAX. 
We also show how gradient ascent can be used to do optimal control.

Requirements
------------

This package requires Python 3.8+ with:

* pyRDDLGym>=2.0
* tqdm>=4.66
* jax>=0.4.12
* optax>=0.1.9
* dm-haiku>=0.0.10 

To compile some discrete sampling operations (Binomial, Negative-Binomial, Multinomial), you will also need:

* tensorflow>=2.13.0
* tensorflow-probability>=0.21.0

To run the hyper-parameter tuning, you will also need:

* bayesian-optimization>=1.4.3


Installation
-----------------

To install pyRDDLGym-jax and all of its requirements via pip:

.. code-block:: shell

    pip install pyRDDLGym-jax

To install the latest pre-release version via git:

.. code-block:: shell

    pip install git+https://github.com/pyrddlgym-project/pyRDDLGym-jax.git


Simulation using JAX
-------------------

pyRDDLGym ordinarily simulates domains using pure Python and NumPy arrays.
If you require additional structure (e.g. gradient calculations) or better simulation performance, 
the environment can be compiled using JAX and swapped with the default simulation backend, as shown below:

.. code-block:: python
	
	import pyRDDLGym
	from pyRDDLGym_jax.core.simulator import JaxRDDLSimulator
	env = pyRDDLGym.make("Cartpole_Continuous_gym", "0", backend=JaxRDDLSimulator)
	
.. note::
   All RDDL syntax (both new and old) is supported in the RDDL-to-JAX compiler. 
   In almost all cases, the JAX backend should return numerical results identical to the default backend.


Differentiable Planning in Deterministic Domains
-------------------

The (open-loop) planning problem for a deterministic environment involves finding a sequence of actions (plan)
that maximize accumulated reward over a fixed horizon

.. math::

	\max_{a_1, \dots a_T} \sum_{t=1}^{T} R(s_t, a_t), \quad s_{t + 1} = f(s_t, a_t)
	
If the state and action spaces are continuous, and f and R are differentiable functions, 
gradient ascent can optimize the actions as described 
`in this paper <https://proceedings.neurips.cc/paper/2017/file/98b17f068d5d9b7668e19fb8ae470841-Paper.pdf>`_.
Specifically, given learning rate :math:`\eta > 0`, gradient ascent updates the plan
:math:`a_\tau'` at decision epoch :math:`\tau` as

.. math::
	
	a_{\tau}' = a_{\tau} + \eta \sum_{t=1}^{T} \nabla_{a_\tau} R(s_t, a_t),
	
where the gradient of the reward at all times :math:`t \geq \tau` is computed via the chain rule:

.. math::

	\nabla_{a_{\tau}} R(s_t, a_t) = \frac{d R(s_t, a_t)}{d s_{t}} \frac{d f(s_\tau, a_\tau)}{d a_{\tau}} \prod_{k=\tau + 1}^{t-1}\frac{d f(s_k, a_k)}{d s_{k}} + \frac{d R(s_t, a_t)}{d a_{\tau}}.
	
In stochastic domains, an open-loop plan could be sub-optimal 
because it fails to correct for deviations in the state from its anticipated course.
One solution is to recompute the plan periodically or after each decision epoch, 
which is often called "replanning". An alternative approach is to learn a policy network 
:math:`a_t \gets \pi_\theta(s_t)` 
as explained `in this paper <https://ojs.aaai.org/index.php/AAAI/article/view/4744>`_. 
pyRDDLGym-jax currently supports both options, which are detailed in a later section of this tutorial.


Differentiable Planning in Stochastic Domains
-------------------

A common problem of planning in stochastic domains is that the gradients of sampling nodes are not well-defined.
pyRDDLGym-jax works around this problem by using the reparameterization trick.

To illustrate, we can write :math:`s_{t+1} = \mathcal{N}(s_t, a_t^2)` as :math:`s_{t+1} = s_t + a_t * \mathcal{N}(0, 1)`, 
although the latter is amenable to backpropagation while the first is not.
The reparameterization trick also works generally, assuming there exists a closed-form function f such that

.. math::

    s_{t+1} = f(s_t, a_t, \xi_t)
    
and :math:`\xi_t` are random variables drawn from some distribution independent of states and actions. 
For a detailed discussion of reparameterization in the context of planning, 
please see `this paper <https://ojs.aaai.org/index.php/AAAI/article/view/4744>`_ 
or `this paper <https://ojs.aaai.org/index.php/AAAI/article/view/21226>`_.

pyRDDLGym-jax automatically performs reparameterization whenever possible. For some special cases,
such as the Bernoulli and Discrete distribution, it applies the Gumbel-softmax trick 
as described `here <https://arxiv.org/pdf/1611.01144.pdf>`_. 
Defining K independent samples from a standard Gumbel distribution :math:`g_1, \dots g_K`, we reparameterize the 
random variable :math:`X` with probability mass function :math:`p_1, \dots p_K` as

.. math::

    X = \arg\!\max_{i=1\dots K} \left(g_i + \log p_i \right)

where the argmax is approximated using the softmax function.

.. warning::
   For general non-reparameterizable distributions, the result of the gradient calculation 
   is fully dependent on the JAX implementation: it could return a zero or NaN gradient, or raise an exception.


Running pyRDDLGym-jax from the Command Line
-------------------

A basic script is provided to run the JAX planner on any domain in rddlrepository, 
provided a config file of hyper-parameters is available 
(currently, custom config files are provided for a limited subset of problems: 
the default config could be suboptimal for other problems). 

The example can be run as follows in a standard shell, from the install directory of pyRDDLGym-jax:

.. code-block:: shell
    
    python -m pyRDDLGym_jax.examples.run_plan <domain> <instance> <method> <episodes>
    
where:

* ``<domain>`` is the domain identifier in rddlrepository, or a path pointing to a valid domain.rddl file
* ``<instance>`` is the instance identifier in rddlrepository, or a path pointing to a valid instance.rddl file
* ``<method>`` is the planning method to use (see below)
* ``<episodes>`` is the (optional) number of episodes to evaluate the final policy.

The ``<method>`` parameter warrants further explanation. Currently we support three possible modes:

* ``slp`` is the straight-line open-loop planner described `in this paper <https://proceedings.neurips.cc/paper/2017/file/98b17f068d5d9b7668e19fb8ae470841-Paper.pdf>`_
* ``drp`` is the deep reactive policy network described `in this paper <https://ojs.aaai.org/index.php/AAAI/article/view/4744>`_
* ``replan`` is the same as ``slp`` except it uses periodic replanning as described above.

For example, the following will perform open-loop control on the Quadcopter domain with 4 drones:

.. code-block:: shell

    python -m pyRDDLGym_jax.examples.run_plan Quadcopter 1 slp
   

Running pyRDDLGym-jax from within Python
-------------------

.. _jax-intro:

pyRDDLGym-jax provides convenient tools to automatically compile a RDDL description 
of a problem to an optimization problem:

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

Here, we have used an open-loop controller. 
To use periodic replanning, simply change the controller type as below:

.. code-block:: python

    controller = JaxOnlineController(planner, **train_args)	

To use a deep reactive policy, simply change the ``plan`` type as follows:

.. code-block:: python

    plan = JaxDeepReactivePolicy(**plan_args)

.. note::
   All controllers are instances of pyRDDLGym's ``BaseAgent`` and support the ``evaluate()`` function. 

The ``**planner_args`` and ``**train_args`` are keyword arguments passed during initialization, 
but we strongly recommend creating and loading a configuration file as discussed next.


Configuring pyRDDLGym-jax
-------------------

The recommended way to manage planner settings is to write a configuration file 
with all the necessary hyper-parameters. 
Below is the basic structure of a configuration file for straight-line planning:

.. code-block:: shell

    [Model]
    logic='FuzzyLogic'
    logic_kwargs={'weight': 20}
    tnorm='ProductTNorm'
    tnorm_kwargs={}

    [Optimizer]
    method='JaxStraightLinePlan'
    method_kwargs={}
    optimizer='rmsprop'
    optimizer_kwargs={'learning_rate': 0.001}
    batch_size_train=1
    batch_size_test=1
    rollout_horizon=5

    [Training]
    key=42
    epochs=5000
    train_seconds=30

The configuration file contains three sections:

* the ``[Model]`` section dictates how non-differentiable expressions are handled (as discussed later in the tutorial)
* the ``[Optimizer]`` section contains a ``method`` argument to indicate the type of plan/policy, its hyper-parameters, the ``optax`` SGD optimizer and its hyper-parameters, etc.
* the ``[Training]`` section indicates budget on iterations or time, hyper-parameters for the policy, etc.

The configuration file can then be parsed and passed to the planner as follows:

.. code-block:: python

    from pyRDDLGym_jax.core.planner import load_config
    planner_args, plan_args, train_args = load_config("/path/to/config.cfg")
    
    # continue as described above
    plan = ...
    planner = ...
    controller = ...

.. note::
   The ``rollout_horizon`` in the configuration file is optional, and defaults to the horizon specified in the RDDL description. 
   For replanning methods, we recommend setting this parameter manually for best results.

To configure a policy network instead, change the ``method`` in the ``[Optimizer]`` section of the config file:

.. code-block:: shell

    ...
    [Optimizer]
    method='JaxDeepReactivePolicy'
    method_kwargs={'topology': [128, 64]}
    ...

This creates a neural network policy with the default ``tanh`` activation 
and two hidden layers with 128 and 64 neurons, respectively.

.. note::
   ``JaxStraightlinePlan`` and ``JaxDeepReactivePolicy`` are instances of the abstract class ``JaxPlan``. 
   Other policy representations could be defined by overriding this class and its abstract methods.

The full list of settings that can be specified in the configuration files are as follows:

.. list-table:: ``[Model]``
   :widths: 60 60
   :header-rows: 1

   * - Setting
     - Description
   * - comparison
     - Type of ``core.logic.SigmoidComparison``, how comparisons are relaxed
   * - comparison_kwargs
     - kwargs to pass to comparison object constructor
   * - complement
     - Type of ``core.logic.Complement``, how logical complement is relaxed
   * - complement_kwargs
     - kwargs to pass to complement object constructor
   * - logic
     - Type of ``core.logic.FuzzyLogic``, how non-diff. expressions are relaxed
   * - logic_kwargs
     - kwargs to pass to logic object constructor
   * - sampling
     - Type of ``core.logic.RandomSampling``, how to sample discrete distributions
   * - sampling_kwargs
     - kwargs to pass to sampling object constructor
   * - tnorm
     - Type of ``core.logic.TNorm``, how logical expressions are relaxed
   * - tnorm_kwargs
     - kwargs to pass to tnorm object constructor (see next table for options)


.. list-table:: ``logic_kwargs`` in ``[Model]``
   :widths: 60 60
   :header-rows: 1

   * - Setting
     - Description
   * - debias
     - Set of operations with exact calculation on the forward pass
   * - eps
     - Small parameter to control underflow
   * - verbose
     - Whether to print messages during compilation to the console
   * - weight 
     - Parameter for sigmoid and softmax approximations


.. list-table:: ``[Optimizer]``
   :widths: 60 60
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
     - Model relaxations are not applied to non-fluent expressions
   * - cpfs_without_grad
     - A set of CPFs that do not allow gradients to flow through them
   * - method
     - Type of ``core.planner.JaxPlan``, specifies the policy class
   * - method_kwargs
     - kwargs to pass to policy constructor (see next two tables for options)
   * - optimizer
     - Name of optimizer from optax to use
   * - optimizer_kwargs
     - kwargs to pass to optimizer constructor
   * - rollout_horizon
     - Rollout horizon of the computation graph
   * - use64bit
     - Whether to use 64 bit precision, instead of the default 32
   * - use_symlog_reward
     - Whether to apply the symlog transform to the immediate reward
   * - utility
     - A utility function to optimize instead of expected return
   * - utility_kwargs
     - kwargs to pass hyper-parameters to the utility



.. list-table:: ``method_kwargs`` in ``[Optimizer]`` for ``JaxStraightLinePlan``
   :widths: 60 60
   :header-rows: 1

   * - Setting
     - Description
   * - initializer
     - Type of ``jax.nn.initializers``, specifies parameter initialization
   * - initializer_kwargs
     - kwargs to pass to the initializer
   * - max_constraint_iter
     - Maximum iterations of gradient projection for boolean action preconditions
   * - min_action_prob
     - Minimum probability of boolean action to avoid sigmoid saturation
   * - use_new_projection
     - Whether to use new gradient projection for boolean action preconditions
   * - wrap_non_bool
     - Whether to wrap non-boolean actions with nonlinearity for box constraints
   * - wrap_sigmoid
     - Whether to wrap boolean actions with sigmoid
   * - wrap_softmax
     - Whether to wrap with softmax to satisfy boolean action preconditions


.. list-table:: ``method_kwargs`` in ``[Optimizer]`` for ``JaxDeepReactivePolicy``
   :widths: 60 60
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
     - Whether to apply layer norm to inputs
   * - normalize_per_layer
     - Whether to apply layer norm to each input individually
   * - normalizer_kwargs
     - kwargs to pass to ``haiku.LayerNorm`` constructor for layer norm
   * - topology
     - List specifying number of neurons per hidden layer
   * - wrap_non_bool
     - Whether to wrap non-boolean actions with nonlinearity for box constraints   


.. list-table:: ``[Training]``
   :widths: 60 60
   :header-rows: 1

   * - Setting
     - Description
   * - epochs
     - Maximum number of iterations of gradient descent   
   * - key
     - An integer to seed the RNG with for reproducibility
   * - model_params
     - Dictionary of hyper-parameter values to pass to the model relaxation
   * - plot_kwargs
     - kwargs to pass to plan visualizer constructor (see next table for options)
   * - plot_step
     - How often to update the plan visualizer
   * - policy_hyperparams
     - Dictionary of hyper-parameter values to pass to the policy
   * - print_progress
     - Whether to print the progress bar from the planner to console
   * - print_summary
     - Whether to print summary information from the planner to console
   * - test_rolling_window
     - Smoothing window over which to calculate test return
   * - train_seconds
     - Maximum seconds to train for


.. list-table:: ``plot_kwargs`` in ``[Training]``
   :widths: 60 60
   :header-rows: 1

   * - Setting
     - Description   
   * - show_action
     - Whether to plot the average actions of the policy
   * - show_violin
     - Whether to plot the return distribution
     

Boolean Actions
-------------------

By default, boolean actions are wrapped using the sigmoid function:

.. math::
    
    a = \frac{1}{1 + e^{-w \theta}},

where :math:`\theta` denotes the trainable action parameters, and :math:`w` denotes a 
hyper-parameter that controls the sharpness of the approximation.

.. warning::
   If the sigmoid wrapping is used, then the weights ``w`` should be specified in 
   ``policy_hyperparams`` for each boolean action fluent when interfacing with the planner.
   
At test time, the action is aliased by evaluating the expression 
:math:`a > 0.5`, or equivalently :math:`\theta > 0`.
The sigmoid wrapper can be disabled by setting ``wrap_sigmoid = False``, 
but this is not recommended.


Constraints on Action Fluents
-------------------

Currently, the JAX planner supports two different kind of actions constraints.

Box constraints are useful for bounding each action fluent independently within some range.
Box constraints typically do not need to be specified manually, since they are automatically 
parsed from the ``action_preconditions`` as defined in the RDDL domain description file.

However, if the user wishes, it is possible to override these bounds
by passing a dictionary of bounds for each action fluent into the ``action_bounds`` argument. 
The syntax for specifying optional box constraints in the ``[Optimizer]`` section of the config file is:

.. code-block:: shell
	
    [Optimizer]
    ...
    action_bounds={ <action_name1>: (lower1, upper1), <action_name2>: (lower2, upper2), ... }
   
where ``lower#`` and ``upper#`` can be any list or nested list.

By default, the box constraints on actions are enforced using the projected gradient method.
An alternative approach is to map the actions to the box via a differentiable transformation, 
as described by `equation 6 in this paper <https://ojs.aaai.org/index.php/AAAI/article/view/4744>`_.
In the JAX planner, this can be enabled by setting ``wrap_non_bool = True``. 

Concurrency constraints are typically of the form :math:`\sum_i a_i \leq B` for some constant :math:`B`.
If the ``max-nondef-actions`` property in the RDDL instance is less 
than the total number of boolean action fluents, then ``JaxBackpropPlanner`` will automatically 
apply a projected gradient step to ensure this constraint is satisfied at each optimization step, as described 
`in this paper <https://ojs.aaai.org/index.php/ICAPS/article/view/3467>`_.

.. note::
   Concurrency constraints on action-fluents are applied to boolean actions only: 
   e.g., real and int actions are currently ignored.


Reward Normalization
-------------------

Some domains yield rewards that vary significantly in magnitude between time steps, 
making optimization difficult without some kind of normalization.
Following `this paper <https://arxiv.org/pdf/2301.04104v1.pdf>`_, pyRDDLGym-jax can apply a 
symlog transform to the sampled rewards during backprop:

.. math::
    
    \mathrm{symlog}(x) = \mathrm{sign}(x) * \ln(|x| + 1)

which compresses the magnitudes of large positive or negative outcomes.
This can be enabled by setting ``use_symlog_reward = True`` in ``JaxBackpropPlanner``.


Utility Optimization
-------------------

By default, the JAX planner will optimize the expected sum of future reward, 
which may not be desirable for risk-sensitive applications where tail risk of the returns is important.
Following `this paper <https://ojs.aaai.org/index.php/AAAI/article/view/21226>`_, 
it is possible to optimize a non-linear utility of the return instead.

The JAX planner currently supports several utility functions:

* "mean" is the risk-neutral or ordinary expected return
* "mean_var" is the variance penalized return
* "entropic" is the entropic or exponential utility
* "cvar" is the conditional value at risk.

The utility function can be specified by passing a string or function to the ``utility`` argument of the planner,
and its hyper-parameters can be passed through the ``utility_kwargs`` argument, 
which accepts a dictionary of name, value pairs.

For example, to set the CVAR utility at 5 percent:

.. code-block:: python

    planner = JaxBackpropPlanner(..., utility="cvar", utility_kwargs={'alpha': 0.05})
   
Similarly, to set the entropic utility with risk aversion parameter 2:

.. code-block:: python

    planner = JaxBackpropPlanner(..., utility="entropic", utility_kwargs={'beta': 2.0})

The utility function could also be provided explicitly as a callable that maps a JAX array to a scalar, 
with additional arguments specifying the hyper-parameters of the utility function referred to by name:

.. code-block:: python
    import jax

    @jax.jit
    def my_utility_function(x: jax.numpy.ndarray, aversion: float=1.0) -> float:
        return ...
        
    planner = JaxBackpropPlanner(..., utility=my_utility_function, utility_kwargs={'aversion': 2.0})
    

Using Another Planning Algorithm
-------------------

In the :ref:`introductory example <jax-intro>`, we defined the planning algorithm separately from the controller.
Therefore, it is possible to incorporate new planning algorithms simply by extending the 
``JaxBackpropPlanner`` class. 

pyRDDLGym-jax currently provides one such extension based on
`backtracking line-search <https://en.wikipedia.org/wiki/Backtracking_line_search>`_, which 
adaptively selects a learning rate at each iteration whose gradient update 
provides the greatest improvement in the return. 

This optimizer can be used as a drop-in replacement for ``JaxBackpropPlanner`` as follows:

.. code-block:: python

    from pyRDDLGym_jax.core.planner import JaxLineSearchPlanner, JaxOfflineController
    
    planner = JaxLineSearchPlanner(env.model, **planner_args)
    controller = JaxOfflineController(planner, **train_args)

Like the default planner, the line-search planner is compatible with offline and online controllers, 
and straight-line plans and deep reactive policies.


Automatically Tuning Hyper-Parameters
-------------------

pyRDDLGym-jax provides a Bayesian optimization algorithm for automatically tuning 
key hyper-parameters of the planner, which:

* supports multi-processing by evaluating multiple hyper-parameter settings in parallel
* leverages Bayesian optimization to search the hyper-parameter space more efficiently
* supports straight-line planning, replanning, and deep reactive policies.

The automatic tuning can be performed as follows:

.. code-block:: python

    import pyRDDLGym
    from pyRDDLGym_jax.core.tuning import JaxParameterTuningSLP
    
    # set up the environment   
    env = pyRDDLGym.make(domain, instance, vectorized=True)
    
    # set up the tuning instance
    tuning = JaxParameterTuningSLP(env=env,
                                   train_epochs=epochs,
                                   timeout_training=timeout,
                                   eval_trials=trials,
                                   planner_kwargs=planner_args,
                                   plan_kwargs=plan_args,
                                   num_workers=workers,
                                   gp_iters=iters)

    # tune and report the best hyper-parameters found
    best = tuning.tune(key=key, filename="/path/to/log.csv")
    print(f'best parameters found: {best}')
    
The ``__init__`` method requires the ``num_workers`` parameter to specify the 
number of parallel processes and the ``gp_iters`` to specify the number of iterations of Bayesian optimization. 

Upon executing this code, a dictionary of the best hyper-parameters 
(e.g. learning rate, policy network architecture, model hyper-parameters, etc.) is returned.
A log of the previous sets of hyper-parameters suggested by the algorithm is also recorded.

Policy networks and replanning can be tuned by replacing ``JaxParameterTuningSLP`` with 
``JaxParameterTuningDRP`` and ``JaxParameterTuningSLPReplan``, respectively. 
This will also tune the architecture (number of neurons, layers) of the policy network 
and the ``rollout_horizon`` for replanning.


Dealing with Non-Differentiable Expressions
-------------------

Many RDDL programs contain expressions that do not support derivatives.
A common technique to deal with this is to rewrite non-differentiable operations as similar differentiable ones.

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

pyRDDLGym-jax works around these limitations by approximating such operations with 
JAX expressions that support derivatives.
For instance, the ``classify`` function above could be implemented as follows:
 
.. code-block:: python

    from pyRDDLGym_jax.core.logic import FuzzyLogic

    logic = FuzzyLogic()    
    And, _ = logic.logical_and()
    Not, _ = logic.logical_not()
    Gre, _ = logic.greater()
    Or, _ = logic.logical_or()
    If, _ = logic.control_if()

    def approximate_classify(x1, x2, w):
        q1 = And(Gre(x1, 0, w), Gre(x2, 0, w), w)
        q2 = And(Not(Gre(x1, 0, w), w), Not(Gre(x2, 0, w), w), w)
        cond = Or(q1, q2, w)
        return If(cond, +1, -1, w)

Calling ``approximate_classify`` with ``x=0.5``, ``y=1.5`` and ``w=10`` returns 0.98661363, 
which is very close to 1.

The ``FuzzyLogic`` instance can be passed to a planner through the config file, or directly as follows:

.. code-block:: python
    
    from pyRDDLGym.core.logic import FuzzyLogic
    planner = JaxBackpropPlanner(model, ..., logic=FuzzyLogic())

By default, ``FuzzyLogic`` uses the `product t-norm <https://en.wikipedia.org/wiki/T-norm_fuzzy_logics#Motivation>`_
to approximate the logical operations, the standard complement :math:`\sim a \approx 1 - a`, and
sigmoid approximations for other relational and functional operations.

The latter introduces model hyper-parameters :math:`w`, which control the "sharpness" of the operation.
Higher values mean the approximation approaches its exact counterpart, at the cost of sparse and 
possibly numerically unstable gradients. 

These hyper-parameters be retrieved and modified at run-time, such as during optimization, as follows:

.. code-block:: python

    model_params = planner.compiled.model_params
    model_params[key] = ...
    planner.optimize(..., model_params=model_params)

The following table summarizes the default rules used in ``FuzzyLogic``.

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
     - :math:`c * a + (1 - c) * b`
   * - :math:`a == b`
     - :math:`1 - \tanh^2(w * (a - b))`
   * - :math:`a > b`, :math:`a >= b`
     - :math:`\mathrm{sigmoid}(w * (a - b))`
   * - :math:`\mathrm{signum}(a)`
     - :math:`\tanh(w * a)`
   * - argmax_{?p : type} x(?p)
     - :math:`\sum_{i = 1, 2, \dots |\mathrm{type}|} i * \mathrm{softmax}(w * x)[i]`
   * - Bernoulli(p)
     - Gumbel-Softmax trick
   * - Discrete(type, {cases ...} )
     - Gumbel-Softmax trick

It is possible to control these rules by subclassing ``FuzzyLogic``, or by 
passing different values to the ``tnorm`` or ``complement`` arguments.

   
Manual Gradient Calculation
-------------------

As of version 0.3, it is possible to export the optimization problem for a given problem
to another optimizer (for example scipy). 

To do this, call the ``as_optimization_problem`` function on the planner to rewrite
the planning problem in terms of functions over 1D numpy arrays:

.. code-block:: python
    
    planner = JaxBackpropPlanner(rddl=..., **planner_args)
    loss_fn, grad_fn, guess, unravel_fn = planner.as_optimization_problem()

The loss function ``loss_fn`` and gradient map ``grad_fn`` express policy parameters as 1D numpy arrays,
so they can be used as inputs for other packages that do not make use of JAX. The 
``unravel_fn`` allows the 1D array to be mapped back to a JAX pytree.

For example, to optimize and evaluate a policy using scipy, please see the 
`worked example here <https://github.com/pyrddlgym-project/pyRDDLGym-jax/blob/main/pyRDDLGym_jax/examples/run_scipy.py>`_.

The API also supports manual return gradient calculations for custom applications.
For details, please see the 
`worked example here <https://github.com/pyrddlgym-project/pyRDDLGym-jax/blob/main/pyRDDLGym_jax/examples/run_gradient.py>`_.


Limitations
-------------------

We cite several limitations of the current JAX planner:

* Not all operations have natural differentiable relaxations. Currently, the following are not supported:
	* nested fluents such as ``fluent1(fluent2(?p))``
	* distributions that are not naturally reparameterizable such as Poisson, Gamma and Beta
* Some relaxations can accumulate high error
	* this is particularly problematic when stacking CPFs for long roll-out horizons, so we recommend reducing or tuning the rollout-horizon for best results
* Some relaxations may not be mathematically consistent with one another:
	* no guarantees are provided about dichotomy of equality, e.g. a == b, a > b and a < b do not necessarily "sum" to one, but in many cases should be close
	* if this is a concern, it is recommended to override some operations in ``ProductLogic`` to suit the user's needs
* Termination conditions and state/action constraints are not considered in the optimization
	* constraints are logged in the optimizer callback and can be used to define loss functions that take the constraints into account
* The optimizer can fail to make progress when the structure of the problem is largely discrete:
	* to diagnose this, compare the training loss to the test loss over time, and at the time of convergence
	* a low, or drastically improving, training loss with a similar test loss indicates that the continuous model relaxation is likely accurate around the optimum
	* on the other hand, a low training loss and a high test loss indicates that the continuous model relaxation is poor.

The goal of the JAX planner is to provide a simple baseline that can be easily built upon.
However, we welcome any suggestions or modifications about how to improve the robustness of the JAX planner 
on a broader subset of RDDL.


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

If you use the utility optimization setting, please include:

.. code-block:: bibtex

    @inproceedings{patton2022distributional,
        title={A distributional framework for risk-sensitive end-to-end planning in continuous mdps},
        author={Patton, Noah and Jeong, Jihwan and Gimelfarb, Mike and Sanner, Scott},
        booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
        volume={36},
        number={9},
        pages={9894--9901},
        year={2022}
    }
    
