.. _jaxplan:

pyRDDLGym-jax: JAX Compiler and Planner
===============

In this tutorial, we discuss how a RDDL model can be compiled into a differentiable simulator using JAX. 
We also show how gradient ascent can be used to estimate optimal actions.

Requirements
------------
This package requires Python 3.8+

* pyRDDLGym>=2.0
* tqdm>=4.66
* jax>=0.4.12
* optax>=0.1.9
* dm-haiku>=0.0.10 
* tensorflow>=2.13.0
* tensorflow-probability>=0.21.0

To run the hyper-parameter tuning, you will also need

* bayesian-optimization>=1.4.3


Installing via pip
-----------------

You can install pyRDDLGym-jax and all of its requirements via pip:

.. code-block:: shell

    pip install pyRDDLGym-jax


Installing the Pre-Release Version via git
---------
.. code-block:: shell

    pip install git+https://github.com/pyrddlgym-project/pyRDDLGym-jax.git


Changing the Simulation Backend to JAX
-------------------

By default, pyRDDLGym simulates using Python and stores the outputs of intermediate expressions in NumPy arrays.
However, if additional structure such as gradients are required, or if simulation is slow using the default backend, 
the environment can be compiled using JAX by changing the backend:

.. code-block:: python
	
	import pyRDDLGym
	from pyRDDLGym_jax.core.simulator import JaxRDDLSimulator
	env = pyRDDLGym.make("Cartpole_Continuous_gym", "0", backend=JaxRDDLSimulator)
	
This is designed to be interchangeable with the default backend, so the numerical results should be the same.

.. note::
   All RDDL syntax (both new and old) is supported in the RDDL-to-JAX compiler.


Differentiable Planning: Deterministic Domains
-------------------

The planning problem for a deterministic environment involves finding actions 
that maximize accumulated reward over a fixed horizon (an open-loop plan)

.. math::

	\max_{a_1, \dots a_T} \sum_{t=1}^{T} R(s_t, a_t),\\
	s_{t + 1} = f(s_t, a_t)
	
If the state and action spaces are continuous, and f and R are differentiable functions, 
it is possible to apply gradient ascent to optimize the actions directly as described 
`in this paper <https://proceedings.neurips.cc/paper/2017/file/98b17f068d5d9b7668e19fb8ae470841-Paper.pdf>`_.
Given a learning rate :math:`\eta > 0` and "guess" :math:`a_\tau`, gradient ascent produces an estimate of the optimal 
action :math:`a_\tau'` at time :math:`\tau` as

.. math::
	
	a_{\tau}' = a_{\tau} + \eta \sum_{t=1}^{T} \nabla_{a_\tau} R(s_t, a_t),
	
where the gradient of the reward at all times :math:`t \geq \tau` can be computed using the chain rule:

.. math::

	\nabla_{a_\tau} R(s_t, a_t) = \frac{\mathrm{d}R(s_t,a_t)}{\mathrm{d}s_t} \frac{\mathrm{d}s_t}{\mathrm{d}a_\tau} + \frac{\mathrm{d}R(s_t,a_t)}{\mathrm{d}a_t}\frac{\mathrm{d}a_t}{\mathrm{d}a_\tau}.

In domains with stochastic transitions, an open-loop plan could be sub-optimal 
as it does not correct for deviations in the state from its expected course as anticipated during optimization.
One solution is to recompute the plan periodically or after each decision epoch, and is often called "replanning".

An alternative approach to replanning is to learn a policy network 
:math:`a_t \gets \pi_\theta(s_t)` that maps the states to actions, such as a feed-forward neural network
as explained `in this paper <https://ojs.aaai.org/index.php/AAAI/article/view/4744>`_. 


Differentiable Planning: Stochastic Domains
-------------------

A common problem of planning in stochastic domains is that the gradients are no longer well-defined.
pyRDDLGym-jax works around this problem by using the reparameterization trick.

To illustrate, we can write :math:`s_{t+1} = \mathcal{N}(s_t, a_t^2)` as :math:`s_{t+1} = s_t + a_t * \mathcal{N}(0, 1)`, 
although the latter is amenable to backpropagation while the first is not. 

The reparameterization trick also works generally, assuming there exists a closed-form function f such that

.. math::

    s_{t+1} = f(s_t, a_t, \xi_t)
    
and :math:`\xi_t` are random variables drawn from some distribution independent of states or actions. 
For a detailed discussion of reparameterization in the context of planning, 
please see `this paper <https://ojs.aaai.org/index.php/AAAI/article/view/4744>`_ 
or `this one <https://ojs.aaai.org/index.php/AAAI/article/view/21226>`_.

pyRDDLGym-jax automatically performs reparameterization whenever possible. For some special cases,
such as the Bernoulli and Discrete distribution, it applies the Gumbel-softmax trick 
as described `in this paper <https://arxiv.org/pdf/1611.01144.pdf>`_. 
Defining K independent samples from a standard Gumbel distribution :math:`g_1, \dots g_K`, we reparameterize the 
random variable :math:`X` with probability mass function :math:`p_1, \dots p_K` as

.. math::

    X = \arg\!\max_{i=1\dots K} \left(g_i + \log p_i \right)

where the argmax is approximated using the softmax function.

.. warning::
   For general non-reparameterizable distributions, the result of the gradient calculation 
   is fully dependent on the JAX implementation: it could return a zero or NaN gradient, or raise an exception.


Running the Basic Example
-------------------

A basic run script is provided to run the Jax Planner on any domain in rddlrepository, 
provided a config file of hyper-parameters is available (currently, only a limited subset of configs are provided). 
The example can be run as follows in a standard shell, from the install directory of pyRDDLGym-jax:

.. code-block:: shell
    
    python -m pyRDDLGym_jax.examples.run_plan <domain> <instance> <method> <episodes>
    
where:

* ``<domain>`` is the domain identifier as specified in rddlrepository, or a path pointing to a valid domain.rddl file
* ``<instance>`` is the instance identifier in rddlrepository, or a path pointing to a valid instance.rddl file
* ``<method>`` is the planning method to use (see below)
* ``<episodes>`` is the (optional) number of episodes to evaluate the learned policy.

The ``<method>`` parameter warrants further explanation. Currently we support three possible modes:

* ``slp`` is the straight-line open-loop planner described `in this paper <https://proceedings.neurips.cc/paper/2017/file/98b17f068d5d9b7668e19fb8ae470841-Paper.pdf>`_
* ``drp`` is the deep reactive policy network described `in this paper <https://ojs.aaai.org/index.php/AAAI/article/view/4744>`_
* ``replan`` is the same as ``slp`` except it uses periodic replanning as described above.

For example, copy and pasting the following will train the JAX Planner on the Quadcopter domain with 4 drones:

.. code-block:: shell

    python -m pyRDDLGym_jax.examples.run_plan Quadcopter 1 slp
   

Running from the Python API
-------------------

pyRDDLGym-jax provides convenient tools to automatically compile a RDDL description of a problem to the above optimization problem:

.. code-block:: python

    import pyRDDLGym
    from pyRDDLGym_jax.core.planner import JaxBackpropPlanner, JaxOfflineController

    # set up the environment (note the vectorized option must be True)
    env = pyRDDLGym.make("domain", "instance", vectorized=True)

    # create the planning algorithm
    planner = JaxBackpropPlanner(rddl=env.model, **planner_args)
    controller = JaxOfflineController(planner, **train_args)

    # evaluate the planner
    controller.evaluate(env, episodes=1, verbose=True, render=True)

    env.close()

Here, we have used an open-loop (offline) controller. 
To use periodic replanning, simply change the controller type to online:

.. code-block:: python

    controller = JaxOnlineController(planner, **train_args)	

.. note::
   All controllers are instances of pyRDDLGym's ``BaseAgent`` and support the ``evaluate()`` function. 

The ``**planner_args`` and ``**train_args`` are keyword arguments passed during initialization, 
but we strongly recommend creating and loading a configuration file as discussed next.


Writing Configuration Files for Custom Problems
-------------------

The recommended way to load planner and training arguments is to write a configuration file with all the necessary hyper-parameters. 
The basic structure of a configuration file is provided below for open-loop planning or replanning:

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
    planner_args, _, train_args = load_config("/path/to/config.cfg")
    
    # continue as described above
    planner = ...
    controller = ...

.. note::
   The ``rollout_horizon`` in the configuration file is optional, and defaults to the horizon specified in the RDDL description. 
   For replanning methods, we recommend setting this parameter manually, and tuning it to get the best result.


Writing Configuration Files for Policy Networks
-------------------

To use a policy network instead of an open-loop plan or replanning, 
change the ``method`` in the ``[Optimizer]`` section of the config file:

.. code-block:: shell

    ...
    [Optimizer]
    method='JaxDeepReactivePolicy'
    method_kwargs={'topology': [128, 64]}
    ...

This creates a neural network policy with the default ReLU activations, and two hidden layers with 128 and 64 neurons.

.. note::
   ``JaxStraightlinePlan`` and ``JaxDeepReactivePolicy`` are instances of the abstract class ``JaxPlan``. 
   Other policy representations could be defined by overriding this class and its abstract methods.


Boolean Actions
-------------------

By default, boolean actions are wrapped using the sigmoid function:

.. math::
    
    a = \frac{1}{1 + e^{-w \theta}},

where :math:`\theta` denotes the trainable action parameters, and :math:`w` denotes a 
hyper-parameter that controls the sharpness of the approximation.

.. warning::
   If the sigmoid wrapping is used, then the weights ``w`` must be specified in 
   ``policy_hyperparams`` for each boolean action fluent when interfacing with the planner.
   
At test time, the action is aliased by evaluating the expression :math:`a > 0.5`, or equivalently :math:`\theta > 0`.
The use of sigmoid for boolean actions can be disabled by setting ``wrap_sigmoid = False``, but this is not recommended.


Constraints on Action Fluents
-------------------

Currently, the JAX planner supports two different kind of actions constraints: box constraints and concurrency constraints. 

Box constraints are useful for bounding each action fluent independently into some range.
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
In the JAX planner, it is possible to switch to the transformation method by setting ``wrap_non_bool = True``. 

The JAX planner also supports concurrency constraints on actions of the form 
:math:`\sum_i a_i \leq B` for some constant :math:`B`.
If the ``max-nondef-actions`` property in the RDDL instance is less 
than the total number of boolean action fluents, then ``JaxRDDLBackpropPlanner`` will automatically 
apply a projected gradient step to ensure this constraint is satisfied at each optimization step, as described 
`in this paper <https://ojs.aaai.org/index.php/ICAPS/article/view/3467>`_.

.. note::
   Concurrency constraints on action-fluents are applied to boolean actions only: e.g., real and int actions are ignored.


Reward Normalization
-------------------

Some domains yield rewards that vary significantly in magnitude between time steps, 
making optimization difficult without some form of normalization.
Following `this paper <https://arxiv.org/pdf/2301.04104v1.pdf>`_, pyRDDLGym-jax can apply a 
symlog transform to the sampled rewards during backprop:

.. math::
    
    \mathrm{symlog}(x) = \mathrm{sign}(x) * \ln(|x| + 1)

which compresses the magnitudes of large positive and negative outcomes.
The use of symlog can be enabled by setting ``use_symlog_reward = True`` in ``JaxBackpropPlanner``.


Utility Optimization
-------------------

By default, the JAX planner will optimize the expected sum of future reward, which may not be desirable for risk-sensitive applications.
Following the framework `in this paper <https://ojs.aaai.org/index.php/AAAI/article/view/21226>`_, 
it is possible to optimize a non-linear utility of the return instead.

For example, the entropic utility with risk-aversion parameter :math:`\beta` is

.. math::
    
    U(a_1, \dots a_T) = -\frac{1}{\beta} \log \mathbb{E}\left[e^{-\beta \sum_t R(s_t, a_t)} \right]

This can be passed to the planner as follows:

.. code-block:: python

    import jax.numpy as jnp
    
    def entropic(x, beta=0.00001):
        return (-1.0 / beta) * jnp.log(jnp.mean(jnp.exp(-beta * x)) + 1e-12)
       
    planner = JaxRDDLBackpropPlanner(..., utility=entropic)
    ...


Changing the Planning Algorithm
-------------------

In the introductory example, you may have noticed that we defined the planning algorithm separately from the controller.
Therefore, it is possible to incorporate new planning algorithms simply by extending the ``JaxBackpropPlanner`` class. 

pyRDDLGym-jax currently provides one such extension based on backtracking line-search, which 
adaptively selects a learning rate at each iteration whose gradient update 
provides the greatest improvement in the return objective. 

This optimizer can be used as a drop-in replacement for ``JaxRDDLBackpropPlanner`` as follows:

.. code-block:: python

    from pyRDDLGym_jax.core.planner import JaxRDDLArmijoLineSearchPlanner, JaxOfflineController
    
    planner = JaxRDDLArmijoLineSearchPlanner(env.model, **planner_args)
    controller = JaxOfflineController(planner, **train_args)

Like the default planner, the line-search planner is compatible with offline and online controllers, 
and straight-line plans and deep reactive policies.


Automatically Tuning Hyper-Parameters
-------------------

pyRDDLGym-jax provides a Bayesian optimization algorithm for automatically tuning key hyper-parameters of the planner. 
It:

* supports multi-processing by evaluating multiple hyper-parameter settings in parallel
* leverages Bayesian optimization to perform more efficient search than random or grid search
* supports straight-line planning and deep reactive policies

The key hyper-parameters can be tuned as follows:

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
A log of the previous sets of hyper-parameters suggested by the algorithm is also recorded in the specified output file.

Policy networks and replanning can be tuned by replacing ``JaxParameterTuningSLP`` with 
``JaxParameterTuningDRP`` and ``JaxParameterTuningSLPReplan``, respectively. 
This will also tune the architecture (number of neurons, layers) of the policy network and the ``rollout_horizon`` for replanning.


Dealing with Non-Differentiable Expressions
-------------------

Many RDDL programs contain expressions that do not support derivatives.
A common technique to deal with this is to rewrite non-differentiable operations as similar differentiable ones.
For instance, consider the following problem of classifying points (x, y) in 2D-space as 
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

pyRDDLGym-jax works around these limitations by approximating such operations with JAX expressions that support derivatives.
For instance, the ``classify`` function above could be implemented as follows:
 
.. code-block:: python

    from pyRDDLGym_jax.core.logic import FuzzyLogic

    logic = FuzzyLogic()    
    And, _ = logic.And()
    Not, _ = logic.Not()
    Gre, _ = logic.greater()
    Or, _ = logic.Or()
    If, _ = logic.If()

    def approximate_classify(x1, x2, w):
        q1 = And(Gre(x1, 0, w), Gre(x2, 0, w), w)
        q2 = And(Not(Gre(x1, 0, w), w), Not(Gre(x2, 0, w), w), w)
        cond = Or(q1, q2, w)
        return If(cond, +1, -1, w)

Calling ``approximate_classify`` with ``x=0.5``, ``y=1.5`` and ``w=10`` returns 0.98661363, which is very close to 1.

The ``FuzzyLogic`` instance can be passed to a planner through the config file, or directly as follows:

.. code-block:: python
    
    from pyRDDLGym.core.logic import FuzzyLogic
    planner = JaxRDDLBackpropPlanner(model, ..., logic=FuzzyLogic())

By default, ``FuzzyLogic`` uses the `product t-norm <https://en.wikipedia.org/wiki/T-norm_fuzzy_logics#Motivation>`_
to approximate the logical operations, the standard complement :math:`\sim a \approx 1 - a`, and
sigmoid approximations for other relational and functional operations.

The latter introduces model hyper-parameters :math:`w`, which control the "sharpness" of the operation.
Higher values mean the approximation approaches its exact counterpart, at the cost of sparse and possibly numerically unstable gradients. 

These can be retrieved and modified at run-time, such as during optimization, as follows:

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
     - :math:`\frac{\mathrm{sigmoid}(w * (a - b + 0.5)) - \mathrm{sigmoid}(w * (a - b - 0.5))}{\tanh(0.25 * w)}`
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
passing different values to the ``tnorm`` or ``complement`` arguments to replace the product t-norm logic and
standard complement, respectively.


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
* Termination conditions and state/action constraints are not considered in the optimization (but can be checked at test-time).
* The optimizer can fail to make progress when the structure of the problem is largely discrete:
	* to diagnose this, compare the training loss to the test loss over time, and at the time of convergence
	* a low, or drastically improving, training loss with a similar test loss indicates that the continuous model relaxation is likely accurate around the optimum
	* on the other hand, a low training loss and a high test loss indicates that the continuous model relaxation is poor, in which case the optimality of the solution should be questioned.

The goal of the JAX planner was not to replicate the state-of-the-art, but to provide a simple baseline that can be easily built-on.
However, we welcome any suggestions or modifications about how to improve this algorithm on a broader subset of RDDL.
