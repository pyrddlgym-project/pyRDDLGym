Baselines: The Gradient-Based JAX Planner
===============

In this tutorial, we discuss how a RDDL model can be compiled into a differentiable simulator using JAX. 
We also show how gradient descent can be used to estimate optimal actions/controls.

Changing the Simulation Backend
-------------------

By default, ``RDDLEnv`` simulates using Python and stores intermediate expressions in NumPy arrays.
However, if additional structure such as gradients are required, or if simulation is slow, 
the environment can be compiled using JAX by changing the backend:

.. code-block:: python
	
	from pyRDDLGym.Core.Jax.JaxRDDLSimulator import JaxRDDLSimulator
	env = RDDLEnv.build(info, instance, backend=JaxRDDLSimulator)
	
which is designed to be interchangeable with the default backend.

.. note::
   All RDDL syntax (both new and old) is supported in the RDDL-to-JAX compiler.

Open-Loop Planning with JAX
-------------------

We now turn our attention to optimization. The planning problem for a deterministic environment involves finding actions 
that maximize accumulated reward over a fixed horizon

.. math::

	\max_{a_1, \dots a_T} \sum_{t=1}^{T} R(s_t, a_t),\\
	s_{t + 1} = f(s_t, a_t)
	
If the state and action spaces are continuous, and f and R are differentiable functions, 
it is possible to apply gradient ascent to optimize the actions directly.
Given a learning rate :math:`\eta > 0` and "guess" :math:`a_\tau`, gradient ascent produces an estimate of the optimal 
action :math:`a_\tau'` at time :math:`\tau` as

.. math::
	
	a_{\tau}' = a_{\tau} + \eta \sum_{t=1}^{T} \nabla_{a_\tau} R(s_t, a_t),
	
where the gradient of the reward at all times :math:`t \geq \tau` can be computed using the chain rule:

.. math::

	\nabla_{a_\tau} R(s_t, a_t) = \frac{\mathrm{d}R(s_t,a_t)}{\mathrm{d}s_t} \frac{\mathrm{d}s_t}{\mathrm{d}a_\tau} + \frac{\mathrm{d}R(s_t,a_t)}{\mathrm{d}a_t}\frac{\mathrm{d}a_t}{\mathrm{d}a_\tau}.
	
This approach is further described 
`in this paper <https://proceedings.neurips.cc/paper/2017/file/98b17f068d5d9b7668e19fb8ae470841-Paper.pdf>`_.

We now describe the process of creating a differentiable planner for solving the above problem.

First, a configuration file with extension ``.cfg`` is created to store and pass hyper-parameter settings to the planner.
It is also possible to pass parameters directly to the planner, but this is highly discouraged since the
planner is quite complex and requires a large number of parameters. A number of sample configuration files for different 
environments are provided in the ``JaxPlanConfigs`` directory of pyRDDLGym. 

The typical structure of the configuration file is:

.. code-block:: shell

    [Model]
    logic='FuzzyLogic'
    logic_kwargs={'weight': 100}
    tnorm='ProductTNorm'
    tnorm_kwargs={}
	
    [Optimizer]
    method='JaxStraightLinePlan'
    method_kwargs={}
    optimizer='rmsprop'
    optimizer_kwargs={'learning_rate': 0.01}
    batch_size_train=32
    batch_size_test=32

    [Training]
    key=42
    epochs=1000
    train_seconds=30
    policy_hyperparams=...

There are three sections, corresponding to model, optimizer and training hyper-parameters:

* the ``[Model]`` section dictates how non-differentiable expressions are handled (as discussed later in the tutorial)
* the ``[Optimizer]`` section contains a ``method`` argument to indicate the type of plan/policy, its hyper-parameters, the ``optax`` SGD optimizer and its hyper-parameters, etc.
* the ``[Training]`` section indicates budget on iterations or time, hyper-parameters for the policy, etc.

Configuration files can be loaded through a convenience function which returns the 
parsed parameters for the planner, the plan/policy, and the training arguments:

.. code-block:: python

    from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import load_config
    planner_args, plan_args, train_args = load_config(config_path)

Next, a planning algorithm must be initialized by feeding the parameters above, which provides the training/optimization loop:

.. code-block:: python

    from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxRDDLBackpropPlanner
    planner = JaxRDDLBackpropPlanner(env.model, **planner_args)

Finally, a controller must be initialized, which is a policy that calls the planning algorithm above to produce optimal actions. 
The controller is a policy instance in pyRDDLGym, so the usual ``sample_action()`` and ``evaluate()`` functions allow easy interaction with the environment.

.. code-block:: python

    from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxOfflineController
    controller = JaxOfflineController(planner, **train_args)
    controller.evaluate(env, verbose=True, render=True)

Putting this all together into a working example:

.. code-block:: python

    from pyRDDLGym.Core.Env.RDDLEnv import RDDLEnv
    from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import load_config
    from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxRDDLBackpropPlanner
    from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxOfflineController
    from pyRDDLGym.Examples.ExampleManager import ExampleManager

    # create the environment
    info = ExampleManager.GetEnvInfo(domain)    
    env = RDDLEnv.build(info, instance)
    
    # load the config file with planner settings from the JaxPlanConfigs
    planner_args, _, train_args = load_config(config_path)
    
    # create the planning algorithm, controller and begin training immediately
    planner = JaxRDDLBackpropPlanner(env.model, **planner_args)
    controller = JaxOfflineController(planner, **train_args)
    controller.evaluate(env, verbose=True, render=True)

Open-Loop Replanning with Periodic Revision
-------------------

In domains with stochastic transitions, an open-loop plan could be sub-optimal 
since it does correct for any deviations in the state from its anticipated trajectory during optimization.
One solution is to recompute the plan periodically or after each decision epoch, and is often called "replanning".

Another problem of planning in stochastic domains is that the gradients are no longer well-defined.
pyRDDLGym works around this problem by using the reparameterization trick.
To illustrate this, if :math:`s_{t+1} = \mathcal{N}(s_t, a_t^2)`, then 
after reparametization :math:`s_{t+1} = s_t + a_t * \mathcal{N}(0, 1)`, and backpropagation can now be performed.
The reparameterization trick also works for general probability distributions, assuming there exists a closed-form function f such that

.. math::

    s_{t+1} = f(s_t, a_t, \xi_t)
    
where :math:`\xi_t` are random variables drawn from some concrete distribution independent of states or actions. 
For a detailed discussion of reparameterization in the context of planning, 
please see `this paper <https://ojs.aaai.org/index.php/AAAI/article/view/4744>`_ 
or `this one <https://ojs.aaai.org/index.php/AAAI/article/view/21226>`_.

pyRDDLGym will automatically perform reparameterization as needed assuming it is possible.
However, some probability distributions, such as the Beta distribution, do not have tractable reparameterizations.
For a small subset, like the Bernoulli and Discrete distribution, pyRDDLGym offers efficient 
approximations (see the Gumbel-softmax discussion below). 

.. warning::
   For general non-reparameterizable distributions, the result of the gradient calculation 
   is fully dependent on the JAX implementation: it could return a zero or NaN gradient, or raise an exception.

Replanning is easy by modifying the previous example. Instead of creating an 
``JaxOfflineController``, create an ``JaxOnlineController``:

.. code-block:: python

    from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxRDDLBackpropPlanner
    from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxOnlineController

    planner = JaxRDDLBackpropPlanner(env.model, **planner_args)
    controller = JaxOnlineController(planner, **train_args)
    controller.evaluate(env, verbose=True, render=True)

By comparing the realized return to the one obtained earlier, 
we observe that replanning performs much better in some cases than straight-line planning.

.. note::
   For replanning, we recommend specifying the ``rollout_horizon`` in the the configuration file to instruct 
   the planner to limit its simulation lookahead horizon during optimization, otherwise it will default to the 
   full horizon as defined in the RDDL environment which can lead to slower optimization times.

Policy Networks for Closed-Loop Planning
-------------------

An alternative approach to replanning is to learn a policy network 
:math:`a_t \gets \pi_\theta(s_t)` mapping the state to action, which is typically parameterized as a feed-forward neural network.

To do this, the configuration file must set the [Optimizer] method to ``JaxDeepReactivePolicy``, 
and must specify the number of layers, the number of neurons, and an activation function to use:

.. code-block:: python

    [Model]
    logic='FuzzyLogic'
    logic_kwargs={'weight': 100}
    tnorm='ProductTNorm'
    tnorm_kwargs={}

    [Optimizer]
    method='JaxDeepReactivePolicy'
    method_kwargs={'topology': [64, 64]}
    optimizer='rmsprop'
    optimizer_kwargs={'learning_rate': 0.01}
    batch_size_train=1
    batch_size_test=1
    action_bounds={'power-x': (-0.0999, 0.0999), 'power-y': (-0.0999, 0.0999)}

    [Training]
    key=42
    epochs=500
    train_seconds=30

Then an online or offline controller can be instantiated and trained using one of the previous code examples given.

.. note::
   ``JaxStraightlinePlan`` and ``JaxDeepReactivePolicy`` are instances of the abstract class ``JaxPlan``. 
   Other agent representations could be defined by overriding the ``JaxPlan`` class and its abstract methods.
   
Details about the implementation of the deep reactive policy for planning are explained 
`in this paper <https://ojs.aaai.org/index.php/AAAI/article/view/4744>`_. 

Box Constraints on Action Fluents
-------------------

Currently, the JAX planner supports two different kind of actions constraints: box constraints and concurrency constraints. 

Box constraints are useful for bounding each action fluent independently into some range, 
and can be specified by passing a dictionary of bounds for each action fluent into the ``action_bounds`` argument. 
The syntax for specifying box constraints in the [Optimizer] section of the configuration file is:

.. code-block:: shell
	
	[Optimizer]
	...
    action_bounds={ <action_name1>: (lower1, upper1), <action_name2>: (lower2, upper2), ... }
   
where ``lower#`` and ``upper#`` can be any floating point value, including positive and negative infinity. 
Passing ``None`` as a value to ``lower`` or ``upper`` indicates that a bound is not enforced, 
i.e. ``(10.0, None)`` indicates an action must be at least 10.

The bounds on actions are enforced by default using the projected gradient method.

Boolean Actions
-------------------

By default, boolean actions are wrapped using the sigmoid function:

.. math::
    
    a = \frac{1}{1 + e^{-w \theta}},

where :math:`\theta` denotes the trainable action parameters, and :math:`w` denotes a 
hyper-parameter that controls the sharpness of the approximation.

.. note::
   If ``wrap_sigmoid = True``, then the weights ``w`` as defined above must be specified in 
   ``policy_hyperparams`` for each boolean action fluent when interfacing with the planner.
   
At test time, the action is aliased by evaluating the expression :math:`a > 0.5`, or equivalently :math:`\theta > 0`.
The use of sigmoid for boolean actions can be controlled by setting ``wrap_sigmoid`` to True.
Non-boolean action fluents can also be wrapped in a similar way, bypassing the projected gradient trick, 
by setting ``wrap_non_bool = True``. This approach is `equation 6 in this paper <https://ojs.aaai.org/index.php/AAAI/article/view/4744>`_.
   
Concurrency Constraints on Action Fluents
-------------------

The JAX planner also supports concurrency constraints on actions of the form 
:math:`\sum_i a_i \leq B` for some constant :math:`B`.
Specifically, if the ``max-nondef-actions`` property in the RDDL instance is less 
than the total number of boolean action fluents, then ``JaxRDDLBackpropPlanner`` will automatically 
apply a projected gradient step to ensure ``max_nondef_actions`` is satisfied at each optimization step.

Two methods are provided to ensure constraint satisfaction: one is a simplified new projection technique and the other is 
described `in this paper <https://ojs.aaai.org/index.php/ICAPS/article/view/3467>`_
The choice of method can be controlled through the ``use_new_projection`` argument of the planner. 

.. note::
   Concurrency constraints on action-fluents are applied to boolean actions only: e.g., real and int actions are ignored.

Reward Normalization
-------------------

Some domains yield rewards that vary significantly in magnitude between time steps, 
making optimization difficult without some form of normalization.
Following `this paper <https://arxiv.org/pdf/2301.04104v1.pdf>`_, pyRDDLGym can apply a 
symlog transform to the sampled rewards during backprop:

.. math::
    
    \mathrm{symlog}(x) = \mathrm{sign}(x) * \ln(|x| + 1)

which compresses the magnitudes of large positive and negative outcomes.
The use of symlog can be enabled by setting ``use_symlog_reward`` to True in ``JaxBackpropPlanner``.

Utility Optimization
-------------------

By default, the JAX planner will optimize the expected sum of future reward, which may not be desirable for risk-sensitive applications.
Following the framework `in this paper <https://ojs.aaai.org/index.php/AAAI/article/view/21226>`_, 
it is possible to optimize a non-linear utility of the return instead.
For example, the entropic utility with risk-aversion parameter :math:`\beta` is

.. math::
    
    U(a_1, \dots a_T) = -\frac{1}{\beta} \log \mathbb{E}\left[e^{-\beta \sum_t R(s_t, a_t)} \right]

and can be passed to the planner as follows:

.. code-block:: python

    import jax.numpy as jnp
    
    def entropic(x, beta=0.00001):
        return (-1.0 / beta) * jnp.log(jnp.mean(jnp.exp(-beta * x)) + 1e-12)
       
    planner = JaxRDDLBackpropPlanner(..., utility=entropic)
    ...

Changing the Planning Algorithm
-------------------

In the introductory example given at the top of this tutorial, you may have noticed that we 
defined the planning algorithm or training loop (``JaxBackpropPlanner``) separately from the controller.
Therefore, it is possible to incorporate new planning algorithms into pyRDDLGym simply 
by extending the ``JaxBackpropPlanner`` class. 

pyRDDLGym currently provides one such extension based on backtracking line-search, which 
adaptively selects a learning rate at each iteration whose gradient update 
provides the greatest improvement in the return objective. 

This optimizer can be used as a drop-in replacement for ``JaxRDDLBackpropPlanner`` as follows:

.. code-block:: python

    from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxRDDLArmijoLineSearchPlanner
    from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxOfflineController

    planner = JaxRDDLArmijoLineSearchPlanner(env.model, **planner_args)
    controller = JaxOfflineController(planner, **train_args)
    controller.evaluate(env, verbose=True, render=True)

Like the default planner, the line-search planner is compatible with both offline and online controllers, 
and both straight-line and deep reactive policy implementations.

Automatically Tuning Hyper-Parameters
-------------------

The JAX planner requires many hyper-parameters, many of which can significantly affect performance.
pyRDDLGym provides a Bayesian optimization algorithm for automatically tuning key hyper-parameters, with the following features:

* supports multi-processing by evaluating multiple hyper-parameter settings in parallel
* leverages Bayesian optimization to perform more efficient search than random or grid search
* supports straight-line planning and deep reactive policies

Tuning of hyper-parameters can be done with only a slight modification of the previous codes:

.. code-block:: python

    from pyRDDLGym.Core.Env.RDDLEnv import RDDLEnv
    from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import load_config
    from pyRDDLGym.Core.Jax.JaxParameterTuning import JaxParameterTuningSLP
    from pyRDDLGym.Examples.ExampleManager import ExampleManager

    # create the environment
    info = ExampleManager.GetEnvInfo(domain)    
    env = RDDLEnv.build(info, instance)
    
    # load the config file to provide the non-tunable parameters
    planner_args, plan_args, train_args = load_config(config_path)
    
    # create the tuning algorithm
    tuning = JaxParameterTuningSLP(
        env=env,
        train_epochs=train_args['epochs'],
        timeout_training=train_args['train_seconds'],
        planner_kwargs=planner_args,
        plan_kwargs=plan_args,
        num_workers=workers, ...)
    
    # perform tuning
    best = tuning.tune(key=train_args['key'], filename='outputfile')
    print(f'best parameters found = {best}')

The ``__init__`` method requires the ``num_workers`` parameter to specify the 
number of parallel processes and the ``gp_iters`` to specify the number of iterations of Bayesian optimization. 

Upon executing this code, a dictionary of the best hyper-parameters (e.g. learning rate, policy network architecture, model hyper-parameters, etc.) is returned.
A log of the previous sets of hyper-parameters suggested by the algorithm is also recorded in the specified output file.
Deep reactive policies and replanning can be tuned by replacing ``JaxParameterTuningSLP`` with ``JaxParameterTuningDRP`` and ``JaxParameterTuningSLPReplan``, respectively.

Dealing with Non-Differentiable Expressions
-------------------

Many RDDL programs contain expressions that do not support derivatives.
A common technique to deal with this is to rewrite non-differentiable operations as similar differentiable ones.
For instance, consider the following problem of classifying points (x, y) in 2D-space as +1 if they lie in the top-right or bottom-left quadrants, and -1 otherwise:

.. code-block:: python

    def classify(x, y):
        if x > 0 and y > 0 or not x > 0 and not y > 0:
            return +1
        else:
            return -1
		    
Relational expressions such as ``x > 0`` and ``y > 0`` and logical expressions such as ``and`` and ``or`` do not have obvious derivatives. 
To complicate matters further, the ``if`` statement depends on both ``x`` and ``y`` so it does not have partial derivatives with respect to ``x`` nor ``y``.

``JaxRDDLBackpropPlanner`` works around these limitations by replacing such operations with JAX expressions that support derivatives.
Specifically, the ``classify`` function above could be written as follows:
 
.. code-block:: python

    from pyRDDLGym.Core.Jax.JaxRDDLLogic import FuzzyLogic

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
        pred = If(cond, +1, -1, w)
        return pred

For illustration, calling ``approximate_classify`` with ``x=0.5``, ``y=1.5`` and ``w=10`` returns 0.98661363, which is very close to 1.

It is possible to gain more control over how pyRDDLGym performs differentiable relaxations.
The abstract class ``FuzzyLogic`` can be sub-classed to specify how each mathematical operation should be approximated in JAX.
This logic can be passed to the planner as an optimal argument:

.. code-block:: python
    
    from pyRDDLGym.Core.Jax.JaxRDDLLogic import FuzzyLogic
    planner = JaxRDDLBackpropPlanner(model, ..., logic=FuzzyLogic())

Customizing the Differentiable Relaxations
-------------------

As of the time of this writing, pyRDDLGym only contains one implementation of differentiable logic, 
``ProductLogic``, which is based on the `product t-norm fuzzy logic <https://en.wikipedia.org/wiki/T-norm_fuzzy_logics#Motivation>`_.
The mathematical operations and their substitutions are summarized in the following table.
Here, the user-specified parameter :math:`w` specifies the "sharpness" of the operation -- higher values mean the approximation becomes closer to its exact counterpart. 

.. list-table:: Differentiable Mathematical Operations in ``ProductLogic``
   :widths: 60 60
   :header-rows: 1

   * - Exact RDDL Operation
     - ``ProductLogic`` Operation
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
    
The Gumbel-softmax trick, which is useful for (approximately) reparameterizing discrete distributions on the finite support, 
works by sampling K standard Gumbel random variables :math:`g_1, \dots g_K`.
Then, a random variable :math:`X` with probability mass function :math:`p_1, \dots p_K` can be reparameterized as

.. math::

    X = \arg\!\max_{i=1\dots K} \left(g_i + \log p_i \right)

where the approximation rule in the above table is used for argmax.
Further details about Gumbel-softmax can be found `in this paper <https://arxiv.org/pdf/1611.01144.pdf>`_.

Any operation(s) can be replaced by the user by sub-classing ``FuzzyLogic``.
For example, the RDDL operation :math:`a \text{ ^ } b` can be replaced with a user-specified one by sub-classing as follows:

.. code-block:: python
 
    class NewLogic(FuzzyLogic):
        
        def And(self):
            
            def jax_and_operation(a, b, param):
                ...
            
            new_parameter = (('weight', 'logical_and') 
            
            return jax_and_operation, new_parameter

Here, ``jax_and_operation`` represents an inner jax expression that computes the value of ``a ^ b``, and is returned as part of the ``And()`` call.
The ``new_parameter`` describes any new parameters that are introduced that must be passed to the ``jax_and_operation``.
These take the form ``((<param_type>, <expr_type>), <default_value>)``, where the inner tuple forms a key ``<param_type>_<expr_type>`` 
used to refer to parameters inside the compiled jax expression, and ``<default_value>`` is a default numeric value of the parameter(s).
A new instance of ``NewLogic`` can then be passed to ``JaxRDDLBackpropPlanner`` as described above.

The parameters of JAX logic expressions can be modified at run-time (e.g. during training). 
To do this, retrieve the parameters in the computation graph, modify them as needed, then pass them back to the planner:

.. code-block:: python

    model_params = planner.compiled.model_params
    model_params[key] = ...
    planner.optimize(..., model_params=model_params)

Limitations
-------------------

We cite several limitations of the current baseline JAX optimizer:

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

The goal of the JAX optimizer was not to replicate the state-of-the-art, but to provide a simple baseline that can be easily built-on.
However, we welcome any suggestions or modifications about how to improve this algorithm on a broader subset of RDDL.
