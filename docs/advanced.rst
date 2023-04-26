Advanced Topics
===============

Changing the Backend
-------------------

By default, RDDLEnv simulates all RDDL control flow using Python and stores intermediate expressions in numpy arrays.
However, if performance is a bottleneck, or if additional structure is required (e.g. gradients), then it is possible to compile and simulate the RDDL problem using JAX.
In pyRDDLGym, this can be done easily by specifying the backend:

.. code-block:: python
	
	from pyRDDLGym.Core.Jax.JaxRDDLSimulator import JaxRDDLSimulator
	
	myEnv = RDDLEnv.RDDLEnv(domain=EnvInfo.get_domain(),
                            instance=EnvInfo.get_instance(0),
                            backend=JaxRDDLSimulator)
	
For the purpose of simulation, the default backend and the ``JaxRDDLSimulator`` are designed to be as interchangeable as possible, so the latter can be used in place of the former with identical outputs in most cases.

.. note::
   All RDDL syntax (both new and old!) is supported in the RDDL-to-JAX compiler.

Logging RDDL Compilation
-------------------

For purposes such as debugging, it is possible to log information about the RDDL compilation to a file.

.. code-block:: python
	
	myEnv = RDDLEnv.RDDLEnv(domain=EnvInfo.get_domain(),
                            instance=EnvInfo.get_instance(0),
                            debug=True)

Upon executing this command, a log file is created with the name <domain name>_<instance name>.log in the installation's root directory.
Currently, the following information is written in the generated log file:

* description of pvariables as they are stored in memory (e.g., parameters, data type, data shape)
* dependency graph between CPFs
* calculated order of evaluation of CPFs
* information used by the simulator and JAX compiler for operating on pvariables stored as arrays
* simulation bounds for state and action fluents (unbounded or non-box constraints are represented as [-inf, inf])
* for JAX compilation, also prints the JAX compiled expressions corresponding to CPFs, reward and constraint expressions.

Open-Loop Planning with JAX
-------------------

In many applications, such as planning in continuous control problems, it is desirable to compute gradients of RDDL expressions using autodiff. 
For example, the planning problem in a deterministic environment can be formulated as finding the action sequence that maximizes the sum of accumulated reward over a horizon of T time steps

.. math::

	\max_{a_1, \dots a_T} \sum_{t=1}^{T} R(s_t, a_t),\\
	s_{t + 1} = f(s_t, a_t)
	
In continuous action spaces, it is possible to obtain a reasonable solution using gradient ascent. More concretely, given a learning rate parameter :math:`\eta > 0` and a "guess" :math:`a_\tau`, gradient ascent obtains a new estimate of the optimal action :math:`a_\tau'` at time :math:`\tau` via

.. math::
	
	a_{\tau}' = a_{\tau} + \eta \sum_{t=1}^{T} \nabla_{a_\tau} R(s_t, a_t),
	
where the gradient of the reward at all times :math:`t \geq \tau` can be computed following the chain rule:

.. math::

	\nabla_{a_\tau} R(s_t, a_t) = \frac{\mathrm{d}R(s_t,a_t)}{\mathrm{d}s_t} \frac{\mathrm{d}s_t}{\mathrm{d}a_\tau} + \frac{\mathrm{d}R(s_t,a_t)}{\mathrm{d}a_t}\frac{\mathrm{d}a_t}{\mathrm{d}a_\tau}.
	
This requires that the reward function and the CPF expression(s) :math:`f(s_t, a_t)` must both be partially differentiable with respect to either argument.
This approach is introduced and further described `in this paper <https://proceedings.neurips.cc/paper/2017/file/98b17f068d5d9b7668e19fb8ae470841-Paper.pdf>`_.

If the RDDL program is indeed differentiable (or a differentiable approximation exists), it is possible to estimate the optimal plan using a baseline method provided in pyRDDLGym:

.. code-block:: python

    import jax
    import optax
    
    from pyRDDLGym import ExampleManager
    from pyRDDLGym import RDDLEnv
    from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxRDDLBackpropPlanner
    from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxStraightLinePlan
    
    # specify the model
    EnvInfo = ExampleManager.GetEnvInfo('Wildfire')
    myEnv = RDDLEnv.RDDLEnv(domain=EnvInfo.get_domain(), instance=EnvInfo.get_instance(0))
    model = myEnv.model
    
    # initialize the planner
    planner = JaxRDDLBackpropPlanner(
        model,
        batch_size_train=32,
        plan=JaxStraightLinePlan(),
        optimizer=optax.rmsprop,
        optimizer_kwargs={'learning_rate': 0.1})
    
    # train for 1000 epochs using gradient ascent - print progress every 50
    # note that boolean actions are wrapped with sigmoid by default, so the 
    # policy_hyperparams dictionary must be filled with weights for them
    policy_weights = {'cut-out': 10.0, 'put-out': 10.0}
    for callback in planner.optimize(
        jax.random.PRNGKey(42), epochs=1000, step=10, policy_hyperparams=policy_weights):
        print('step={} train_return={:.6f} test_return={:.6f} best_return={:.6f}'.format(
              str(callback['iteration']).rjust(4),
              callback['train_return'],
              callback['test_return'],
              callback['best_return']))

The ``policy_hyperparams`` argument is required whenever the policy or plan representation takes additional hyper-parameters. 
Further details are provided in the "Box Constraints" section below.
The final action sequence can then be easily extracted from the final callback.

.. code-block:: python
	
	plan = planner.get_action(<PRNG key>, callback['params'], <step>, None, policy_weights)
	

Open-Loop Planning with Periodic Revision
-------------------

In domains that have stochastic transitions, an open loop plan can be considerably sub-optimal.
In order to take into account the actual evolution of the state trajectory into the planning problem, it is possible to re-compute the optimal plan periodically in each state.
This is often called "re-planning".

Another problem of planning in stochastic domains is that the state transition function :math:`s_{t + 1} = f(s_t, a_t)` is no longer deterministic, and so the gradients are no longer well-defined in this formulation.
pyRDDLGym works around this problem by using the reparameterization trick.
To illustrate this in action, if :math:`s_{t+1} = \mathcal{N}(s_t, a_t^2)`, then after reparametization this becomes :math:`s_{t+1} = s_t + a_t * \mathcal{N}(0, 1)`, and back-propagation can now be performed with respect to both state and action.
The reparameterization trick can also work for other classes of probability distributions, if there exists a closed-form function f such that

.. math::

    s_{t+1} = f(s_t, a_t, \xi_t)
    
where :math:`\xi_t` are i.i.d. random variables drawn from some concrete distribution. 
For a detailed discussion of reparameterization in the context of planning by back-propagation, please see `this paper <https://ojs.aaai.org/index.php/AAAI/article/view/4744>`_ or `this one <https://ojs.aaai.org/index.php/AAAI/article/view/21226>`_.

pyRDDLGym will automatically perform reparameterization as needed if it is possible to do so.
However, some probability distributions, such as the Beta distribution, do not have tractable reparameterizations.
For a small subset of them, like the Bernoulli and Discrete distribution, pyRDDLGym offers efficient approximations backed by the existing literature (see, e.g. the Gumbel-softmax discussion below). 

.. warning::
   For non-reparameterizable distributions, the result of the gradient calculation is fully dependent on the JAX implementation: it could return a zero or NaN gradient, or raise an exception.

The ``JaxRDDLBackpropPlanner`` makes it relatively easy to do re-planning in stochastic domains inside the usual simulation loop.
To do this, the parameter ``rollout_horizon`` specifies how far ahead the planner will look during optimization at each time step. 
This quantity overrides the default horizon specified in the RDDL instance.

.. code-block:: python

    # specify the model
    EnvInfo = ExampleManager.GetEnvInfo('Wildfire')
    myEnv = RDDLEnv.RDDLEnv(domain=EnvInfo.get_domain(), instance=EnvInfo.get_instance(0))
    model = myEnv.model
    
    # initialize the planner with a roll-out horizon of 5
    planner = JaxRDDLBackpropPlanner(
        model, 
        plan=JaxStraightLinePlan(),
        batch_size_train=32, 
        rollout_horizon=5,
        optimizer=optax.rmsprop,
        optimizer_kwargs={'learning_rate': 0.01})

The optimizer can then be invoked at every decision step (or periodically), as shown below:

.. code-block:: python

    policy_weights = {'put-out': 10.0, 'cut-out': 10.0}
    key = jax.random.PRNGKey(42)
    total_reward = 0
    state = myEnv.reset()
    for step in range(myEnv.horizon):
        key, subkey1, subkey2 = jax.random.split(key, num=3)
        *_, callback = planner.optimize(
            subkey1, epochs=500, step=100, policy_hyperparams=policy_weights, 
            subs=myEnv.sampler.subs)
        action = planner.get_action(
            subkey2, params=callback['params'], step=0, 
            subs=None, policy_hyperparams=policy_weights)
        next_state, reward, done, _ = myEnv.step(action)
        total_reward += reward 
        ...
        
    print(f'episode ended with reward {total_reward}')
    myEnv.close()
    
By executing this code, and comparing the realized return to the one obtained by the code in the previous section, 
it is clear that re-planning can perform much better on average than straight-line planning.

Policy Networks for Closed-Loop Planning
-------------------

An alternative approach to re-planning is to learn a policy network :math:`a_t \gets \pi_\theta(s_t)`, i.e. a feed-forward neural network with parameters :math:`\theta` mapping state to action.
The example below adapts the Wildfire experiment above to use a deep reactive policy instead of a straight-line plan:

.. code-block:: python

    import optax
    import jax
    
    from pyRDDLGym import ExampleManager
    from pyRDDLGym import RDDLEnv
    from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxRDDLBackpropPlanner
    from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxDeepReactivePolicy
    
    # specify the model
    EnvInfo = ExampleManager.GetEnvInfo('Wildfire')
    myEnv = RDDLEnv.RDDLEnv(domain=EnvInfo.get_domain(), instance=EnvInfo.get_instance(0))
    model = myEnv.model
    
    # initialize the planner
    # here we initialize a policy network with two hidden layers of size 128 and 64
    planner = JaxRDDLBackpropPlanner(
        model,
        batch_size_train=32,
        plan=JaxDeepReactivePolicy(topology=[128, 64]),
        optimizer=optax.rmsprop,
        optimizer_kwargs={'learning_rate': 0.001})
    
    # train for 1000 epochs using gradient ascent - print progress every 50
    for callback in planner.optimize(
        jax.random.PRNGKey(42), epochs=1000, step=10):
        print('step={} train_return={:.6f} test_return={:.6f} best_return={:.6f}'.format(
              str(callback['iteration']).rjust(4),
              callback['train_return'],
              callback['test_return'],
              callback['best_return']))

The use of a policy often produces better results than straight-line planning, as shown in the example above.

.. note::
   ``JaxStraightlinePlan`` and ``JaxDeepReactivePolicy`` are instances of the abstract class ``JaxPlan``. 
   Other agent representations could be defined by overriding the ``JaxPlan`` class and its methods `compile` and ``guess_next_epoch``.
   
Details about the implementation of the deep reactive policy for planning are explained further `in this paper <https://ojs.aaai.org/index.php/AAAI/article/view/4744>`_. 

Box Constraints on Action Fluents
-------------------

Currently, the JAX planner supports two different kind of actions constraints: box constraints and concurrency constraints. 

Box constraints are useful for bounding each action-fluent independently into some range during optimization.
Box constraints can be specified by passing a dictionary that maps action-fluent names to box bounds into the ``action_bounds`` keyword argument.
The syntax for specifying box constraints is written as follows:

.. code-block:: python

    action_bounds={ <action_name1>: (lower1, upper1), <action_name2>: (lower2, upper2), ... }
   
where ``lower#`` and ``upper#`` can be any floating point value, including positive and negative infinity. 
Passing ``None`` as a value to ``lower`` or ``upper`` indicates that a bound is not enforced, i.e. ``(10.0, None)`` indicates an action must be at least 10.
The bounds are enforced by default using a projected gradient step that corrects the action parameters at each iteration during optimization.

By default, boolean actions are wrapped using the sigmoid function:

.. math::
    
    a = \frac{1}{1 + e^{-w \theta}},

where :math:`\theta` denotes the trainable action parameters, and :math:`w` denotes a hyper-parameter that controls the sharpness of the approximation.

.. note::
   If ``wrap_sigmoid = True``, then the weights ``w`` as defined above must be specified in ``policy_hyperparams`` for each action when interfacing with the planner methods.
   
At test time, the action is aliased by evaluating the expression :math:`a > 0.5`, or equivalently :math:`\theta > 0`.
The use of sigmoid for boolean actions can be controlled by setting ``wrap_sigmoid`` in ``JaxStraightLinePlan``.
Non-boolean action-fluents can also be wrapped in a similar way, rather than use the projected gradient trick, by setting ``wrap_non_bool = True``.
The details of this approach is described further in `equation 6 in this paper <https://ojs.aaai.org/index.php/AAAI/article/view/4744>`_.
   
Concurrency Constraints on Action Fluents
-------------------

The JAX planner also supports constraints on the maximum number of action-fluents that can be set at any given time. 
This is given mathematically as a constraint of the form :math:`\sum_i a_i \leq B` for some constant :math:`B`.
Specifically, if the ``max-nondef-actions`` property in the RDDL instance is less than the total number of boolean action fluents, then ``JaxRDDLBackpropPlanner`` will automatically apply a projected gradient technique to ensure ``max_nondef_actions`` is satisfied at each optimization step.
Two methods are provided to ensure constraint satisfaction: the exact implementation details of the original method are provided `in this paper <https://ojs.aaai.org/index.php/ICAPS/article/view/3467>`_

.. note::
   Concurrency constraints on action-fluents are applied to boolean actions only: e.g., real and int actions will be ignored.

Reward Normalization
-------------------

Some domains have rewards that vary significantly in magnitude between time steps, making optimization difficult without some form of normalization.
Following the suggestion `in this paper <https://arxiv.org/pdf/2301.04104v1.pdf>`_, pyRDDLGym applies the symlog transform to the sampled rewards during back-prop.
Mathematically, symlog is defined as

.. math::
    
    \mathrm{symlog}(x) = \mathrm{sign}(x) * \ln(|x| + 1)

which compresses the magnitudes of large positive and negative outcomes.
The use of symlog can be enabled by the ``use_symlog_reward`` argument in ``JaxBackpropPlanner``.

Utility Optimization
-------------------

By default, the Jax planner will optimize the expected sum of future reward. In settings that entail risk, this may not always be desirable.
Following the framework `in this paper <https://ojs.aaai.org/index.php/AAAI/article/view/21226>`_, it is possible to optimize some non-linear utility of the return instead.
For example, the entropic utility for risk-aversion parameter :math:`\beta` can be written mathematically as

.. math::
    
    U(a_1, \dots a_T) = -\frac{1}{\beta} \log \mathbb{E}\left[e^{-\beta \sum_t R(s_t, a_t)} \right]

In JAX, this can be passed to the planner as follows:

.. code-block:: python

    import jax.numpy as jnp
    
    def entropic(x, beta=0.00001):
        return (-1.0 / beta) * jnp.log(jnp.mean(jnp.exp(-beta * x)) + 1e-12)
       
    planner = JaxRDDLBackpropPlanner(..., utility=entropic)
    ...

Dealing with Non-Differentiable Expressions
-------------------

Many RDDL programs contain CPFs or reward functions that do not support derivatives.
A common technique to deal with such problems is to rewrite non-differentiable operations as similar differentiable ones.
For instance, consider the following problem of classifying points (x, y) in 2D-space as +1 if they lie in the top-right or bottom-left quadrants, and -1 otherwise:

.. code-block:: python

    def classify(x, y):
        if x > 0 and y > 0 or not x > 0 and not y > 0:
            return +1
        else:
            return -1
		    
Relational expressions such as ``x > 0`` and ``y > 0`` and logical expressions such as ``and`` and ``or`` do not have obvious derivatives. 
To complicate matters further, the ``if`` statement depends on both ``x`` and ``y`` so it does not have partial derivatives with respect to ``x`` nor ``y``.

``JaxRDDLBackpropPlanner`` works around these limitations by replacing such operations with JAX-based expressions that support derivatives.
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

``ProductLogic`` replaces exact boolean (and other) expressions with fuzzy logic rules that are approximately equal to their exact counterparts.
For illustration, calling ``approximate_classify`` with ``x=0.5``, ``y=1.5`` and ``w=10`` returns 0.98661363, which is very close to 1.

It is possible to gain fine-grained control over how pyRDDLGym should perform differentiable relaxations.
The abstract class ``FuzzyLogic``, from which ``ProductLogic`` is derived, can be sub-classed to specify how each mathematical operation should be approximated in JAX.
This logic can be passed to the planner as an optimal argument:

.. code-block:: python

    planner = JaxRDDLBackpropPlanner(
        model, 
        ...,
        logic=FuzzyLogic())

Customizing the Differentiable Operations
-------------------

As of the time of this writing, pyRDDLGym only contains one implementation of differentiable logic, ``ProductLogic``, which is based on the `product t-norm fuzzy logic <https://en.wikipedia.org/wiki/T-norm_fuzzy_logics#Motivation>`_.
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
    
The Gumbel-softmax trick, which is useful for (approximately) reparameterizing discrete distributions on the finite support, works by sampling K standard Gumbel random variables :math:`g_1, \dots g_K`.
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

Here, ``jax_and_operation`` represents an inner jax expression that computes the value of ``a and b``, and is returned as part of the ``And()`` call.
The ``new_parameter`` describes any new parameters that are introduced that must be passed to the ``jax_and_operation``.
These take the form ``((<param_type>, <expr_type>), <default_value>)``, where the inner tuple forms a key ``<param_type>_<expr_type>`` used to refer to parameters inside the compiled jax expression, and ``<default_value>`` is a default numeric value of the parameter(s).
A new instance of ``NewLogic`` can then be passed to ``JaxRDDLBackpropPlanner`` as described above.

The parameters of jax logic expressions can be modified at run-time (e.g. during training). To do this, it is possible to retrieve the names and values of all such parameters in the computation graph as follows:

.. code-block:: python

    model_params = planner.compiled.model_params

During training, these values can be modified before passing to other subroutines in the planner, such as ``update``. 

Limitations
-------------------

We cite several limitations of the current baseline JAX optimizer:

* Not all operations have natural differentiable relaxations. Currently, the following are not supported:
	* nested fluents such as fluent1(fluent2(?p))
	* distributions that are not naturally reparameterizable such as Poisson, Gamma and Beta
* Some relaxations can accumulate a high error relative to their exact counterparts, particularly when stacking CPFs via the chain rule for long roll-out horizons
* Some relaxations may not be mathematically consistent with one another
	* no guarantees are provided about dichotomy of equality, e.g. a == b, a > b and a < b do not necessarily "sum" to one, but in many cases should be close
	* if this is a concern, it is recommended to override some operations in ``ProductLogic`` to suit the user's needs
* Termination conditions and state/action constraints are not considered in the optimization (but can be checked at test-time).

The goal of the JAX optimizer was not to replicate the state-of-the-art, but to provide a simple baseline that can be easily built-on.
However, we welcome any suggestions or modifications about how to improve this algorithm on a broader subset of RDDL.