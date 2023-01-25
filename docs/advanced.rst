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
All RDDL syntax (both new and old!) is already supported in the RDDL-to-JAX compiler.

Autodiff of RDDL using JAX
-------------------

In many applications, such as planning in continuous control problems, it is desirable to compute gradients of RDDL expressions using autodiff. 
For example, the planning problem in a deterministic environment can be formulated as finding the action sequence that maximizes the sum of accumulated reward over a horizon of T time steps

.. math::

	\max_{a_1, \dots a_T} \sum_{t=1}^{T} R(s_t, a_t),\\
	s_{t + 1} = f(s_t, a_t)
	
In continuous action spaces, it is possible to obtain a reasonable solution using gradient descent. More concretely, given a learning rate parameter :math:`\eta > 0` and a "guess" :math:`a_\tau`, gradient descent obtains a new estimate of the optimal action :math:`a_\tau'` at time :math:`\tau` via

.. math::
	
	a_{\tau}' = a_{\tau} - \eta \sum_{t=1}^{T} \nabla_{a_\tau} R(s_t, a_t),
	
where the gradient of the reward at all times :math:`t \geq \tau` can be computed following the chain rule:

.. math::

	\nabla_{a_\tau} R(s_t, a_t) = \frac{\mathrm{d}R(s_t,a_t)}{\mathrm{d}s_t} \frac{\mathrm{d}s_t}{\mathrm{d}a_\tau} + \frac{\mathrm{d}R(s_t,a_t)}{\mathrm{d}a_t}\frac{\mathrm{d}a_t}{\mathrm{d}a_\tau}.
	
This requires that the reward function and the CPF expression(s) :math:`f(s_t, a_t)` must both be partially differentiable with respect to either argument.

If the RDDL program is indeed differentiable (or a differentiable approximation exists), it is possible to estimate the optimal plan using a baseline method provided in pyRDDLGym:

.. code-block:: python
	
    import jax
    import optax  
    
    from pyRDDLGym import ExampleManager
    from pyRDDLGym import RDDLEnv
    from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxRDDLBackpropPlanner
	
    # specify the model
    EnvInfo = ExampleManager.GetEnvInfo('MountainCar')
    myEnv = RDDLEnv.RDDLEnv(domain=EnvInfo.get_domain(), instance=EnvInfo.get_instance(0))
    model = myEnv.model
    
    # initialize the planner
    # note that actions should be constrained to [0, 2] for MountainCar
    planner = JaxRDDLBackpropPlanner(
        model, 
        key=jax.random.PRNGKey(42), 
        batch_size_train=32, 
        batch_size_test=32,
        optimizer=optax.rmsprop(0.01),
        action_bounds={'action': (0.0, 2.0)})
      
    # train for 1000 epochs using gradient descent
    # print progress every 50 epochs
    for callback in planner.optimize(epochs=1000, step=50):
    	print('step={} train_return={:.6f} test_return={:.6f}'.format(
              str(callback['iteration']).rjust(4),
              callback['train_return'],
              callback['test_return']))

The final action sequence can then be easily extracted from the final callback.

.. code-block:: python
	
	plan = planner.get_plan(callback['params'])
	

Dealing with Non-Differentiable Expressions
-------------------

Many RDDL programs contain CPFs or reward functions that do not support derivatives.
A common technique to deal with such problems is to map non-differentiable operations to similar differentiable ones.
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

    from pyRDDLGym.Core.Jax.JaxRDDLLogic import ProductLogic

    logic = ProductLogic()

    def approximate_classify(x, y):
        cond1 = logic.And(logic.greater(x, 0), logic.greater(y, 0))
        cond2 = logic.And(logic.Not(logic.greater(x, 0)), logic.Not(logic.greater(y, 0)))
        return logic.If(logic.Or(cond1, cond2), +1, -1)

``ProductLogic`` replaces exact boolean (and other) expressions with fuzzy logic rules that are approximately equal to their exact counterparts.
For illustration, calling ``approximate_classify`` with ``x=0.5`` and ``y=1.5`` returns 0.98661363, which is very close to 1.

It is possible to gain fine-grained control over how pyRDDLGym should perform differentiable relaxations.
The abstract class ``FuzzyLogic``, from which ``ProductLogic`` is derived, can be subclassed to specify how each mathematical operation should be approximated in JAX.
This logic can be passed to the planner as an optimal argument:

.. code-block:: python

    planner = JaxRDDLBackpropPlanner(
        model, 
        ...,
        logic=ProductLogic())

Customizing the Differentiable Operations
-------------------

As of the time of this writing, pyRDDLGym only contains one implementation of differentiable logic, ``ProductLogic``.
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
   * - :math:`\mathrm{Bernoulli}(p)`
     - Gumbel-Softmax trick
   * - :math:`\mathrm{Discrete}(\mathrm{type}, \lbrace \mathrm{cases} \dots \rbrace )`
     - Gumbel-Softmax trick
    
Any operation(s) can be replaced by the user by subclassing ``FuzzyLogic`` or ``ProductLogic``.
For example, the RDDL operation :math:`a \text{ ^ } b` can be replaced with a user-specified one by sub-classing as follows:

.. code-block:: python
 
    class NewLogic(ProductLogic):
        
        def And(self, a, b):
            ...
            return ...

A new instance of ``NewLogic`` can then be passed to ``JaxRDDLBackpropPlanner`` as described above.

Limitations
-------------------

We cite several limitations of the current baseline JAX optimizer:

* Not all operations have natural differentiable relaxations. Currently, the following are not supported:
	* integer-valued functions such as round, floor, ceil
	* nested fluents such as fluent1(fluent2(?p))
	* distributions that are not naturally reparameterizable such as Poisson, Gamma and Beta
* Some relaxations can accumulate a high error relative to their exact counterparts, particularly when stacking CPFs via the chain rule for long roll-out horizons
* Some relaxations may not be mathematically consistent with one another
	* no guarantees are provided about dichotomy of equality, e.g. a == b, a > b and a < b do not necessarily "sum" to one, but in many cases should be close
	* if this is a concern, we recommend overriding some operations in ``ProductLogic`` to suit the user's needs
* The parameter :math:`w` is fixed: support for annealing or otherwise modifying this value during optimization may be added in the future.

The goal of the JAX optimizer was not to replicate the state-of-the-art, but to provide a simple baseline that can be easily built-on.
However, we welcome any suggestions or modifications about how to improve this algorithm on a broader subset of RDDL.