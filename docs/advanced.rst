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

It is possible to gain fine-grained control over how pyRDDLGym should perform differentiable relaxations as illustrated above.
The abstract class ``FuzzyLogic`` can be subclassed to specify how each mathematical operation should be approximated in JAX.
This logic can then be passed to the planner as an optimal argument:

.. code-block:: python

    planner = JaxRDDLBackpropPlanner(
        model, 
        ...,
        logic=ProductLogic())