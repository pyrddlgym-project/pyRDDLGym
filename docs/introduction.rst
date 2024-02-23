Introduction
============

pyRDDLGym is a Python toolkit for auto-generation of OpenAI Gym environments from 
Relational Dynamic Influence Diagram Language (RDDL) description files. This is 
currently the official parser, simulator and evaluation system for RDDL in Python, 
with new features and enhancements to the RDDL language.

Purpose and Benefits
-----

- Describe your environment in RDDL and use it with your existing workflow for OpenAI gym environments
- Compact, easily modifiable representation language for discrete time control in dynamic stochastic environments
- Object-oriented relational (template) specification allows easy scaling of model instances from 1 object to 1000's of objects without changing the domain model
- Customizable visualization and recording tools facilitate domain debugging and plan interpretation
- Runs out-of-the-box in Python or within Colab
- Compiler tools to extract Dynamic Bayesian Networks (DBNs) and Extended Algebraic Decision Diagrams (XADDs) for symbolic analysis of causal dependencies and transition distributions
- Ready to use with out-of-the-box planners:
	- JaxPlan: Planning through autodifferentiation
	- GurobiPlan: Planning through mixed discrete-continuous optimization
	- PROST: Monte Carlo Tree Search (MCTS)
	- Deep Reinforcement Learning (DQN, PPO, etc.): Popular Reinforcement Learning (RL) algorithms from Stable Baselines and RLlib
	- Symbolic Dynamic Programming: Exact Symbolic regression-based planning and policy evaluation

Status
------

Additional features have been added to the language:

- terminal states can now be explicitly defined in a separate termination block
- direct inquiry of state and action spaces is supported through the standard action space and state space properties of OpenAI gym environments; this is currently only supported for simple constraints such as box constraints
- an effort was made to ensure that enumerated (enum) and object types are as interchangeable as possible, i.e. an aggregation operation could now be performed over either
- parameter equality and disequality are supported for object and enum parameters, i.e., expressions ``?p == ?r`` and ``?p ~= ?q`` can be evaluated to True or False
- arbitrarily-level nested indexing is now supported, e.g., ``fluent'(?p, ?q) = outer(inner(?p, ?q))``
- a very large number of univariate distributions are now supported
- multivariate distributions such as Multivariate normal, Student, Dirichlet, and multinomial are now supported
- matrix algebra operations such as determinant, inverse and Cholesky decomposition are now supported
- ``argmax`` and ``argmin`` over enumerated types are now supported
- simulation is vectorized under-the-hood in order to provide reasonable performance while working in pure Python (a faster JAX compiler and simulator is also available).

The following features have been omitted (or marked as deprecated) from the RDDL language in pyRDDLGym:

- derived-fluent are still supported, but they are considered deprecated and will be removed from future versions
- fluent levels are deprecated and are reasoned automatically, thus specifying levels explicitly is no longer required
- the state-action-constraint block is not implemented and is considered deprecated; only the newer syntax of specifying state-invariants and action-preconditions is supported.

This toolkit was the official evaluation system of the `2023 IPC RL and planning track <https://ataitler.github.io/IPPC2023/>`_.

License
-------
This software is distributed under the MIT License.

Citing pyRDDLGym
-----
Please see our `paper <https://arxiv.org/abs/2211.05939>`_ describing pyRDDLGym. To cite:

.. code-block:: python

    @article{taitler2022pyrddlgym,
      title={pyRDDLGym: From RDDL to Gym Environments},
      author={Taitler, Ayal and Gimelfarb, Michael and Gopalakrishnan, Sriram and Mladenov, Martin and Liu, Xiaotian and Sanner, Scott},
      journal={arXiv preprint arXiv:2211.05939},
      year={2022}}

Contributors
------------
- Ayal Taitler (University of Toronto, CA)
- Michael Gimelfarb (University of Toronto, CA)
- Sriram Gopalakrishnan (Arizona State University/J.P. Morgan, USA)
- Martin Mladenov (Google, BR)
- Jack Liu (University of Toronto, CA)
