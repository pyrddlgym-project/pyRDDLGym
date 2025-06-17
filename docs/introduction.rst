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
	- :ref:`JaxPlan <jaxplan>`: Planning through autodifferentiation
	- :ref:`GurobiPlan <gurobiplan>`: Planning through mixed discrete-continuous optimization
	- :ref:`PROST <prost>`: Monte Carlo Tree Search (MCTS)
	- :ref:`Deep Reinforcement Learning (DQN, PPO, etc.) <rl>`: Popular Reinforcement Learning (RL) algorithms from Stable Baselines and RLlib
	- :ref:`Symbolic Dynamic Programming <xadds>`: Exact Symbolic regression-based planning and policy evaluation

Features and Limitations
------

pyRDDLGym expands on the RDDL language officially defined in rddlsim:

- domains and instances are compiled to standard OpenAI Gym environments, and are usable with existing workflows that rely upon OpenAI Gym
- simulation is vectorized to provide reasonable performance in pure Python (a faster JAX compiler and simulator is available in pyRDDLGym-jax)
- terminal states can be explicitly defined in a separate termination block
- enum and object types are interchangeable in most calculations, enhancing the flexibility of valid RDDL operations
- object-valued fluents are supported
- arbitrarily-level nested evaluation of pvariables is supported, e.g., ``fluent'(?p, ?q) = outer(inner(?p, ?q))``
- ``argmax`` and ``argmin`` aggregations are supported
- new probability distributions such as Laplace, Gumbel, Kumaraswamy are supported
- multivariate distributions such as Normal, Student, Dirichlet and Multinomial are supported
- matrix operations such as determinant, inverse and Cholesky decomposition are supported.

The following features have been omitted (or marked as deprecated) from the RDDL language in pyRDDLGym:

- the state-action-constraint block is not implemented; only the newer syntax including state-invariants and action-preconditions is supported.
- derived-fluent pvariables are still supported but considered deprecated, and may be removed in future versions
- fluent levels are reasoned automatically, thus specifying levels explicitly is no longer required.

This toolkit was the official evaluation system of the `2023 IPC RL and planning track <https://ataitler.github.io/IPPC2023/>`_.

License
-------
This software is distributed under the MIT License.

Citing pyRDDLGym
-----

Please see our `paper <https://arxiv.org/abs/2211.05939>`_ describing pyRDDLGym. To cite:

.. code-block:: bibtex

    @article{taitler2022pyrddlgym,
        title={pyRDDLGym: From RDDL to Gym Environments},
        author={Taitler, Ayal and Gimelfarb, Michael and Gopalakrishnan, Sriram and Mladenov, Martin and Liu, Xiaotian and Sanner, Scott},
        journal={arXiv preprint arXiv:2211.05939},
        year={2022}}
