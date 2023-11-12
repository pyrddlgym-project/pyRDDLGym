Introduction
============

pyRDDLGym is a Python toolkit for auto-generation of OpenAI Gym environments from RDDL description files. It also provides simulation, visualization/recording, and baselines for the planning and reinforcement learning community.

Purpose and Benefits
-----
- describe your environment easily in RDDL, and let pyRDDLGym convert it to a standard OpenAI Gym environment for training and testing your reinforcement learning and planning algorithms in Python
- compiler tools to help you understand the structure of your problem
- visualization and video recording tools for monitoring and documenting the behavior of your algorithm
- support for new language features (i.e. multivariate distributions) not present in older RDDL implementations
- out-of-the-box planning algorithms in Gurobi and JAX that you can use as baselines, or build upon.

Paper
-----
Please see our `paper <https://arxiv.org/abs/2211.05939>`_ describing pyRDDLGym. To cite:

.. code-block:: python

    @article{taitler2022pyrddlgym,
      title={pyRDDLGym: From RDDL to Gym Environments},
      author={Taitler, Ayal and Gimelfarb, Michael and Gopalakrishnan, Sriram and Mladenov, Martin and Liu, Xiaotian and Sanner, Scott},
      journal={arXiv preprint arXiv:2211.05939},
      year={2022}}

Status
------

Additional features have been added to the language to increase expressivity, and to accommodate learning interaction type:

- terminal states can now be explicitly defined in a separate termination block
- action-preconditions are implemented according to the original language, but failure to enforce them now prints a warning instead of an exception; this behavior can be controlled by the user
- direct inquiry of state and action spaces is supported through the standard action space and state space properties of OpenAI gym environments; this is currently only supported for simple constraints such as box constraints
- an effort was made to ensure that enumerated (enum) and object types are as interchangeable as possible, i.e. an aggregation operation could now be performed over either
- parameter equality and disequality are supported for object and enum parameters, i.e., expressions ``?p == ?r`` and ``?p ~= ?q`` can be evaluated to True or False
- arbitrarily-level nested indexing is now supported, e.g., ``fluent'(?p, ?q) = outer(inner(?p, ?q))``
- a very large number of univariate distributions are now supported
- multivariate distributions such as Multivariate normal, Student, Dirichlet, and multinomial are now supported
- matrix algebra operations such as determinant and inverse are now supported
- ``argmax`` and ``argmin`` over enumerated types are now supported
- simulation is vectorized under-the-hood in order to provide reasonable performance while working in pure Python.

The following features have been omitted (or marked as deprecated) from the RDDL language in pyRDDLGym:

- derived-fluent are still supported, but they are considered deprecated and will be removed from future versions
- fluent levels are deprecated and are reasoned automatically, thus specifying levels explicitly is no longer required
- the state-action-constraint block is not implemented and is considered deprecated; only the newer syntax of specifying state-invariants and action-preconditions is supported.

Several RDDL environments are included as examples with pyRDDLGym:

- CartPole Continuous
- CartPole discrete
- Elevators
- MarsRover
- MountainCar
- PowerGeneration
- Quadcopter
- RaceCar
- RecSim
- UAV continuous
- UAV discrete
- UAV mixed
- Wildfire
- Supply Chain

Software for related simulators:

- `rddlsim <https://github.com/ssanner/rddlsim>`_
- `rddlgym <https://github.com/thiagopbueno/rddlgym>`_
- `pddlgym <https://github.com/tomsilver/pddlgym>`_


This toolkit was the official evaluation system of the `2023 IPC RL and planning track <https://ataitler.github.io/IPPC2023/>`_.

License
-------
This software is distributed under the MIT License.

Contributors
------------
- Michael Gimelfarb (University of Toronto, CA)
- Sriram Gopalakrishnan (Arizona State University/J.P. Morgan, USA)
- Martin Mladenov (Google, BR)
- Jack Liu (University of Toronto, CA)
