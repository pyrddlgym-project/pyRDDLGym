Introduction
============

pyRDDLGym is A toolkit for auto-generation of OpenAI Gym environments from RDDL description files.

This toolkit is the official evaluation system of the `2023 IPC RL and planning track <https://ataitler.github.io/IPPC2023/>`_.

Paper
-----
Please see our `paper <https://arxiv.org/abs/2211.05939>`_ describing pyRDDLGym.

Status
------

The following components and structures have been added to the language to increase expressiveness, and to accommodate learning interaction type:

- ``object`` (instance-defined) and ``enum`` (domain-defined) types can be used interchangeably in expressions such as aggregations, and both used as values for p-variables. Exceptions are switch statements that explicitly reference objects of a type in the domain, and are valid for enum objects only.
- Terminal states can now be explicitly defined. The termination block has been added to the language.
- Action-preconditions are implemented according to the original language description.
- Direct Inquiry of variable (states/action) domains is supported through the standard action_space and state_space properties of the environment. 
- Parameter inequality is supported for lifted types, i.e., the following expression ``?p == ?r`` can be evaluated to ``True`` or ``False``.
- Nested indexing is now supported, e.g., ``fluent'(?p,?q) = NEXT(fluent(?p, ?q))``.
- Additional probability distributions are implemented (please see RDDL Language Description section for details)
- Vectorized distributions such as Multivariate normal, Student, Dirichlet, and Multinomial are now supported.
- Basic matrix algebra such as determinant and inverse operation are supported for two appropriate fluents.
- ``argmax`` and ``argmin`` are supported over enumerated types (enums).

The following components are omitted (or marked as deprecated) from the language variant implemented in pyRDDLGym:

- Derived-fluents are supported by the framework as described in the language description. However, they are considered deprecated and will be removed from future versions.
- Fluent levels are deprecated and are reasoned automatically by the framework, specifying levels explicitly is not required.
- State-action-constraints are not implemented and are considered deprecated in the language to avoid ambiguity. 

Several RDDL environments are included as examples with pyRDDLGym:

- CartPole Continuous
- CartPole discrete
- Elevators
- MarsRover
- MountainCar
- PowerGeneration
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

License
-------
This software is distributed under the MIT License.

Contributors
------------
- Michael Gimelfarb (University of Toronto, CA)
- Sriram Gopalakrishnan (Arizona State University/J.P. Morgan, USA)
- Martin Mladenov (Google, BR)
- Jack Liu (University of Toronto, CA)
