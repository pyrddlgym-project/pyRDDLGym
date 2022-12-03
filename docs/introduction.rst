# pyRDDLGym

A toolkit for auto-generation of OpenAI Gym environments from RDDL description files. <br />
This toolkit is the official evaluation system of the [2023 IPC RL and planning track](https://ataitler.github.io/IPPC2023/).

### Paper
Please see our [paper](https://arxiv.org/abs/2211.05939) describing pyRDDLGym.

### Status
As we support a large subset of RDDL, we list what we do not support:
* state-action-constraints -- deprecated in favor of state-invariants and action-preconditions (RDDL2.0).
* action-preconditions are not enforced by the environment, and should be incorporated into the cpfs definitions.
* action-preconditions of structure of action <=/>= deterministic-function (can be of constants or non-fluents),
are supported for the porpuse of gym spaces definitions.
* enums

We have extended the RDDL language and also support the following:
* Automatic reasoning of levels. Levels are no longer required (and ignored by the infrastructure).
* Terminal states can now be explicitly defined. The termination block has been added to the language.

All other features of RDDL are supported according to the language definition.

Several RDDL environments are included as examples with pyRDDLGym:
* CartPole Continuous
* CartPole discrete
* Elevators
* MarsRover
* MountainCar
* PowerGeneration
* RaceCar
* RecSim
* UAV continuous
* UAV discrete
* UAV mixed
* Wildfire

Software for related simulators:
* [rddlsim](https://github.com/ssanner/rddlsim)
* [rddlgym](https://github.com/thiagopbueno/rddlgym)
* [pddlgym](https://github.com/tomsilver/pddlgym)

The parser used in this project is based on the parser from
Thiago Pbueno's [pyrddl](https://github.com/thiagopbueno/pyrddl)
(used in [rddlgym](https://github.com/thiagopbueno/rddlgym)).