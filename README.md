# pyRDDLGym

A toolkit for auto-generation of OpenAI Gym environments from RDDL description files. <br />
This toolkit is the official evaluation system of the [2023 IPC RL and planning track](https://ataitler.github.io/IPPC2023/).

### Paper
Please see our [paper](https://arxiv.org/abs/2211.05939) describing pyRDDLGym.

### Status
pyRDDLGym supports a major subset of the original RDDL language:
* [RDDL](https://users.cecs.anu.edu.au/~ssanner/IPPC_2011/RDDL.pdf)

The following components are omitted (or marked as deprecated) from the language variant implemented in pyRDDLGym:
* derived-fluent are supported by the framework as described in the language description. However, they are considered deprecated and will be removed from future versions.
* Fluent levels are deprecated and are reasoned automatically by the framework, specifying levels explicitly is not required.
* state-action-constraint is not implemented and is considered deprecated in the language to avoid ambiguity. Only the newer syntax of specifying state-invariants and action-preconditions is supported.

Additional components and structures have been added to the language to increase expressivity, and to accommodate learning interaction type. These are listed here:
* Terminal states can now be explicitly defined. The termination block has been added to the language.
* action-preconditions are implemented according to the original language description.
However, they are subject to the user preference. By default the framework _does not_ enforce the expressions in the action-preconditions block,
thus, upon violation a warning will be printed to the user and the simulation will push the actions inside the legal space by using the default value and the simulation will continue. 
To ensure correct behaviour it is expected from the domain designer to include the appropriate logic of how to handle an invalid action within the _cpfs_ block.
In the case where the user does choose to enforce action-preconditions, the simulation will be interrupted and an appropriate exception will be thrown.
* Direct Inquiry of variable (states/action) domains is supported through the standard action\_space and state\_space properties of the environment.
In order for this functionality to work correctly, the domain designer is required to specify each (lifted-)variable bounds within the 
action-preconditions block in the format "_fluent_ OP BOUND" where OP \in {<,>,>=,<=}, and BOUND is a deterministic function of the problem parameter to be evaluated at instantiation.
* Parameter inequality is supported for lifted types. I.e., the following expression ?p == ?r can be evaluated to True or False.
* Nested indexing is now supported, e.g., fluent'(?p,?q) = NEXT(fluent(?p, ?q)).
* Vectorized distributions such as Multivariate normal, Student, Dirichlet, and Multinomial are now supported.
* Basic matrix algebra such as determinant and inverse operation are supported for two appropriate fluents.
* _argmax_ and _argmin_ are supported over enumerated types (enums).


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
* Traffic

A complete archive of past, present, and future RDDL problems including all IPPC problems, is also available to clone\pip
* [rddlrepository](https://github.com/ataitler/rddlrepository) (`pip install rddlrepository`)

Software for related simulators:
* [rddlsim](https://github.com/ssanner/rddlsim)
* [rddlgym](https://github.com/thiagopbueno/rddlgym)
* [pddlgym](https://github.com/tomsilver/pddlgym)

The parser used in this project is based on the parser from 
Thiago Pbueno's [pyrddl](https://github.com/thiagopbueno/pyrddl)
(used in [rddlgym](https://github.com/thiagopbueno/rddlgym)).

### Installation

#### Python version
We require Python 3.8+.

###Requirements:
* ply
* pillow>=9.2.0
* numpy>=1.22
* matplotlib>=3.5.0
* gym>=0.24.0
* pygame

#### Installing via pip
pip install pyRDDLGym

#### Known issues
There are two known issues not documented with RDDL
1. The minus (-) arithmatic operation must have spaces on both sides,
otherwise there is ambiguity is whether it is a mathematical operation of a fluent name.
2. Aggregation union precedence requires for encapsulating parentheses, e.g., (sum_{}[]).

## Usage examples

The two main imports are the environment object and the ExampleManager.
In addition, we supply two simple agents, to illustrate interaction with the environment.
```python
from pyRDDLGym import RDDLEnv
from pyRDDLGym import ExampleManager
from pyRDDLGym.Policies.Agents import RandomAgent
```
The list of examples can be obtained through the ExampleManager object:
```python
ExampleManager.ListExamples()
```

And an instantiation of the ExampleManager object with a specific example will give access to the example information:
```python
# get the environment info
EnvInfo = ExampleManager.GetEnvInfo('MarsRover')
# access to the domain file
EnvInfo.get_domain()
#list all available instances for that domain
EnvInfo.list_instances()
# access to instance 0  
EnvInfo.get_instance(0)
# obtain the dedicated visualizer object of the domain if exists
EnvInfo.get_visualizer()
```

An environment can be initilaized by *.rddl files directly or by the ExampleManager:
```python
# set up the environment class, choose instance 0 because every example has at least one example instance
myEnv = RDDLEnv.RDDLEnv(domain=EnvInfo.get_domain(), instance=EnvInfo.get_instance(0))
# set up the environment visualizer
myEnv.set_visualizer(EnvInfo.get_visualizer())
```

An agent can be initilized:
```python
agent = RandomAgent(action_space=myEnv.action_space, num_actions=myEnv.numConcurrentActions)
```

And the final interaction with the environment is identical to the gym standard interaction:
```python
total_reward = 0
state = myEnv.reset()
for step in range(myEnv.horizon):
    myEnv.render()
    action = agent.sample_action()
    next_state, reward, done, info = myEnv.step(action)
    total_reward += reward
    print()
    print('step       = {}'.format(step))
    print('state      = {}'.format(state))
    print('action     = {}'.format(action))
    print('next state = {}'.format(next_state))
    print('reward     = {}'.format(reward))
    state = next_state
    if done:
        break
print("episode ended with reward {}".format(total_reward))
myEnv.close()
```

__Note__: the _rddlrepository_ package contains an example manager similar to the one included with pyRDDLGym.
It is possible (and encouraged!) to `import rddlrepository.Manager.RDDLRepoManager` and use it in a similar manner to the pyRDDLGym example manager to access the full RDDL problems archive. 

### Observations and actions representation
All observations (POMDP), states (MDP) and actions are represented by DICT objects.
The keys in the DICTs are the appropriate fluents as defined in the RDDL of the problem.

Note of lifted and grounded fluents: a grounded fluent 'fluent(obj)' and obj 'o1' are grounded as 'fluent_o1'.
Thus, we encourage not to give names with underscores (_) for the fluents for easy reverse lifting. 

### Writing custom problems
Writing custom new problems only required to know RDDL, no coding is requireed.

1. Create RDDL file for the domain description.
2. Create RDDL file for the non-fluents and instance.
3. (optional) Create a custom visualizer, by inheriting from pyRDDL.Visualizer.StateViz.

Now the instantiation of the environment for the newly written problem is done by:
```python
myEnv = RDDLEnv.RDDLEnv(domain=<domain path>, instance=<instance path>)
# set up the environment visualizer
myEnv.set_visualizer(<visualizer object>)
```

## License
This software is distributed under the MIT License.

## Contributors
- Michael Gimelfarb (University of Toronto, CA)
- Jihwan Jeong (University of Toronto, CA)
- Sriram Gopalakrishnan (Arizona State University/J.P. Morgan, USA)
- Martin Mladenov (Google, BR)
- Jack Liu (University of Toronto, CA)
