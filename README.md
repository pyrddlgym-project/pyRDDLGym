# pyRDDLGym

A toolkit for auto-generation of OpenAI Gym environments from RDDL description files. <br />
This toolkit is the official evaluation system of the [2023 IPC RL and planning track](https://ataitler.github.io/IPPC2023/).

### Paper
Please see our [paper](https://arxiv.org/abs/2211.05939) describing pyRDDLGym.

### Status
As we support all of RDDL's functionality, we list what we do not support as it is considered:
* state-action-constraints -- deprecated in favor of state-invariants and action-preconditions (RDDL2.0).
* derived-fluents are considered deprecated and should not be used. interm-fluents should be used instead.

We have extended the RDDL language and also support the following:
* Automatic reasoning of levels. Levels are no longer required (and ignored by the infrastructure).
* Terminal states can now be explicitly defined. The termination block has been added to the language.
* New probability distributions (binomial, beta, geometric, pareto, student-t, and many more) and mathematical functions (gamma, beta).
* action-preconditions are optional in the Gym environment. By default action-preconditions are not enforced by the environment, and should be incorporated into the cpfs definitions.
However, if desired the environment can be informed that about the user desire to enforce them, in that case an exception will be raisd upon violation.
* action-preconditions of structure of action <=/>= deterministic-function (can be of constants or non-fluents), 
are supported for the purpose of gym spaces definitions.

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

### Installation

#### Python version
We require Python 3.7+.

###Requirements:
* ply
* pillow>=9.2.0
* numpy
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
agent = RandomAgent(action_space=myEnv.action_space, num_actions=myEnv.NumConcurrentActions)
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
