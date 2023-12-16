
# pyRDDLGym

A Python toolkit for auto-generation of OpenAI Gym environments from RDDL description files.
This is currently the official parser, simulator and evaluation system for RDDL in Python, with new features and enhancements to the RDDL language.<br />

<p align="center">
<img src="Images/cartpole.gif" width="120" height="120" margin=0/>
<img src="Images/drones.gif" width="120" height="120" margin=0/>
<img src="Images/elevators.gif" width="120" height="120" margin=0/>
<img src="Images/mountaincar.gif" width="120" height="120" margin=0/>
<img src="Images/powergen.gif" width="120" height="120" margin=0/>
</p>
<p align="center">
<img src="Images/quadcopter.gif" width="120" height="120" margin=0/>
<img src="Images/racecar.gif" width="120" height="120" margin=0/>
<img src="Images/recsys.gif" width="120" height="120" margin=0/>
<img src="Images/rovers.gif" width="120" height="120" margin=0/>
<img src="Images/wildfire.gif" width="120" height="120" margin=0/>
</p>


## Purpose and Benefits
* describe your environment in RDDL, leverage automated translation tools to convert it to a standard OpenAI gym environment in your existing workflow
* compiler tools to help you understand the structure of your problem
* visualization and video recording tools for monitoring and documenting the behavior of your algorithm
* support for new language features (i.e. multivariate distributions) not present in older RDDL implementations
* out-of-the-box planning algorithms in Gurobi, JAX, and support for stable-baselines RL that you can use as baselines or build upon.

## Paper

Please see our [paper](https://arxiv.org/abs/2211.05939) describing pyRDDLGym. If you found this useful, please consider citing us:

```
@article{taitler2022pyrddlgym,
      title={pyRDDLGym: From RDDL to Gym Environments},
      author={Taitler, Ayal and Gimelfarb, Michael and Gopalakrishnan, Sriram and Mladenov, Martin and Liu, Xiaotian and Sanner, Scott},
      journal={arXiv preprint arXiv:2211.05939},
      year={2022}}
```

## Tutorial

This toolkit was the official evaluation system of the [2023 IPC RL and planning track](https://ataitler.github.io/IPPC2023/).

Please see the following slides and notebook which were given as a tutorial at ICAPS 2023:
* [Part 1 slides](https://github.com/ataitler/pyRDDLGym/raw/main/Tutorial/RDDL_Tutorial_ICAPS_2023_Part_1.pdf)
* [Part 2 slides](https://github.com/ataitler/pyRDDLGym/raw/main/Tutorial/Tutorial_part2.pdf)
* [Tutorial notebook](https://colab.research.google.com/drive/19O-vgPsEX7t32cqV0bABmAdRaSWSMa4g?usp=sharing)
<!---* [Tutorial notebook](https://colab.research.google.com/drive/1wdX0MbjmjpC7NvBRFlTQ8kvi1uuTCQaK?usp=sharing) --->

## Status
pyRDDLGym supports a major subset of the original RDDL language:
* [RDDL](https://users.cecs.anu.edu.au/~ssanner/IPPC_2011/RDDL.pdf)

The following features have been omitted (or marked as deprecated) from the RDDL language in pyRDDLGym:
* derived-fluent are still supported, but they are considered deprecated and will be removed from future versions
* fluent levels are deprecated and are reasoned automatically, thus specifying levels explicitly is no longer required
* the state-action-constraint block is not implemented and is considered deprecated; only the newer syntax of specifying state-invariants and action-preconditions is supported.

Additional features have been added to the language to increase expressivity, and to accommodate learning interaction type:
* terminal states can now be explicitly defined in a separate termination block
* action-preconditions are implemented according to the original language, but failure to enforce them now prints a warning instead of an exception; this behavior can be controlled by the user
* direct inquiry of state and action spaces is supported through the standard action\_space and state\_space properties of OpenAI gym environments; this is currently only supported for simple constraints such as box constraints
* an effort was made to ensure that enumerated (enum) and object types are as interchangeable as possible, i.e. an aggregation operation could now be performed over either
* parameter equality and disequality are supported for object and enum parameters, i.e., expressions ?p == ?r and ?p ~= ?q can be evaluated to True or False
* arbitrarily-level nested indexing is now supported, e.g., fluent'(?p, ?q) = outer(inner(?p, ?q))
* a very large number of univariate distributions are now supported
* multivariate distributions such as Multivariate normal, Student, Dirichlet, and multinomial are now supported
* matrix algebra operations such as determinant and inverse are now supported
* _argmax_ and _argmin_ over enumerated types are now supported
* simulation is vectorized under-the-hood in order to provide reasonable performance while working in pure Python.

A number of representative RDDL environments are included with pyRDDLGym:
* Continuous/discrete cart-pole control
* Continuous/discrete mountain-car control
* Elevator control
* Mars rover control
* Power generation
* Race car control
* Recommendation system
* Continuous/discrete/mixed UAV control
* Quadcopter control
* Forest fire control
* Traffic control

A complete archive of past and present RDDL problems, including all IPPC problems, is also available to clone\pip
* [rddlrepository](https://github.com/ataitler/rddlrepository) (`pip install rddlrepository`)

Software for related simulators:
* [rddlsim](https://github.com/ssanner/rddlsim)
* [rddlgym](https://github.com/thiagopbueno/rddlgym)
* [pddlgym](https://github.com/tomsilver/pddlgym)

The parser used in this project is based on the parser from 
Thiago Pbueno's [pyrddl](https://github.com/thiagopbueno/pyrddl)
(used in [rddlgym](https://github.com/thiagopbueno/rddlgym)).

## Installation

### Requirements
We require Python 3.8+.

As well, the following packages are prerequisite:
* ply
* pillow>=9.2.0
* numpy>=1.22
* matplotlib>=3.5.0
* gym>=0.24.0
* pygame

To run the JAX planner, the following additional packages are required:
* tqdm
* jax>=0.3.25
* optax>=0.1.4
* dm-haiku>=0.0.9
* tensorflow>=2.11.0
* tensorflow-probability>=0.19.0

To run the Gurobi planner, you will need to obtain a valid Gurobi license. 
Then, follow the instructions [here](https://support.gurobi.com/hc/en-us/articles/360044290292-How-do-I-install-Gurobi-for-Python-). We require Gurobi 10+.

To run the stable-baselines3 in pyRDDLGym, you will need this package along with any of its requirements.

### Installing via pip
pip install pyRDDLGym

### Known issues
There are two known issues not documented with RDDL:
1. the minus (-) arithmetic operation must have spaces on both sides,
otherwise there is ambiguity whether it refers to a mathematical operation or to variables
2. aggregation-union-precedence parsing requires for encapsulating parentheses around aggregations, e.g., (sum_{}[]).

## Usage examples
The two main imports are the environment object and the ExampleManager.
In addition, we supply two simple agents, to illustrate interaction with the environment.
```python
from pyRDDLGym import RDDLEnv
from pyRDDLGym import ExampleManager
from pyRDDLGym.Core.Policies.Agents import RandomAgent
```

The list of examples can be obtained through the ExampleManager object:
```python
ExampleManager.ListExamples()
```

And an instantiation of the ExampleManager with a specific example will give access to its information:
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

An environment can be initialized by *.rddl files directly or by the ExampleManager:
```python
# set up the environment class, choose instance 0 because every example has at least one example instance
env = RDDLEnv.RDDLEnv(domain=EnvInfo.get_domain(), instance=EnvInfo.get_instance(0))
# set up the environment visualizer
env.set_visualizer(EnvInfo.get_visualizer())
```

An agent can be initialized:
```python
agent = RandomAgent(action_space=env.action_space, num_actions=env.numConcurrentActions)
```

And the interaction with the environment is identical to the OpenAI gym interaction:
```python
total_reward = 0
state = env.reset()
for step in range(env.horizon):
    env.render()
    action = agent.sample_action(state)
    next_state, reward, done, info = env.step(action)
    print(f'state = {state}, action = {action}, reward = {reward}')
    total_reward += reward
    state = next_state
    if done:
        break
print(f'episode ended with reward {total_reward}')

# release all viz resources, and finish logging if used
env.close()
```

__Note__: the _rddlrepository_ package contains an example manager similar to the one included with pyRDDLGym.
It is possible (and encouraged!) to `import rddlrepository.Manager.RDDLRepoManager` and use it in a similar manner to the pyRDDLGym example manager to access the full RDDL problems archive. 

### Observations and actions representation
All observations (POMDP), states (MDP) and actions are represented by dictionary objects, 
whose keys correspond to the appropriate fluents as defined in the RDDL description.

### Writing custom problems
Writing new OpenAI gym environments only requires knowledge of RDDL, thus no Python coding is required!
1. create RDDL file for the domain description
2. create RDDL file for the non-fluents and instance
3. (optional) create a custom visualizer, by inheriting from pyRDDL.Visualizer.StateViz.

Now the instantiation of the environment for the newly written problem is done as:
```python
env = RDDLEnv.RDDLEnv(domain=<domain path>, instance=<instance path>)
# set up the environment visualizer
env.set_visualizer(<visualizer object>)
```

## License
This software is distributed under the MIT License.

## Contributors
- Michael Gimelfarb (University of Toronto, CA)
- Jihwan Jeong (University of Toronto, CA)
- Sriram Gopalakrishnan (Arizona State University/J.P. Morgan, USA)
- Martin Mladenov (Google, BR)
- Jack Liu (University of Toronto, CA)
