# pyRDDLGym

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
[![PyPI Version](https://img.shields.io/pypi/v/pyRDDLGym.svg)](https://pypi.org/project/pyRDDLGym/)
[![Documentation Status](https://readthedocs.org/projects/pyrddlgym/badge/?version=latest)](https://pyrddlgym.readthedocs.io/en/latest/)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
[![Cumulative PyPI Downloads](https://img.shields.io/pypi/dm/pyrddlgym)](https://pypistats.org/packages/pyrddlgym)

[Purpose](#purpose-and-benefits) | [Installation](#installation) | [Example Scripts](#example-scripts) | [Usage](#usage) | [Status](#status) | [Citing](#citing-pyrddlgym)

> [!WARNING]  
> As of Feb 9, 2024, the pyRDDLGym API has been updated to version 2.0, and is no longer backwards compatible with the previous stable version 1.4.4.
> While we strongly recommend that you update to 2.0, in case you require the old API, you can install the last stable version with pip:
> ``pip install pyRDDLGym==1.4.4``, or directly from github ``pip install git+https://github.com/pyrddlgym-project/pyRDDLGym@version_1.4.4_stable``.

A Python toolkit for auto-generation of OpenAI Gym environments from Relational Dynamic Influence Diagram Language (RDDL) description files.
This is currently the official parser, simulator and evaluation system for RDDL in Python, with new features and enhancements to the RDDL language.<br />

<img src="Images/examples.gif" margin=0/>

## Purpose and Benefits

* Describe your environment in RDDL ([web-based intro](https://ataitler.github.io/IPPC2023/pyrddlgym_rddl_tutorial.html)), ([full tutorial](https://pyrddlgym-project.github.io/AAAI24-lab)), ([language spec](https://pyrddlgym.readthedocs.io/en/latest/rddl.html)) and use it with your existing workflow for OpenAI gym environments
* Compact, easily modifiable representation language for discrete time control in dynamic stochastic environments
    * e.g., [a few lines of RDDL](https://github.com/pyrddlgym-project/rddlrepository/blob/main/rddlrepository/archive/standalone/CartPole/Continuous/domain.rddl#L61) for CartPole vs. [200 lines in direct Python for Gym](https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py#L130)
* Object-oriented relational (template) specification allows easy scaling of model instances from 1 object to 1000's of objects without changing the domain model
    * e.g., [Wildfire (Web Tutorial)](https://ataitler.github.io/IPPC2023/pyrddlgym_rddl_tutorial.html), [Reservoir Control (Colab Tutorial)](https://colab.research.google.com/drive/19O-vgPsEX7t32cqV0bABmAdRaSWSMa4g?usp=sharing)
* Customizable [visualization](https://github.com/pyrddlgym-project/pyRDDLGym?tab=readme-ov-file#creating-your-own-visualizer) and [recording](https://github.com/pyrddlgym-project/pyRDDLGym?tab=readme-ov-file#recording-movies) tools facilitate domain debugging and plan interpretation
    * e.g., a student course project [visualizing Jax plans](https://github.com/CowboyTime/CISC813-Project-USV-Nav/blob/main/CISC813%20Gifs/V2_5Moving_2.gif) in a [sailing domain](https://github.com/CowboyTime/CISC813-Project-USV-Nav/blob/main/Version2/USV_obstacle_nav_v2_Domain.rddl)
* Runs out-of-the-box in [Python](https://github.com/pyrddlgym-project/pyRDDLGym?tab=readme-ov-file#installation) or within [Colab (RDDL Playground)](https://colab.research.google.com/drive/1XjPnlujsJPNUqhHK5EuWSVWQY2Pvxino?usp=sharing)
* Compiler tools to extract [Dynamic Bayesian Networks (DBNs)](https://github.com/pyrddlgym-project/pyRDDLGym-symbolic?tab=readme-ov-file#visualizing-dbns-with-xadd) and [Extended Algebraic Decision Diagrams (XADDs)](https://github.com/pyrddlgym-project/pyRDDLGym-symbolic?tab=readme-ov-file#xadd-compilation-of-cpfs) for symbolic analysis of causal dependencies and transition distributions
* Ready to use with out-of-the-box planners:
    * [JaxPlan](https://github.com/pyrddlgym-project/pyRDDLGym-jax): Planning through autodifferentiation
    * [GurobiPlan](https://github.com/pyrddlgym-project/pyRDDLGym-gurobi): Planning through mixed discrete-continuous optimization
    * [PROST](https://github.com/pyrddlgym-project/pyRDDLGym-prost): Monte Carlo Tree Search (MCTS)
    * [Deep Reinforcement Learning (DQN, PPO, etc.)](https://github.com/pyrddlgym-project/pyRDDLGym-rl): Popular Reinforcement Learning (RL) algorithms from Stable Baselines and RLlib
    * [Symbolic Dynamic Programming](https://github.com/pyrddlgym-project/pyRDDLGym-symbolic): Exact Symbolic regression-based planning and policy evaluation
  
## Installation

To install via pip:

```shell
pip install pyRDDLGym
```

To install the pre-release version via git:

```shell
git clone https://github.com/pyRDDLGym-project/pyRDDLGym.git
```

Since pyRDDLGym does not come with any premade environments, you can either load RDDL documents from your local file system, or install [rddlrepository](https://github.com/pyrddlgym-project/rddlrepository) for easy access to preexisting domains:

```shell
pip install rddlrepository
```

## Example Scripts

The best source of pyRDDLGym related examples is the [example gallery of Jupyter notebooks hosted on our documentation site](https://pyrddlgym.readthedocs.io/en/latest/examples.html).

Several example scripts are packaged with pyRDDLGym to highlight the core usage:
* [run_gym.py](https://github.com/pyrddlgym-project/pyRDDLGym/blob/main/pyRDDLGym/examples/run_gym.py) launches a pyRDDLGym environment and evaluates the random policy
* [run_gym2.py](https://github.com/pyrddlgym-project/pyRDDLGym/blob/main/pyRDDLGym/examples/run_gym2.py) is similar to the above but illustrates the environment interaction explicitly
* [run_ground.py](https://github.com/pyrddlgym-project/pyRDDLGym/blob/main/pyRDDLGym/examples/run_ground.py) illustrates grounding a domain and instance
* [run_intervals.py](https://github.com/pyrddlgym-project/pyRDDLGym/blob/main/pyRDDLGym/examples/run_intervals.py) computes lower and upper bounds on the policy value using interval arithmetic
* [run_server.py](https://github.com/pyrddlgym-project/pyRDDLGym/blob/main/pyRDDLGym/examples/run_server.py) illustrates how to set up pyRDDLGym to send and receive messages through TCP

## Usage

This section outlines some of the basic python API functions of pyRDDLGym.

### Loading an Environment

Instantiation of an existing environment follows OpenAI gym (i.e., instance 0 of CartPole):

```python
import pyRDDLGym
env = pyRDDLGym.make("CartPole_Continuous_gym", "0")
```

You can also load your own domain and instance files:

```python
env = pyRDDLGym.make("/path/to/domain.rddl", "/path/to/instance.rddl")
```

Both versions above instantiate ``env`` as an OpenAI gym environment, so that the usual ``reset()`` and ``step()`` calls work as intended.

You can also pass custom settings to make (i.e., to validate actions at each step):

```python
env = pyRDDLGym.make("CartPole_Continuous_gym", "0", enforce_action_constraints=True, ...)
```

### Interacting with an Environment

Policies map states to actions through the ``sample_action(obs)`` function, and can be used to interact with an environment. For example, to initialize a random agent:

```python
from pyRDDLGym.core.policy import RandomAgent
agent = RandomAgent(action_space=env.action_space, num_actions=env.max_allowed_actions)
```

You can use the policy to directly interact with the environment:

```python
state, _ = env.reset()
for epoch in range(env.horizon):
    env.render()
    action = agent.sample_action(state)
    next_state, reward, terminated, truncated, _ = env.step(action)
    state = next_state
    if terminated or truncated:
        break
env.close()
```

All agent instances support one-line evaluation in a given environment:

```python
stats = agent.evaluate(env, episodes=1, verbose=True, render=True)
```

which returns a dictionary of summary statistics (e.g. "mean", "std", etc...).

> [!NOTE]  
> All observations (for a POMDP), states (for an MDP) and actions are represented by ``dict`` objects, whose keys correspond to the appropriate fluents as defined in the RDDL description.
> Here, the syntax is ``pvar-name___o1__o2...``, where ``pvar-name`` is the pvariable name, followed by 3 underscores, and object parameters ``o1``, ``o2``... are separated by 2 underscores.

> [!WARNING] 
> There are two known issues not documented with RDDL:
> 1. the minus (-) arithmetic operation must have spaces on both sides, otherwise there is ambiguity whether it refers to a mathematical operation or to variables
> 2. aggregation-union-precedence parsing requires for encapsulating parentheses around aggregations, e.g., (sum_{}[]).

### Visualization

pyRDDLGym supports several standard visualization modes:

```python
env.set_visualizer("chart")
env.set_visualizer("heatmap")
env.set_visualizer("text")
```

You can also design your own visualizer by subclassing ``pyRDDLGym.core.visualizer.viz.BaseViz`` and overriding the ``render(state)`` method. Then, changing the visualizer of the environment is easy:

```python
viz_class = ...   # the type/class of your custom viz
env.set_visualizer(viz_class)
```

### Recording Movies

You can record an animated gif or movie when interacting with an environment. 
Simply pass a ``MovieGenerator`` object to the ``set_visualizer`` method:

```python
from pyRDDLGym.core.visualizer.movie import MovieGenerator
recorder = MovieGenerator("/path/where/to/save", "env_name")
env.set_visualizer(viz_class, movie_gen=recorder)

# continue with normal interaction
```

## Status

A complete archive of past and present RDDL problems, including all IPPC problems, is also available to clone\pip
* [rddlrepository](https://github.com/pyRDDLGym-project/rddlrepository) (``pip install rddlrepository``)

Software for related simulators:
* [rddlsim](https://github.com/ssanner/rddlsim)
* [rddlgym](https://github.com/thiagopbueno/rddlgym)
* [pddlgym](https://github.com/tomsilver/pddlgym)

The parser used in this project is based on the parser from 
Thiago Pbueno's [pyrddl](https://github.com/thiagopbueno/pyrddl)
(used in [rddlgym](https://github.com/thiagopbueno/rddlgym)).

## Citing pyRDDLGym

Please see our [paper](https://arxiv.org/abs/2211.05939) describing pyRDDLGym. If you found this useful, please consider citing us:

```
@article{taitler2022pyrddlgym,
      title={pyRDDLGym: From RDDL to Gym Environments},
      author={Taitler, Ayal and Gimelfarb, Michael and Gopalakrishnan, Sriram and Mladenov, Martin and Liu, Xiaotian and Sanner, Scott},
      journal={arXiv preprint arXiv:2211.05939},
      year={2022}}
```

## License
This software is distributed under the MIT License.
