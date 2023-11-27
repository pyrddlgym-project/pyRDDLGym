# Symbolic Solvers via Dynamic Programming (SDP)

Author: [Jihwan Jeong](jihwan-jeong.netlify.app)


In this directory, we provide solvers that solve a Markov decision process specified via RDDL domain and instance files using Symbolic Dynamic Programming (SDP). 


## Installation

```
# Create a new conda environment
conda create -n sdp python=3.11     # Python 3.12 won't work with Gurobi 10.

# Install the XADDPy package
pip install xaddpy

# Install Gurobipy (make sure you have a license)
python -m pip install gurobipy

# Go to the directory where pyRDDLGym is cloned
cd ~/path/to/pyRDDLGym

# Then, install pyRDDLGym
pip install -e .
```

In addition, you'll need to install `pygraphviz` to visualize XADDs in the graph format. For this, check out the content here: [installing-graphviz](https://github.com/jihwan-jeong/xaddpy#step-1-installing-graphviz).

## Running Example 1: Value Iteration (VI)

With the [run_vi.py](../../../run_examples/run_vi.py) file, you can run a value iteration solver.

Here, we provide a detailed dissection of the run script.

### XADD compilation of a given RDDL domain/instance.

```python
    env_info = ExampleManager.GetEnvInfo(args.env)
    domain = env_info.get_domain()
    instance = env_info.get_instance(args.inst)

    # Read and parse domain and instance.
    reader = RDDLReader(domain, instance)
    domain = reader.rddltxt
    rddl_parser = RDDLParser(None, False)
    rddl_parser.build()

    # Parse RDDL file.
    rddl_ast = rddl_parser.parse(domain)

    # Ground domain.
    grounder = RDDLGrounder(rddl_ast)
    model = grounder.Ground()

    # XADD compilation.
    xadd_model = RDDLModelWXADD(model, simulation=False)
    xadd_model.compile()
    ...
```

In the above code snippet (lines 17-36), we read the RDDL domain and instance files and perform the XADD compilation of the problem (`xadd_model.compile()`).

### Constructing the MDP problem with the associated XADD model

```python
    mdp_parser = Parser()
    mdp = mdp_parser.parse(
        xadd_model,
        xadd_model.discount,
        concurrency=rddl_ast.instance.max_nondef_actions,
        is_linear=args.is_linear,
        include_noop=not args.skip_noop,
        is_vi=True,
    )
```
Then, in lines 38 - 46, we instantiate an `MDPParser` object that has the `parse` method, which interprets the XADD RDDL model and construct some necessary attributes, like CPFs and such.

Some important operations that happen within the parser are as follows:

- __Bound analysis on continuous variables (lines 51-58 and lines 100-103):__

```python
# Configure the bounds of continuous states.
cont_s_vars = set()
for s in model.states:
    if model.gvar_to_type[s] != 'real':
        continue
    cont_s_vars.add(model.ns[s])
cont_state_bounds = self.configure_bounds(mdp, model.invariants, cont_s_vars)
mdp.cont_state_bounds = cont_state_bounds
...
...
# Configure the bounds of continuous actions.
if len(mdp.cont_a_vars) > 0:
    cont_action_bounds = self.configure_bounds(mdp, model.preconditions, mdp.cont_a_vars)
    mdp.cont_action_bounds = cont_action_bounds
```
Here, the parser has a method called `configure_bounds` in which we perform the analysis on bounds of continuous variables. Specifically, the bound information has to be provided in `state-invariants` and `action-preconditions` blocks of the original RDDL domain file. If no bounds are provided for a variable, we assume `[-inf, inf]` as its bounds.

Once configured, this bound information is then updated to the `XADD` context object such that each continuous symbolic variable is associated with its upper and lower bounds.

- __Handling of concurrent boolean actions (lines 77 - 91)__
```python
if is_vi:
    # Need to consider all combinations of boolean actions.
    # Note: there's always an implicit no-op action with which
    # none of the boolean actions are set to True.
    total_bool_actions = tuple(
        _truncated_powerset(
            bool_actions,
            mdp.max_allowed_actions,
            include_noop=include_noop,
    ))
    for actions in total_bool_actions:
        names = tuple(a.name for a in actions)
        symbols = tuple(a.symbol for a in actions)
        action = BActions(names, symbols, model)
        mdp.add_action(action)
```
This part is where we handle concurrent actions, specifically for Value Iteration. Here we have a few modeling assumptions. First, continuous actions will always be concurrent, so we only specifically handle concurrent Boolean actions. Second, we provide an option to either use or not use a `no-op` action, which sets all Boolean action values to `False`.

Now, let's say we have 2 Boolean actions: `move___a_1` and `pick___a_1`. When the concurrency is set to `1` and we allow the `noop` action, then we'll have the following Boolean actions:

- `noop` (i.e., `{move___a_1: False, pick___a_1: False}`)
- `{move___a_1: True, pick___a_1: False}`
- `{move___a_1: False, pick___a_1: True}`

On the other hand, if the concurrency is set to `2`, then we will have the following concurrent Boolean actions:
- `noop` (i.e., `{move___a_1: False, pick___a_1: False}`)
- `{move___a_1: True, pick___a_1: False}`
- `{move___a_1: False, pick___a_1: True}`
- `{move___a_1: True, pick___a_1: True}`

That is, the concurrency value specifies the maximum number of Boolean actions that can be taken at each time step, so we should consider all possible combinations, which is done by the `_truncated_powerset` helper function.

We define a class `BActions` that can handle any of these concurrent actions. More importantly, the class implements a `restrict` method in which we restrict a given XADD with the associated action values.

- Constructing the full CPFs for Boolean next state and interm variables (line 112)

By calling `mdp.update(is_vi=is_vi)`, we update the CPFs of Boolean next state and interm variables to fully consider `P(b'=0|...)`. This is a necessary step as in the RDDL file, we have only specified the probability of a Boolean variable being `True`. Also, the `update` method links updated CPFs with each action.

### Solving the MDP

Finally, we call `vi_solver.solve()` which will perform SDP to obtain the optimal symbolic value function.

Notice that the `solve` method is shared by the `ValueIteration` and `PolicyEvaluation` solvers; hence, it's defined in [base.py](./base.py). The method will return the integer ID of the optimal value function at a set iteration number.

## Running Example 2: Policy Evaluation (PE)

With the [run_pe.py](../../../run_examples/run_pe.py) file, you can run a policy evaluation solver.

The script is exactly the same as run_vi.py until the XADD RDDL model compilation is done. Then, a slight difference of PE from VI is what we pass to the `MDPParser.parse` function. 

```python
mdp = mdp_parser.parse(
        xadd_model,
        xadd_model.discount,
        concurrency=rddl_ast.instance.max_nondef_actions,
        is_linear=args.is_linear,
        is_vi=False,
    )
```
In PE, we do not have to specify the maximum concurrency value to the parser as that should be implicitly determined by the given policy. Instead, we set `is_vi=False` such that we do not create `BActions` objects.

Then, in lines 47 - 53, we instantiate a `PolicyParser` object and parse the policy provided in a json format, specified by the argument `--policy_fpath`. An example policy json file looks like the following ([p1.json](pyRDDLGym/Examples/SDPTestDomains/1D/policy/p1.json)):

```json
{
    "action-fluents": ["a"],
    "a": "pyRDDLGym/Examples/SDPTestDomains/1D/policy/a.xadd"
}
```
A policy json file should have the following field:
- "action-fluents": a list of grounded action variable names.

Then, it should be followed by "action-name": "file path" pairs for all actions specified in "action-fluents". This json file should specify the file path of each and every action fluent of a given problem; otherwise, an assertion error will occur from the parser.

The value of one action variable points to the file path where the XADD of that action is defined. The `PolicyParser` will read in the XADD and perform some checks (e.g., type and dependency checks). Check out the comments in the [policy_parser.py](./helper/policy_parser.py) file for more detailed information.

### Assertion for concurrency

The `PolicyParser` class implements an assertion that in the entire state space no more than the set `concurrency` number of Boolean actions can be set to `True`. Check out the `_assert_concurrency` method in lines 150 - 172 of [policy_parser.py](./helper/policy_parser.py).

### Substitution of the policy into CPFs and reward function

A unique step in PE is where we substitute in the policy XADDs into the CPFs and the reward function. See lines 21 - 59 of [pe.py](./pe.py). Note how we handle the continuous and Boolean action variables differently. 

Once all the CPFs and reward function are restricted with the given policy XADDs, the remaining steps are identical to VI, except that we do not have to iterate over actions as they have all been already incorporated into CPFs.

