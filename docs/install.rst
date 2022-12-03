#### Python version
We require Python 3.7+.

### Requirements:
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
