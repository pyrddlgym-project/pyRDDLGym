

# overall

todo  LATER: handle type matching in expressions 

Handle nested sum, product, exists operations. 
    this will apply to cpfs, reward, constraints

Handle AVG which is / n first 

Handle For all, and Exists statements

# Simulator interface
reach out to Mike

# Grounding CPFS

In a for loop, need to expand the dictionary for aggregate ops

use the grounding dictionaries in the recursion and ground subseq.
if a non-fluent check fails, stop that grounding chain, and
return. When the leaf is reached, add to the successfully grounded
list (passed through the recursions)
