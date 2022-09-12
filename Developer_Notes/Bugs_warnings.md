# RDDLGrounder.py
## _groundCPF   
post variable grounding, in the cpf expression
we still see ?x , ?y, ?x2, etc. unsure if this is intentionally or pending
completion.
### possible fix  : nested dicts
during recursion of the expression tree, we would
need to update our dictionary for other variables (?x2, ?y2)
that obey the conditions in the cpf. These variables may have a
LIST of possible values, and each value maybe a tuple, 
example (?x2 = x3_4, ?y2 = y_3_2). What adds difficulty is that
associated to each instance maybe another nested list of variables
i.e. for (?x2 = x3_4, ?y2 = y_3_2), a nested expression may
require instantiation of (?x3,?y3), and the grounded value for
them may only be valid for a particular (?x2, ?y2).
SO...I think what we need is a dict of dicts. example:
{(?x,?y):{ (x1,y2) : { (?x2,?y2): { ... }}}
The leaf values in this nested dictionary would be a dictionary with 
only key values , which is in effect a list. Not the most elegant, but 
manages our needs , I think. Each time we recurse, we add a dict, 
each time we return, we remove the lowest level of that dictionary (still needs thought)