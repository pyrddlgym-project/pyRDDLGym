#support for max/ min

do I compile this into an if statement with a > b, etc. 
handle cases for a == b and so forth

# if-else statements: are else statements required, or do we insert a "zero" or default value for the variable

I assume we DO require an else condition?
            # or do we manually have to insert an else with "default" value for cpfs ??
            

#constraints in aggregate statements 
Sum( <set def> : s.t. condition )
the conditions are typically in if statements, but need to review the RDDL language spec for this. 
are the if statements insertable into a sum def. FOR NOW: only type fluents are set

# aggregate statements with only one instance in set
There could be a sum statement, where only one instance statisfies the type def with constraints
