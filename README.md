# RDDLGym

A toolkit for autogeneration of OpenAI Gym environments from RDDL description files. 

Bug fix:
* Parsing:
  * param_list was not properly defined, now defined as '(' str_list ')' or empty
  * interm-fluent was not defined without level parameter
  * derived-fluents were not defined in parser (only in lexer)
  * derived-fluents field were not defined in pvariable class
  * derived cpfs were not defined in domain class
  * observ-fluents were not defined in parser (only in lexer)
  * power unit commitment domain had non-fluent ambiguity definition in instance (implicit in instance)
  * Aggregation union precedence parsing still exists (the bug is traced back to the JAVA code).<br/> 
  Temp fix: put parenthesis around aggregations - (sum_{}[])


Features
* support added for separate domain and instance (+non-fluents) files
* informative exception is raised when there is a problem with a block (domain, non-fluents, instance)
* RDDL text generator from parsed ast (partial support of the language)