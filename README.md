# RDDLGym

A toolkit for autogeneration of OpenAI Gym environments from RDDL description files. 

Bug fix:
* Parsing:
  * param_list was not properly defined, now defined as '(' str_list ')' or empty
  * interm-fluent was not defined without level parameter
  * derived-fluents were not defined in parser (only in lexer)
  * observ-fluents were not defined in parser (only in lexer)
  * power unit commitment domain had non-fluent ambiguity definition in instance (implicit in instance)


Features
* support added for separate domain and instance (+non-fluents) files
* informative exception is raise when there is a problem with a block (domain, non-fluents, instance)