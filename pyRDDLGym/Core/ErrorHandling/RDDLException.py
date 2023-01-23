

class RDDLActionPreconditionNotSatisfiedError(ValueError):
    pass


class RDDLInvalidActionError(ValueError):
    pass


class RDDLInvalidDependencyInCPFError(SyntaxError):
    pass


class RDDLInvalidExpressionError(SyntaxError):
    pass


class RDDLInvalidNumberOfArgumentsError(SyntaxError):
    pass


class RDDLInvalidObjectError(SyntaxError):
    pass


class RDDLMissingCPFDefinitionError(SyntaxError):
    pass


class RDDLNotImplementedError(NotImplementedError):
    pass


class RDDLParseError(SyntaxError):
    pass


class RDDLRepeatedVariableError(SyntaxError):
    pass
    

class RDDLStateInvariantNotSatisfiedError(ValueError):
    pass


class RDDLTypeError(TypeError):
    pass


class RDDLUndefinedCPFError(SyntaxError):
    pass


class RDDLUndefinedVariableError(SyntaxError):
    pass


class RDDLValueOutOfRangeError(ValueError):
    pass


class RDDLEnvironmentNotExist(ValueError):
    pass


class RDDLInstanceNotExist(ValueError):
    pass
