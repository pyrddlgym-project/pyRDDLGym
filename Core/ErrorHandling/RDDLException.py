

class RDDLActionPreconditionNotSatisfiedError(ValueError):
    pass


class RDDLInvalidDependencyInCPFError(SyntaxError):
    pass


class RDDLInvalidExpressionError(SyntaxError):
    pass


class RDDLInvalidNumberOfArgumentsError(SyntaxError):
    pass


class RDDLMissingCPFDefinitionError(SyntaxError):
    pass


class RDDLNotImplementedError(NotImplementedError):
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

