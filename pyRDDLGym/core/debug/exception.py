from pyRDDLGym.core.debug.decompiler import RDDLDecompiler
from pyRDDLGym.core.parser.expr import Expression

ERROR_MESSAGE_DECOMPILER = RDDLDecompiler()


def print_stack_trace(expr):
    if isinstance(expr, Expression):
        trace = ERROR_MESSAGE_DECOMPILER.decompile_expr(expr)
    else:
        trace = str(expr)
    return f'>> {trace}'


def print_stack_trace_root(expr, root):
    return print_stack_trace(expr) + '\n' + f'Please check expression for {root}.' 


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


class RDDLEnvironmentNotExistError(ValueError):
    pass


class RDDLInstanceNotExistError(ValueError):
    pass


class RDDLLogFolderError(ValueError):
    pass


class RDDLEpisodeAlreadyEndedError(RuntimeError):
    pass
