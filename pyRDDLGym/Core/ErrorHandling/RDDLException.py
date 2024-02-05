from pyRDDLGym.Core.Compiler.RDDLDecompiler import RDDLDecompiler
from pyRDDLGym.Core.Parser.expr import Expression

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


class RDDLEnvironmentNotExist(ValueError):
    pass


class RDDLInstanceNotExist(ValueError):
    pass


class RDDLLogFolderError(ValueError):
    pass


class RDDLEpisodeAlreadyEndedError(RuntimeError):
    pass


class RDDLRandPolicyVecNotImplemented(NotImplementedError):
    pass