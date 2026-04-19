from pyRDDLGym.core.debug.decompiler import RDDLDecompiler
from pyRDDLGym.core.parser.expr import Expression

import warnings
try:
    import termcolor
except:
    termcolor = None

ERROR_MESSAGE_DECOMPILER = RDDLDecompiler()


def print_stack_trace(expr) -> str:
    '''Prints a stack trace for the given expression, showing the sequence of operations 
    that led to it.'''
    if isinstance(expr, Expression):
        trace = ERROR_MESSAGE_DECOMPILER.decompile_expr(expr)
    else:
        trace = str(expr)
    return f'>> {trace}'


def print_stack_trace_root(expr, root) -> str:
    '''Prints a stack trace for the given expression, showing the sequence of operations
    that led to it, and also includes the root expression for context.'''
    return print_stack_trace(expr) + '\n' + f'Please check expression for {root}.' 


def raise_warning(message: str, color: str='yellow') -> None:
    '''Raises a warning with the given message, optionally colored.'''
    if termcolor is not None:
        message = termcolor.colored(message, color)
    warnings.warn(message)
    
    
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


class RDDLRandPolicyVecNotImplemented(NotImplementedError):
    pass