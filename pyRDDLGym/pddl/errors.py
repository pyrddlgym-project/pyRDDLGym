from typing import Optional


class PDDLError(Exception):
    '''Base class for errors raised while reading PDDL, with an optional location.'''

    def __init__(self, message: str, line: Optional[int]=None,
                 col: Optional[int]=None) -> None:
        self.message = message
        self.line = line
        self.col = col
        if line is not None:
            message = f'line {line}, col {col}: {message}'
        super().__init__(message)


class PDDLParseError(PDDLError):
    '''The PDDL text is malformed or refers to undeclared or mistyped symbols.'''
    pass


class PDDLUnsupportedError(PDDLError):
    '''The PDDL text is valid but uses a construct that cannot be converted to RDDL.'''
    pass
