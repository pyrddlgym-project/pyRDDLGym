from typing import Optional


class PDDLError(Exception):

    def __init__(self, message: str, line: Optional[int]=None, 
                 col: Optional[int]=None) -> None:
        self.message = message
        self.line = line
        self.col = col
        if line is not None:
            message = f'line {line}, col {col}: {message}'
        super().__init__(message)


class PDDLParseError(PDDLError):
    pass


class PDDLUnsupportedError(PDDLError):
    pass
