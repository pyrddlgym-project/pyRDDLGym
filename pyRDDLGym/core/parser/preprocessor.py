from abc import ABCMeta, abstractmethod
from itertools import product
import re


class RDDLPreprocessor(metaclass=ABCMeta):
    '''Abstract base class for RDDL preprocessors.'''

    @abstractmethod
    def preprocess(self, rddl_str: str) -> str:
        '''Preprocess the given RDDL string.'''
        pass


class RDDLPreprocessorChain(RDDLPreprocessor):
    ''''A chain of RDDL preprocessors that applies them sequentially.'''

    def __init__(self, preprocessors: list[RDDLPreprocessor]):
        self.preprocessors = preprocessors

    def preprocess(self, rddl_str: str) -> str:
        for preprocessor in self.preprocessors:
            rddl_str = preprocessor.preprocess(rddl_str)
        return rddl_str


class RDDLPreprocessorIdentity(RDDLPreprocessor):
    '''A preprocessor that returns the input string unchanged.'''

    def preprocess(self, rddl_str: str) -> str:
        return rddl_str


class RDDLEnumPreprocessor(RDDLPreprocessor):
    '''A preprocessor that replaces enum definitions with their corresponding values.'''

    OUTER = re.compile(r'\{\{\s*@([\w-]*)((?:\[\s*\d+\s*,\s*\d+\s*\])+)\s*\}\}')
    RANGE  = re.compile(r'\[\s*(\d+)\s*,\s*(\d+)\s*\]')

    @staticmethod
    def _expand(match):
        ident  = match.group(1)
        ranges = [(int(a), int(b)) 
                  for a, b in RDDLEnumPreprocessor.RANGE.findall(match.group(2))]
        combos = product(*(range(lo, hi + 1) for lo, hi in ranges))
        prefix = f'@{ident}' if ident else '@'
        values = [prefix + '_'.join(str(i) for i in combo) for combo in combos]
        return '{ ' + ', '.join(values) + ' }'

    def preprocess(self, rddl_str: str) -> str:
        return RDDLEnumPreprocessor.OUTER.sub(RDDLEnumPreprocessor._expand, rddl_str)
