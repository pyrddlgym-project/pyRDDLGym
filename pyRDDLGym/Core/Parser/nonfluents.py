# This file is based on thiago pbueno's pyrddl.
# https://github.com/thiagopbueno/pyrddl
# it was adapted and extended for pyRDDLGym

from typing import Dict, List, Sequence, Tuple, Union

FluentTuple = Tuple[Union[str, None]]
Value = Union[bool, int, float]
FluentInitializer = Tuple[FluentTuple, Value]
FluentInitializerList = List[FluentInitializer]

ObjectsList = List[Tuple[str, List[str]]]


class NonFluents(object):
    '''NonFluents class for accessing RDDL non-fluents sections.
    Note:
        This class is intended to be solely used by the parser and compiler.
        Do not attempt to directly use this class to build a NonFluents object.
    Args:
        name: Name of RDDL non-fluents.
        sections: Mapping from string to non-fluents section.
    Attributes:
        name (str): Name of RDDL non-fluents block.
        domain (str): Name of RDDL domain block.
        objects (:obj:`ObjectsList`): List of RDDL objects for each type.
        init_non_fluent (:obj:`FluentInitializerList`): List of non-fluent initializers.
    '''

    def __init__(self, name: str, sections: Dict[str, Sequence]) -> None:
        self.name = name
        self.objects = []
        self.__dict__.update(sections)
