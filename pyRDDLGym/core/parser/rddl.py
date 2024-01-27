# This file is part of thiago pbueno's pyrddl.
# https://github.com/thiagopbueno/pyrddl

from pyRDDLGym.Core.Parser.domain import Domain
from pyRDDLGym.Core.Parser.instance import Instance
from pyRDDLGym.Core.Parser.nonfluents import NonFluents

import collections
import itertools
from typing import Dict, List, Sequence, Optional, Tuple, Union

Block = Union[Domain, NonFluents, Instance]
ObjectStruct = Dict[str, Union[int, Dict[str, int], List[str]]]
ObjectTable = Dict[str, ObjectStruct]
FluentParamsList = Sequence[Tuple[str, List[str]]]


class RDDL(object):
    '''RDDL class for accessing RDDL blocks.
    Note:
        This class is intended to be solely used by the parser and compiler.
        Do not attempt to directly use this class to build a RDDL object.
    Args:
        blocks: Mapping from string to RDDL block.
    Attributes:
        domain (:obj:`Domain`): RDDL domain block.
        non_fluents (:obj:`NonFluents`): RDDL non-fluents block.
        instance (:obj:`Instance`): RDDL instance block.
        object_table (:obj:`ObjectTable`): The object table for each RDDL type.
    '''

    def __init__(self, blocks: Dict[str, Block]) -> None:
        self.domain = blocks['domain']
        self.non_fluents = blocks['non_fluents']
        self.instance = blocks['instance']

    def build(self):
        self.domain.build()
        self._build_object_table()
        self._build_fluent_table()

    def _build_object_table(self):
        '''Builds the object table for each RDDL type.'''
        types = self.domain.types
        objects = dict(self.non_fluents.objects)
        self.object_table = dict()
        for name, value in self.domain.types:
            if value == 'object':
                objs = objects[name]
                idx = { obj: i for i, obj in enumerate(objs) }
                self.object_table[name] = {
                    'size': len(objs),
                    'idx': idx,
                    'objects': objs
                }

    def _build_fluent_table(self):
        '''Builds the fluent table for each RDDL pvariable.'''
        self.fluent_table = collections.OrderedDict()

        for name, size in zip(self.domain.non_fluent_ordering, self.non_fluent_size):
            non_fluent = self.domain.non_fluents[name]
            self.fluent_table[name] = (non_fluent, size)

        for name, size in zip(self.domain.state_fluent_ordering, self.state_size):
            fluent = self.domain.state_fluents[name]
            self.fluent_table[name] = (fluent, size)

        for name, size in zip(self.domain.action_fluent_ordering, self.action_size):
            fluent = self.domain.action_fluents[name]
            self.fluent_table[name] = (fluent, size)

        for name, size in zip(self.domain.interm_fluent_ordering, self.interm_size):
            fluent = self.domain.intermediate_fluents[name]
            self.fluent_table[name] = (fluent, size)

    @property
    def non_fluent_variables(self) -> FluentParamsList:
        '''Returns the instantiated non-fluents in canonical order.
        Returns:
            Sequence[Tuple[str, List[str]]]: A tuple of pairs of fluent name
            and a list of instantiated fluents represented as strings.
        '''
        fluents = self.domain.non_fluents
        ordering = self.domain.non_fluent_ordering
        return self._fluent_params(fluents, ordering)

    @property
    def state_fluent_variables(self) -> FluentParamsList:
        '''Returns the instantiated state fluents in canonical order.
        Returns:
            Sequence[Tuple[str, List[str]]]: A tuple of pairs of fluent name
            and a list of instantiated fluents represented as strings.
        '''
        fluents = self.domain.state_fluents
        ordering = self.domain.state_fluent_ordering
        return self._fluent_params(fluents, ordering)

    @property
    def interm_fluent_variables(self) -> FluentParamsList:
        '''Returns the instantiated intermediate fluents in canonical order.
        Returns:
            Sequence[Tuple[str, List[str]]]: A tuple of pairs of fluent name
            and a list of instantiated fluents represented as strings.
        '''
        fluents = self.domain.intermediate_fluents
        ordering = self.domain.interm_fluent_ordering
        return self._fluent_params(fluents, ordering)

    @property
    def action_fluent_variables(self) -> FluentParamsList:
        '''Returns the instantiated action fluents in canonical order.
        Returns:
            Sequence[Tuple[str, List[str]]]: A tuple of pairs of fluent name
            and a list of instantiated fluents represented as strings.
        '''
        fluents = self.domain.action_fluents
        ordering = self.domain.action_fluent_ordering
        return self._fluent_params(fluents, ordering)

    @property
    def non_fluent_size(self) -> Sequence[Sequence[int]]:
        '''The size of each non-fluent in canonical order.
        Returns:
            Sequence[Sequence[int]]: A tuple of tuple of integers
            representing the shape and size of each non-fluent.
        '''
        fluents = self.domain.non_fluents
        ordering = self.domain.non_fluent_ordering
        return self._fluent_size(fluents, ordering)

    @property
    def state_size(self) -> Sequence[Sequence[int]]:
        '''The size of each state fluent in canonical order.
        Returns:
            Sequence[Sequence[int]]: A tuple of tuple of integers
            representing the shape and size of each fluent.
        '''
        fluents = self.domain.state_fluents
        ordering = self.domain.state_fluent_ordering
        return self._fluent_size(fluents, ordering)

    @property
    def action_size(self) -> Sequence[Sequence[int]]:
        '''The size of each action fluent in canonical order.
        Returns:
            Sequence[Sequence[int]]: A tuple of tuple of integers
            representing the shape and size of each fluent.
        '''
        fluents = self.domain.action_fluents
        ordering = self.domain.action_fluent_ordering
        return self._fluent_size(fluents, ordering)

    @property
    def interm_size(self)-> Sequence[Sequence[int]]:
        '''The size of each intermediate fluent in canonical order.
        Returns:
            Sequence[Sequence[int]]: A tuple of tuple of integers
            representing the shape and size of each fluent.
        '''
        fluents = self.domain.intermediate_fluents
        ordering = self.domain.interm_fluent_ordering
        return self._fluent_size(fluents, ordering)

    @property
    def state_range_type(self) -> Sequence[str]:
        '''The range type of each state fluent in canonical order.
        Returns:
            Sequence[str]: A tuple of range types representing
            the range of each fluent.
        '''
        fluents = self.domain.state_fluents
        ordering = self.domain.state_fluent_ordering
        return self._fluent_range_type(fluents, ordering)

    @property
    def action_range_type(self) -> Sequence[str]:
        '''The range type of each action fluent in canonical order.
        Returns:
            Sequence[str]: A tuple of range types representing
            the range of each fluent.
        '''
        fluents = self.domain.action_fluents
        ordering = self.domain.action_fluent_ordering
        return self._fluent_range_type(fluents, ordering)

    @property
    def interm_range_type(self) -> Sequence[str]:
        '''The range type of each intermediate fluent in canonical order.
        Returns:
            Sequence[str]: A tuple of range types representing
            the range of each fluent.
        '''
        fluents = self.domain.intermediate_fluents
        ordering = self.domain.interm_fluent_ordering
        return self._fluent_range_type(fluents, ordering)

    @classmethod
    def _fluent_range_type(cls, fluents, ordering) -> Sequence[str]:
        '''Returns the range types of `fluents` following the given `ordering`.
        Returns:
            Sequence[str]: A tuple of range types representing
            the range of each fluent.
        '''
        range_types = []
        for name in ordering:
            fluent = fluents[name]
            range_type = fluent.range
            range_types.append(range_type)
        return tuple(range_types)

    def _fluent_params(self, fluents, ordering) -> FluentParamsList:
        '''Returns the instantiated `fluents` for the given `ordering`.
        For each fluent in `fluents`, it instantiates each parameter
        type w.r.t. the contents of the object table.
        Returns:
            Sequence[Tuple[str, List[str]]]: A tuple of pairs of fluent name
            and a list of instantiated fluents represented as strings.
        '''
        variables = []
        for fluent_id in ordering:
            fluent = fluents[fluent_id]
            param_types = fluent.param_types
            objects = ()
            names = []
            if param_types is None:
                names = [fluent.name]
            else:
                objects = tuple(self.object_table[ptype]['objects'] for ptype in param_types)
                for values in itertools.product(*objects):
                    values = ','.join(values)
                    var_name = '{}({})'.format(fluent.name, values)
                    names.append(var_name)
            variables.append((fluent_id, names))
        return tuple(variables)

    def _fluent_size(self, fluents, ordering) -> Sequence[Sequence[int]]:
        '''Returns the sizes of `fluents` following the given `ordering`.
        Returns:
            Sequence[Sequence[int]]: A tuple of tuple of integers
            representing the shape and size of each fluent.
        '''
        shapes = []
        for name in ordering:
            fluent = fluents[name]
            shape = self._param_types_to_shape(fluent.param_types)
            shapes.append(shape)
        return tuple(shapes)

    def _param_types_to_shape(self, param_types: Optional[str]) -> Sequence[int]:
        '''Returns the fluent shape given its `param_types`.'''
        param_types = [] if param_types is None else param_types
        shape = tuple(self.object_table[ptype]['size'] for ptype in param_types)
        return shape

    def get_dependencies(self, expr):
        deps = set()

        expressions = [expr]
        while expressions:
            expr = expressions.pop()

            for name in expr.scope:
                fluent, _ = self.fluent_table[name]

                if fluent.is_intermediate_fluent():
                    cpf = self.domain.get_intermediate_cpf(name)
                    expressions.append(cpf.expr)
                else:
                    deps.add(fluent)

        return deps
