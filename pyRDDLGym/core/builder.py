from typing import Any, Iterable


class RDDLBuilder:
    '''A general class for building RDDL domain and instance code programmatically.'''

    def __init__(self) -> None:

        # domain definitions
        self.object_types = set()
        self.enum_types = {}
        self.pvariable_defs = {}
        self.cpf_defs = {}
        self.reward_def = None
        self.termination_defs = []
        self.invariant_defs = []
        self.precondition_defs = []

        # instance definitions
        self.object_values = {}
        self.nonfluent_inits = {}
        self.init_states = {}
        self.maxnondef = None
        self.horizon = None
        self.discount = None
    
    # ===========================================================================
    # domain construction
    # ===========================================================================
    
    def add_object_type(self, name: str) -> None:
        self.object_types.add(name)
    
    def add_enum_type(self, name: str, objects: Iterable[str]) -> None:
        self.enum_types[name] = list(objects)
    
    def add_pvariable(self, name: str, params: Iterable[str], ptype: str, prange: str, 
                      default: Any) -> None:
        params = tuple(params)
        if ptype in {'derived-fluent', 'interm-fluent', 'observ-fluent'}:
            default = None
        if ptype not in {'non-fluent', 'derived-fluent', 'interm-fluent', 
                         'action-fluent', 'state-fluent', 'observ-fluent'}:
            raise ValueError(f'<{ptype}> is not a valid pvariable type.')
        if prange not in {'bool', 'int', 'real'} \
        and prange not in self.object_types and prange not in self.enum_types:
            raise ValueError(f'<{prange}> is not a valid pvariable range.')
        for param in params:
            if param not in self.object_types and param not in self.enum_types:
                raise ValueError(f'<{param}> is not a valid pvariable parameter type.')
        self.pvariable_defs[name] = (params, ptype, prange, default)
    
    def add_cpf(self, name: str, params: Iterable[str], expr: str) -> None:
        params = tuple(params)
        if name.endswith('\''):
            name = name[:-1]
        self.cpf_defs[name] = (params, expr)
    
    def add_reward(self, expr: str) -> None:
        self.reward_def = expr
    
    def add_termination(self, expr: str) -> None:
        self.termination_defs.append(expr)
    
    def add_invariant(self, expr: str) -> None:
        self.invariant_defs.append(expr)

    def add_precondition(self, expr: str) -> None:
        self.precondition_defs.append(expr)
    
    def build_domain(self, name: str) -> str:
        result = f'domain {name} {{\n\n'

        # write types block
        if self.object_types or self.enum_types:
            result += '\ttypes {\n'
            for otype in self.object_types:
                result += f'\t\t{otype} : object;\n'
            for (etype, objects) in self.enum_types.items():
                objects_str = ', '.join(objects)
                result += f'\t\t{etype} : {{ {objects_str} }};\n'
            result += '\t};\n\n'

        # write pvariables block
        result += '\tpvariables {\n'
        for (pname, (params, ptype, prange, default)) in self.pvariable_defs.items():
            if params:
                params_str = '(' + ', '.join(params) + ')'
            else:
                params_str = ''
            if default is None:
                default_str = ''
            else:
                default_str = f', default={default}'
            result += f'\t\t{pname}{params_str} : {{ {ptype}, {prange}{default_str} }};\n'
        result += '\t};\n\n'

        # write cpfs block
        result += '\tcpfs {\n'
        for (pname, (params, expr)) in self.cpf_defs.items():
            _, ptype, *_ = self.pvariable_defs[pname]
            if params:
                params_str = '(' + ', '.join(params) + ')'
            else:
                params_str = ''
            if ptype == 'state-fluent':
                pname += '\''
            result += f'\t\t{pname}{params_str} = {expr};\n'
        result += '\t};\n\n'    
        
        # write reward
        result += f'\treward = {self.reward_def};\n'

        # write termination block
        if self.termination_defs:
            result += '\n\ttermination {\n'
            for expr in self.termination_defs:
                result += f'\t\t{expr};\n'
            result += '\t};\n'

        # write state-invariants block
        if self.invariant_defs:
            result += '\n\tstate-invariants {\n'
            for expr in self.invariant_defs:
                result += f'\t\t{expr};\n'
            result += '\t};\n'
        
        # write action-preconditions block
        if self.precondition_defs:
            result += '\n\taction-preconditions {\n'
            for expr in self.precondition_defs:
                result += f'\t\t{expr};\n'
            result += '\t};\n'
        
        result += '}'
        return result

    # ===========================================================================
    # instance construction
    # ===========================================================================
    
    def add_object_values(self, name: str, values: Iterable[str]) -> None:
        self.object_values.setdefault(name, []).extend(values)
    
    def add_nonfluent_init(self, pvar: str, params: Iterable[str], value: Any) -> None:
        params = tuple(params)
        key = pvar
        if params:
            key += '(' + ', '.join(params) + ')'
        self.nonfluent_inits[key] = value
    
    def add_init_state(self, pvar: str, params: Iterable[str], value: Any) -> None:
        params = tuple(params)
        key = pvar
        if params:
            key += '(' + ', '.join(params) + ')'
        self.init_states[key] = value
    
    def add_max_nondef_actions(self, value: int) -> None:
        self.maxnondef = value

    def add_discount(self, value: float) -> None:
        self.discount = value

    def add_horizon(self, value: int) -> None:
        self.horizon = value
    
    def build_instance(self, domain_name: str, instance_name: str, nonfluents_name: str) -> str:
        result = f'non-fluents {nonfluents_name} {{\n\n'
        result += f'\tdomain = {domain_name};\n'

        # write objects block
        if self.object_types:
            result += '\n\tobjects {\n'
            for otype in self.object_types:
                objects_str = ', '.join(self.object_values[otype])
                result += f'\t\t{otype} : {{ {objects_str} }};\n'
            result += '\t};\n'
        
        # write non-fluents block
        if self.nonfluent_inits:
            result += '\n\tnon-fluents {\n'
            for (name, value) in self.nonfluent_inits.items():
                result += f'\t\t{name} = {value};\n'
            result += '\t};\n'
        
        result += '}\n\n'
        result += f'instance {instance_name} {{\n\n'
        result += f'\tdomain = {domain_name};\n'
        result += f'\tnon-fluents = {nonfluents_name};\n'

        # write init-state block
        if self.init_states:
            result += '\n\tinit-state {\n'
            for (name, value) in self.init_states.items():
                result += f'\t\t{name} = {value};\n'
            result += '\t};\n\n'
        
        # write constants
        result += f'\tmax-nondef-actions = {self.maxnondef};\n'
        result += f'\thorizon = {self.horizon};\n'
        result += f'\tdiscount = {self.discount};\n'

        result += '}'
        return result
