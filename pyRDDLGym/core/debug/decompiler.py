import numpy as np
from typing import Dict, List, Union, TYPE_CHECKING
if TYPE_CHECKING:
    from pyRDDLGym.core.compiler.model import RDDLPlanningModel

from pyRDDLGym.core.parser.expr import Expression
from pyRDDLGym.core.parser.cpf import CPF


class RDDLDecompiler:
    '''Converts AST representation (e.g., Expression) to a string that represents
    the corresponding expression in RDDL.'''
    
    # ===========================================================================
    # main subroutines
    # ===========================================================================

    def decompile_expr(self, expr: Expression, level: int=0) -> str:
        '''Converts an AST expression to a string representing valid RDDL code.
        
        :param expr: the expression to convert
        :param level: indentation level
        '''
        return self._decompile(expr, False, level)
    
    def decompile_cpf(self, cpf: CPF, level: int=0) -> str:
        '''Converts a CPF object to its equivalent string as it would appear in
        the cpfs {...} block in a RDDL domain description. 
        
        :param cpf: the CPF object to convert
        :param level: indentation level
        '''
        lhs = self._symbolic(*cpf.pvar[1], aggregation=False)
        rhs = self.decompile_expr(cpf.expr, level)
        return f'{lhs} = {rhs};'
    
    def decompile_exprs(self, rddl: 'RDDLPlanningModel', level: int=0) -> Dict[str, Union[str, Dict[str, str], List[str]]]:
        '''Converts a RDDL model to a dictionary of decompiled expression strings,
        as they would appear in the domain description file.'''
        decompiled = {}
        decompiled['cpfs'] = {name: self.decompile_expr(expr, level)
                              for (name, (_, expr)) in rddl.cpfs.items()}
        decompiled['reward'] = self.decompile_expr(rddl.reward, level)
        decompiled['invariants'] = [self.decompile_expr(expr, level) 
                                    for expr in rddl.invariants]
        decompiled['preconditions'] = [self.decompile_expr(expr, level) 
                                       for expr in rddl.preconditions]
        decompiled['terminations'] = [self.decompile_expr(expr, level) 
                                      for expr in rddl.terminations]
        return decompiled
    
    def decompile_domain(self, rddl: 'RDDLPlanningModel') -> str:
        '''Converts a RDDL model to a RDDL domain description file.'''
        decompiled = self.decompile_exprs(rddl, level=2)
        
        # decompile types
        decompiled_types = ''
        if rddl.type_to_objects:
            decompiled_types = '\n\ttypes {'
            for (name, values) in rddl.type_to_objects.items():
                if name in rddl.enum_types:
                    decompiled_types += f'\n\t\t{name}: {{ ' + \
                        ', '.join([f'@{v}' for v in values]) + ' };'
                else:
                    decompiled_types += f'\n\t\t{name}: object;'
            decompiled_types += '\n\t};'
        
        # decompile pvars {} block
        decompiled_pvars = '\n\tpvariables {'
        for pvars in (rddl.non_fluents, rddl.derived_fluents, rddl.interm_fluents, 
                      rddl.state_fluents, rddl.observ_fluents, rddl.action_fluents):
            if pvars:
                decompiled_pvars += '\n'
            for name in pvars:
                prange = rddl.variable_ranges[name]
                ptype = rddl.variable_types[name]
                decompiled_params = ''
                if rddl.variable_params[name]:
                    decompiled_params = '(' + ', '.join(rddl.variable_params[name]) + ')' 
                dv = self._value_to_string(rddl.variable_defaults[name])
                if prange not in ['real', 'int', 'bool']:
                    dv = f'@{dv}'
                if ptype in ['interm-fluent', 'derived-fluent', 'observ-fluent']:
                    decompiled_pvars += f'\n\t\t{name}{decompiled_params} : ' + \
                                        f'{{ {ptype}, {prange} }};'
                else:
                    decompiled_pvars += f'\n\t\t{name}{decompiled_params} : ' + \
                                        f'{{ {ptype}, {prange}, default = {dv} }};'
        decompiled_pvars += '\n\t};'
        
        # decompile cpfs {} block
        decompiled_cpfs = '\n\tcpfs {'
        for (name, expr) in decompiled['cpfs'].items():
            params_and_types, _ = rddl.cpfs[name]
            if params_and_types:
                params = ', '.join(p for p, _ in params_and_types)
                name = f'{name}({params})'
            decompiled_cpfs += f'\n\n\t\t{name} = {expr};'
        decompiled_cpfs += '\n\t};'
        
        # decompile reward function
        decompiled_reward = f'\n\treward = {decompiled["reward"]};'
        
        # decompile state-invariants {} block
        decompiled_invariants = ''
        if decompiled['invariants']:
            decompiled_invariants = '\n\n\tstate-invariants {'
            for expr in decompiled['invariants']:
                decompiled_invariants += f'\n\t\t{expr};'
            decompiled_invariants += '\n\t};'
        
        # decompile action-preconditions {} block
        decompiled_preconds = ''
        if decompiled['preconditions']:
            decompiled_preconds = '\n\n\taction-preconditions {'
            for expr in decompiled['preconditions']:
                decompiled_preconds += f'\n\t\t{expr};'
            decompiled_preconds += '\n\t};'
        
        # decompile terminations {} block
        decompiled_terminals = ''
        if decompiled['terminations']:
            decompiled_terminals = '\n\n\ttermination {'
            for expr in decompiled['terminations']:
                decompiled_terminals += f'\n\t\t{expr};'
            decompiled_terminals += '\n\t};'
        
        # decompile domain {} block
        decompiled_domain = (
            f'domain {rddl.domain_name} {{'
            f'\n{decompiled_types}'
            f'\n{decompiled_pvars}'
            f'\n{decompiled_cpfs}'
            f'\n{decompiled_reward}'
            f'{decompiled_invariants}'
            f'{decompiled_preconds}'
            f'{decompiled_terminals}'
            f'\n}}')
        return decompiled_domain
    
    def decompile_instance(self, rddl: 'RDDLPlanningModel') -> str:
        '''Converts a RDDL model to a RDDL instance description file.'''

        # domain name statement
        decompiled_domain_name = f'\tdomain = {rddl.domain_name};'

        # objects {} block
        all_objects = {name: objects 
                       for (name, objects) in rddl.type_to_objects.items()
                       if name not in rddl.enum_types}
        decompiled_objects = ''
        if all_objects:
            decompiled_objects += '\n\tobjects {'
            for (name, objects) in all_objects.items():
                objects_str = ', '.join(objects)
                decompiled_objects += f'\n\t\t{name} : {{ {objects_str} }};'
            decompiled_objects += '\n\t};'
        
        # nonfluents {} block
        decompiled_nonfluents = ''
        nonfluents_statements = []
        for (name, values) in rddl.non_fluents.items():
            default_value = rddl.variable_defaults[name]
            if isinstance(values, (list, tuple, set)):
                for (gname, gvalue) in rddl.ground_var_with_values(name, values):
                    if gvalue != default_value:
                        gvar, objects = rddl.parse_grounded(gname)
                        objects_str = ('(' + ','.join(objects) + ')') if objects else ''
                        gvalue = self._value_to_string(gvalue)
                        assign_expr = f'\t\t{gvar}{objects_str} = {gvalue};'
                        nonfluents_statements.append(assign_expr)
            else:
                if values != default_value:
                    values = self._value_to_string(values)
                    assign_expr = f'\t\t{name} = {values};'
                    nonfluents_statements.append(assign_expr)
        if nonfluents_statements:
            decompiled_nonfluents = '\n\tnon-fluents {'
            decompiled_nonfluents += '\n' + '\n'.join(nonfluents_statements)
            decompiled_nonfluents += '\n\t};'
        
        # decompile non-fluents {} overall block
        decompiled_nonfluents_block = (
            f'non-fluents {rddl.instance_name}_nf {{'
            f'\n{decompiled_domain_name}'
            f'\n{decompiled_objects}'
            f'\n{decompiled_nonfluents}'
            f'\n}}'
        )

        # decompile non-fluents name
        decompiled_nonfluents_name = f'\tnon-fluents = {rddl.instance_name}_nf;'

        # decompile init-state {} block
        decompiled_initstate = ''
        initstate_statements = []
        for (name, values) in rddl.state_fluents.items():
            default_value = rddl.variable_defaults[name]
            if isinstance(values, (list, tuple, set)):
                for (gname, gvalue) in rddl.ground_var_with_values(name, values):
                    if gvalue != default_value:
                        gvar, objects = rddl.parse_grounded(gname)
                        objects_str = ('(' + ','.join(objects) + ')') if objects else ''
                        gvalue = self._value_to_string(gvalue)
                        assign_expr = f'\t\t{gvar}{objects_str} = {gvalue};'
                        initstate_statements.append(assign_expr)
            else:
                if values != default_value:
                    values = self._value_to_string(values)
                    assign_expr = f'\t\t{name} = {values};'
                    initstate_statements.append(assign_expr)
        if initstate_statements:
            decompiled_initstate = '\n\tinit-state {'
            decompiled_initstate += '\n' + '\n'.join(initstate_statements)
            decompiled_initstate += '\n\t};'

        # decompile constants
        max_allowed_actions = getattr(rddl.ast.instance, 'max_nondef_actions', 'pos-inf')
        decompiled_maxnondefactions = f'\tmax-nondef-actions = {max_allowed_actions};'
        decompiled_horizon = f'\thorizon = {rddl.horizon};'
        decompiled_discount = f'\tdiscount = {rddl.discount};'

        # decompile instance overall block
        decompiled_instance_block = (
            f'instance {rddl.instance_name} {{'
            f'\n{decompiled_domain_name}'
            f'\n{decompiled_nonfluents_name}'
            f'\n{decompiled_initstate}'
            f'\n{decompiled_maxnondefactions}'
            f'\n{decompiled_horizon}'
            f'\n{decompiled_discount}'
            f'\n}}'
        )

        decompiled_instance = decompiled_nonfluents_block + '\n\n' + decompiled_instance_block
        return decompiled_instance

    # ===========================================================================
    # helper subroutines
    # ===========================================================================

    def _decompile(self, expr, enclose, level):
        etype, _ = expr.etype
        if etype == 'constant':
            return self._decompile_constant(expr, enclose, level)
        elif etype == 'pvar':
            return self._decompile_pvar(expr, enclose, level)
        elif etype in {'arithmetic', 'relational', 'boolean'}:
            return self._decompile_math(expr, enclose, level)
        elif etype == 'aggregation':
            return self._decompile_aggregation(expr, enclose, level)   
        elif etype == 'func':
            return self._decompile_func(expr, enclose, level) 
        elif etype == 'control':
            return self._decompile_control(expr, enclose, level)        
        elif etype == 'randomvar':
            return self._decompile_random(expr, enclose, level)
        elif etype == 'randomvector':
            return self._decompile_random_vector(expr, enclose, level)
        elif etype == 'matrix':
            return self._decompile_matrix(expr, enclose, level)
        else:
            return ''
    
    def _symbolic(self, value, params, aggregation):
        value = str(value)
        if params is not None and params:
            if aggregation:
                args = ', '.join(f'{k}: {v}' for (k, v) in params)
                value += f'_{{{args}}}'
            else:
                args = ', '.join(map(str, params))
                value += f'({args})'
        return value

    @staticmethod
    def _value_to_string(value):
        if value is None:
            return 'None'
        elif isinstance(value, float):
            value = np.format_float_positional(value)
            if value.endswith('.'):
                value += '0'
            return value
        else:
            value = str(value)
            if value == 'True':
                value = 'true'
            elif value == 'False':
                value = 'false'
            return value

    def _decompile_constant(self, expr, enclose, level):
        return self._value_to_string(expr.args)
        
    def _decompile_pvar(self, expr, enclose, level):
        _, name = expr.etype
        _, params = expr.args 
        if params is not None:
            params = ((self._decompile(arg, False, 0) 
                       if isinstance(arg, Expression) 
                       else arg)
                      for arg in params)
        return self._symbolic(name, params, aggregation=False)
        
    def _decompile_math(self, expr, enclose, level):
        _, op = expr.etype
        args = expr.args
        if len(args) == 1:
            arg, = args
            value = str(op) + self._decompile(arg, True, level)
        else:
            sep = ' ' + str(op) + ' '
            value = sep.join(self._decompile(arg, True, level) for arg in args)
            
        if enclose:
            value = f'( {value} )'
        return value
        
    def _decompile_aggregation(self, expr, enclose, level):
        _, op = expr.etype
        * pvars, arg = expr.args
        params = [pvar for (_, pvar) in pvars]
        agg = self._symbolic(op, params, aggregation=True)
        decompiled = self._decompile(arg, False, level)        
        return f'( {agg} [ {decompiled} ] )'
        
    def _decompile_func(self, expr, enclose, level):
        _, op = expr.etype
        decompiled = ', '.join(self._decompile(arg, False, level) 
                               for arg in expr.args)
        return f'{op}[{decompiled}]'
            
    def _decompile_control(self, expr, enclose, level):
        _, op = expr.etype
        indent = '\t' * (level + 1)
        
        if op == 'if':
            pred, if_true, if_false = expr.args
            pred = self._decompile(pred, False, level)
            if_true = self._decompile(if_true, True, level + 1)
            if_false = self._decompile(if_false, True, level + 1)
            value = f'if ({pred})\n{indent}then {if_true}\n{indent}else {if_false}'

        else:  # switch
            pvar, *args = expr.args
            pred = self._decompile(pvar, False, level)
            cases = [''] * len(args)
            for (i, _case) in enumerate(args):
                case_type, value = _case
                if case_type == 'case':
                    literal, arg = value
                    decompiled = self._decompile(arg, False, level + 1)
                    cases[i] = f'case {literal} : {decompiled}'
                else:  # default
                    decompiled = self._decompile(value, False, level + 1)
                    cases[i] = f'default : {decompiled}'
            cases = f',\n{indent}'.join(cases)
            indentm1 = '\t' * level
            value = f'switch({pred}) {{ \n{indent}{cases}\n{indentm1} }}'

        if enclose: 
            value = f'( {value} )'
        return value
    
    def _decompile_random(self, expr, enclose, level):
        _, op = expr.etype
        
        if op == 'Discrete' or op == 'UnnormDiscrete':
            (_, var), *args = expr.args
            cases = [var] + [''] * len(args)
            for (i, _case) in enumerate(args):
                _, (literal, arg) = _case
                decompiled = self._decompile(arg, False, level + 1)
                cases[i + 1] = f'{literal} : {decompiled}'
            indent = '\t' * (level + 1)
            value = f',\n{indent}'.join(cases)
        
        elif op == 'Discrete(p)' or op == 'UnnormDiscrete(p)':
            op = op[:-3]
            * pvars, args = expr.args
            params = [pvar for (_, pvar) in pvars]
            op = self._symbolic(op, params, aggregation=True)    
            value = ', '.join(self._decompile(arg, False, level) 
                              for arg in args)
            
        else:  # Normal, exponential, etc...
            value = ', '.join(self._decompile(arg, False, level) 
                              for arg in expr.args)
            
        return f'{op}({value})'
    
    def _decompile_random_vector(self, expr, enclose, level):
        _, op = expr.etype
        pvars, args = expr.args
        args = ', '.join(self._decompile(arg, False, level) for arg in args)
        pvars = ', '.join(pvars)
        return f'{op}[{pvars}]({args})'
            
    def _decompile_matrix(self, expr, enclose, level):
        _, op = expr.etype
        if op == 'det':
            *pvars, arg = expr.args
            params = [pvar for (_, pvar) in pvars]
            agg = self._symbolic(op, params, aggregation=True)
            decompiled = self._decompile(arg, False, level)        
            return f'{agg}[ {decompiled} ]'
        elif op == 'inverse' or op == 'pinverse' or op == 'cholesky':
            pvars, arg = expr.args
            prow, pcol = pvars
            params = f'row={prow}, col={pcol}'
            decompiled = self._decompile(arg, False, level)
            return f'{op}[{params}][ {decompiled} ]'
