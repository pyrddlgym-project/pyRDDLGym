from typing import Dict, List, Union

from pyRDDLGym.Core.Parser.expr import Expression
from pyRDDLGym.Core.Parser.cpf import CPF


class RDDLDecompiler:
    '''Converts AST representation (e.g., Expression) to a string that represents
    the corresponding expression in RDDL.'''
    
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
    
    def decompile_exprs(self, rddl, level: int=0) -> Dict[str, Union[str, Dict[str, str], List[str]]]:
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
                                      for expr in rddl.terminals]
        return decompiled
    
    def decompile_domain(self, rddl) -> str:
        '''Converts a RDDL model to a RDDL domain description file.'''
        decompiled = self.decompile_exprs(rddl, level=2)
        
        decompiled_types = ''
        if rddl.objects:
            decompiled_types = '\n\ttypes {'
            for (name, values) in rddl.objects.items():
                if name in rddl.enums:
                    decompiled_types += f'\n\t\t{name}: {{ ' + \
                        ', '.join([f'@{v}' for v in values]) + ' };'
                else:
                    decompiled_types += f'\n\t\t{name}: object;'
            decompiled_types += '\n\t};'
        
        decompiled_pvars = '\n\tpvariables {'
        for pvars in [rddl.nonfluents, rddl.derived, rddl.interm, 
                      rddl.states, rddl.observ, rddl.actions]:
            if pvars:
                decompiled_pvars += '\n'
            for name in pvars:
                prange = rddl.variable_ranges[name]
                ptype = rddl.variable_types[name]
                decompiled_params = ''
                if rddl.param_types[name]:
                    decompiled_params = '(' + ', '.join(rddl.param_types[name]) + ')' 
                dv = str(rddl.default_values[name])
                if dv == 'True' or dv == 'False':
                    dv = dv.lower()
                if prange not in ['real', 'int', 'bool']:
                    dv = f'@{dv}'
                if ptype in ['interm-fluent', 'derived-fluent', 'observ-fluent']:
                    decompiled_pvars += f'\n\t\t{name}{decompiled_params} : ' + \
                                        f'{{ {ptype}, {prange} }};'
                else:
                    decompiled_pvars += f'\n\t\t{name}{decompiled_params} : ' + \
                                        f'{{ {ptype}, {prange}, default = {dv} }};'
        decompiled_pvars += '\n\t};'
        
        decompiled_cpfs = '\n\tcpfs {'
        for (name, expr) in decompiled['cpfs'].items():
            decompiled_cpfs += f'\n\n\t\t{name} = {expr};'
        decompiled_cpfs += '\n\t};'
        
        decompiled_reward = f'\n\treward = {decompiled["reward"]};'
        
        decompiled_invariants = ''
        if decompiled['invariants']:
            decompiled_invariants = '\n\n\tstate-invariants {'
            for expr in decompiled['invariants']:
                decompiled_invariants += f'\n\t\t{expr};'
            decompiled_invariants += '\n\t};'
        
        decompiled_preconds = ''
        if decompiled['preconditions']:
            decompiled_preconds = '\n\n\taction-preconditions {'
            for expr in decompiled['preconditions']:
                decompiled_preconds += f'\n\t\t{expr};'
            decompiled_preconds += '\n\t};'
        
        decompiled_terminals = ''
        if decompiled['terminations']:
            decompiled_terminals = '\n\n\ttermination {'
            for expr in decompiled['terminations']:
                decompiled_terminals += f'\n\t\t{expr};'
            decompiled_terminals += '\n\t};'
        
        decompiled_domain = (
            f'domain {rddl.domainName()} {{'
            f'\n{decompiled_types}'
            f'\n{decompiled_pvars}'
            f'\n{decompiled_cpfs}'
            f'\n{decompiled_reward}'
            f'{decompiled_invariants}'
            f'{decompiled_preconds}'
            f'{decompiled_terminals}'
            f'\n}}')
        return decompiled_domain
        
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

    def _decompile_constant(self, expr, enclose, level):
        value = expr.args
        value = str(value)
        if value == 'True':
            value = 'true'
        elif value == 'False':
            value = 'false'
        return value
        
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
            value = f'switch({pred}) {{ \n{indent}{cases}\n }}'

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
