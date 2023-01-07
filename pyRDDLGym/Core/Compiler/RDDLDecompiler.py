from pyRDDLGym.Core.Parser.expr import Expression
from pyRDDLGym.Core.Parser.cpf import CPF


class RDDLDecompiler:
    '''Converts AST representation (e.g., Expression) to a string that represents
    the corresponding expression in RDDL.
    '''
    
    def _symbolic(self, value, params):
        value = str(value)
        if params is not None and params:
            if isinstance(params, dict):
                args = ', '.join(f'{k}:{v}' for k, v in params.items())
                args = f'_{{{args}}}'
            else:
                args = ', '.join(map(str, params))
                args = f'({args})'
            value += args
        return value

    def decompile_expr(self, expr: Expression) -> str:
        '''Converts an AST expression to a string representing valid RDDL code.
        
        :param expr: the expression to convert
        '''
        return self._decompile(expr, False, 0)
    
    def decompile_cpf(self, cpf: CPF) -> str:
        '''Converts a CPF object to its equivalent string as it would appear in
        the cpfs {...} block in a RDDL domain description. 
        
        :param cpf: the CPF object to convert
        '''
        lhs = self._symbolic(*cpf.pvar[1])
        rhs = self.decompile_expr(cpf.expr)
        return f'{lhs} = {rhs};'
        
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
        else:
            raise Exception(f'Internal error: type {etype} is undefined.')
    
    def _decompile_constant(self, expr, enclose, level):
        return str(expr.args)
        
    def _decompile_pvar(self, expr, enclose, level):
        _, name = expr.etype
        _, params = expr.args        
        return self._symbolic(name, params)
        
    def _decompile_math(self, expr, enclose, level):
        _, op = expr.etype
        decompiled = [self._decompile(arg, True, level) for arg in expr.args]
        if len(decompiled) == 1:
            value = str(op) + decompiled[0]
        else:
            value = (' ' + str(op) + ' ').join(decompiled)
        if enclose:
            value = f'( {value} )'
        return value
        
    def _decompile_aggregation(self, expr, enclose, level):
        * pvars, arg = expr.args
        _, op = expr.etype
        params = dict(p[1] for p in pvars)
        agg = self._symbolic(op, params)
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
            pred = self._decompile(expr.args[0], False, level)
            if_true = self._decompile(expr.args[1], True, level + 1)
            if_false = self._decompile(expr.args[2], True, level + 1)
            value = f'if ({pred})\n{indent}then {if_true}\n{indent}else {if_false}'

        else:  # switch
            pvar, *args = expr.args
            pred = self._decompile(pvar, False, level)
            cases = []
            for (case_type, value) in args:
                if case_type == 'case':
                    literal, arg = value
                    decompiled = self._decompile(arg, False, level + 1)
                    cases.append(f'case {literal} : {decompiled}')
                else:  # default
                    decompiled = self._decompile(value, False, level + 1)
                    cases.append(f'default : {decompiled}')
            cases = f',\n{indent}'.join(cases)
            value = f'switch({pred}) {{ \n{indent}{cases}\n }}'

        if enclose: 
            value = f'( {value} )'
        return value
    
    def _decompile_random(self, expr, enclose, level):
        _, op = expr.etype
        
        if op == 'Discrete' or op == 'UnnormDiscrete':
            (_, var), *args = expr.args
            cases = [var]
            for (_, (literal, arg)) in args:
                decompiled = self._decompile(arg, False, level + 1)
                cases.append(f'{literal} : {decompiled}')
            indent = '\t' * (level + 1)
            value = f',\n{indent}'.join(cases)
        
        else:
            value = ', '.join(self._decompile(arg, False, level) 
                              for arg in expr.args)
            
        return f'{op}({value})'
            