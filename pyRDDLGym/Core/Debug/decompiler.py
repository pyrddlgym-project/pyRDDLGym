class TreeNode:
    
    def __init__(self, etype, value, *args):
        self.etype = etype
        self.value = value
        self.args = tuple(args)
    
    def _str(self, level=0):
        value = ' ' * level + str(self.value)
        for arg in self.args:
            value += '\n' + arg._str(level + 1)
        return value
    
    def __str__(self):
        return self._str()
    

class TreeBuilder:
    
    def build_cpf(self, cpf):
        _, (name, params) = cpf.pvar
        value = name
        if params:
            value += '(' + ', '.join(params) + ')'
        arg = self.build_expr(cpf.expr)
        return TreeNode('cpf', value, arg)
        
    def build_expr(self, expr):
        etype, _ = expr.etype        
        if etype == 'constant':
            return self._build_const(expr)                
        elif etype == 'pvar':
            return self._build_pvar(expr)            
        elif etype == 'aggregation':
            return self._build_aggregation(expr)
        elif etype == 'control':
            return self._build_control(expr)
        elif etype == 'randomvar':
            return self._build_random(expr)
        elif etype == 'func':
            return self._build_function(expr)
        else:
            return self._build_arithmetic(expr)
    
    def _build_const(self, expr):
        return TreeNode('const', expr.args)
    
    def _build_pvar(self, expr):
        _, value = expr.etype  
        _, params = expr.args
        if params:
            value += '(' + ', '.join(params) + ')'
        return TreeNode('pvar', value)
    
    def _build_aggregation(self, expr):
        _, value = expr.etype        
        params, exprs = [], []
        for arg in expr.args:
            if isinstance(arg, tuple):
                param = '{}:{}'.format(*arg[1])
                params.append(param)
            else:
                exprs.append(arg)        
        if params:
            value += '_' + '{' + ', '.join(params) + '}'
        args = map(self.build_expr, exprs)
        return TreeNode('agg', value, *args)
    
    def _build_control(self, expr):
        _, value = expr.etype        
        args = map(self.build_expr, expr.args)
        return TreeNode('if', value, *args)
    
    def _build_random(self, expr):
        _, value = expr.etype  
        args = map(self.build_expr, expr.args)
        return TreeNode('random', value, *args)
    
    def _build_function(self, expr):
        _, value = expr.etype  
        args = map(self.build_expr, expr.args)
        return TreeNode('func', value, *args)
    
    def _build_arithmetic(self, expr):
        _, value = expr.etype        
        args = map(self.build_expr, expr.args)
        if len(expr.args) == 1:
            return TreeNode('unary', value, *args)
        elif len(expr.args) == 2:
            return TreeNode('binary', value, *args)
        else:
            return TreeNode('nary', value, *args)


class RDDLDecompiler:
    
    def decompile_cpf(self, cpf):
        tree = TreeBuilder().build_cpf(cpf)
        return self._decompile(tree, False, 0)
        
    def decompile_expr(self, expr):
        tree = TreeBuilder().build_expr(expr)
        return self._decompile(tree, False, 0)
    
    def _decompile(self, tree, enclose, level):
        if tree.etype == 'cpf':
            format_str = '{} = {};'
            decompiled = self._decompile(tree.args[0], False, level)
            value = format_str.format(tree.value, decompiled)
            enclose = False
        
        elif tree.etype in {'const', 'pvar'}:
            value = str(tree.value)
            enclose = False

        elif tree.etype == 'agg':
            format_str = '{} [ {} ]'
            decompiled = self._decompile(tree.args[0], False, level)
            value = format_str.format(tree.value, decompiled)
            enclose = True
        
        elif tree.etype == 'if':
            indent = '\t' * (level + 1)
            format_str = 'if ({})' + '\n' + indent + 'then {}' + '\n' + indent + 'else {}'
            pred = self._decompile(tree.args[0], False, level)
            if_true = self._decompile(tree.args[1], True, level + 1)
            if_false = self._decompile(tree.args[2], True, level + 1)
            value = format_str.format(pred, if_true, if_false)
        
        elif tree.etype == 'random':
            format_str = '{}({})'
            decompiled = (self._decompile(arg, False, level) for arg in tree.args)
            value = format_str.format(tree.value, ', '.join(decompiled))
            enclose = False
        
        elif tree.etype == 'func':
            format_str = '{}({})'
            decompiled = (self._decompile(arg, False, level) for arg in tree.args)
            value = format_str.format(tree.value, ', '.join(decompiled))
            enclose = False
        
        elif tree.etype == 'unary':
            format_str = '{}'
            decompiled = self._decompile(tree.args[0], True, level)
            value = format_str.format(tree.value + decompiled)
        
        elif tree.etype == 'binary':
            lhs = self._decompile(tree.args[0], True, level)
            rhs = self._decompile(tree.args[1], True, level)
            format_str = '{} {} {}'
            value = format_str.format(lhs, tree.value, rhs)
        
        elif tree.etype == 'nary':
            decompiled = (self._decompile(arg, True, level) for arg in tree.args)
            op = ' ' + tree.value + ' '
            value = op.join(decompiled)
        
        if enclose:
            value = '( {} )'.format(value)
        
        return value