#
# class TreeNode:
#
#     def __init__(self, etype, value, *args, params=None, commutes: bool=False):
#         self.etype = etype
#         self.value = value
#         self.args = tuple(args)
#         if params is None:
#             params = []
#         if not isinstance(params, dict):
#             params = tuple(params)
#         self.params = params
#         self.commutes = commutes
#
#     def symbolic(self):
#         value = str(self.value)
#         if self.params:
#             if isinstance(self.params, dict):
#                 args = ('{}:{}'.format(*p) for p in self.params.items())
#                 args = '_{{{}}}'.format(','.join(args))
#             else:
#                 args = '({})'.format(','.join(map(str, self.params)))
#             value += args
#         return value
#
#     def _str(self, level=0):
#         value = ' ' * level + self.symbolic()
#         for arg in self.args:
#             value += '\n' + arg._str(level + 1)
#         return value
#
#     Y = TypeVar('T')
#
#     def bfs(self,
#             tr: Callable[['TreeNode'], Y],
#             acc: Callable[[Y, Y], Y]):
#         y = tr(self)
#         for node in self.args:
#             yn = node.bfs(tr, acc)
#             y = acc(y, yn)
#         return y
#
#     def dfs(self,
#             tr: Callable[['TreeNode'], Y],
#             acc: Callable[[Y, Y], Y]):
#         y = None
#         for i, node in enumerate(self.args):
#             yn = node.dfs(tr, acc)
#             y = acc(y, yn) if i else yn
#         yn = tr(self)
#         y = acc(y, yn) if self.args else yn
#         return y
#
#     def __str__(self):
#         return self._str()
#
#     def __hash__(self):
#         args = self.args
#         if self.commutes:
#             args = tuple(sorted(args, key=hash))
#         args = (self.etype, self.value, self.params) + args
#         return hash(args)
#
#     def __eq__(self, other):
#         if not isinstance(other, TreeNode): return False
#         if self.etype != other.etype: return False
#         if self.value != other.value: return False
#         if self.commutes != other.commutes: return False
#         if len(self.args) != len(other.args): return False
#         if self.params != other.params: return False
#
#         if self.commutes:
#             arg1 = Counter(self.args)
#             arg2 = Counter(other.args)
#         else:
#             arg1 = self.args
#             arg2 = other.args
#         return arg1 == arg2
#
#
# class TreeBuilder:
#
#     COMMUTATIVE = {'+', '*',
#                    '==', '~=',
#                    '|', '^', '~', '<=>',
#                    'minimum', 'maximum', 'sum', 'avg', 'prod',
#                    'forall', 'exists',
#                    'min', 'max'}
#
#     def build_cpf(self, cpf: CPF) -> TreeNode:
#         _, (name, params) = cpf.pvar
#         arg = self.build_expr(cpf.expr)
#         return TreeNode('cpf', name, arg, params=params)
#
#     def build_expr(self, expr: Expression) -> TreeNode:
#         etype, _ = expr.etype        
#         if etype == 'constant':
#             return self._build_const(expr)                
#         elif etype == 'pvar':
#             return self._build_pvar(expr)            
#         elif etype == 'aggregation':
#             return self._build_aggregation(expr)
#         elif etype == 'control':
#             return self._build_control(expr)
#         elif etype == 'randomvar':
#             return self._build_random(expr)
#         elif etype == 'func':
#             return self._build_function(expr)
#         elif etype in {'relational', 'arithmetic', 'boolean'}:
#             return self._build_arithmetic(expr)
#         else:
#             raise Exception(f'Internal error: type {etype} is not supported.')
#
#     def _build_const(self, expr):
#         return TreeNode('const', expr.args)
#
#     def _build_pvar(self, expr):
#         _, name = expr.etype  
#         _, params = expr.args
#         return TreeNode('pvar', name, params=params)
#
#     def _build_aggregation(self, expr):
#         _, op = expr.etype        
#         params, exprs = [], []
#         for arg in expr.args:
#             if isinstance(arg, tuple):
#                 params.append(arg[1])
#             else:
#                 exprs.append(arg)  
#         params = dict(params)
#         args = map(self.build_expr, exprs)
#         commutes = op in TreeBuilder.COMMUTATIVE
#         return TreeNode('agg', op, *args, params=params, commutes=commutes)
#
#     def _build_control(self, expr):
#         _, op = expr.etype        
#         args = map(self.build_expr, expr.args)
#         commutes = op in TreeBuilder.COMMUTATIVE
#         return TreeNode('if', op, *args, commutes=commutes)
#
#     def _build_random(self, expr):
#         _, op = expr.etype  
#         if op == 'Discrete':
#             args = tuple(arg[1] for arg in expr.args)
#             print(args)
#         else:
#             args = map(self.build_expr, expr.args)
#         commutes = op in TreeBuilder.COMMUTATIVE
#         return TreeNode('random', op, *args, commutes=commutes)
#
#     def _build_function(self, expr):
#         _, op = expr.etype
#         args = map(self.build_expr, expr.args)
#         commutes = op in TreeBuilder.COMMUTATIVE
#         return TreeNode('func', op, *args, commutes=commutes)
#
#     def _build_arithmetic(self, expr):
#         _, op = expr.etype        
#         args = map(self.build_expr, expr.args)
#         commutes = op in TreeBuilder.COMMUTATIVE
#         if len(expr.args) == 1:
#             return TreeNode('unary', op, *args, commutes=commutes)
#         elif len(expr.args) == 2:
#             return TreeNode('binary', op, *args, commutes=commutes)
#         else:
#             return TreeNode('nary', op, *args, commutes=commutes)
#
# if __name__ == '__main__':
#
#     # build a simple expression 'x + y - x * y'
#     x = TreeNode('pvar', 'x')
#     y = TreeNode('pvar', 'y')
#     s = TreeNode('binary', '+', x, y)
#     p = TreeNode('binary', '*', x, y)
#     expr = TreeNode('binary', '-', s, p)
#
#     # print the tree
#     print(expr)
#
#     # count maximum depth of the tree
#     print('\ndepth')
#     print(expr.bfs(tr=lambda _: 1, acc=lambda c, y: max(c, 1 + y)))
#
#     # count size
#     print('\nsize')
#     print(expr.dfs(tr=lambda _: 1, acc=lambda c, y: c + y))
#
#     # count leaf nodes
#     print('\nleaves')
#     print(expr.dfs(tr=lambda n: int(n.etype in {'pvar', 'const'}),
#                    acc=lambda c, y: c + y))
#
#     # list the variables
#     print('\nvariables')
#     print(expr.dfs(tr=lambda n: [n.value] if n.etype == 'pvar' else [],
#                    acc=lambda c, y: c + y))
#
#     # list the unique variables
#     print(expr.dfs(tr=lambda n: {n.value} if n.etype == 'pvar' else set(),
#                    acc=lambda c, y: c | y))
#
#     # count occurrences of each variable
#     print('\noccurrences')
#     print(expr.dfs(tr=lambda n: Counter({n.value: 1} if n.etype == 'pvar' else {}),
#                    acc=lambda c, y: c + y))
#
#     # list the unique sub-expressions
#     print('\nunique sub-expressions')
#     unique = expr.dfs(tr=lambda n: {n}, acc=lambda c, y: c | y)
#     print('\n-----\n'.join(map(str, unique)))
#

