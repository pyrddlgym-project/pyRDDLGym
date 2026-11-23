'''Parser for the subset of PDDL and PPDDL that can be converted to RDDL.

This module uses only the Python standard library. It reads PDDL text into an
intermediate representation (IR) of frozen dataclasses mirroring the PDDL syntax,
and checks that all symbols are declared, have the correct arity and are well-typed.
Constructs that are valid PDDL but cannot be converted to RDDL (durative actions,
processes, events, PDDL3 constraints and preferences, timed initial literals and
object-valued functions) are recognized and raise PDDLUnsupportedError.
'''

import re
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

from pyRDDLGym.pddl.errors import PDDLParseError, PDDLUnsupportedError


# ===========================================================================
# intermediate representation
# ===========================================================================

@dataclass(frozen=True)
class Loc:
    line: int
    col: int


def _loc_field():
    return field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class Param:
    '''A typed variable; types has more than one entry for (either ...).'''
    name: str
    types: Tuple[str, ...]
    loc: Optional[Loc] = _loc_field()


# ----- terms -----

@dataclass(frozen=True)
class Var:
    name: str
    loc: Optional[Loc] = _loc_field()


@dataclass(frozen=True)
class Obj:
    name: str
    loc: Optional[Loc] = _loc_field()


@dataclass(frozen=True)
class Num:
    value: float
    loc: Optional[Loc] = _loc_field()


@dataclass(frozen=True)
class FTerm:
    '''A numeric function application such as (f ?x).'''
    name: str
    args: Tuple[Union[Var, Obj], ...]
    loc: Optional[Loc] = _loc_field()


@dataclass(frozen=True)
class Arith:
    '''Arithmetic: op is one of + - * /; (- x) is unary negation.'''
    op: str
    args: tuple
    loc: Optional[Loc] = _loc_field()


# ----- formulas -----

@dataclass(frozen=True)
class Atom:
    pred: str
    args: Tuple[Union[Var, Obj], ...]
    loc: Optional[Loc] = _loc_field()


@dataclass(frozen=True)
class Equals:
    '''Object equality (= t1 t2).'''
    left: Union[Var, Obj]
    right: Union[Var, Obj]
    loc: Optional[Loc] = _loc_field()


@dataclass(frozen=True)
class Not:
    arg: object
    loc: Optional[Loc] = _loc_field()


@dataclass(frozen=True)
class And:
    args: tuple
    loc: Optional[Loc] = _loc_field()


@dataclass(frozen=True)
class Or:
    args: tuple
    loc: Optional[Loc] = _loc_field()


@dataclass(frozen=True)
class Imply:
    left: object
    right: object
    loc: Optional[Loc] = _loc_field()


@dataclass(frozen=True)
class Exists:
    params: Tuple[Param, ...]
    body: object
    loc: Optional[Loc] = _loc_field()


@dataclass(frozen=True)
class Forall:
    params: Tuple[Param, ...]
    body: object
    loc: Optional[Loc] = _loc_field()


@dataclass(frozen=True)
class Compare:
    '''Numeric comparison: op is one of < <= > >= =.'''
    op: str
    left: object
    right: object
    loc: Optional[Loc] = _loc_field()


# ----- effects -----

@dataclass(frozen=True)
class Add:
    atom: Atom
    loc: Optional[Loc] = _loc_field()


@dataclass(frozen=True)
class Del:
    atom: Atom
    loc: Optional[Loc] = _loc_field()


@dataclass(frozen=True)
class NumEffect:
    '''op is one of assign increase decrease scale-up scale-down.'''
    op: str
    target: FTerm
    expr: object
    loc: Optional[Loc] = _loc_field()


@dataclass(frozen=True)
class AndEffect:
    effects: tuple
    loc: Optional[Loc] = _loc_field()


@dataclass(frozen=True)
class ForallEffect:
    params: Tuple[Param, ...]
    effect: object
    loc: Optional[Loc] = _loc_field()


@dataclass(frozen=True)
class When:
    condition: object
    effect: object
    loc: Optional[Loc] = _loc_field()


@dataclass(frozen=True)
class Probabilistic:
    '''PPDDL (probabilistic p1 e1 ... pk ek): branches is ((p1, e1), ..., (pk, ek)).
    With probability 1 - sum(pi) nothing happens.'''
    branches: tuple
    loc: Optional[Loc] = _loc_field()


# ----- declarations -----

@dataclass(frozen=True)
class Action:
    name: str
    params: Tuple[Param, ...]
    precondition: object  # None if absent or empty
    effect: object
    loc: Optional[Loc] = _loc_field()


@dataclass(frozen=True)
class DerivedRule:
    '''One (:derived (name params) body) rule; a predicate may have several.'''
    name: str
    params: Tuple[Param, ...]
    body: object
    loc: Optional[Loc] = _loc_field()


@dataclass
class Domain:
    name: str
    requirements: frozenset
    types: Dict[str, Tuple[str, ...]]          # type -> parent types (excludes object)
    constants: Dict[str, Tuple[str, ...]]      # constant -> types
    predicates: Dict[str, Tuple[Param, ...]]
    functions: Dict[str, Tuple[Param, ...]]    # all numeric
    actions: Dict[str, Action]
    derived: Tuple[DerivedRule, ...]

    def is_subtype(self, child: str, parent: str) -> bool:
        '''Whether child equals parent or (transitively) derives from it.'''
        if parent == 'object' or child == parent:
            return True
        stack, seen = [child], set()
        while stack:
            t = stack.pop()
            if t == parent:
                return True
            if t not in seen:
                seen.add(t)
                stack.extend(self.types.get(t, ()))
        return False

    def fits(self, arg_types: Tuple[str, ...], param_types: Tuple[str, ...]) -> bool:
        '''Whether every object of a type in arg_types belongs to some type in
        param_types.'''
        return all(any(self.is_subtype(a, p) for p in param_types) for a in arg_types)


@dataclass
class Problem:
    name: str
    domain_name: str
    objects: Dict[str, Tuple[str, ...]]        # excludes domain constants
    init_atoms: Tuple[Atom, ...]
    init_numeric: Dict[Tuple[str, Tuple[str, ...]], float]
    goal: object
    metric: Optional[Tuple[str, object]]       # ('minimize'|'maximize', expr)
    goal_reward: Optional[float]               # PPDDL (:goal-reward n)


# ===========================================================================
# lexer and s-expression reader
# ===========================================================================

class Sym(str):
    '''A lower-cased token with its source location.'''

    def __new__(cls, text, line, col):
        obj = super().__new__(cls, text)
        obj.line = line
        obj.col = col
        return obj


class SList(list):
    '''A parenthesized list with the location of its opening parenthesis.'''

    def __init__(self, line, col):
        super().__init__()
        self.line = line
        self.col = col


_NUMBER = re.compile(r'^[-+]?(\d+/\d+|\d+(\.\d*)?|\.\d+)$')  # PPDDL allows rationals
_DELIMITERS = '();'


def _err(node, message):
    return PDDLParseError(message, getattr(node, 'line', None), getattr(node, 'col', None))


def _unsupported(node, message):
    return PDDLUnsupportedError(message, getattr(node, 'line', None),
                                getattr(node, 'col', None))


def _read(text: str) -> SList:
    '''Reads the text as exactly one s-expression.'''
    stack = []
    top = None
    line, col, i, n = 1, 1, 0, len(text)
    while i < n:
        ch = text[i]
        if ch == '\n':
            line, col, i = line + 1, 1, i + 1
        elif ch.isspace():
            col, i = col + 1, i + 1
        elif ch == ';':
            while i < n and text[i] != '\n':
                i += 1
        elif ch == '(':
            stack.append(SList(line, col))
            col, i = col + 1, i + 1
        elif ch == ')':
            if not stack:
                raise PDDLParseError('unmatched ")"', line, col)
            done = stack.pop()
            if stack:
                stack[-1].append(done)
            elif _head(done) != 'in-package':  # legacy Lisp header, ignored
                if top is not None:
                    raise _err(done, 'unexpected text after the end of the definition')
                top = done
            col, i = col + 1, i + 1
        else:
            start = i
            while i < n and not text[i].isspace() and text[i] not in _DELIMITERS:
                i += 1
            token = Sym(text[start:i].lower(), line, col)
            if not stack:
                raise _err(token, f'unexpected token "{token}" outside of parentheses')
            stack[-1].append(token)
            col += i - start
    if stack:
        raise _err(stack[-1], 'unclosed "("')
    if top is None:
        raise PDDLParseError('empty input', 1, 1)
    return top


def _is_sym(node) -> bool:
    return isinstance(node, Sym)


def _is_number(node) -> bool:
    return isinstance(node, Sym) and _NUMBER.match(node) is not None


def _number(node) -> float:
    '''The value of a number token, including PPDDL rationals such as 1/7.'''
    if '/' in node:
        (num, den) = node.split('/')
        if int(den) == 0:
            raise _err(node, f'division by zero in "{node}"')
        return int(num) / int(den)
    return float(node)


def _is_name(node) -> bool:
    return isinstance(node, Sym) and not node.startswith('?') \
        and node != '-' and not _is_number(node)


def _loc(node) -> Loc:
    return Loc(node.line, node.col)


def _head(node) -> Optional[str]:
    if isinstance(node, SList) and node and _is_sym(node[0]):
        return str(node[0])
    return None


# ===========================================================================
# shared parsing logic
# ===========================================================================

_UNSUPPORTED_DOMAIN_SECTIONS = {
    ':durative-action': 'durative actions',
    ':process': 'PDDL+ processes',
    ':event': 'PDDL+ events',
    ':constraints': 'PDDL3 constraints',
}

_PDDL3_MODAL = {'always', 'sometime', 'within', 'at-most-once', 'sometime-after',
                'sometime-before', 'always-within', 'hold-during', 'hold-after'}

_COMPARISONS = {'<', '<=', '>', '>=', '='}
_ARITHMETIC = {'+', '-', '*', '/'}
_NUMERIC_EFFECTS = {'assign', 'increase', 'decrease', 'scale-up', 'scale-down'}
_PROB_TOLERANCE = 1e-9


class _Parser:
    '''Turns s-expressions into IR, given the declarations in scope.'''

    def __init__(self, domain: Domain, objects: Dict[str, Tuple[str, ...]]) -> None:
        self.domain = domain
        self.objects = objects

    # ----- typed lists -----

    def typed_list(self, items, kind, default=('object',)):
        '''Parses "a b - t c - (either u v) d" into [(Sym, types), ...].
        kind is 'var' (?x), 'name' (plain names) or 'fterm' (function declarations).'''
        result, pending = [], []
        i = 0
        while i < len(items):
            item = items[i]
            if _is_sym(item) and item == '-':
                if not pending:
                    raise _err(item, 'expected a name before "-"')
                if i + 1 >= len(items):
                    raise _err(item, 'expected a type after "-"')
                types = self.type_spec(items[i + 1])
                result.extend((p, types) for p in pending)
                pending = []
                i += 2
                continue
            if kind == 'var':
                ok = _is_sym(item) and item.startswith('?') and len(item) > 1
            elif kind == 'name':
                ok = _is_name(item)
            else:
                ok = isinstance(item, SList) and len(item) > 0 and _is_name(item[0])
            if not ok:
                expected = {'var': 'a variable', 'name': 'a name',
                            'fterm': 'a function declaration'}[kind]
                raise _err(item, f'expected {expected}, got {_show(item)}')
            pending.append(item)
            i += 1
        result.extend((p, default) for p in pending)
        return result

    def type_spec(self, node) -> Tuple[str, ...]:
        if _is_name(node):
            return (str(node),)
        if _head(node) == 'either' and len(node) > 1 and all(map(_is_name, node[1:])):
            return tuple(str(t) for t in node[1:])
        raise _err(node, f'expected a type, got {_show(node)}')

    def check_types_declared(self, types, node):
        for t in types:
            if t != 'object' and t not in self.domain.types:
                raise _err(node, f'undeclared type "{t}"')

    def params(self, node, allow_duplicates=False) -> Tuple[Param, ...]:
        '''Parses a typed variable list. Duplicate names are allowed only in
        predicate and function declarations, where names carry no meaning.'''
        if not isinstance(node, SList):
            raise _err(node, f'expected a parameter list, got {_show(node)}')
        result, seen = [], set()
        for (var, types) in self.typed_list(node, 'var'):
            self.check_types_declared(types, var)
            if var in seen and not allow_duplicates:
                raise _err(var, f'duplicate parameter "{var}"')
            seen.add(var)
            result.append(Param(str(var), types, _loc(var)))
        return tuple(result)

    # ----- terms -----

    def obj_term(self, node, scope):
        '''A variable or object/constant argument; returns (term, types).'''
        if _is_sym(node) and node.startswith('?'):
            if node not in scope:
                raise _err(node, f'undeclared variable "{node}"')
            return Var(str(node), _loc(node)), scope[node]
        if _is_name(node):
            if node not in self.objects:
                raise _err(node, f'undeclared object or constant "{node}"')
            return Obj(str(node), _loc(node)), self.objects[node]
        raise _err(node, f'expected a variable or object, got {_show(node)}')

    def args(self, node, decl, scope, what):
        '''Checks arity and argument types of (name args...) against decl.'''
        if len(node) - 1 != len(decl):
            raise _err(node, f'{what} "{node[0]}" expects {len(decl)} argument(s), '
                             f'got {len(node) - 1}')
        result = []
        for (arg, param) in zip(node[1:], decl):
            term, types = self.obj_term(arg, scope)
            if not self.domain.fits(types, param.types):
                raise _err(arg, f'argument "{arg}" of type {_show_types(types)} does not '
                                f'match parameter {param.name} of type '
                                f'{_show_types(param.types)} in {what} "{node[0]}"')
            result.append(term)
        return tuple(result)

    def num_term(self, node, scope):
        if _is_number(node):
            return Num(_number(node), _loc(node))
        if _is_sym(node):
            if node in ('?duration', '#t'):
                raise _unsupported(node, f'"{node}" is only meaningful in temporal PDDL')
            raise _err(node, f'expected a numeric expression, got "{node}"')
        head = _head(node)
        if head is None:
            raise _err(node, f'expected a numeric expression, got {_show(node)}')
        if head in _ARITHMETIC:
            args = tuple(self.num_term(a, scope) for a in node[1:])
            valid = {'+': len(args) >= 2, '*': len(args) >= 2,
                     '-': len(args) in (1, 2), '/': len(args) == 2}[head]
            if not valid:
                raise _err(node, f'wrong number of arguments for "{head}"')
            return Arith(head, args, _loc(node))
        if head in ('total-time', 'is-violated'):
            raise _unsupported(node, f'"{head}" (temporal or PDDL3 metric) is not supported')
        if head in self.domain.functions:
            args = self.args(node, self.domain.functions[head], scope, 'function')
            return FTerm(head, args, _loc(node))
        raise _err(node, f'undeclared function "{head}"')

    def fterm(self, node, scope) -> FTerm:
        head = _head(node)
        if head is None or head not in self.domain.functions:
            raise _err(node, f'expected a numeric fluent, got {_show(node)}')
        return self.num_term(node, scope)

    # ----- formulas -----

    def atom(self, node, scope) -> Atom:
        head = _head(node)
        if head is None:
            raise _err(node, f'expected an atom, got {_show(node)}')
        if head not in self.domain.predicates:
            if head == 'preference':
                raise _unsupported(node, 'PDDL3 preferences are not supported')
            if head in _PDDL3_MODAL:
                raise _unsupported(node, f'PDDL3 modal operator "{head}" is not supported')
            raise _err(node, f'undeclared predicate "{head}"')
        args = self.args(node, self.domain.predicates[head], scope, 'predicate')
        return Atom(head, args, _loc(node))

    def formula(self, node, scope):
        head = _head(node)
        loc = _loc(node) if isinstance(node, SList) else None
        if head is not None and head not in self.domain.predicates:
            if head in ('and', 'or'):
                args = tuple(self.formula(a, scope) for a in node[1:])
                return (And if head == 'and' else Or)(args, loc)
            if head == 'not':
                self.expect_len(node, 2)
                return Not(self.formula(node[1], scope), loc)
            if head == 'imply':
                self.expect_len(node, 3)
                return Imply(self.formula(node[1], scope), self.formula(node[2], scope), loc)
            if head in ('exists', 'forall'):
                self.expect_len(node, 3)
                params = self.params(node[1])
                inner = dict(scope)
                inner.update((p.name, p.types) for p in params)
                body = self.formula(node[2], inner)
                return (Exists if head == 'exists' else Forall)(params, body, loc)
            if head in _COMPARISONS:
                self.expect_len(node, 3)
                left, right = node[1], node[2]
                if head == '=' and not any(isinstance(a, SList) or _is_number(a)
                                           for a in (left, right)):
                    return Equals(self.obj_term(left, scope)[0],
                                  self.obj_term(right, scope)[0], loc)
                return Compare(head, self.num_term(left, scope),
                               self.num_term(right, scope), loc)
        return self.atom(node, scope)

    def expect_len(self, node, n):
        if len(node) != n:
            raise _err(node, f'"{node[0]}" expects {n - 1} argument(s), got {len(node) - 1}')

    # ----- effects -----

    def effect(self, node, scope):
        if isinstance(node, SList) and not node:
            return AndEffect((), _loc(node))
        head = _head(node)
        loc = _loc(node) if isinstance(node, SList) else None
        if head is not None and head not in self.domain.predicates:
            if head == 'and':
                return AndEffect(tuple(self.effect(e, scope) for e in node[1:]), loc)
            if head == 'not':
                self.expect_len(node, 2)
                return Del(self.atom(node[1], scope), loc)
            if head == 'forall':
                self.expect_len(node, 3)
                params = self.params(node[1])
                inner = dict(scope)
                inner.update((p.name, p.types) for p in params)
                return ForallEffect(params, self.effect(node[2], inner), loc)
            if head == 'when':
                self.expect_len(node, 3)
                return When(self.formula(node[1], scope), self.effect(node[2], scope), loc)
            if head in _NUMERIC_EFFECTS:
                self.expect_len(node, 3)
                return NumEffect(head, self.fterm(node[1], scope),
                                 self.num_term(node[2], scope), loc)
            if head == 'probabilistic':
                return self.probabilistic(node, scope)
        return Add(self.atom(node, scope), loc)

    def probabilistic(self, node, scope):
        items = node[1:]
        if not items or len(items) % 2 != 0:
            raise _err(node, '"probabilistic" expects pairs of probability and effect')
        branches, total = [], 0.0
        for i in range(0, len(items), 2):
            prob = self.num_term(items[i], scope)
            if isinstance(prob, Num):
                if prob.value < 0:
                    raise _err(items[i], f'negative probability {prob.value}')
                total += prob.value
            branches.append((prob, self.effect(items[i + 1], scope)))
        if total > 1 + _PROB_TOLERANCE:
            raise _err(node, f'probabilities sum to {total}, which exceeds 1')
        return Probabilistic(tuple(branches), _loc(node))


def _show(node) -> str:
    if isinstance(node, SList):
        return '(' + ' '.join(_show(n) for n in node) + ')'
    return f'"{node}"'


def _show_types(types) -> str:
    return types[0] if len(types) == 1 else '(either ' + ' '.join(types) + ')'


def _sections(node, start, unsupported):
    '''Splits the (:keyword ...) sections of a define block.'''
    result = []
    for section in node[start:]:
        key = _head(section)
        if key is None or not key.startswith(':'):
            raise _err(section, f'expected a section such as (:keyword ...), got {_show(section)}')
        if key in unsupported:
            raise _unsupported(section, f'{unsupported[key]} ({key}) are not supported')
        result.append((key, section))
    return result


def _define_header(root, kind):
    if _head(root) != 'define' or len(root) < 2 or _head(root[1]) != kind \
    or len(root[1]) != 2 or not _is_name(root[1][1]):
        raise _err(root, f'expected (define ({kind} <name>) ...)')
    return str(root[1][1])


# ===========================================================================
# domain
# ===========================================================================

def parse_domain(text: str) -> Domain:
    '''Parses PDDL or PPDDL domain text into a Domain.'''
    root = _read(text)
    name = _define_header(root, 'domain')
    domain = Domain(name=name, requirements=frozenset(), types={}, constants={},
                    predicates={}, functions={}, actions={}, derived=())
    parser = _Parser(domain, domain.constants)
    sections = _sections(root, 2, _UNSUPPORTED_DOMAIN_SECTIONS)

    # declarations are processed before actions and derived predicates
    seen = set()
    for (key, section) in sections:
        if key in (':action', ':derived'):
            continue
        if key in seen:
            raise _err(section, f'duplicate {key} section')
        seen.add(key)
    order = [':requirements', ':types', ':constants', ':predicates', ':functions']
    for key in order:
        for (k, section) in sections:
            if k == key:
                _DOMAIN_SECTIONS[key](parser, section)
    if _has_rewards(domain) and 'reward' not in domain.functions:
        domain.functions['reward'] = ()

    derived = []
    for (key, section) in sections:
        if key == ':action':
            action = _parse_action(parser, section)
            if action.name in domain.actions:
                raise _err(section, f'duplicate action "{action.name}"')
            domain.actions[action.name] = action
        elif key == ':derived':
            derived.append(_parse_derived(parser, section))
        elif key not in order:
            raise _err(section, f'unknown domain section {key}')
    domain.derived = tuple(derived)
    return domain


def _has_rewards(domain) -> bool:
    '''Whether the domain declares the PPDDL reward fluent (:mdp is mdpsim's alias).'''
    return bool(domain.requirements & {':rewards', ':mdp'})


def _parse_requirements(parser, section):
    for req in section[1:]:
        if not _is_sym(req) or not req.startswith(':'):
            raise _err(req, f'expected a requirement such as :strips, got {_show(req)}')
    parser.domain.requirements = frozenset(str(r) for r in section[1:])


def _parse_types(parser, section):
    types = parser.domain.types
    for (name, parents) in parser.typed_list(section[1:], 'name'):
        if name == 'object':
            if parents != ('object',):
                raise _err(name, 'type "object" cannot have a parent type')
            continue
        merged = types.get(name, ())
        types[name] = merged + tuple(p for p in parents if p not in merged)
    for parents in list(types.values()):
        for p in parents:
            if p != 'object' and p not in types:
                types[p] = ('object',)

    # reject cycles in the hierarchy
    state = {}

    def visit(t):
        if state.get(t) == 1:
            raise _err(section, f'cyclic type hierarchy involving "{t}"')
        if state.get(t) is None:
            state[t] = 1
            for p in types.get(t, ()):
                visit(p)
            state[t] = 2

    for t in types:
        visit(t)


def _declare_objects(parser, items, table, existing=None):
    for (obj, types) in parser.typed_list(items, 'name'):
        parser.check_types_declared(types, obj)
        if existing is not None and obj in existing:
            if existing[obj] != types:
                raise _err(obj, f'object "{obj}" redeclares a domain constant '
                                f'with a different type')
            continue
        if obj in table:
            raise _err(obj, f'duplicate object "{obj}"')
        table[str(obj)] = types


def _parse_constants(parser, section):
    _declare_objects(parser, section[1:], parser.domain.constants)


def _parse_predicates(parser, section):
    preds = parser.domain.predicates
    for decl in section[1:]:
        head = _head(decl)
        if head is None or not _is_name(decl[0]):
            raise _err(decl, f'expected a predicate declaration, got {_show(decl)}')
        if head in preds:
            raise _err(decl, f'duplicate predicate "{head}"')
        if head in _COMPARISONS:
            raise _err(decl, f'"{head}" cannot be declared as a predicate')
        preds[head] = parser.params(_tail(decl), allow_duplicates=True)


def _tail(node):
    tail = SList(node.line, node.col)
    tail.extend(node[1:])
    return tail


def _parse_functions(parser, section):
    funcs = parser.domain.functions
    for (decl, types) in parser.typed_list(section[1:], 'fterm', default=('number',)):
        if types != ('number',):
            raise _unsupported(decl, f'object-valued function "{decl[0]}" of type '
                                     f'{_show_types(types)} is not supported')
        if decl[0] in funcs or decl[0] in parser.domain.predicates:
            raise _err(decl, f'duplicate function "{decl[0]}"')
        funcs[str(decl[0])] = parser.params(_tail(decl), allow_duplicates=True)


_DOMAIN_SECTIONS = {
    ':requirements': _parse_requirements,
    ':types': _parse_types,
    ':constants': _parse_constants,
    ':predicates': _parse_predicates,
    ':functions': _parse_functions,
}


def _parse_action(parser, section):
    if len(section) < 2 or not _is_name(section[1]):
        raise _err(section, 'expected an action name')
    name = str(section[1])
    fields = {}
    rest = section[2:]
    if len(rest) % 2 != 0:
        raise _err(section, f'action "{name}" has a keyword without a value')
    for i in range(0, len(rest), 2):
        key = rest[i]
        if key == ':vars':
            raise _unsupported(key, 'the PDDL 1.2 action field :vars is not supported')
        if not _is_sym(key) or key not in (':parameters', ':precondition', ':effect'):
            raise _err(key, f'unknown action field {_show(key)}')
        if key in fields:
            raise _err(key, f'duplicate action field {key}')
        fields[str(key)] = rest[i + 1]
    params = parser.params(fields[':parameters']) if ':parameters' in fields else ()
    scope = {p.name: p.types for p in params}
    pre = fields.get(':precondition')
    if pre is None or (isinstance(pre, SList) and not pre):
        precondition = None
    else:
        precondition = parser.formula(pre, scope)
    eff = fields.get(':effect')
    effect = AndEffect(()) if eff is None else parser.effect(eff, scope)
    return Action(name, params, precondition, effect, _loc(section))


def _parse_derived(parser, section):
    if len(section) != 3 or _head(section[1]) is None:
        raise _err(section, 'expected (:derived (<predicate> <params>) <formula>)')
    decl = section[1]
    name = str(decl[0])
    if name not in parser.domain.predicates:
        raise _err(decl, f'derived predicate "{name}" is not declared in :predicates')
    params = parser.params(_tail(decl))
    declared = parser.domain.predicates[name]
    if len(params) != len(declared) or not all(
            parser.domain.fits(p.types, d.types) for (p, d) in zip(params, declared)):
        raise _err(decl, f'derived predicate "{name}" does not match its declaration')
    scope = {p.name: p.types for p in params}
    body = parser.formula(section[2], scope)
    return DerivedRule(name, params, body, _loc(section))


# ===========================================================================
# problem
# ===========================================================================

def parse_problem(text: str, domain: Domain) -> Problem:
    '''Parses PDDL or PPDDL problem text for the given domain into a Problem.'''
    root = _read(text)
    name = _define_header(root, 'problem')
    sections = _sections(root, 2, {':constraints': 'PDDL3 constraints'})

    objects = dict(domain.constants)
    parser = _Parser(domain, objects)
    problem = Problem(name=name, domain_name='', objects={}, init_atoms=(),
                      init_numeric={}, goal=None, metric=None, goal_reward=None)

    handlers = {
        ':domain': _parse_problem_domain,
        ':requirements': lambda *args: None,
        ':objects': _parse_objects,
        ':init': _parse_init,
        ':goal': _parse_goal,
        ':metric': _parse_metric,
        ':goal-reward': _parse_goal_reward,
        ':length': lambda *args: None,
    }
    seen = set()
    for key in handlers:  # process in a fixed order: objects before init and goal
        for (k, section) in sections:
            if k == key:
                if k in seen:
                    raise _err(section, f'duplicate {k} section')
                seen.add(k)
                handlers[k](parser, section, problem)
    for (key, section) in sections:
        if key not in handlers:
            raise _err(section, f'unknown problem section {key}')
    if ':domain' not in seen:
        raise _err(root, 'missing (:domain <name>)')
    if ':goal' not in seen and not _has_rewards(domain):  # reward-only MDPs have no goal
        raise _err(root, 'missing (:goal ...)')
    return problem


def _parse_problem_domain(parser, section, problem):
    if len(section) != 2 or not _is_name(section[1]):
        raise _err(section, 'expected (:domain <name>)')
    if section[1] != parser.domain.name:
        raise _err(section, f'problem refers to domain "{section[1]}", '
                            f'but the domain is "{parser.domain.name}"')
    problem.domain_name = str(section[1])


def _parse_objects(parser, section, problem):
    _declare_objects(parser, section[1:], problem.objects, parser.domain.constants)
    parser.objects.update(problem.objects)


def _parse_init(parser, section, problem):
    atoms, negated, numeric = [], [], {}
    for item in section[1:]:
        head = _head(item)
        if head == 'probabilistic' and head not in parser.domain.predicates:
            raise _unsupported(item, 'probabilistic initial states are not supported')
        if head == 'at' and len(item) == 3 and _is_number(item[1]):
            raise _unsupported(item, 'timed initial literals are not supported')
        if head == 'not' and 'not' not in parser.domain.predicates:
            # redundant under the closed-world assumption, but must be consistent
            parser.expect_len(item, 2)
            negated.append((parser.atom(item[1], {}), item))
        elif head == '=' and len(item) == 3 and _head(item[1]) in parser.domain.functions:
            target = parser.fterm(item[1], {})
            if not _is_number(item[2]):
                raise _err(item, f'expected a number, got {_show(item[2])}')
            key = (target.name, tuple(a.name for a in target.args))
            if key in numeric:
                raise _err(item, f'duplicate initial value for {_show(item[1])}')
            numeric[key] = _number(item[2])
        else:
            atoms.append(parser.atom(item, {}))
    positive = set(atoms)
    for (atom, item) in negated:
        if atom in positive:
            raise _err(item, f'{_show(item[1])} is declared both true and false in :init')
    problem.init_atoms = tuple(atoms)
    problem.init_numeric = numeric


def _parse_goal(parser, section, problem):
    if len(section) != 2:
        raise _err(section, 'expected (:goal <formula>)')
    problem.goal = parser.formula(section[1], {})


def _parse_metric(parser, section, problem):
    if len(section) != 3 or section[1] not in ('minimize', 'maximize'):
        raise _err(section, 'expected (:metric minimize|maximize <expression>)')
    problem.metric = (str(section[1]), parser.num_term(section[2], {}))


def _parse_goal_reward(parser, section, problem):
    if len(section) != 2 or not _is_number(section[1]):
        raise _err(section, 'expected (:goal-reward <number>)')
    problem.goal_reward = _number(section[1])
