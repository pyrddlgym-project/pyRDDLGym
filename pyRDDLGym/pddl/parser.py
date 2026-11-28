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

    @staticmethod
    def unset():
        '''The field spec for an optional location that is ignored by == and repr.'''
        return field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class Param:
    '''A typed variable; types has more than one entry for (either ...).'''
    name: str
    types: Tuple[str, ...]
    loc: Optional[Loc] = Loc.unset()


# ----- terms -----

@dataclass(frozen=True)
class Var:
    name: str
    loc: Optional[Loc] = Loc.unset()


@dataclass(frozen=True)
class Obj:
    name: str
    loc: Optional[Loc] = Loc.unset()


@dataclass(frozen=True)
class Num:
    value: float
    loc: Optional[Loc] = Loc.unset()


@dataclass(frozen=True)
class FTerm:
    '''A numeric function application such as (f ?x).'''
    name: str
    args: Tuple[Union[Var, Obj], ...]
    loc: Optional[Loc] = Loc.unset()


@dataclass(frozen=True)
class Arith:
    '''Arithmetic: op is one of + - * /; (- x) is unary negation.'''
    op: str
    args: tuple
    loc: Optional[Loc] = Loc.unset()


# ----- formulas -----

@dataclass(frozen=True)
class Atom:
    pred: str
    args: Tuple[Union[Var, Obj], ...]
    loc: Optional[Loc] = Loc.unset()


@dataclass(frozen=True)
class Equals:
    '''Object equality (= t1 t2).'''
    left: Union[Var, Obj]
    right: Union[Var, Obj]
    loc: Optional[Loc] = Loc.unset()


@dataclass(frozen=True)
class Not:
    arg: object
    loc: Optional[Loc] = Loc.unset()


@dataclass(frozen=True)
class And:
    args: tuple
    loc: Optional[Loc] = Loc.unset()


@dataclass(frozen=True)
class Or:
    args: tuple
    loc: Optional[Loc] = Loc.unset()


@dataclass(frozen=True)
class Imply:
    left: object
    right: object
    loc: Optional[Loc] = Loc.unset()


@dataclass(frozen=True)
class Exists:
    params: Tuple[Param, ...]
    body: object
    loc: Optional[Loc] = Loc.unset()


@dataclass(frozen=True)
class Forall:
    params: Tuple[Param, ...]
    body: object
    loc: Optional[Loc] = Loc.unset()


@dataclass(frozen=True)
class Compare:
    '''Numeric comparison: op is one of < <= > >= =.'''
    op: str
    left: object
    right: object
    loc: Optional[Loc] = Loc.unset()


# ----- effects -----

@dataclass(frozen=True)
class Add:
    atom: Atom
    loc: Optional[Loc] = Loc.unset()


@dataclass(frozen=True)
class Del:
    atom: Atom
    loc: Optional[Loc] = Loc.unset()


@dataclass(frozen=True)
class NumEffect:
    '''op is one of assign increase decrease scale-up scale-down.'''
    op: str
    target: FTerm
    expr: object
    loc: Optional[Loc] = Loc.unset()


@dataclass(frozen=True)
class AndEffect:
    effects: tuple
    loc: Optional[Loc] = Loc.unset()


@dataclass(frozen=True)
class ForallEffect:
    params: Tuple[Param, ...]
    effect: object
    loc: Optional[Loc] = Loc.unset()


@dataclass(frozen=True)
class When:
    condition: object
    effect: object
    loc: Optional[Loc] = Loc.unset()


@dataclass(frozen=True)
class Probabilistic:
    '''PPDDL (probabilistic p1 e1 ... pk ek): branches is ((p1, e1), ..., (pk, ek)).
    With probability 1 - sum(pi) nothing happens.'''
    branches: tuple
    loc: Optional[Loc] = Loc.unset()


# ----- declarations -----

@dataclass(frozen=True)
class Action:
    name: str
    params: Tuple[Param, ...]
    precondition: object  # None if absent or empty
    effect: object
    loc: Optional[Loc] = Loc.unset()


@dataclass(frozen=True)
class DerivedRule:
    '''One (:derived (name params) body) rule; a predicate may have several.'''
    name: str
    params: Tuple[Param, ...]
    body: object
    loc: Optional[Loc] = Loc.unset()


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

    def has_rewards(self) -> bool:
        '''Whether the domain declares the PPDDL reward fluent (:mdp is mdpsim's alias).'''
        return bool(self.requirements & {':rewards', ':mdp'})


@dataclass
class Problem:
    '''The initial state is init_atoms and init_numeric, then the PPDDL init_effects
    applied to it as one simultaneous effect (deletes first, then adds, as in mdpsim):
    a branch may delete an atom of init_atoms, and an add beats a delete. Every
    init_effects branch is a ground Add, Del, AndEffect, Probabilistic or
    NumEffect('assign', ..., Num).'''
    name: str
    domain_name: str
    objects: Dict[str, Tuple[str, ...]]        # excludes domain constants
    init_atoms: Tuple[Atom, ...]
    init_numeric: Dict[Tuple[str, Tuple[str, ...]], float]
    init_effects: Tuple[Probabilistic, ...]    # PPDDL (probabilistic ...) in :init
    goal: object
    metric: Optional[Tuple[str, object]]       # ('minimize'|'maximize', expr)
    goal_reward: Optional[float]               # PPDDL (:goal-reward n)


# ===========================================================================
# lexer tokens
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


# ===========================================================================
# parser
# ===========================================================================

class PDDLParser:
    '''Turns PDDL text into IR. An instance holds the declarations in scope; the
    entry points are the static methods parse_domain and parse_problem.'''

    NUMBER = re.compile(r'^[-+]?(\d+/\d+|\d+(\.\d*)?|\.\d+)$')  # PPDDL allows rationals
    DELIMITERS = '();'

    UNSUPPORTED_DOMAIN_SECTIONS = {
        ':durative-action': 'durative actions',
        ':process': 'PDDL+ processes',
        ':event': 'PDDL+ events',
        ':constraints': 'PDDL3 constraints',
    }

    PDDL3_MODAL = {'always', 'sometime', 'within', 'at-most-once', 'sometime-after',
                   'sometime-before', 'always-within', 'hold-during', 'hold-after'}

    COMPARISONS = {'<', '<=', '>', '>=', '='}
    ARITHMETIC = {'+', '-', '*', '/'}
    NUMERIC_EFFECTS = {'assign', 'increase', 'decrease', 'scale-up', 'scale-down'}
    PROB_TOLERANCE = 1e-9

    def __init__(self, domain: Domain, objects: Dict[str, Tuple[str, ...]]) -> None:
        self.domain = domain
        self.objects = objects

    # ===========================================================================
    # entry points
    # ===========================================================================

    @staticmethod
    def parse_domain(text: str) -> Domain:
        '''Parses PDDL or PPDDL domain text into a Domain.'''
        (root, name) = PDDLParser._definition(PDDLParser._read(text), 'domain')
        domain = Domain(name=name, requirements=frozenset(), types={}, constants={},
                        predicates={}, functions={}, actions={}, derived=())
        parser = PDDLParser(domain, domain.constants)
        sections = PDDLParser._sections(root, 2, PDDLParser.UNSUPPORTED_DOMAIN_SECTIONS)

        # declarations are processed before actions and derived predicates
        seen = set()
        for (key, section) in sections:
            if key in (':action', ':derived'):
                continue
            if key in seen:
                raise PDDLParser._err(section, f'duplicate {key} section')
            seen.add(key)
        handlers = {
            ':requirements': parser._parse_requirements,
            ':types': parser._parse_types,
            ':constants': parser._parse_constants,
            ':predicates': parser._parse_predicates,
            ':functions': parser._parse_functions,
        }
        for (key, handler) in handlers.items():
            for (k, section) in sections:
                if k == key:
                    handler(section)
        if domain.has_rewards() and 'reward' not in domain.functions:
            domain.functions['reward'] = ()

        derived = []
        for (key, section) in sections:
            if key == ':action':
                action = parser._parse_action(section)
                if action.name in domain.actions:
                    raise PDDLParser._err(section, f'duplicate action "{action.name}"')
                domain.actions[action.name] = action
            elif key == ':derived':
                derived.append(parser._parse_derived(section))
            elif key not in handlers:
                raise PDDLParser._err(section, f'unknown domain section {key}')
        domain.derived = tuple(derived)
        return domain

    @staticmethod
    def parse_problem(text: str, domain: Domain) -> Problem:
        '''Parses PDDL or PPDDL problem text for the given domain into a Problem.'''
        (root, name) = PDDLParser._definition(PDDLParser._read(text), 'problem')
        sections = PDDLParser._sections(root, 2, {':constraints': 'PDDL3 constraints'})

        parser = PDDLParser(domain, dict(domain.constants))
        problem = Problem(name=name, domain_name='', objects={}, init_atoms=(),
                          init_numeric={}, init_effects=(), goal=None, metric=None,
                          goal_reward=None)

        handlers = {
            ':domain': parser._parse_problem_domain,
            ':requirements': lambda *args: None,
            ':objects': parser._parse_objects,
            ':init': parser._parse_init,
            ':goal': parser._parse_goal,
            ':metric': parser._parse_metric,
            ':goal-reward': parser._parse_goal_reward,
            ':length': lambda *args: None,
        }
        seen = set()
        for key in handlers:  # process in a fixed order: objects before init and goal
            for (k, section) in sections:
                if k == key:
                    if k in seen:
                        raise PDDLParser._err(section, f'duplicate {k} section')
                    seen.add(k)
                    handlers[k](section, problem)
        for (key, section) in sections:
            if key not in handlers:
                raise PDDLParser._err(section, f'unknown problem section {key}')
        if ':domain' not in seen:
            raise PDDLParser._err(root, 'missing (:domain <name>)')
        if ':goal' not in seen and not domain.has_rewards():  # reward-only MDPs have no goal
            raise PDDLParser._err(root, 'missing (:goal ...)')
        return problem

    # ===========================================================================
    # s-expression reader and token helpers
    # ===========================================================================

    @staticmethod
    def _err(node, message):
        return PDDLParseError(message, getattr(node, 'line', None),
                              getattr(node, 'col', None))

    @staticmethod
    def _unsupported(node, message):
        return PDDLUnsupportedError(message, getattr(node, 'line', None),
                                    getattr(node, 'col', None))

    @staticmethod
    def _read(text: str) -> list:
        '''Reads the text as a list of top-level s-expressions.'''
        stack = []
        tops = []
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
                elif PDDLParser._head(done) != 'in-package':  # legacy Lisp header, ignored
                    tops.append(done)
                col, i = col + 1, i + 1
            else:
                start = i
                while i < n and not text[i].isspace() \
                and text[i] not in PDDLParser.DELIMITERS:
                    i += 1
                token = Sym(text[start:i].lower(), line, col)
                if not stack:
                    raise PDDLParser._err(
                        token, f'unexpected token "{token}" outside of parentheses')
                if len(token) > 1 and token[0] == '-' and token[1].isalpha():
                    # "?x -type": names start with a letter, so this is "- type"
                    stack[-1].append(Sym('-', line, col))
                    token = Sym(token[1:], line, col + 1)
                stack[-1].append(token)
                col += i - start
        if stack:
            raise PDDLParser._err(stack[-1], 'unclosed "("')
        if not tops:
            raise PDDLParseError('empty input', 1, 1)
        return tops

    @staticmethod
    def _is_sym(node) -> bool:
        return isinstance(node, Sym)

    @staticmethod
    def _is_number(node) -> bool:
        return isinstance(node, Sym) and PDDLParser.NUMBER.match(node) is not None

    @staticmethod
    def _number(node) -> float:
        '''The value of a number token, including PPDDL rationals such as 1/7.'''
        if '/' in node:
            (num, den) = node.split('/')
            if int(den) == 0:
                raise PDDLParser._err(node, f'division by zero in "{node}"')
            return int(num) / int(den)
        return float(node)

    @staticmethod
    def _is_name(node) -> bool:
        return isinstance(node, Sym) and not node.startswith('?') \
            and node != '-' and not PDDLParser._is_number(node)

    @staticmethod
    def _loc(node) -> Loc:
        return Loc(node.line, node.col)

    @staticmethod
    def _head(node) -> Optional[str]:
        if isinstance(node, SList) and node and isinstance(node[0], Sym):
            return str(node[0])
        return None

    @staticmethod
    def _tail(node) -> SList:
        tail = SList(node.line, node.col)
        tail.extend(node[1:])
        return tail

    @staticmethod
    def _show(node) -> str:
        if isinstance(node, SList):
            return '(' + ' '.join(PDDLParser._show(n) for n in node) + ')'
        return f'"{node}"'

    @staticmethod
    def _show_types(types) -> str:
        return types[0] if len(types) == 1 else '(either ' + ' '.join(types) + ')'

    @staticmethod
    def _sections(node, start, unsupported):
        '''Splits the (:keyword ...) sections of a define block.'''
        result = []
        for section in node[start:]:
            key = PDDLParser._head(section)
            if key is None or not key.startswith(':'):
                raise PDDLParser._err(section, f'expected a section such as (:keyword ...), '
                                               f'got {PDDLParser._show(section)}')
            if key in unsupported:
                raise PDDLParser._unsupported(
                    section, f'{unsupported[key]} ({key}) are not supported')
            result.append((key, section))
        return result

    @staticmethod
    def _definition(forms, kind):
        '''Returns (form, name) for the one (define (<kind> <name>) ...) among the
        top-level forms. A file may hold a domain and its problem, as IPPC files do.'''
        found = []
        for form in forms:
            if PDDLParser._head(form) != 'define' or len(form) < 2 \
            or PDDLParser._head(form[1]) not in ('domain', 'problem') \
            or len(form[1]) != 2 or not PDDLParser._is_name(form[1][1]):
                raise PDDLParser._err(form, f'expected (define ({kind} <name>) ...)')
            if form[1][0] == kind:
                found.append(form)
        if not found:
            raise PDDLParser._err(forms[0], f'expected (define ({kind} <name>) ...)')
        if len(found) > 1:
            raise PDDLParser._err(found[1], f'more than one {kind} definition')
        return (found[0], str(found[0][1][1]))

    # ===========================================================================
    # typed lists
    # ===========================================================================

    def typed_list(self, items, kind, default=('object',)):
        '''Parses "a b - t c - (either u v) d" into [(Sym, types), ...].
        kind is 'var' (?x), 'name' (plain names) or 'fterm' (function declarations).'''
        result, pending = [], []
        i = 0
        while i < len(items):
            item = items[i]
            if self._is_sym(item) and item == '-':
                if not pending:
                    raise self._err(item, 'expected a name before "-"')
                if i + 1 >= len(items):
                    raise self._err(item, 'expected a type after "-"')
                types = self.type_spec(items[i + 1])
                result.extend((p, types) for p in pending)
                pending = []
                i += 2
                continue
            if kind == 'var':
                ok = self._is_sym(item) and item.startswith('?') and len(item) > 1
            elif kind == 'name':
                ok = self._is_name(item)
            else:
                ok = isinstance(item, SList) and len(item) > 0 and self._is_name(item[0])
            if not ok:
                expected = {'var': 'a variable', 'name': 'a name',
                            'fterm': 'a function declaration'}[kind]
                raise self._err(item, f'expected {expected}, got {self._show(item)}')
            pending.append(item)
            i += 1
        result.extend((p, default) for p in pending)
        return result

    def type_spec(self, node) -> Tuple[str, ...]:
        if self._is_name(node):
            return (str(node),)
        if self._head(node) == 'either' and len(node) > 1 \
        and all(map(self._is_name, node[1:])):
            return tuple(str(t) for t in node[1:])
        raise self._err(node, f'expected a type, got {self._show(node)}')

    def check_types_declared(self, types, node):
        for t in types:
            if t != 'object' and t not in self.domain.types:
                raise self._err(node, f'undeclared type "{t}"')

    def params(self, node, allow_duplicates=False) -> Tuple[Param, ...]:
        '''Parses a typed variable list. Duplicate names are allowed only in
        predicate and function declarations, where names carry no meaning.'''
        if not isinstance(node, SList):
            raise self._err(node, f'expected a parameter list, got {self._show(node)}')
        result, seen = [], set()
        for (var, types) in self.typed_list(node, 'var'):
            self.check_types_declared(types, var)
            if var in seen and not allow_duplicates:
                raise self._err(var, f'duplicate parameter "{var}"')
            seen.add(var)
            result.append(Param(str(var), types, self._loc(var)))
        return tuple(result)

    # ===========================================================================
    # terms
    # ===========================================================================

    def obj_term(self, node, scope):
        '''A variable or object/constant argument; returns (term, types).'''
        if self._is_sym(node) and node.startswith('?'):
            if node not in scope:
                raise self._err(node, f'undeclared variable "{node}"')
            return Var(str(node), self._loc(node)), scope[node]
        if self._is_name(node):
            if node not in self.objects:
                raise self._err(node, f'undeclared object or constant "{node}"')
            return Obj(str(node), self._loc(node)), self.objects[node]
        raise self._err(node, f'expected a variable or object, got {self._show(node)}')

    def args(self, node, decl, scope, what):
        '''Checks arity and argument types of (name args...) against decl.'''
        if len(node) - 1 != len(decl):
            raise self._err(node, f'{what} "{node[0]}" expects {len(decl)} argument(s), '
                                  f'got {len(node) - 1}')
        result = []
        for (arg, param) in zip(node[1:], decl):
            term, types = self.obj_term(arg, scope)
            if not self.domain.fits(types, param.types):
                raise self._err(arg, f'argument "{arg}" of type {self._show_types(types)} '
                                     f'does not match parameter {param.name} of type '
                                     f'{self._show_types(param.types)} in {what} "{node[0]}"')
            result.append(term)
        return tuple(result)

    def num_term(self, node, scope):
        if self._is_number(node):
            return Num(self._number(node), self._loc(node))
        if self._is_sym(node):
            if node in ('?duration', '#t'):
                raise self._unsupported(node, f'"{node}" is only meaningful in temporal PDDL')
            if self._is_bare_fterm(node):
                return FTerm(str(node), (), self._loc(node))
            raise self._err(node, f'expected a numeric expression, got "{node}"')
        head = self._head(node)
        if head is None:
            raise self._err(node, f'expected a numeric expression, got {self._show(node)}')
        if head in self.ARITHMETIC:
            args = tuple(self.num_term(a, scope) for a in node[1:])
            valid = {'+': len(args) >= 2, '*': len(args) >= 2,
                     '-': len(args) in (1, 2), '/': len(args) == 2}[head]
            if not valid:
                raise self._err(node, f'wrong number of arguments for "{head}"')
            return Arith(head, args, self._loc(node))
        if head in ('total-time', 'is-violated'):
            raise self._unsupported(
                node, f'"{head}" (temporal or PDDL3 metric) is not supported')
        if head in self.domain.functions:
            args = self.args(node, self.domain.functions[head], scope, 'function')
            return FTerm(head, args, self._loc(node))
        raise self._err(node, f'undeclared function "{head}"')

    def _is_bare_fterm(self, node) -> bool:
        '''Whether node is a 0-ary function written without parentheses, such as
        "reward", which PDDL 2.1 allows (<f-head> ::= <function-symbol>).'''
        return self._is_sym(node) and self.domain.functions.get(node) == () \
            and node not in self.objects

    def fterm(self, node, scope) -> FTerm:
        head = self._head(node)
        if self._is_bare_fterm(node):
            head = str(node)
        if head is None or head not in self.domain.functions:
            raise self._err(node, f'expected a numeric fluent, got {self._show(node)}')
        return self.num_term(node, scope)

    # ===========================================================================
    # formulas
    # ===========================================================================

    def atom(self, node, scope) -> Atom:
        head = self._head(node)
        if head is None:
            raise self._err(node, f'expected an atom, got {self._show(node)}')
        if head not in self.domain.predicates:
            if head == 'preference':
                raise self._unsupported(node, 'PDDL3 preferences are not supported')
            if head in self.PDDL3_MODAL:
                raise self._unsupported(
                    node, f'PDDL3 modal operator "{head}" is not supported')
            raise self._err(node, f'undeclared predicate "{head}"')
        args = self.args(node, self.domain.predicates[head], scope, 'predicate')
        return Atom(head, args, self._loc(node))

    def formula(self, node, scope):
        head = self._head(node)
        loc = self._loc(node) if isinstance(node, SList) else None
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
            if head in self.COMPARISONS:
                self.expect_len(node, 3)
                left, right = node[1], node[2]
                if head == '=' and not any(isinstance(a, SList) or self._is_number(a)
                                           or self._is_bare_fterm(a) for a in (left, right)):
                    return Equals(self.obj_term(left, scope)[0],
                                  self.obj_term(right, scope)[0], loc)
                return Compare(head, self.num_term(left, scope),
                               self.num_term(right, scope), loc)
        return self.atom(node, scope)

    def expect_len(self, node, n):
        if len(node) != n:
            raise self._err(node, f'"{node[0]}" expects {n - 1} argument(s), '
                                  f'got {len(node) - 1}')

    # ===========================================================================
    # effects
    # ===========================================================================

    def effect(self, node, scope):
        if isinstance(node, SList) and not node:
            return AndEffect((), self._loc(node))
        head = self._head(node)
        loc = self._loc(node) if isinstance(node, SList) else None
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
            if head in self.NUMERIC_EFFECTS:
                self.expect_len(node, 3)
                return NumEffect(head, self.fterm(node[1], scope),
                                 self.num_term(node[2], scope), loc)
            if head == 'probabilistic':
                return self.probabilistic(node, scope)
        return Add(self.atom(node, scope), loc)

    def probabilistic(self, node, scope, init=False):
        '''Parses (probabilistic p1 e1 ... pk ek). In :init (init=True) the
        probabilities must be numbers and each ei is an init element.'''
        items = node[1:]
        if not items or len(items) % 2 != 0:
            raise self._err(node, '"probabilistic" expects pairs of probability and effect')
        branches, total = [], 0.0
        for i in range(0, len(items), 2):
            prob = self.num_term(items[i], scope)
            if isinstance(prob, Num):
                if prob.value < 0:
                    raise self._err(items[i], f'negative probability {prob.value}')
                total += prob.value
            elif init:
                raise self._err(items[i], f'probabilities in :init must be numbers, '
                                          f'got {self._show(items[i])}')
            if init:
                branch = self.init_element(items[i + 1])
            else:
                branch = self.effect(items[i + 1], scope)
            branches.append((prob, branch))
        if total > 1 + self.PROB_TOLERANCE:
            raise self._err(node, f'probabilities sum to {total}, which exceeds 1')
        return Probabilistic(tuple(branches), self._loc(node))

    # ===========================================================================
    # domain sections
    # ===========================================================================

    def _parse_requirements(self, section):
        for req in section[1:]:
            if not self._is_sym(req) or not req.startswith(':'):
                raise self._err(
                    req, f'expected a requirement such as :strips, got {self._show(req)}')
        self.domain.requirements = frozenset(str(r) for r in section[1:])

    def _parse_types(self, section):
        types = self.domain.types
        for (name, parents) in self.typed_list(section[1:], 'name'):
            if name == 'object':
                if parents != ('object',):
                    raise self._err(name, 'type "object" cannot have a parent type')
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
                raise self._err(section, f'cyclic type hierarchy involving "{t}"')
            if state.get(t) is None:
                state[t] = 1
                for p in types.get(t, ()):
                    visit(p)
                state[t] = 2

        for t in types:
            visit(t)

    def _declare_objects(self, items, table, existing=None):
        for (obj, types) in self.typed_list(items, 'name'):
            self.check_types_declared(types, obj)
            if existing is not None and obj in existing:
                if existing[obj] != types:
                    raise self._err(obj, f'object "{obj}" redeclares a domain constant '
                                         f'with a different type')
                continue
            if obj in table:
                raise self._err(obj, f'duplicate object "{obj}"')
            table[str(obj)] = types

    def _parse_constants(self, section):
        self._declare_objects(section[1:], self.domain.constants)

    def _parse_predicates(self, section):
        preds = self.domain.predicates
        for decl in section[1:]:
            head = self._head(decl)
            if head is None or not self._is_name(decl[0]):
                raise self._err(
                    decl, f'expected a predicate declaration, got {self._show(decl)}')
            if head in preds:
                raise self._err(decl, f'duplicate predicate "{head}"')
            if head in self.COMPARISONS:
                raise self._err(decl, f'"{head}" cannot be declared as a predicate')
            preds[head] = self.params(self._tail(decl), allow_duplicates=True)

    def _parse_functions(self, section):
        funcs = self.domain.functions
        for (decl, types) in self.typed_list(section[1:], 'fterm', default=('number',)):
            if types != ('number',):
                raise self._unsupported(
                    decl, f'object-valued function "{decl[0]}" of type '
                          f'{self._show_types(types)} is not supported')
            if decl[0] in funcs or decl[0] in self.domain.predicates:
                raise self._err(decl, f'duplicate function "{decl[0]}"')
            funcs[str(decl[0])] = self.params(self._tail(decl), allow_duplicates=True)

    def _parse_action(self, section):
        if len(section) < 2 or not self._is_name(section[1]):
            raise self._err(section, 'expected an action name')
        name = str(section[1])
        fields = {}
        rest = section[2:]
        if len(rest) % 2 != 0:
            raise self._err(section, f'action "{name}" has a keyword without a value')
        for i in range(0, len(rest), 2):
            key = rest[i]
            if key == ':vars':
                raise self._unsupported(key, 'the PDDL 1.2 action field :vars is not supported')
            if not self._is_sym(key) or key not in (':parameters', ':precondition', ':effect'):
                raise self._err(key, f'unknown action field {self._show(key)}')
            if key in fields:
                raise self._err(key, f'duplicate action field {key}')
            fields[str(key)] = rest[i + 1]
        params = self.params(fields[':parameters']) if ':parameters' in fields else ()
        scope = {p.name: p.types for p in params}
        pre = fields.get(':precondition')
        if pre is None or (isinstance(pre, SList) and not pre):
            precondition = None
        else:
            precondition = self.formula(pre, scope)
        eff = fields.get(':effect')
        effect = AndEffect(()) if eff is None else self.effect(eff, scope)
        return Action(name, params, precondition, effect, self._loc(section))

    def _parse_derived(self, section):
        if len(section) != 3 or self._head(section[1]) is None:
            raise self._err(section, 'expected (:derived (<predicate> <params>) <formula>)')
        decl = section[1]
        name = str(decl[0])
        if name not in self.domain.predicates:
            raise self._err(decl, f'derived predicate "{name}" is not declared in :predicates')
        params = self.params(self._tail(decl))
        declared = self.domain.predicates[name]
        if len(params) != len(declared) or not all(
                self.domain.fits(p.types, d.types) for (p, d) in zip(params, declared)):
            raise self._err(decl, f'derived predicate "{name}" does not match its declaration')
        scope = {p.name: p.types for p in params}
        body = self.formula(section[2], scope)
        return DerivedRule(name, params, body, self._loc(section))

    # ===========================================================================
    # problem sections
    # ===========================================================================

    def _parse_problem_domain(self, section, problem):
        if len(section) != 2 or not self._is_name(section[1]):
            raise self._err(section, 'expected (:domain <name>)')
        if section[1] != self.domain.name:
            raise self._err(section, f'problem refers to domain "{section[1]}", '
                                     f'but the domain is "{self.domain.name}"')
        problem.domain_name = str(section[1])

    def _parse_objects(self, section, problem):
        self._declare_objects(section[1:], problem.objects, self.domain.constants)
        self.objects.update(problem.objects)

    def _is_init_value(self, item) -> bool:
        return self._head(item) == '=' and len(item) == 3 \
            and (self._head(item[1]) in self.domain.functions or self._is_bare_fterm(item[1]))

    def _init_value(self, item):
        '''Parses (= (f o...) n) in :init into (FTerm, value).'''
        target = self.fterm(item[1], {})
        if not self._is_number(item[2]):
            raise self._err(item, f'expected a number, got {self._show(item[2])}')
        return (target, self._number(item[2]))

    def init_element(self, node):
        '''Parses one element of a (probabilistic ...) branch in :init: a ground
        literal, a numeric value (= (f ...) n), (and ...) or a nested
        (probabilistic ...).'''
        if isinstance(node, SList) and not node:
            return AndEffect((), self._loc(node))
        head = self._head(node)
        loc = self._loc(node) if isinstance(node, SList) else None
        if head is not None and head not in self.domain.predicates:
            if head == 'and':
                return AndEffect(tuple(self.init_element(e) for e in node[1:]), loc)
            if head == 'not':
                self.expect_len(node, 2)
                return Del(self.atom(node[1], {}), loc)
            if head == 'probabilistic':
                return self.probabilistic(node, {}, init=True)
            if self._is_init_value(node):
                (target, value) = self._init_value(node)
                return NumEffect('assign', target, Num(value, self._loc(node[2])), loc)
        return Add(self.atom(node, {}), loc)

    def _parse_init(self, section, problem):
        atoms, negated, numeric, effects = [], [], {}, []
        for item in section[1:]:
            head = self._head(item)
            if head == 'at' and len(item) == 3 and self._is_number(item[1]):
                raise self._unsupported(item, 'timed initial literals are not supported')
            if head == 'probabilistic' and head not in self.domain.predicates:
                effects.append(self.probabilistic(item, {}, init=True))
            elif head == 'not' and 'not' not in self.domain.predicates:
                # redundant under the closed-world assumption, but must be consistent
                self.expect_len(item, 2)
                negated.append((self.atom(item[1], {}), item))
            elif self._is_init_value(item):
                (target, value) = self._init_value(item)
                key = (target.name, tuple(a.name for a in target.args))
                if key in numeric:
                    raise self._err(item, f'duplicate initial value for {self._show(item[1])}')
                numeric[key] = value
            else:
                atoms.append(self.atom(item, {}))
        positive = set(atoms)
        for (atom, item) in negated:
            if atom in positive:
                raise self._err(
                    item, f'{self._show(item[1])} is declared both true and false in :init')
        self._check_init_assignments(effects, numeric)
        problem.init_atoms = tuple(atoms)
        problem.init_numeric = numeric
        problem.init_effects = tuple(effects)

    def _check_init_assignments(self, effects, numeric):
        '''Each numeric fluent assigned inside a probabilistic :init must have a value
        in every outcome: a deterministic value, or an assignment that happens with
        probability 1. Two assignments that can happen together must agree.'''

        # collect every assignment with its path of (block, branch index) choices
        found = {}

        def collect(effect, path):
            if isinstance(effect, AndEffect):
                for e in effect.effects:
                    collect(e, path)
            elif isinstance(effect, Probabilistic):
                for (i, (_, branch)) in enumerate(effect.branches):
                    collect(branch, path + ((id(effect), i),))
            elif isinstance(effect, NumEffect):
                key = (effect.target.name, tuple(a.name for a in effect.target.args))
                found.setdefault(key, []).append((dict(path), effect))

        for block in effects:
            collect(block, ())

        def assigns_surely(effect, key):
            if isinstance(effect, AndEffect):
                return any(assigns_surely(e, key) for e in effect.effects)
            if isinstance(effect, Probabilistic):
                total = sum(p.value for (p, _) in effect.branches)
                return total >= 1 - self.PROB_TOLERANCE and all(
                    assigns_surely(branch, key) for (_, branch) in effect.branches)
            if isinstance(effect, NumEffect):
                return (effect.target.name,
                        tuple(a.name for a in effect.target.args)) == key
            return False

        for (key, assignments) in found.items():
            for (i, (path1, eff1)) in enumerate(assignments):
                for (path2, eff2) in assignments[:i]:
                    exclusive = any(path1.get(block, branch) != branch
                                    for (block, branch) in path2.items())
                    if not exclusive and eff1.expr.value != eff2.expr.value:
                        raise self._err(
                            eff1.loc, f'{self._show_fterm(eff1.target)} can be initialized to '
                                  f'both {eff2.expr.value} and {eff1.expr.value} '
                                  f'in the same outcome of :init')
            if key not in numeric and not any(assigns_surely(b, key) for b in effects):
                eff = assignments[0][1]
                raise self._unsupported(
                    eff.loc, f'{self._show_fterm(eff.target)} is initialized only in some '
                         f'outcomes of a probabilistic :init and has no deterministic '
                         f'initial value')

    @staticmethod
    def _show_fterm(term) -> str:
        return '"(' + ' '.join((term.name,) + tuple(a.name for a in term.args)) + ')"'

    def _parse_goal(self, section, problem):
        if len(section) != 2:
            raise self._err(section, 'expected (:goal <formula>)')
        problem.goal = self.formula(section[1], {})

    def _parse_metric(self, section, problem):
        if len(section) != 3 or section[1] not in ('minimize', 'maximize'):
            raise self._err(section, 'expected (:metric minimize|maximize <expression>)')
        expr = section[2]
        if isinstance(expr, SList) and expr == ['goal-probability'] \
                and 'goal-probability' not in self.domain.functions:
            # PPDDL: the probability of reaching the goal, as a 0-ary function
            problem.metric = (str(section[1]), FTerm('goal-probability', (), self._loc(expr)))
            return
        problem.metric = (str(section[1]), self.num_term(expr, {}))

    def _parse_goal_reward(self, section, problem):
        if len(section) != 2 or not self._is_number(section[1]):
            raise self._err(section, 'expected (:goal-reward <number>)')
        problem.goal_reward = self._number(section[1])
