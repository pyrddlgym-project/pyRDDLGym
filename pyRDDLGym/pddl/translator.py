'''Translates the PDDL IR built by PDDLParser into lifted RDDL (PDDL_PLAN.md, step 2).

Generated names contain upper-case letters (IS-t, OBJ-c, DEFINED-f, Object). The
parser lower-cases every PDDL name, so generated names never collide with PDDL ones.
'''

import dataclasses
import math
import re
from typing import Dict, Optional, Tuple

from pyRDDLGym.core.builder import RDDLBuilder
from pyRDDLGym.core.debug.decompiler import RDDLDecompiler
from pyRDDLGym.core.parser.parser import RDDLlex
from pyRDDLGym.pddl.errors import PDDLUnsupportedError
from pyRDDLGym.pddl.parser import (
    Domain, Problem, Param, Var, Num, FTerm, Arith, Atom, Equals, Not, And, Or, Imply,
    Exists, Forall, Compare, Add, Del, NumEffect, When, Probabilistic,
)


class PDDLToRDDLTranslator:
    '''Translates a parsed PDDL domain and problem into RDDL through RDDLBuilder.'''

    RESERVED = frozenset(RDDLlex().reserved)
    OBJECT = 'Object'  # the RDDL type of a group that contains PDDL's "object"

    def __init__(self, domain: Domain, problem: Problem) -> None:
        self.domain = domain
        self.problem = problem
        self.builder = RDDLBuilder()
        self.objects = {**domain.constants, **problem.objects}  # object -> PDDL types
        self.names: Dict[Tuple[str, str], str] = {}  # (kind, PDDL name) -> RDDL name
        self._taken = {'var': set(), 'obj': set()}
        self.primed = False  # whether formulas read state-fluents on the next state
        self._check_supported()
        self._translate_names()
        self._translate_types()
        self._translate_fluents()

    @staticmethod
    def _unsupported(message, node=None):
        loc = getattr(node, 'loc', None)
        return PDDLUnsupportedError(message, getattr(loc, 'line', None),
                                    getattr(loc, 'col', None))

    def _check_supported(self) -> None:
        '''PPDDL is converted in step 3; until then it is rejected.'''
        for block in self.nodes_in(self.domain.actions, Probabilistic):
            raise self._unsupported('probabilistic effects are not supported', block)
        if self.problem.init_effects:
            raise self._unsupported('probabilistic initial states are not supported',
                                    self.problem.init_effects[0])
        if self.domain.has_rewards() or self.problem.goal_reward is not None:
            raise self._unsupported('PPDDL rewards are not supported')

    # ===========================================================================
    # IR helpers
    # ===========================================================================

    @staticmethod
    def nodes_in(node, cls):
        '''Yields every IR node of class cls inside node, recursively.'''
        if isinstance(node, cls):
            yield node
        if dataclasses.is_dataclass(node):
            for f in dataclasses.fields(node):
                yield from PDDLToRDDLTranslator.nodes_in(getattr(node, f.name), cls)
        elif isinstance(node, (tuple, list)):
            for item in node:
                yield from PDDLToRDDLTranslator.nodes_in(item, cls)
        elif isinstance(node, dict):
            for item in node.values():
                yield from PDDLToRDDLTranslator.nodes_in(item, cls)

    # ===========================================================================
    # names
    # ===========================================================================

    @staticmethod
    def mangle(name: str) -> str:
        '''Makes a lower-case PDDL name a valid RDDL identifier. "__" is also removed,
        because pyRDDLGym uses it to separate the objects of a grounded name.'''
        name = re.sub(r'[^a-z0-9_-]', '-', name)
        name = re.sub(r'_{2,}', '_', name)
        if not name[0].isalpha():
            name = 'n' + name
        if not name[-1].isalnum():
            name += '0'
        return name

    def _assign(self, kind: str, name: str) -> None:
        '''Gives (kind, name) a unique RDDL name. Types, predicates, functions and
        actions share one RDDL namespace and objects another. A name that clashes with
        an RDDL reserved word or an earlier name gets the kind as a prefix, and then
        a number.'''
        space = 'obj' if kind == 'obj' else 'var'
        base = candidate = self.mangle(name)
        n = 1
        while candidate in self.RESERVED or candidate in self._taken[space]:
            candidate = f'{kind}-{base}' if n == 1 else f'{kind}-{base}-{n}'
            n += 1
        self._taken[space].add(candidate)
        self.names[(kind, name)] = candidate

    def _translate_names(self) -> None:
        for (kind, names) in (('type', self.domain.types), ('pred', self.domain.predicates),
                              ('func', self.domain.functions), ('action', self.domain.actions),
                              ('obj', self.objects)):
            for name in names:
                self._assign(kind, name)

    @staticmethod
    def fresh(used: set, name: str) -> str:
        '''A new RDDL variable for the PDDL variable name, distinct from used.'''
        base = candidate = '?' + PDDLToRDDLTranslator.mangle(name.lstrip('?'))
        n = 2
        while candidate in used:
            candidate = f'{base}-{n}'
            n += 1
        used.add(candidate)
        return candidate

    # ===========================================================================
    # types
    # ===========================================================================

    def roots(self, t: str) -> frozenset:
        '''The top-level types of t (t itself when its parent is object).'''
        if t == 'object':
            return frozenset({'object'})
        result = set()
        for p in self.domain.types[t]:
            result |= {t} if p == 'object' else self.roots(p)
        return frozenset(result)

    def _translate_types(self) -> None:
        '''Groups PDDL's top-level types into RDDL object types with union-find: two
        groups merge when some type, parameter or object spans both. A group that
        contains "object" contains every type. A type narrower than its group gets a
        non-fluent membership guard IS-t.'''
        parent = {}

        def find(x):
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(types):
            roots = [r for t in types for r in sorted(self.roots(t))]
            for r in roots:
                parent[find(r)] = find(roots[0])

        for t in self.domain.types:
            union((t,))
        for p in self.nodes_in((self.domain.predicates, self.domain.functions,
                                self.domain.actions, self.domain.derived,
                                self.problem.goal, self.problem.metric), Param):
            union(p.types)
        for types in self.objects.values():
            union(types)
        if 'object' in parent:
            union(('object',) + tuple(self.domain.types))

        # name each group: Object if it contains "object", else after its first root
        group_roots = {}
        for t in parent:
            group_roots.setdefault(find(t), []).append(t)
        self.group = {}  # PDDL type -> RDDL type
        for roots in group_roots.values():
            if 'object' in roots:
                name = self.OBJECT
            else:
                name = self.names[('type', next(t for t in self.domain.types if t in roots))]
            for r in roots:
                self.group[r] = name
        for t in self.domain.types:
            self.group[t] = self.group[next(iter(self.roots(t)))]
        self.group_roots = {self.group[rep]: roots for (rep, roots) in group_roots.items()}

        # objects per RDDL type, in declaration order (constants first); RDDL has no
        # empty types, so an empty group gets the object NONE, which every guard excludes
        members = {name: [] for name in self.group_roots}
        for (obj, types) in self.objects.items():
            members[self.group[types[0]]].append(self.names[('obj', obj)])
        self.padded = set()
        for (name, objs) in members.items():
            if not objs:
                if 'object' in self.group_roots[name]:
                    raise PDDLUnsupportedError('the problem has no objects')
                self.padded.add(name)
                objs = ['NONE']
            self.builder.add_object_type(name)
            self.builder.add_object_values(name, objs)

        # membership guards
        for t in self.domain.types:
            if self.needs_guard(t):
                guard = 'IS-' + self.names[('type', t)]
                self.builder.add_pvariable(guard, [self.group[t]], 'non-fluent', 'bool', False)
                for obj in self.members((t,)):
                    self.builder.add_nonfluent_init(guard, [self.names[('obj', obj)]], True)

    def members(self, types: Tuple[str, ...]) -> list:
        '''The PDDL objects that have one of the types.'''
        return [o for (o, ot) in self.objects.items()
                if any(self.domain.is_subtype(a, t) for a in ot for t in types)]

    def needs_guard(self, t: str) -> bool:
        '''Whether the RDDL type of t also holds objects outside t.'''
        group = self.group[t]
        return t != 'object' and (self.group_roots[group] != [t] or group in self.padded)

    def rddl_type(self, types: Tuple[str, ...]) -> str:
        return self.group[types[0]]

    def guard(self, var: str, types: Tuple[str, ...]) -> Optional[str]:
        '''The RDDL condition that var has one of the PDDL types, or None if every
        object of its RDDL type does.'''
        if not all(map(self.needs_guard, types)):
            return None
        tests = [f'IS-{self.names[("type", t)]}({var})' for t in types]
        return tests[0] if len(tests) == 1 else '(' + ' | '.join(tests) + ')'

    def obj_test(self, var: str, obj: str) -> str:
        '''The RDDL condition var == obj. pyRDDLGym allows only enum literals in domain
        expressions, so each referenced object o gets a non-fluent OBJ-o that is true
        only for o.'''
        test = 'OBJ-' + self.names[('obj', obj)]
        if test not in self.builder.pvariable_defs:
            self.builder.add_pvariable(test, [self.rddl_type(self.objects[obj])],
                                       'non-fluent', 'bool', False)
            self.builder.add_nonfluent_init(test, [self.names[('obj', obj)]], True)
        return f'{test}({var})'

    # ===========================================================================
    # fluents
    # ===========================================================================

    def _translate_fluents(self) -> None:
        '''Predicates and functions that an effect changes become state-fluents, derived
        predicates derived-fluents, and everything else non-fluents. Functions in the
        metric that nothing reads become reward terms (step 5), not fluents. A function
        that the problem initializes only for some objects gets a non-fluent DEFINED-f,
        and every read of it is guarded by it.'''
        domain, problem = self.domain, self.problem
        derived = {r.name for r in domain.derived}
        changed_preds = {e.atom.pred for e in self.nodes_in(domain.actions, (Add, Del))}
        changed_funcs = {e.target.name for e in self.nodes_in(domain.actions, NumEffect)}
        assigned = {e.target.name for e in self.nodes_in(domain.actions, NumEffect)
                    if e.op == 'assign'}
        for e in self.nodes_in(domain.actions, (Add, Del)):
            if e.atom.pred in derived:
                raise self._unsupported(f'an effect changes the derived predicate '
                                        f'"{e.atom.pred}"', e)

        # functions read anywhere other than as the target of an effect
        reads = [problem.goal, [a.precondition for a in domain.actions.values()],
                 [r.body for r in domain.derived],
                 [w.condition for w in self.nodes_in(domain.actions, When)],
                 [e.expr for e in self.nodes_in(domain.actions, NumEffect)]]
        read_funcs = {t.name for t in self.nodes_in(reads, FTerm)}
        self.reward_funcs = {t.name for t in self.nodes_in(problem.metric, FTerm)} - read_funcs

        self.cpf_vars = {}  # RDDL state-fluent -> its CPF parameters
        for (pred, params) in domain.predicates.items():
            kind = 'state-fluent' if pred in changed_preds | derived else 'non-fluent'
            self._declare(self.names[('pred', pred)], params, kind, 'bool', False)

        defined = {}
        for (f, args) in problem.init_numeric:
            defined.setdefault(f, []).append(args)
        self.partial = set()
        for (func, params) in domain.functions.items():
            if func in self.reward_funcs:
                continue
            kind = 'state-fluent' if func in changed_funcs else 'non-fluent'
            name = self.names[('func', func)]
            self._declare(name, params, kind, 'real', 0.0)
            total = math.prod(len(self.members(p.types)) for p in params)
            if len(defined.get(func, ())) < total:
                # definedness is fixed unless an assign can define a value
                if func in assigned:
                    raise self._unsupported(
                        f'the numeric fluent "{func}" is assigned by effects but the problem '
                        f'does not initialize it for every object')
                self.partial.add(func)
                self._declare('DEFINED-' + name, params, 'non-fluent', 'bool', False)
                for args in defined.get(func, ()):
                    self._init('DEFINED-' + name, args, True)

        # initial values
        for atom in problem.init_atoms:
            if atom.pred in derived:
                raise self._unsupported(f'the derived predicate "{atom.pred}" is in :init', atom)
            self._init(self.names[('pred', atom.pred)], [a.name for a in atom.args], True)
        for ((func, args), value) in problem.init_numeric.items():
            if func not in self.reward_funcs:
                self._init(self.names[('func', func)], args, value)

        self._translate_derived()

    def _declare(self, name, params, kind, prange, default) -> None:
        self.builder.add_pvariable(name, [self.rddl_type(p.types) for p in params], kind,
                                   prange, default)
        if kind == 'state-fluent':  # unchanged unless an effect changes it (step 4)
            used = set()
            self.cpf_vars[name] = [self.fresh(used, p.name) for p in params]
            self.builder.add_cpf(name, self.cpf_vars[name], self.call(name, self.cpf_vars[name]))

    def _init(self, name, args, value) -> None:
        kind = self.builder.pvariable_defs[name][1]
        objs = [self.names[('obj', a)] for a in args]
        if kind == 'state-fluent':
            self.builder.add_init_state(name, objs, value)
        else:
            self.builder.add_nonfluent_init(name, objs, value)

    def _translate_derived(self) -> None:
        '''pyRDDLGym does not allow derived or interm fluents in preconditions or
        termination, so each derived predicate becomes a state-fluent that always holds
        the disjunction of its rules: its CPF evaluates the rules on the next state, and
        a lifted init-state entry evaluates them on the initial state. Recursive rules
        raise.'''
        rules = {}
        for rule in self.domain.derived:
            rules.setdefault(rule.name, []).append(rule)
        state = {}

        def visit(pred, rule):
            if state.get(pred) == 1:
                raise self._unsupported(f'the derived predicate "{pred}" is recursive', rule)
            if state.get(pred) is None:
                state[pred] = 1
                for r in rules[pred]:
                    for atom in self.nodes_in(r.body, Atom):
                        if atom.pred in rules:
                            visit(atom.pred, r)
                state[pred] = 2

        for (pred, rs) in rules.items():
            visit(pred, rs[0])
            name = self.names[('pred', pred)]
            cpf_vars = self.cpf_vars[name]
            declared = self.domain.predicates[pred]
            for primed in (False, True):
                self.primed = primed
                disjuncts = []
                for rule in rs:
                    scope = {p.name: v for (p, v) in zip(rule.params, cpf_vars)}
                    body = self.formula(rule.body, scope)
                    guards = [self.guard(v, p.types) for (p, d, v) in
                              zip(rule.params, declared, cpf_vars) if p.types != d.types]
                    guards = [g for g in guards if g is not None]
                    disjuncts.append(f'({" ^ ".join(guards + [body])})' if guards else body)
                if primed:
                    self.builder.add_cpf(name, cpf_vars, ' | '.join(disjuncts))
                else:
                    self.builder.add_init_state(name, cpf_vars, ' | '.join(disjuncts))
            self.primed = False

    # ===========================================================================
    # formulas and numeric expressions
    # ===========================================================================

    @staticmethod
    def call(name, args) -> str:
        return f'{name}({", ".join(args)})' if args else name

    def apply(self, name, args, scope, aggregation) -> str:
        '''name(args) for PDDL terms. RDDL domain expressions cannot name objects, so
        each object argument becomes a fresh variable fixed to the object by OBJ-o
        inside an aggregation: "exists" for booleans, "sum" for numbers.'''
        used = set(scope.values())
        terms, decls, tests = [], [], []
        for arg in args:
            if isinstance(arg, Var):
                terms.append(scope[arg.name])
            else:
                var = self.fresh(used, arg.name)
                terms.append(var)
                decls.append(f'{var} : {self.rddl_type(self.objects[arg.name])}')
                tests.append(self.obj_test(var, arg.name))
        if self.primed and self.builder.pvariable_defs[name][1] == 'state-fluent':
            name += "'"
        expr = self.call(name, terms)
        if not decls:
            return expr
        if aggregation == 'exists':
            return f'exists_{{{", ".join(decls)}}}[{" ^ ".join(tests + [expr])}]'
        return f'sum_{{{", ".join(decls)}}}[({" ^ ".join(tests)}) * {expr}]'

    def bind(self, scope, params):
        '''Binds quantified PDDL params to fresh RDDL variables; returns the inner scope,
        the RDDL declarations and the type guards.'''
        inner = dict(scope)
        used = set(scope.values())
        decls, guards = [], []
        for p in params:
            var = self.fresh(used, p.name)
            inner[p.name] = var
            decls.append(f'{var} : {self.rddl_type(p.types)}')
            guard = self.guard(var, p.types)
            if guard is not None:
                guards.append(guard)
        return (inner, decls, guards)

    def formula(self, node, scope) -> str:
        '''The RDDL boolean expression for a PDDL formula; scope maps PDDL variables to
        RDDL variables.'''
        if isinstance(node, Atom):
            return self.apply(self.names[('pred', node.pred)], node.args, scope, 'exists')
        if isinstance(node, Equals):
            terms = [scope[t.name] if isinstance(t, Var) else t.name
                     for t in (node.left, node.right)]
            (left, right) = (node.left, node.right)
            if isinstance(left, Var) and isinstance(right, Var):
                return f'({terms[0]} == {terms[1]})'
            if isinstance(left, Var):
                return self.obj_test(terms[0], right.name)
            if isinstance(right, Var):
                return self.obj_test(terms[1], left.name)
            return 'true' if left.name == right.name else 'false'
        if isinstance(node, Not):
            return f'~({self.formula(node.arg, scope)})'
        if isinstance(node, (And, Or)):
            if not node.args:
                return 'true' if isinstance(node, And) else 'false'
            op = ' ^ ' if isinstance(node, And) else ' | '
            return '(' + op.join(self.formula(a, scope) for a in node.args) + ')'
        if isinstance(node, Imply):
            return f'({self.formula(node.left, scope)} => {self.formula(node.right, scope)})'
        if isinstance(node, (Exists, Forall)):
            (inner, decls, guards) = self.bind(scope, node.params)
            body = self.formula(node.body, inner)
            if not decls:
                return body
            if isinstance(node, Exists):
                return f'exists_{{{", ".join(decls)}}}[{" ^ ".join(guards + [body])}]'
            if guards:
                body = f'({" ^ ".join(guards)}) => {body}'
            return f'forall_{{{", ".join(decls)}}}[{body}]'
        if isinstance(node, Compare):
            op = '==' if node.op == '=' else node.op
            test = f'({self.num(node.left, scope)} {op} {self.num(node.right, scope)})'
            return self.defined(node, scope, test)
        raise TypeError(f'not a formula: {node}')

    def defined(self, node, scope, expr) -> str:
        '''expr guarded by DEFINED-f for every partially initialized function in node.'''
        tests = [self.apply('DEFINED-' + self.names[('func', t.name)], t.args, scope, 'exists')
                 for t in self.nodes_in(node, FTerm) if t.name in self.partial]
        return f'({" ^ ".join(tests + [expr])})' if tests else expr

    def num(self, node, scope) -> str:
        '''The RDDL real expression for a PDDL numeric expression.'''
        if isinstance(node, Num):
            value = RDDLDecompiler._value_to_string(float(node.value))
            return f'({value})' if value.startswith('-') else value
        if isinstance(node, FTerm):
            return self.apply(self.names[('func', node.name)], node.args, scope, 'sum')
        if isinstance(node, Arith):
            args = [self.num(a, scope) for a in node.args]
            if len(args) == 1:
                return f'(-{args[0]})'
            return '(' + f' {node.op} '.join(args) + ')'
        raise TypeError(f'not a numeric expression: {node}')
