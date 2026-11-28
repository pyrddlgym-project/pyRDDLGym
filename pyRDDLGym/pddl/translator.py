'''Translates the PDDL and PPDDL IR built by PDDLParser into lifted RDDL (PDDL_PLAN.md,
steps 2 and 3).

Generated names contain upper-case letters (IS-t, OBJ-c, DEFINED-f, Object, OUT-a-n,
INIT-OUT-n, Outcome). The parser lower-cases every PDDL name, so generated names never
collide with PDDL ones.
'''

import dataclasses
import math
import os
import re
from collections import Counter
from typing import Dict, Optional, Tuple

from pyRDDLGym.core.builder import RDDLBuilder
from pyRDDLGym.core.debug.decompiler import RDDLDecompiler
from pyRDDLGym.core.parser.parser import RDDLlex
from pyRDDLGym.pddl.errors import PDDLUnsupportedError
from pyRDDLGym.pddl.parser import (
    PDDLParser, Domain, Problem, Param, Var, Num, FTerm, Arith, Atom, Equals, Not, And, Or, Imply,
    Exists, Forall, Compare, Add, Del, NumEffect, AndEffect, ForallEffect, When, Probabilistic,
)


class PDDLToRDDLTranslator:
    '''Translates a parsed PDDL domain and problem into RDDL through RDDLBuilder.'''

    RESERVED = frozenset(RDDLlex().reserved)
    OBJECT = 'Object'  # the RDDL type of a group that contains PDDL's "object"

    def __init__(self, domain: Domain, problem: Problem, noop_cost: float=1.0) -> None:
        self.domain = domain
        self.problem = problem
        self.builder = RDDLBuilder()
        self.objects = {**domain.constants, **problem.objects}  # object -> PDDL types
        self.names: Dict[Tuple[str, str], str] = {}  # (kind, PDDL name) -> RDDL name
        self._taken = {'var': set(), 'obj': set()}
        self.primed = False  # whether formulas read state-fluents on the next state
        self._translate_names()
        self._translate_types()
        self._translate_fluents()
        self._translate_actions()
        self.occurrences = self._occurrences()
        self._translate_effects()
        self._translate_goal_and_reward(noop_cost)

    @staticmethod
    def _unsupported(message, node=None):
        loc = getattr(node, 'loc', None)
        return PDDLUnsupportedError(message, getattr(loc, 'line', None),
                                    getattr(loc, 'col', None))

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

        # outcomes of probabilistic blocks with several branches; @O0 is "no effect"
        blocks = self.nodes_in((self.domain.actions, self.problem.init_effects), Probabilistic)
        self.outcomes = max((len(b.branches) for b in blocks), default=0)
        if self.outcomes > 1:
            self.builder.add_enum_type('Outcome', [f'@O{i}' for i in range(self.outcomes + 1)])

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
        '''Predicates and functions that an effect or a random :init changes become
        state-fluents, and so do derived predicates; everything else is a non-fluent.
        Functions in the metric that nothing reads become reward terms, not fluents. A function
        that the problem initializes only for some objects gets a non-fluent DEFINED-f,
        and every read of it is guarded by it.'''
        domain, problem = self.domain, self.problem
        derived = {r.name for r in domain.derived}
        # a random :init needs init-state expressions, so what it sets is a state-fluent
        changers = (domain.actions, problem.init_effects)
        changed_preds = {e.atom.pred for e in self.nodes_in(changers, (Add, Del))}
        changed_funcs = {e.target.name for e in self.nodes_in(changers, NumEffect)}
        assigned = {}  # function -> its first assign effect
        for e in self.nodes_in(domain.actions, NumEffect):
            if e.op == 'assign':
                assigned.setdefault(e.target.name, e)
        for e in self.nodes_in(domain.actions, (Add, Del)):
            if e.atom.pred in derived:
                raise self._unsupported(f'an effect changes the derived predicate '
                                        f'"{e.atom.pred}"', e)

        # functions read anywhere other than as the target of an effect
        reads = [problem.goal, [a.precondition for a in domain.actions.values()],
                 [r.body for r in domain.derived],
                 [w.condition for w in self.nodes_in(domain.actions, When)],
                 [e.expr for e in self.nodes_in(domain.actions, NumEffect)],
                 [p for b in self.nodes_in(domain.actions, Probabilistic)
                  for (p, _) in b.branches]]
        read_funcs = {t.name for t in self.nodes_in(reads, FTerm)}
        self.reward_funcs = {t.name for t in self.nodes_in(problem.metric, FTerm)} - read_funcs

        self.cpf_vars = {}  # RDDL state-fluent -> its CPF parameters
        for (pred, params) in domain.predicates.items():
            kind = 'state-fluent' if pred in changed_preds | derived else 'non-fluent'
            self._declare(self.names[('pred', pred)], params, kind, 'bool', False)

        # the parser ensures a function set by a random :init has a value in every outcome
        defined = {}  # function -> {defined args: None}, ordered for deterministic output
        for (f, args) in problem.init_numeric:
            defined.setdefault(f, {})[args] = None
        for e in self.nodes_in(problem.init_effects, NumEffect):
            defined.setdefault(e.target.name, {})[tuple(a.name for a in e.target.args)] = None
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
                        f'does not initialize it for every object', assigned[func])
                self.partial.add(func)
                self._declare('DEFINED-' + name, params, 'non-fluent', 'bool', False)
                for args in defined.get(func, ()):
                    self._init('DEFINED-' + name, args, True)

        # initial values
        random_atoms = [e.atom for e in self.nodes_in(problem.init_effects, (Add, Del))]
        for atom in problem.init_atoms + tuple(random_atoms):
            if atom.pred in derived:
                raise self._unsupported(f'the derived predicate "{atom.pred}" is in :init', atom)
        for atom in problem.init_atoms:
            self._init(self.names[('pred', atom.pred)], [a.name for a in atom.args], True)
        for ((func, args), value) in problem.init_numeric.items():
            if func not in self.reward_funcs:
                self._init(self.names[('func', func)], args, value)
        self._translate_random_init()

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

    def _translate_random_init(self) -> None:
        '''A PPDDL (probabilistic ...) in :init is applied to the deterministic initial
        state as one simultaneous effect, as in mdpsim: an add beats a delete, and a
        branch may delete a deterministic atom. Each block gets an outcome state-fluent
        INIT-OUT-n sampled at reset, and each fluent it sets gets an init-state
        expression over those outcomes.'''
        changes = {}  # (kind, PDDL name, objects) -> [(effect, condition)]

        def walk(effect, conds):
            if isinstance(effect, AndEffect):
                for e in effect.effects:
                    walk(e, conds)
            elif isinstance(effect, Probabilistic):
                name = self._outcome(f'INIT-OUT-{len(blocks) + 1}', effect, (), init=True)
                blocks.append(name)
                for (i, (_, branch)) in enumerate(effect.branches):
                    walk(branch, conds + [self.outcome_test(name, i, ())])
            else:
                if isinstance(effect, NumEffect):
                    if effect.target.name in self.reward_funcs:
                        return
                    (kind, term) = ('func', effect.target)
                else:
                    (kind, term) = ('pred', effect.atom)
                key = (kind, term.pred if kind == 'pred' else term.name,
                       tuple(a.name for a in term.args))
                changes.setdefault(key, []).append((effect, ' ^ '.join(conds)))

        blocks = []
        for block in self.problem.init_effects:
            walk(block, [])

        atoms = {(a.pred, tuple(t.name for t in a.args)) for a in self.problem.init_atoms}
        for ((kind, pddl_name, args), effects) in changes.items():
            name = self.names[(kind, pddl_name)]
            if kind == 'pred':
                adds = [f'({c})' for (e, c) in effects if isinstance(e, Add)]
                dels = [f'({c})' for (e, c) in effects if isinstance(e, Del)]
                if (pddl_name, args) in atoms:
                    if not dels:
                        continue
                    adds.append(f'~({" | ".join(dels)})')
                if adds:
                    self._init(name, args, ' | '.join(adds))
            else:
                # the parser ensures that values which can be set together agree, and
                # that without a deterministic value some condition always holds
                cases = [(c, self.num(e.expr, {})) for (e, c) in effects]
                value = self.problem.init_numeric.get((pddl_name, args))
                expr = self.num(Num(value), {}) if value is not None else cases.pop()[1]
                for (c, v) in reversed(cases):
                    expr = f'if ({c}) then {v} else {expr}'
                self._init(name, args, expr)

    def _outcome(self, name, block, binders, init=False) -> str:
        '''Declares name to sample which branch of a probabilistic block happens: a bool
        Bernoulli for a single branch, else an Outcome Discrete in which @O0 means no
        branch. Inside an action (binders given), it is an interm-fluent with one value
        per grounding of the binders; in :init (init=True), a state-fluent sampled at
        reset.'''
        used = set()
        variables = [self.fresh(used, p.name) for p in binders]
        scope = {p.name: v for (p, v) in zip(binders, variables)}
        probs = [p for (p, _) in block.branches]
        for t in self.nodes_in(probs, FTerm):
            if t.name in self.partial:
                raise self._unsupported(f'a probability reads the partially initialized '
                                        f'function "{t.name}"', block)
        probs = [self.num(p, scope) for p in probs]
        if len(probs) == 1:
            (prange, default, expr) = ('bool', False, f'Bernoulli({probs[0]})')
        else:
            if all(isinstance(p, Num) for (p, _) in block.branches):
                rest = sum(p.value for (p, _) in block.branches)
                rest = self.num(Num(max(0.0, round(1.0 - rest, 12))), {})  # 1 - 0.7 = 0.3
            else:
                rest = f'(1.0 - ({" + ".join(probs)}))'
            cases = [rest] + probs + ['0.0'] * (self.outcomes - len(probs))
            (prange, default) = ('Outcome', '@O0')
            expr = 'Discrete(Outcome, ' + ', '.join(
                f'@O{i} : {p}' for (i, p) in enumerate(cases)) + ')'
        if init:
            self._declare(name, (), 'state-fluent', prange, default)
            self.builder.add_init_state(name, (), expr)
        else:
            self.builder.add_pvariable(name, [self.rddl_type(p.types) for p in binders],
                                       'interm-fluent', prange, None)
            self.builder.add_cpf(name, variables, expr)
        return name

    def outcome_test(self, name, i, args) -> str:
        '''The condition that branch i of the block sampled by name happens.'''
        call = self.call(name, args)
        return call if self.builder.pvariable_defs[name][2] == 'bool' else f'({call} == @O{i + 1})'

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
    # actions
    # ===========================================================================

    def _translate_actions(self) -> None:
        '''Each action schema becomes a bool action-fluent. At most one action is taken
        per step, and taking none is allowed. When an action is taken, its precondition
        and the type guards of its parameters must hold.'''
        self.action_scope = {}  # PDDL action -> (scope, RDDL declarations, call)
        for (act, action) in self.domain.actions.items():
            name = self.names[('action', act)]
            self.builder.add_pvariable(name, [self.rddl_type(p.types) for p in action.params],
                                       'action-fluent', 'bool', False)
            (scope, decls, guards) = self.bind({}, action.params)
            call = self.call(name, [scope[p.name] for p in action.params])
            self.action_scope[act] = (scope, decls, call)
            conds = list(guards)
            if action.precondition is not None:
                conds.append(self.formula(action.precondition, scope))
            if conds:
                rule = f'{call} => ({" ^ ".join(conds)})'
                self.builder.add_precondition(
                    f'(forall_{{{", ".join(decls)}}}[{rule}])' if decls else rule)
        self.builder.add_max_nondef_actions(1)

    # ===========================================================================
    # effects
    # ===========================================================================

    def _occurrences(self) -> list:
        '''Every Add, Del and NumEffect of every action, as (action, binders, conditions,
        effect). binders are the action parameters then the enclosing forall parameters;
        conditions are the enclosing (when ...) conditions as (number of binders in
        scope, formula) and the enclosing probabilistic branches as (number of binders
        in scope, (outcome fluent, branch index)). Declares the outcome fluents, so it
        runs once.'''
        result = []

        def walk(effect, act, binders, conds):
            if isinstance(effect, AndEffect):
                for e in effect.effects:
                    walk(e, act, binders, conds)
            elif isinstance(effect, ForallEffect):
                walk(effect.effect, act, binders + effect.params, conds)
            elif isinstance(effect, When):
                walk(effect.effect, act, binders, conds + ((len(binders), effect.condition),))
            elif isinstance(effect, Probabilistic):
                blocks.append(effect)
                name = self._outcome(f'OUT-{self.names[("action", act)]}-{len(blocks)}',
                                     effect, binders)
                for (i, (_, branch)) in enumerate(effect.branches):
                    walk(branch, act, binders, conds + ((len(binders), (name, i)),))
            else:
                result.append((act, binders, conds, effect))

        for (act, action) in self.domain.actions.items():
            blocks = []
            walk(action.effect, act, action.params, ())
        return result

    def _context(self, occ, cpf_vars=(), args=(), sampled=True):
        '''Returns (decls, conjuncts, scope) for an effect occurrence: the RDDL
        declarations of its free variables, the conditions under which it fires (the
        action, the type guards, the when conditions read on the current state and,
        if sampled, the probabilistic outcomes) and the scope for its expressions.
        When cpf_vars are given, args are the PDDL arguments of the changed fluent: a
        binder that occurs once in args is replaced by the CPF variable at its
        position, and the other positions are matched by equalities.'''
        (act, binders, conds, _) = occ
        innermost = {p.name: i for (i, p) in enumerate(binders)}
        counts = Counter(a.name for a in args if isinstance(a, Var))
        subst = {innermost[a.name]: v for (a, v) in zip(args, cpf_vars)
                 if isinstance(a, Var) and counts[a.name] == 1}
        used = set(cpf_vars)
        names = [subst[i] if i in subst else self.fresh(used, p.name)
                 for (i, p) in enumerate(binders)]

        # quantifiers inside conditions must avoid every variable of the context, so
        # the scopes also hold them under keys that are not PDDL variables
        hidden = {('#', v): v for v in list(cpf_vars) + names}
        scopes = [dict(hidden)]
        decls, conjuncts = [], []
        for (i, (p, var)) in enumerate(zip(binders, names)):
            scopes.append({**scopes[-1], p.name: var})
            if i not in subst:
                decls.append(f'{var} : {self.rddl_type(p.types)}')
            guard = self.guard(var, p.types)
            if guard is not None:
                conjuncts.append(guard)
        params = self.domain.actions[act].params
        action_scope = scopes[len(params)]
        conjuncts.insert(0, self.call(self.names[('action', act)],
                                      [action_scope[p.name] for p in params]))
        for (depth, c) in conds:
            if not isinstance(c, tuple):
                conjuncts.append(self.formula(c, scopes[depth]))
            elif sampled:
                conjuncts.append(self.outcome_test(c[0], c[1], names[:depth]))
        scope = scopes[-1]
        for (a, v) in zip(args, cpf_vars):
            if isinstance(a, Var):
                if scope[a.name] != v:
                    conjuncts.append(f'({v} == {scope[a.name]})')
            else:
                conjuncts.append(self.obj_test(v, a.name))
        return (decls, conjuncts, scope)

    @staticmethod
    def exists(decls, conjuncts) -> str:
        body = ' ^ '.join(conjuncts)
        return f'(exists_{{{", ".join(decls)}}}[{body}])' if decls else f'({body})'

    @staticmethod
    def fired_sum(decls, conjuncts, value) -> str:
        '''The sum of value over the groundings of decls where the conjuncts hold.'''
        cond = ' ^ '.join(conjuncts)
        if decls:
            return f'(sum_{{{", ".join(decls)}}}[({cond}) * {value}])'
        return f'(if ({cond}) then {value} else 0.0)'

    def _translate_effects(self) -> None:
        '''Builds each state-fluent's CPF from the effects that change it. For a boolean
        fluent, p' = adds | (p ^ ~deletes): an add wins over a delete, and every
        condition is read on the current state. For a numeric fluent,
        f' = if (assign) then value else f * scale-ups / scale-downs + increases -
        decreases. An assign therefore wins over other effects on the same fluent in
        the same step, which PDDL forbids anyway. Effects on the metric's reward-only
        functions are left to the reward.'''
        changes = {}
        for occ in self.occurrences:
            effect = occ[3]
            if isinstance(effect, NumEffect):
                if effect.target.name in self.reward_funcs:
                    continue
                name = self.names[('func', effect.target.name)]
            else:
                name = self.names[('pred', effect.atom.pred)]
            changes.setdefault(name, []).append(occ)
            self._require_defined(occ)

        for (name, occs) in changes.items():
            cpf_vars = self.cpf_vars[name]
            current = self.call(name, cpf_vars)
            terms = {'add': [], 'del': [], 'assign': [], 'increase': [], 'decrease': [],
                     'scale-up': [], 'scale-down': []}
            for occ in occs:
                effect = occ[3]
                if isinstance(effect, (Add, Del)):
                    (decls, conj, _) = self._context(occ, cpf_vars, effect.atom.args)
                    terms['add' if isinstance(effect, Add) else 'del'].append(
                        self.exists(decls, conj))
                    continue
                (decls, conj, scope) = self._context(occ, cpf_vars, effect.target.args)
                value = self.num(effect.expr, scope)
                cond = ' ^ '.join(conj)
                if effect.op == 'assign':
                    total = f'(sum_{{{", ".join(decls)}}}[({cond}) * {value}])' if decls \
                        else value
                    terms['assign'].append((self.exists(decls, conj), total))
                elif effect.op in ('increase', 'decrease'):
                    terms[effect.op].append(self.fired_sum(decls, conj, value))
                else:
                    terms[effect.op].append(
                        f'(prod_{{{", ".join(decls)}}}[if ({cond}) then {value} else 1.0])'
                        if decls else f'(if ({cond}) then {value} else 1.0)')

            if self.builder.pvariable_defs[name][2] == 'bool':
                expr = current
                if terms['del']:
                    expr = f'({current} ^ ~({" | ".join(terms["del"])}))'
                if terms['add']:
                    expr = f'{" | ".join(terms["add"])} | {expr}'
            else:
                expr = current
                for t in terms['scale-up']:
                    expr = f'({expr} * {t})'
                for t in terms['scale-down']:
                    expr = f'({expr} / {t})'
                for t in terms['increase']:
                    expr = f'({expr} + {t})'
                for t in terms['decrease']:
                    expr = f'({expr} - {t})'
                # ponytail: assumes at most one assign of a grounding fires per step, as
                # PDDL requires; several with different values would be summed
                for (cond, total) in reversed(terms['assign']):
                    expr = f'(if {cond} then {total} else {expr})'
            self.builder.add_cpf(name, cpf_vars, expr)

    def _require_defined(self, occ) -> None:
        '''PDDL cannot apply an effect that reads or changes an undefined numeric value,
        so for partially initialized functions the action's precondition requires
        DEFINED-f wherever the effect fires. Preconditions cannot read sampled outcomes,
        so DEFINED-f is required whatever the outcome.'''
        effect = occ[3]
        if not isinstance(effect, NumEffect):
            return
        fterms = [t for t in self.nodes_in((effect.target, effect.expr), FTerm)
                  if t.name in self.partial]
        if not fterms:
            return
        (decls, conj, scope) = self._context(occ, sampled=False)
        tests = [self.apply('DEFINED-' + self.names[('func', t.name)], t.args, scope, 'exists')
                 for t in fterms]
        rule = f'({" ^ ".join(conj)}) => ({" ^ ".join(tests)})'
        self.builder.add_precondition(f'(forall_{{{", ".join(decls)}}}[{rule}])'
                                      if decls else rule)

    # ===========================================================================
    # goal and reward
    # ===========================================================================

    def _translate_goal_and_reward(self, noop_cost: float) -> None:
        '''The goal becomes termination. The reward of a step is minus its cost, and
        noop_cost is charged when no action is taken. Without a metric each action
        costs 1. With (:metric minimize M), M must be a linear combination of
        reward-only functions (changed only by increase and decrease, and read by
        nothing), and the cost of a step is the change in M, so the total reward is
        minus the final value of M (plus its initial value); maximize flips the sign.
        PPDDL's goal reward and (goal-probability) are paid when the next state is a
        goal.'''
        if self.problem.goal is not None:
            self.builder.add_termination(self.formula(self.problem.goal, {}))

        taken = [self.exists(decls, [call]) for (_, decls, call) in self.action_scope.values()]
        metric = self.problem.metric
        if metric is None:
            terms = [f'(sum_{{{", ".join(decls)}}}[{call}])' if decls else call
                     for (_, decls, call) in self.action_scope.values()]
            reward = f'-({" + ".join(terms)})' if terms else '0'
        else:
            (sense, expr) = metric
            coefs = self._linear(expr)
            terms = []
            # PPDDL's (:goal-reward n) increases (reward) by n when the goal is reached,
            # and (goal-probability) is 1 once it is reached
            weight = coefs.get('reward', 0.0) * (self.problem.goal_reward or 0.0) + \
                coefs.get('goal-probability', 0.0)
            weight *= -1 if sense == 'minimize' else 1
            if weight and self.problem.goal is not None:
                self.primed = True
                goal = self.formula(self.problem.goal, {})
                self.primed = False
                terms.append(f'({self.num(Num(weight), {})} * {goal})')
            for occ in self.occurrences:
                effect = occ[3]
                if not (isinstance(effect, NumEffect) and effect.target.name in coefs):
                    continue
                if effect.op not in ('increase', 'decrease'):
                    raise self._unsupported(
                        f'the metric function "{effect.target.name}" is changed by '
                        f'{effect.op}; only increase and decrease are supported', effect)
                (decls, conj, scope) = self._context(occ)
                weight = coefs[effect.target.name] * (1 if effect.op == 'increase' else -1)
                weight *= -1 if sense == 'minimize' else 1
                total = self.fired_sum(decls, conj, self.num(effect.expr, scope))
                terms.append(f'({self.num(Num(weight), {})} * {total})')
            reward = ' + '.join(terms) if terms else '0'
        if noop_cost and taken:
            reward = f'{reward} - ({self.num(Num(noop_cost), {})} * ~({" | ".join(taken)}))'
        self.builder.add_reward(reward)

    def _linear(self, expr) -> Dict[str, float]:
        '''The coefficient of each function in a linear metric expression; constant
        terms do not change and are dropped.'''
        if isinstance(expr, Num):
            return {}
        if isinstance(expr, FTerm):
            if expr.name not in self.reward_funcs or expr.args:
                raise self._unsupported(
                    f'the metric uses "{expr.name}", which is read by the domain or has '
                    f'arguments; only functions that are only increased and decreased are '
                    f'supported', expr)
            return {expr.name: 1.0}
        if isinstance(expr, Arith):
            parts = [self._linear(a) for a in expr.args]
            if expr.op == '+' or (expr.op == '-' and len(parts) == 2):
                result = dict(parts[0])
                for part in parts[1:]:
                    sign = -1 if expr.op == '-' else 1
                    for (f, c) in part.items():
                        result[f] = result.get(f, 0.0) + sign * c
                return result
            if expr.op == '-':
                return {f: -c for (f, c) in parts[0].items()}
            if expr.op == '*' and sum(map(bool, parts)) <= 1:
                factor = math.prod(self._constant(a) for (a, p) in zip(expr.args, parts) if not p)
                return {f: c * factor for part in parts for (f, c) in part.items()}
            if expr.op == '/' and not parts[1]:
                return {f: c / self._constant(expr.args[1]) for (f, c) in parts[0].items()}
        raise self._unsupported('the metric is not linear', expr)

    def _constant(self, expr) -> float:
        '''The value of a numeric expression without functions.'''
        if isinstance(expr, Num):
            return expr.value
        values = [self._constant(a) for a in expr.args]
        if len(values) == 1:
            return -values[0]
        if expr.op == '+':
            return sum(values)
        if expr.op == '*':
            return math.prod(values)
        if expr.op == '-':
            return values[0] - values[1]
        return values[0] / values[1]

    # ===========================================================================
    # entry point
    # ===========================================================================

    @staticmethod
    def translate(domain: str, problem: str, *, horizon: int,
                  noop_cost: float=1.0) -> Tuple[str, str]:
        '''Translates PDDL into lifted RDDL; returns (domain RDDL, instance RDDL).

        domain and problem may each be a file path or PDDL text, and may be the same
        when one file holds both. horizon is the number of decision steps; the
        discount is 1. noop_cost is charged on steps where no action is taken.'''
        if not (isinstance(horizon, int) and horizon >= 1):
            raise ValueError(f'horizon must be a positive integer, got {horizon!r}')
        texts = []
        for source in (domain, problem):
            if os.path.isfile(source):
                with open(source, encoding='utf-8', errors='replace') as f:
                    source = f.read()
            texts.append(source)
        dom = PDDLParser.parse_domain(texts[0])
        prob = PDDLParser.parse_problem(texts[1], dom)
        translator = PDDLToRDDLTranslator(dom, prob, noop_cost)
        builder = translator.builder
        builder.add_horizon(horizon)
        builder.add_discount(1.0)
        (dname, iname) = (PDDLToRDDLTranslator.block_name(n) for n in (dom.name, prob.name))
        return (builder.build_domain(dname), builder.build_instance(dname, iname, iname + '-nf'))

    @staticmethod
    def block_name(name: str) -> str:
        '''A valid RDDL name for a domain or instance block.'''
        name = PDDLToRDDLTranslator.mangle(name)
        return 'pddl-' + name if name in PDDLToRDDLTranslator.RESERVED else name

    # ===========================================================================
    # formulas and numeric expressions
    # ===========================================================================

    @staticmethod
    def call(name, args) -> str:
        return f'{name}({", ".join(args)})' if args else name

    # RDDL aggregations extend as far right as possible ("forall_{?x : t}[a] | b" is
    # "forall_{?x : t}([a] | b)"), so every aggregation emitted here is parenthesized

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
            return f'(exists_{{{", ".join(decls)}}}[{" ^ ".join(tests + [expr])}])'
        return f'(sum_{{{", ".join(decls)}}}[({" ^ ".join(tests)}) * {expr}])'

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
                return f'(exists_{{{", ".join(decls)}}}[{" ^ ".join(guards + [body])}])'
            if guards:
                body = f'({" ^ ".join(guards)}) => {body}'
            return f'(forall_{{{", ".join(decls)}}}[{body}])'
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
