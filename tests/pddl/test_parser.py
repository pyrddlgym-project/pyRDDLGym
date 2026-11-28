import ast
import os
import sys

import pytest

from pyRDDLGym.pddl.errors import PDDLError, PDDLParseError, PDDLUnsupportedError
from pyRDDLGym.pddl.parser import (
    PDDLParser,
    Param, Var, Obj, Num, FTerm, Arith,
    Atom, Equals, Not, And, Or, Imply, Exists, Forall, Compare,
    Add, Del, NumEffect, AndEffect, ForallEffect, When, Probabilistic,
)

parse_domain = PDDLParser.parse_domain
parse_problem = PDDLParser.parse_problem


def line_of(text, marker):
    '''1-based line number of the first line containing marker.'''
    for (i, line) in enumerate(text.splitlines(), start=1):
        if marker in line:
            return i
    raise AssertionError(f'marker {marker!r} not found')


BLOCKS = '''
; a comment
(define (domain Blocks)
  (:requirements :strips :typing)
  (:types block)
  (:predicates (on ?x - block ?y - block) (clear ?x - block) (handempty)
               (holding ?x - block) (ontable ?x - block))
  (:action pick-up
    :parameters (?x - block)
    :precondition (and (clear ?x) (ontable ?x) (handempty))
    :effect (and (not (ontable ?x)) (not (clear ?x)) (not (handempty))
                 (holding ?x))))
'''

BLOCKS_PROBLEM = '''
(define (problem p1) (:domain blocks)
  (:objects a b - block)
  (:init (clear a) (clear b) (ontable a) (ontable b) (handempty))
  (:goal (and (on a b)))
  (:length (:serial 4)))
'''


def minimal_domain(body, requirements=':strips :typing', extra=''):
    return f'''(define (domain d)
  (:requirements {requirements})
  (:types t)
  (:constants c - t)
  (:predicates (p ?x - t) (q ?x - t ?y - t))
  (:functions (f ?x - t) (g))
  {extra}
  {body})'''


# =============================================================================
# lexer / reader
# =============================================================================

def test_case_insensitive_and_comments():
    dom = parse_domain(BLOCKS)
    assert dom.name == 'blocks'
    assert 'pick-up' in dom.actions
    assert dom.requirements == frozenset({':strips', ':typing'})


def test_unbalanced_parenthesis_reports_opening_line():
    text = '(define (domain d)\n  (:predicates (p)\n'
    with pytest.raises(PDDLParseError) as e:
        parse_domain(text)
    assert e.value.line == 2


def test_trailing_tokens_rejected():
    with pytest.raises(PDDLParseError):
        parse_domain(BLOCKS + '\n(extra)')


def test_error_message_contains_location():
    text = minimal_domain('(:action a :parameters () :effect (r))')
    with pytest.raises(PDDLParseError) as e:
        parse_domain(text)
    assert f'line {e.value.line}' in str(e.value)


# =============================================================================
# domain structure
# =============================================================================

def test_strips_action_structure():
    act = parse_domain(BLOCKS).actions['pick-up']
    assert act.params == (Param('?x', ('block',)),)
    x = Var('?x')
    assert act.precondition == And((Atom('clear', (x,)), Atom('ontable', (x,)),
                                    Atom('handempty', ())))
    assert act.effect == AndEffect((Del(Atom('ontable', (x,))),
                                    Del(Atom('clear', (x,))),
                                    Del(Atom('handempty', ())),
                                    Add(Atom('holding', (x,)))))


def test_nodes_carry_locations():
    act = parse_domain(BLOCKS).actions['pick-up']
    assert act.loc.line == line_of(BLOCKS, ':action pick-up')
    assert act.precondition.args[0].loc.line == line_of(BLOCKS, ':precondition')


def test_type_hierarchy():
    text = '''(define (domain d) (:requirements :typing)
      (:types truck car - vehicle vehicle place - object pkg (either) )
      (:predicates (at ?v - vehicle ?p - place)))'''
    with pytest.raises(PDDLParseError):
        parse_domain(text)  # (either) as a type name is malformed

    text = '''(define (domain d) (:requirements :typing)
      (:types truck car - vehicle vehicle place - object pkg)
      (:predicates (at ?v - (either vehicle pkg) ?p - place)))'''
    dom = parse_domain(text)
    assert dom.types == {'truck': ('vehicle',), 'car': ('vehicle',),
                         'vehicle': ('object',), 'place': ('object',),
                         'pkg': ('object',)}
    assert dom.predicates['at'] == (Param('?v', ('vehicle', 'pkg')),
                                    Param('?p', ('place',)))
    assert dom.is_subtype('truck', 'object')
    assert dom.is_subtype('car', 'vehicle')
    assert not dom.is_subtype('vehicle', 'car')


def test_implicit_parent_type_is_declared():
    text = '''(define (domain d) (:requirements :typing)
      (:types truck - vehicle) (:predicates (at ?v - vehicle)))'''
    dom = parse_domain(text)
    assert dom.types == {'truck': ('vehicle',), 'vehicle': ('object',)}


def test_cyclic_types_rejected():
    text = '''(define (domain d) (:requirements :typing)
      (:types a - b b - a) (:predicates (p ?x - a)))'''
    with pytest.raises(PDDLParseError):
        parse_domain(text)


def test_untyped_parameters_default_to_object():
    text = '''(define (domain d) (:requirements :strips)
      (:predicates (p ?x) (q ?x ?y))
      (:action a :parameters (?x ?y) :precondition (q ?x ?y) :effect (p ?x)))'''
    dom = parse_domain(text)
    assert dom.predicates['q'] == (Param('?x', ('object',)), Param('?y', ('object',)))
    assert dom.actions['a'].params == (Param('?x', ('object',)), Param('?y', ('object',)))


def test_constants():
    dom = parse_domain(minimal_domain(''))
    assert dom.constants == {'c': ('t',)}


def test_functions_default_and_explicit_number():
    text = minimal_domain('', extra='').replace(
        '(:functions (f ?x - t) (g))', '(:functions (f ?x - t) - number (g) (total-cost) - number)')
    dom = parse_domain(text)
    assert dom.functions == {'f': (Param('?x', ('t',)),), 'g': (), 'total-cost': ()}


def test_object_valued_function_unsupported():
    text = minimal_domain('').replace('(:functions (f ?x - t) (g))',
                                      '(:functions (loc ?x - t) - t)')
    with pytest.raises(PDDLUnsupportedError) as e:
        parse_domain(text)
    assert e.value.line == line_of(text, ':functions')


def test_formulas():
    body = '''(:action a :parameters (?x - t)
      :precondition (and (or (p ?x) (not (p c)))
                         (imply (p ?x) (= ?x c))
                         (exists (?y - t) (q ?x ?y))
                         (forall (?y ?z - t) (not (q ?y ?z)))
                         (< (f ?x) (+ (g) 1 2))
                         (= (f ?x) (- (g)))
                         (>= (* 2 (f c)) (/ (g) 3.5)))
      :effect (and))'''
    act = parse_domain(minimal_domain(body)).actions['a']
    x, y, z, c = Var('?x'), Var('?y'), Var('?z'), Obj('c')
    assert act.precondition == And((
        Or((Atom('p', (x,)), Not(Atom('p', (c,))))),
        Imply(Atom('p', (x,)), Equals(x, c)),
        Exists((Param('?y', ('t',)),), Atom('q', (x, y))),
        Forall((Param('?y', ('t',)), Param('?z', ('t',))), Not(Atom('q', (y, z)))),
        Compare('<', FTerm('f', (x,)), Arith('+', (FTerm('g', ()), Num(1.0), Num(2.0)))),
        Compare('=', FTerm('f', (x,)), Arith('-', (FTerm('g', ()),))),
        Compare('>=', Arith('*', (Num(2.0), FTerm('f', (c,)))),
                Arith('/', (FTerm('g', ()), Num(3.5)))),
    ))
    assert act.effect == AndEffect(())


def test_empty_precondition_is_none():
    body = '(:action a :parameters (?x - t) :effect (p ?x))'
    act = parse_domain(minimal_domain(body)).actions['a']
    assert act.precondition is None
    assert act.effect == Add(Atom('p', (Var('?x'),)))


def test_effects():
    body = '''(:action a :parameters (?x - t)
      :effect (and (forall (?y - t) (when (q ?x ?y) (and (p ?y) (not (q ?x ?y)))))
                   (increase (f ?x) 1) (decrease (g) (f ?x))
                   (assign (f c) 0) (scale-up (g) 2) (scale-down (g) 2)))'''
    act = parse_domain(minimal_domain(body)).actions['a']
    x, y, c = Var('?x'), Var('?y'), Obj('c')
    assert act.effect == AndEffect((
        ForallEffect((Param('?y', ('t',)),),
                     When(Atom('q', (x, y)),
                          AndEffect((Add(Atom('p', (y,))), Del(Atom('q', (x, y))))))),
        NumEffect('increase', FTerm('f', (x,)), Num(1.0)),
        NumEffect('decrease', FTerm('g', ()), FTerm('f', (x,))),
        NumEffect('assign', FTerm('f', (c,)), Num(0.0)),
        NumEffect('scale-up', FTerm('g', ()), Num(2.0)),
        NumEffect('scale-down', FTerm('g', ()), Num(2.0)),
    ))


def test_derived_predicates():
    text = minimal_domain('', extra='''
      (:derived (p ?x - t) (exists (?y - t) (q ?x ?y)))
      (:derived (p ?x - t) (q ?x ?x))''')
    dom = parse_domain(text)
    assert len(dom.derived) == 2
    rule = dom.derived[0]
    assert rule.name == 'p'
    assert rule.params == (Param('?x', ('t',)),)
    assert rule.body == Exists((Param('?y', ('t',)),), Atom('q', (Var('?x'), Var('?y'))))


def test_derived_must_match_declaration():
    text = minimal_domain('', extra='(:derived (p ?x - t ?y - t) (q ?x ?y))')
    with pytest.raises(PDDLParseError):
        parse_domain(text)


# =============================================================================
# symbol checks
# =============================================================================

@pytest.mark.parametrize('body, marker', [
    ('(:action a :parameters (?x - t) :precondition (r ?x) :effect (p ?x))', '(r ?x)'),
    ('(:action a :parameters (?x - t) :precondition (p ?x ?x) :effect (p ?x))', '(p ?x ?x)'),
    ('(:action a :parameters (?x - t) :precondition (p ?y) :effect (p ?x))', '(p ?y)'),
    ('(:action a :parameters (?x - u) :effect (p ?x))', '?x - u'),
    ('(:action a :parameters (?x - t) :effect (p d))', '(p d)'),
    ('(:action a :parameters (?x - t) :effect (increase (h ?x) 1))', '(h ?x)'),
    ('(:action a :parameters (?x - t) :effect (increase (f) 1))', '(f)'),
    ('(:action a :parameters (?x - t ?x - t) :effect (p ?x))', ':parameters'),
    ('(:action a :parameters (?x - t) :effect (p ?x))\n(:action a :parameters (?y - t) :effect (p ?y))',
     '(:action a :parameters (?y - t)'),
    ('(:action a :parameters (?x - t) :precondition (forall (?y - t) (p ?z)) :effect (p ?x))', '(p ?z)'),
    ('(:action a :parameters (?x - t) :effect (when (p ?x) (sometimes ?x)))', '(sometimes ?x)'),
    ('(:action a :parameters (?x - t) :unknown (p ?x))', ':unknown'),
])
def test_symbol_errors(body, marker):
    text = minimal_domain(body)
    with pytest.raises(PDDLParseError) as e:
        parse_domain(text)
    assert e.value.line == line_of(text, marker.splitlines()[0])


def test_argument_type_mismatch():
    text = '''(define (domain d) (:requirements :typing)
      (:types a b)
      (:predicates (p ?x - a))
      (:action act :parameters (?y - b)
        :effect (p ?y)))'''
    with pytest.raises(PDDLParseError) as e:
        parse_domain(text)
    assert e.value.line == line_of(text, ':effect (p ?y)')


def test_subtype_argument_accepted():
    text = '''(define (domain d) (:requirements :typing)
      (:types a - b)
      (:predicates (p ?x - b))
      (:action act :parameters (?y - a) :effect (p ?y)))'''
    parse_domain(text)


def test_either_argument_must_fit():
    text = '''(define (domain d) (:requirements :typing)
      (:types a b c)
      (:predicates (p ?x - (either a b)))
      (:action ok :parameters (?y - a) :effect (p ?y))
      (:action bad :parameters (?y - (either a c)) :effect (p ?y)))'''
    with pytest.raises(PDDLParseError) as e:
        parse_domain(text)
    assert e.value.line == line_of(text, ':action bad')


def test_duplicate_predicate():
    text = '''(define (domain d) (:requirements :strips)
      (:predicates (p) (p)))'''
    with pytest.raises(PDDLParseError):
        parse_domain(text)


def test_unknown_domain_section():
    text = minimal_domain('(:axiom (p c))')
    with pytest.raises(PDDLParseError) as e:
        parse_domain(text)
    assert e.value.line == line_of(text, ':axiom')


# =============================================================================
# unsupported constructs are recognized
# =============================================================================

@pytest.mark.parametrize('body, marker', [
    ('(:durative-action a :parameters () :duration (= ?duration 1) :condition (and) :effect (and))',
     ':durative-action'),
    ('(:process pr :parameters () :precondition (and) :effect (and))', ':process'),
    ('(:event ev :parameters () :precondition (and) :effect (and))', ':event'),
    ('(:constraints (always (p c)))', ':constraints'),
    ('(:action a :parameters () :precondition (preference p1 (p c)) :effect (and))', 'preference'),
])
def test_unsupported_domain_constructs(body, marker):
    text = minimal_domain(body)
    with pytest.raises(PDDLUnsupportedError) as e:
        parse_domain(text)
    assert e.value.line == line_of(text, marker)
    assert isinstance(e.value, PDDLError)


# =============================================================================
# problems
# =============================================================================

def test_problem_structure():
    dom = parse_domain(BLOCKS)
    prob = parse_problem(BLOCKS_PROBLEM, dom)
    assert prob.name == 'p1'
    assert prob.domain_name == 'blocks'
    assert prob.objects == {'a': ('block',), 'b': ('block',)}
    a, b = Obj('a'), Obj('b')
    assert prob.init_atoms == (Atom('clear', (a,)), Atom('clear', (b,)),
                               Atom('ontable', (a,)), Atom('ontable', (b,)),
                               Atom('handempty', ()))
    assert prob.init_numeric == {}
    assert prob.goal == And((Atom('on', (a, b)),))
    assert prob.metric is None
    assert prob.goal_reward is None


def numeric_domain():
    return parse_domain(minimal_domain(
        '(:action a :parameters (?x - t) :effect (increase (f ?x) 1))'
    ).replace('(:functions (f ?x - t) (g))', '(:functions (f ?x - t) (g) (total-cost))'))


def test_problem_numeric_init_and_metric():
    text = '''(define (problem p) (:domain d)
      (:objects o1 o2 - t)
      (:init (p o1) (= (f o1) 3) (= (f c) -2.5) (= (total-cost) 0))
      (:goal (> (f o1) 5))
      (:metric minimize (total-cost)))'''
    prob = parse_problem(text, numeric_domain())
    assert prob.objects == {'o1': ('t',), 'o2': ('t',)}
    assert prob.init_atoms == (Atom('p', (Obj('o1'),)),)
    assert prob.init_numeric == {('f', ('o1',)): 3.0, ('f', ('c',)): -2.5,
                                 ('total-cost', ()): 0.0}
    assert prob.goal == Compare('>', FTerm('f', (Obj('o1'),)), Num(5.0))
    assert prob.metric == ('minimize', FTerm('total-cost', ()))


def test_negative_init_literals_are_redundant_and_ignored():
    # closed-world assumption: atoms not listed in :init are false anyway
    text = '''(define (problem p) (:domain d) (:objects o - t)
      (:init (not (p o)) (p c)) (:goal (p o)))'''
    prob = parse_problem(text, numeric_domain())
    assert prob.init_atoms == (Atom('p', (Obj('c'),)),)


def test_in_package_header_ignored():
    dom = parse_domain('(in-package "PDDL")\n' + BLOCKS)
    assert dom.name == 'blocks'
    prob = parse_problem('(in-package "PDDL")\n' + BLOCKS_PROBLEM, dom)
    assert prob.name == 'p1'


def test_only_one_definition_allowed():
    with pytest.raises(PDDLParseError) as e:
        parse_domain(BLOCKS + '\n(define (domain other))')
    assert e.value.line == line_of(BLOCKS + '\n(define (domain other))', 'domain other')


def test_predicate_declaration_may_repeat_variable_names():
    text = '''(define (domain d) (:requirements :strips)
      (:predicates (in ?obj ?obj)))'''
    dom = parse_domain(text)
    assert len(dom.predicates['in']) == 2


def test_pddl12_vars_field_unsupported():
    text = minimal_domain('''(:action a :parameters (?x - t)
      :vars (?y - t) :effect (q ?x ?y))''')
    with pytest.raises(PDDLUnsupportedError) as e:
        parse_domain(text)
    assert e.value.line == line_of(text, ':vars')


def test_problem_redeclared_constant_with_same_type_allowed():
    text = '''(define (problem p) (:domain d) (:objects c o - t)
      (:init) (:goal (p c)))'''
    prob = parse_problem(text, numeric_domain())
    assert prob.objects == {'o': ('t',)}


@pytest.mark.parametrize('text, marker, exc, message', [
    ('(define (problem p) (:domain other) (:objects o - t) (:init) (:goal (p o)))',
     ':domain other', PDDLParseError, 'refers to domain'),
    ('(define (problem p) (:domain d) (:objects o - t) (:init (p z)) (:goal (p o)))',
     '(p z)', PDDLParseError, 'undeclared object'),
    ('(define (problem p) (:domain d) (:objects o - t)\n(:init (p o) (not (p o))) (:goal (p o)))',
     '(not (p o))', PDDLParseError, 'both true and false'),
    ('(define (problem p) (:domain d) (:objects o - u) (:init) (:goal (p o)))',
     'o - u', PDDLParseError, 'undeclared type'),
    ('(define (problem p) (:domain d) (:objects o - t)\n(:init (= (f o) x)) (:goal (p o)))',
     '(= (f o) x)', PDDLParseError, 'expected a number'),
    ('(define (problem p) (:domain d) (:objects o o - t) (:init) (:goal (p o)))',
     ':objects', PDDLParseError, 'duplicate object'),
    ('(define (problem p) (:domain d) (:objects c - object) (:init) (:goal (p c)))',
     ':objects', PDDLParseError, 'different type'),
    ('(define (problem p) (:domain d) (:objects o - t)\n(:init (= (f o) 1) (= (f o) 2)) (:goal (p o)))',
     '(= (f o) 1)', PDDLParseError, 'duplicate initial value'),
    ('(define (problem p) (:domain d) (:objects o - t) (:init)\n(:goal (p o)) (:bogus))',
     ':bogus', PDDLParseError, 'unknown problem section'),
    ('(define (problem p) (:domain d) (:objects o - t) (:init))',
     'define', PDDLParseError, 'missing (:goal'),
    ('(define (problem p) (:domain d) (:objects o - t)\n(:init (at 10 (p o))) (:goal (p o)))',
     '(at 10', PDDLUnsupportedError, 'timed initial literals'),
    ('(define (problem p) (:domain d) (:objects o - t) (:init) (:goal (p o))\n(:constraints (always (p o))))',
     ':constraints', PDDLUnsupportedError, 'constraints'),
    ('(define (problem p) (:domain d) (:objects o - t) (:init)\n(:goal (preference g1 (p o))))',
     'preference', PDDLUnsupportedError, 'preferences'),
    ('(define (problem p) (:domain d) (:objects o - t) (:init) (:goal (p o))\n(:metric minimize (total-time)))',
     'total-time', PDDLUnsupportedError, 'total-time'),
])
def test_problem_errors(text, marker, exc, message):
    with pytest.raises(exc) as e:
        parse_problem(text, numeric_domain())
    assert e.value.line == line_of(text, marker)
    assert message in e.value.message


# =============================================================================
# PPDDL
# =============================================================================

PPDDL = '''(define (domain coin)
  (:requirements :typing :probabilistic-effects :rewards)
  (:types t)
  (:predicates (heads ?x - t) (tails ?x - t))
  (:action flip :parameters (?x - t)
    :effect (and (probabilistic 0.5 (heads ?x)
                                0.3 (and (tails ?x) (probabilistic 0.1 (not (heads ?x)))))
                 (increase (reward) 2))))
'''


def test_probabilistic_effects_and_reward():
    dom = parse_domain(PPDDL)
    assert dom.functions == {'reward': ()}
    x = Var('?x')
    assert dom.actions['flip'].effect == AndEffect((
        Probabilistic((
            (Num(0.5), Add(Atom('heads', (x,)))),
            (Num(0.3), AndEffect((Add(Atom('tails', (x,))),
                                  Probabilistic(((Num(0.1), Del(Atom('heads', (x,)))),))))),
        )),
        NumEffect('increase', FTerm('reward', ()), Num(2.0)),
    ))


def test_ppddl_problem_goal_reward():
    text = '''(define (problem p) (:domain coin) (:objects o - t) (:init)
      (:goal (heads o)) (:goal-reward 500) (:metric maximize (reward)))'''
    prob = parse_problem(text, parse_domain(PPDDL))
    assert prob.goal_reward == 500.0
    assert prob.metric == ('maximize', FTerm('reward', ()))


@pytest.mark.parametrize('effect', [
    '(probabilistic 0.7 (heads ?x) 0.4 (tails ?x))',
    '(probabilistic -0.1 (heads ?x))',
])
def test_invalid_probabilities(effect):
    text = PPDDL.replace('(increase (reward) 2)', effect)
    with pytest.raises(PDDLParseError) as e:
        parse_domain(text)
    assert e.value.line == line_of(text, effect)


def test_rational_probabilities():
    text = PPDDL.replace('(increase (reward) 2)',
                         '(probabilistic 1/3 (heads ?x) 1/3 (tails ?x) 1/3 (not (heads ?x)))')
    eff = parse_domain(text).actions['flip'].effect.effects[1]
    assert [p.value for (p, _) in eff.branches] == [1 / 3] * 3


def test_mdp_requirement_implies_rewards():
    text = PPDDL.replace(':probabilistic-effects :rewards', ':mdp')
    assert parse_domain(text).functions == {'reward': ()}


def test_goal_optional_when_domain_has_rewards():
    text = '''(define (problem p) (:domain coin) (:objects o - t) (:init)
      (:metric maximize (reward)))'''
    prob = parse_problem(text, parse_domain(PPDDL))
    assert prob.goal is None


def test_probabilistic_initial_state():
    text = '''(define (problem p) (:domain coin) (:objects o u - t)
      (:init (heads o)
             (probabilistic 0.5 (tails o)
                            0.25 (and (tails u) (not (heads o))
                                      (probabilistic 1/3 (heads u)))))
      (:goal (heads o)))'''
    prob = parse_problem(text, parse_domain(PPDDL))
    o, u = Obj('o'), Obj('u')
    assert prob.init_atoms == (Atom('heads', (o,)),)
    assert prob.init_effects == (Probabilistic((
        (Num(0.5), Add(Atom('tails', (o,)))),
        (Num(0.25), AndEffect((Add(Atom('tails', (u,))), Del(Atom('heads', (o,))),
                               Probabilistic(((Num(1 / 3), Add(Atom('heads', (u,)))),))))),
    )),)
    assert prob.init_effects[0].loc.line == line_of(text, '(probabilistic 0.5')


def test_probabilistic_initial_numeric_values():
    text = '''(define (problem p) (:domain d) (:objects o - t)
      (:init (= (f o) 1) (= (g) 0)
             (probabilistic 0.5 (= (f o) 3) 0.5 (and (= (f o) 4) (p o)))
             (probabilistic 0.2 (= (g) 7)))
      (:goal (p o)))'''
    prob = parse_problem(text, numeric_domain())
    assert prob.init_numeric == {('f', ('o',)): 1.0, ('g', ()): 0.0}
    (fo, g) = (FTerm('f', (Obj('o'),)), FTerm('g', ()))
    assert prob.init_effects == (
        Probabilistic(((Num(0.5), NumEffect('assign', fo, Num(3.0))),
                       (Num(0.5), AndEffect((NumEffect('assign', fo, Num(4.0)),
                                             Add(Atom('p', (Obj('o'),)))))))),
        Probabilistic(((Num(0.2), NumEffect('assign', g, Num(7.0))),)),
    )


@pytest.mark.parametrize('init', [
    '(probabilistic 0.5 (= (f o) 3) 0.5 (= (f o) 4))',            # always assigned
    '(probabilistic 1 (probabilistic 0.5 (= (f o) 3) 0.5 (= (f o) 3)))',
    '(probabilistic 0.5 (= (f o) 3)) (probabilistic 0.5 (= (f o) 3)) (= (f o) 0)',
])
def test_probabilistic_initial_values_accepted(init):
    text = f'''(define (problem p) (:domain d) (:objects o - t)
      (:init {init}) (:goal (p o)))'''
    parse_problem(text, numeric_domain())


@pytest.mark.parametrize('init, marker, exc, message', [
    ('(probabilistic 0.5 (p o))\n(probabilistic (g) (p o))', '(probabilistic (g)',
     PDDLParseError, 'must be numbers'),
    ('(probabilistic 0.7 (p o)\n0.4 (p c))', '(probabilistic 0.7', PDDLParseError,
     'exceeds 1'),
    ('(probabilistic 0.5\n(at 10 (p o)))', '(at 10', PDDLParseError, 'undeclared predicate'),
    ('(probabilistic 0.5 (and (p o)\n(= (f o) x)))', '(= (f o) x)', PDDLParseError,
     'expected a number'),
    ('(probabilistic 0.5\n(= (f o) 3))', '(= (f o) 3)', PDDLUnsupportedError,
     'no deterministic initial value'),
    ('(probabilistic 0.5\n(= (f o) 3) 0.4 (= (f o) 4))', '(= (f o) 3)', PDDLUnsupportedError,
     'no deterministic initial value'),
    ('(= (f o) 0) (probabilistic 0.5 (= (f o) 3))\n(probabilistic 0.5 (= (f o) 4))',
     '(probabilistic 0.5 (= (f o) 4))', PDDLParseError, 'both 3.0 and 4.0'),
    ('(= (f o) 0) (probabilistic 0.5 (and (= (f o) 3)\n(probabilistic 0.5 (= (f o) 4))))',
     '(probabilistic 0.5 (= (f o) 4))', PDDLParseError, 'both 3.0 and 4.0'),
])
def test_probabilistic_initial_state_errors(init, marker, exc, message):
    text = f'''(define (problem p) (:domain d) (:objects o - t)
      (:init {init}) (:goal (p o)))'''
    with pytest.raises(exc) as e:
        parse_problem(text, numeric_domain())
    assert e.value.line == line_of(text, marker)
    assert message in e.value.message


def test_bare_zero_ary_atom_rejected():
    text = PPDDL.replace('(increase (reward) 2)', '(probabilistic 0.5 heads)')
    with pytest.raises(PDDLParseError):
        parse_domain(text)


def test_reward_fluent_requires_rewards_requirement():
    text = PPDDL.replace(':rewards', '')
    with pytest.raises(PDDLParseError):
        parse_domain(text)


# =============================================================================
# packaging constraint: parser is stdlib-only
# =============================================================================

def test_parser_imports_only_stdlib():
    import pyRDDLGym.pddl as pkg
    root = os.path.dirname(pkg.__file__)
    for fname in ('parser.py', 'errors.py'):
        with open(os.path.join(root, fname), encoding='utf-8') as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level > 0:
                    names = [f'pyRDDLGym.pddl.{node.module or ""}']
                else:
                    names = [node.module]
            else:
                continue
            for name in names:
                top = name.split('.')[0]
                assert top in sys.stdlib_module_names or name == 'pyRDDLGym.pddl.errors', \
                    f'{fname} imports non-stdlib module {name}'
