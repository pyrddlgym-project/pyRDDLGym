from collections import Counter

import numpy as np
import pytest

from pyRDDLGym.core.env import RDDLEnv
from pyRDDLGym.pddl.translator import PDDLToRDDLTranslator
from pyRDDLGym.pddl.errors import PDDLUnsupportedError
from pyRDDLGym.pddl.parser import PDDLParser


def translate_pddl(domain, problem):
    dom = PDDLParser.parse_domain(domain)
    return PDDLToRDDLTranslator(dom, PDDLParser.parse_problem(problem, dom))


def make_envs(tr, tmp_path):
    '''RDDLEnvs (non-vectorized, vectorized) for the compiled RDDL.'''
    b = tr.builder
    if b.reward_def is None:
        b.add_reward('0')
    b.add_horizon(10)
    b.add_discount(1.0)
    (tmp_path / 'd.rddl').write_text(b.build_domain('d'))
    (tmp_path / 'i.rddl').write_text(b.build_instance('d', 'i', 'nf'))
    return [RDDLEnv(str(tmp_path / 'd.rddl'), str(tmp_path / 'i.rddl'), vectorized=v)
            for v in (False, True)]


# =============================================================================
# names
# =============================================================================

@pytest.mark.parametrize('name, expected', [
    ('at', 'at'), ('a.b', 'a-b'), ('x__y', 'x_y'), ('x___y', 'x_y'),
    ('a-', 'a-0'), ('a_', 'a_0'), ('1st', 'n1st'), ('l-1-1', 'l-1-1'),
])
def test_mangle(name, expected):
    assert PDDLToRDDLTranslator.mangle(name) == expected


def test_reserved_words_and_namespace_collisions():
    domain = '''(define (domain d) (:requirements :typing) (:types real move)
      (:predicates (if ?x - real) (move ?x - move))
      (:action move :parameters (?x - move) :effect (move ?x)))'''
    problem = '''(define (problem p) (:domain d)
      (:objects true - real m.1 m-1 - move) (:init) (:goal (move m-1)))'''
    names = translate_pddl(domain, problem).names
    assert names[('type', 'real')] == 'type-real'
    assert names[('type', 'move')] == 'move'
    assert names[('pred', 'if')] == 'pred-if'
    assert names[('pred', 'move')] == 'pred-move'
    assert names[('action', 'move')] == 'action-move'
    assert names[('obj', 'true')] == 'obj-true'
    assert (names[('obj', 'm.1')], names[('obj', 'm-1')]) == ('m-1', 'obj-m-1')


# =============================================================================
# types and objects
# =============================================================================

TYPED = '''(define (domain d) (:requirements :typing)
  (:types truck car - vehicle vehicle place - object pkg)
  (:constants depot - place)
  (:predicates (at ?v - (either vehicle pkg) ?p - place) (big ?t - truck)))'''


def test_either_merges_groups_and_subtypes_get_guards():
    tr = translate_pddl(TYPED, '''(define (problem p) (:domain d)
      (:objects t1 - truck c1 - car b1 - pkg x - place) (:init) (:goal (big t1)))''')
    assert {t: tr.group[t] for t in ('truck', 'car', 'vehicle', 'pkg', 'place')} == {
        'truck': 'vehicle', 'car': 'vehicle', 'vehicle': 'vehicle', 'pkg': 'vehicle',
        'place': 'place'}
    b = tr.builder
    assert b.object_types == ['vehicle', 'place']
    assert b.object_values == {'vehicle': ['t1', 'c1', 'b1'], 'place': ['depot', 'x']}
    assert [p for p in b.pvariable_defs if p.startswith('IS-')] == [
        'IS-truck', 'IS-car', 'IS-vehicle', 'IS-pkg']
    assert b.pvariable_defs['IS-vehicle'] == (('vehicle',), 'non-fluent', 'bool', False)
    assert b.nonfluent_inits == {'IS-truck(t1)': True, 'IS-car(c1)': True,
                                 'IS-vehicle(t1)': True, 'IS-vehicle(c1)': True,
                                 'IS-pkg(b1)': True, 'OBJ-t1(t1)': True}  # goal object
    assert tr.guard('?v', ('vehicle', 'pkg')) == '(IS-vehicle(?v) | IS-pkg(?v))'
    assert tr.guard('?p', ('place',)) is None


def test_object_parameter_merges_everything():
    domain = TYPED.replace('(big ?t - truck)', '(big ?t - truck) (any ?o - object)')
    tr = translate_pddl(domain, '''(define (problem p) (:domain d)
      (:objects t1 - truck x - place) (:init) (:goal (big t1)))''')
    assert tr.builder.object_types == ['Object']
    assert tr.builder.object_values == {'Object': ['depot', 't1', 'x']}
    assert tr.guard('?o', ('object',)) is None
    assert tr.guard('?p', ('place',)) == 'IS-place(?p)'


def test_untyped_domain():
    domain = '''(define (domain d) (:requirements :strips)
      (:predicates (p ?x)) (:action a :parameters (?x) :effect (p ?x)))'''
    tr = translate_pddl(domain, '(define (problem q) (:domain d) (:objects a b) (:init) (:goal (p a)))')
    assert tr.builder.object_values == {'Object': ['a', 'b']}
    assert not any(p.startswith('IS-') for p in tr.builder.pvariable_defs)
    assert tr.guard('?x', ('object',)) is None


def test_empty_group_gets_guarded_dummy_object():
    tr = translate_pddl(TYPED, '''(define (problem p) (:domain d)
      (:objects x - place) (:init) (:goal (and)))''')
    assert tr.builder.object_values == {'vehicle': ['NONE'], 'place': ['depot', 'x']}
    assert tr.guard('?v', ('vehicle',)) == 'IS-vehicle(?v)'
    assert 'IS-vehicle(NONE)' not in tr.builder.nonfluent_inits


def test_no_objects_at_all():
    domain = '''(define (domain d) (:requirements :strips) (:predicates (p ?x)))'''
    with pytest.raises(PDDLUnsupportedError):
        translate_pddl(domain, '(define (problem q) (:domain d) (:init) (:goal (and)))')


# =============================================================================
# fluents
# =============================================================================

FLUENTS = '''(define (domain d) (:requirements :typing :fluents :derived-predicates)
  (:types loc pkg)
  (:constants home - loc)
  (:predicates (at ?p - pkg ?l - loc) (road ?a ?b - loc) (near ?p - pkg) (far ?p - pkg))
  (:functions (dist ?a ?b - loc) (fuel) (total-cost))
  (:derived (near ?p - pkg) (exists (?l - loc) (and (at ?p ?l) (road ?l home))))
  (:derived (far ?p - pkg) (not (near ?p)))
  (:action move :parameters (?p - pkg ?a ?b - loc)
    :precondition (and (at ?p ?a) (road ?a ?b) (> (fuel) (dist ?a ?b)))
    :effect (and (not (at ?p ?a)) (at ?p ?b) (decrease (fuel) (dist ?a ?b))
                 (increase (total-cost) 1))))'''

FLUENTS_PROBLEM = '''(define (problem p) (:domain d) (:objects x y - loc b - pkg)
  (:init (at b x) (road x y) (road y home) (= (fuel) 10) (= (dist x y) 2)
         (= (total-cost) 0))
  (:goal (at b home)) (:metric minimize (total-cost)))'''


def test_fluent_kinds_and_initial_values():
    b = translate_pddl(FLUENTS, FLUENTS_PROBLEM).builder
    kinds = {name: d[1] for (name, d) in b.pvariable_defs.items()}
    assert kinds == {'at': 'state-fluent', 'road': 'non-fluent', 'near': 'state-fluent',
                     'far': 'state-fluent', 'dist': 'non-fluent', 'DEFINED-dist': 'non-fluent',
                     'fuel': 'state-fluent', 'OBJ-home': 'non-fluent',
                     'move': 'action-fluent', 'OBJ-b': 'non-fluent'}
    assert b.pvariable_defs['dist'] == (('loc', 'loc'), 'non-fluent', 'real', 0.0)
    assert b.init_states['at(b, x)'] is True and b.init_states['fuel'] == 10.0
    assert b.nonfluent_inits['road(x, y)'] is True
    assert b.nonfluent_inits['dist(x, y)'] == 2.0
    assert b.nonfluent_inits['DEFINED-dist(x, y)'] is True


def test_derived_predicates_are_state_fluents_with_next_state_cpfs():
    b = translate_pddl(FLUENTS, FLUENTS_PROBLEM).builder
    near = ('(exists_{?l : loc}[(at%s(?p, ?l) ^ '
            '(exists_{?home : loc}[OBJ-home(?home) ^ road(?l, ?home)]))])')
    assert b.cpf_defs['near'] == (('?p',), near % "'")
    assert b.init_states['near(?p)'] == near % ''
    assert b.cpf_defs['far'] == (('?p',), "~(near'(?p))")
    assert b.init_states['far(?p)'] == '~(near(?p))'


def test_derived_values_in_env(tmp_path):
    tr = translate_pddl(FLUENTS, FLUENTS_PROBLEM)
    for env in make_envs(tr, tmp_path):
        env.reset(seed=0)
        subs = env.sampler.subs
        near = subs['near___b'] if 'near___b' in subs else subs['near'][0]
        far = subs['far___b'] if 'far___b' in subs else subs['far'][0]
        assert (bool(near), bool(far)) == (False, True)  # b is at x; road(x, home) is false


def test_formulas():
    tr = translate_pddl(FLUENTS, FLUENTS_PROBLEM)
    p = PDDLParser.parse_domain(FLUENTS)
    pre = p.actions['move'].precondition
    scope = {'?p': '?p', '?a': '?a', '?b': '?b'}
    assert tr.formula(pre, scope) == (
        '(at(?p, ?a) ^ road(?a, ?b) ^ (DEFINED-dist(?a, ?b) ^ (fuel > dist(?a, ?b))))')
    goal = translate_pddl(FLUENTS, FLUENTS_PROBLEM).problem.goal
    assert tr.formula(goal, {}) == (
        '(exists_{?b : pkg, ?home : loc}[OBJ-b(?b) ^ OBJ-home(?home) ^ at(?b, ?home)])')


def test_quantifier_guards_and_fresh_variables():
    domain = TYPED.replace('(big ?t - truck)', '(big ?t - truck) (any)').replace(
        '(:predicates', '(:predicates (q ?x - vehicle)') + ''
    domain = domain[:-1] + '''
      (:derived (any) (exists (?x - truck) (forall (?x - car) (and (q ?x) (= ?x ?x))))))'''
    tr = translate_pddl(domain, '''(define (problem p) (:domain d)
      (:objects t1 - truck c1 - car b1 - pkg x - place) (:init) (:goal (any)))''')
    assert tr.builder.init_states['any'] == (
        '(exists_{?x : vehicle}[IS-truck(?x) ^ (forall_{?x-2 : vehicle}[(IS-car(?x-2)) => '
        '(q(?x-2) ^ (?x-2 == ?x-2))])])')


def test_reward_only_functions_are_not_fluents():
    b = translate_pddl(FLUENTS, FLUENTS_PROBLEM).builder
    assert 'total-cost' not in b.pvariable_defs


def test_partially_initialized_fluents():
    # increased but only partly defined: allowed, definedness never changes
    text = FLUENTS.replace('(decrease (fuel) (dist ?a ?b))', '(increase (dist ?a ?b) 1)')
    b = translate_pddl(text, FLUENTS_PROBLEM).builder
    assert b.pvariable_defs['dist'][1] == 'state-fluent'
    assert b.init_states['dist(x, y)'] == 2.0
    assert b.nonfluent_inits['DEFINED-dist(x, y)'] is True
    # assigned and only partly defined: rejected
    text = FLUENTS.replace('(decrease (fuel) (dist ?a ?b))', '(assign (dist ?a ?b) 1)')
    with pytest.raises(PDDLUnsupportedError, match='assigned'):
        translate_pddl(text, FLUENTS_PROBLEM)


@pytest.mark.parametrize('old, new, message', [
    ('(:derived (far ?p - pkg) (not (near ?p)))',
     '(:derived (far ?p - pkg) (not (near ?p))) (:derived (near ?p - pkg) (far ?p))', 'recursive'),
    ('(at ?p ?b) (decrease', '(at ?p ?b) (near ?p) (decrease', 'changes the derived predicate'),
])
def test_unsupported_fluents(old, new, message):
    with pytest.raises(PDDLUnsupportedError, match=message):
        translate_pddl(FLUENTS.replace(old, new), FLUENTS_PROBLEM)


def test_derived_atom_in_init_rejected():
    with pytest.raises(PDDLUnsupportedError, match='in :init'):
        translate_pddl(FLUENTS, FLUENTS_PROBLEM.replace('(at b x)', '(at b x) (near b)'))


# =============================================================================
# actions and preconditions
# =============================================================================

def test_action_fluents_and_preconditions():
    b = translate_pddl(FLUENTS, FLUENTS_PROBLEM).builder
    assert b.pvariable_defs['move'] == (('pkg', 'loc', 'loc'), 'action-fluent', 'bool', False)
    assert b.precondition_defs == [
        '(forall_{?p : pkg, ?a : loc, ?b : loc}[move(?p, ?a, ?b) => '
        '((at(?p, ?a) ^ road(?a, ?b) ^ (DEFINED-dist(?a, ?b) ^ (fuel > dist(?a, ?b)))))])',
        '(forall_{?p : pkg, ?a : loc, ?b : loc}[(move(?p, ?a, ?b)) => (DEFINED-dist(?a, ?b))])']
    assert b.maxnondef == 1


def test_action_parameter_guards_and_no_precondition():
    domain = TYPED[:-1] + '''
      (:action drive :parameters (?t - truck ?p - place) :effect (at ?t ?p))
      (:action noop :parameters () :effect (and)))'''
    b = translate_pddl(domain, '''(define (problem p) (:domain d)
      (:objects t1 - truck c1 - car x - place) (:init) (:goal (at t1 x)))''').builder
    assert b.precondition_defs == [
        '(forall_{?t : vehicle, ?p : place}[drive(?t, ?p) => (IS-truck(?t))])']
    assert b.pvariable_defs['noop'] == ((), 'action-fluent', 'bool', False)


def test_preconditions_in_env(tmp_path):
    from pyRDDLGym.core.debug.exception import (
        RDDLActionPreconditionNotSatisfiedError, RDDLInvalidActionError)
    tr = translate_pddl(FLUENTS, FLUENTS_PROBLEM)
    (plain, vec) = make_envs(tr, tmp_path)
    plain.reset(seed=0)
    plain.step({})                              # no action is allowed
    plain.step({'move___b__x__y': True})        # applicable: at(b, x), road(x, y), 10 > 2
    with pytest.raises(RDDLActionPreconditionNotSatisfiedError):
        plain.step({'move___b__y__x': True})    # road(y, x) is false
    with pytest.raises(RDDLInvalidActionError):
        plain.step({'move___b__x__y': True, 'move___b__y__home': True})
    vec.reset(seed=0)
    move = vec.sampler.subs['move'].copy()
    loc = {o: i for (i, o) in enumerate(['home', 'x', 'y'])}
    move[0, loc['x'], loc['y']] = True
    vec.step({'move': move})


def test_quantifiers_do_not_swallow_what_follows(tmp_path):
    # RDDL reads "forall_{?p : t}[a] | b" as "forall_{?p : t}([a] | b)"
    domain = '''(define (domain d) (:requirements :typing :adl) (:types p)
      (:predicates (vip ?x - p) (served ?x - p) (done))
      (:action stop :parameters ()
        :precondition (or (forall (?x - p) (imply (vip ?x) (served ?x)))
                          (exists (?x - p) (served ?x)))
        :effect (done)))'''
    tr = translate_pddl(domain, '''(define (problem q) (:domain d) (:objects a b - p)
      (:init (vip a)) (:goal (done)))''')
    for env in make_envs(tr, tmp_path):
        env.reset(seed=0)
        assert env.sampler.check_action_preconditions(
            {'stop': True} if env.vectorized else {'stop': True}, silent=True) is False



# =============================================================================
# effects
# =============================================================================

def test_effect_cpfs():
    b = translate_pddl(FLUENTS, FLUENTS_PROBLEM).builder
    assert b.cpf_defs['at'] == (('?p', '?l'), (
        '(exists_{?a : loc}[move(?p, ?a, ?l)]) | '
        '(at(?p, ?l) ^ ~((exists_{?b : loc}[move(?p, ?l, ?b)])))'))
    assert b.cpf_defs['fuel'] == ((), (
        '(fuel - (sum_{?p : pkg, ?a : loc, ?b : loc}[(move(?p, ?a, ?b)) * dist(?a, ?b)]))'))
    assert 'road' not in b.cpf_defs


EFFECTS = '''(define (domain e) (:requirements :typing :adl :fluents :derived-predicates)
  (:types box - item item)
  (:constants k - item)
  (:predicates (on ?x - item) (link ?x ?y - item) (lit) (any-on))
  (:functions (count) (w ?x - item))
  (:derived (any-on) (exists (?x - item) (on ?x)))
  (:action flip :parameters (?x - item) :effect (and (on ?x) (not (on ?x))))
  (:action spread :parameters (?x - item)
    :effect (forall (?y - item) (when (link ?x ?y) (and (on ?y) (increase (count) 1)))))
  (:action self-link :parameters (?x - item) :effect (link ?x ?x))
  (:action light :parameters () :effect (and (lit) (on k)))
  (:action box-only :parameters (?b - box) :effect (not (on ?b)))
  (:action weigh :parameters (?x - item)
    :effect (and (assign (w ?x) (+ (w ?x) 4)) (scale-up (count) 2) (decrease (count) (w ?x))))
  (:action shift :parameters (?x ?y - item)
    :effect (and (when (on ?x) (on ?y)) (when (on ?x) (not (on ?x))))))'''

EFFECTS_PROBLEM = '''(define (problem q) (:domain e) (:objects a - item b - box)
  (:init (link a b) (link a k) (= (count) 0) (= (w a) 1) (= (w b) 1) (= (w k) 1))
  (:goal (lit)))'''


def grounded_state(env, obs):
    '''The observation as {grounded name: bool or float}, in either mode.'''
    if env.vectorized:
        obs = dict(env.model.ground_vars_with_values(obs))
    return {k: bool(v) if np.asarray(v).dtype == bool else
            float(v) if np.asarray(v).dtype.kind in 'iuf' else str(v) for (k, v) in obs.items()}


def vector_action(env, grounded):
    from pyRDDLGym.core.compiler.model import RDDLPlanningModel
    actions = {}
    for (key, value) in grounded.items():
        (var, objs) = RDDLPlanningModel.parse_grounded(key)
        array = actions.setdefault(var, np.zeros_like(env.sampler.subs[var]))  # all bool
        array[tuple(env.model.object_to_index[o] for o in objs)] = value
    return actions


def run(env, steps):
    states = [grounded_state(env, env.reset(seed=0)[0])]
    for action in steps:
        obs = env.step(vector_action(env, action) if env.vectorized else action)[0]
        states.append(grounded_state(env, obs))
    return states


def test_transitions(tmp_path):
    steps = [{'flip___a': True},          # add wins over delete
             {'spread___a': True},        # forall + when: on(b), on(k), count += 2
             {'self-link___b': True},     # repeated variable
             {'weigh___a': True},         # assign, scale-up and decrease read the old state
             {'shift___b__a': True},      # conditions read the current state
             {'shift___a__a': True},      # add beats delete on the same atom
             {'light': True}]             # constant argument
    results = [run(env, steps) for env in make_envs(translate_pddl(EFFECTS, EFFECTS_PROBLEM),
                                                    tmp_path)]
    assert results[0] == results[1]  # vectorized and non-vectorized agree
    s = results[0]
    assert (s[0]['on___a'], s[0]['any-on']) == (False, False)
    assert (s[1]['on___a'], s[1]['any-on']) == (True, True)
    assert (s[2]['on___b'], s[2]['on___k'], s[2]['count']) == (True, True, 2.0)
    assert (s[3]['link___b__b'], s[3]['link___a__a']) == (True, False)
    assert (s[4]['w___a'], s[4]['w___b'], s[4]['count']) == (5.0, 1.0, 2 * 2.0 - 1.0)
    assert (s[5]['on___a'], s[5]['on___b']) == (True, False)
    assert s[6]['on___a'] is True
    assert (s[7]['lit'], s[7]['on___k']) == (True, True)


def test_subtype_guard_on_action_parameter(tmp_path):
    from pyRDDLGym.core.debug.exception import RDDLActionPreconditionNotSatisfiedError
    (env, _) = make_envs(translate_pddl(EFFECTS, EFFECTS_PROBLEM), tmp_path)
    env.reset(seed=0)
    env.step({'box-only___b': True})
    with pytest.raises(RDDLActionPreconditionNotSatisfiedError):
        env.step({'box-only___a': True})


def test_effect_on_undefined_value_is_not_applicable(tmp_path):
    from pyRDDLGym.core.debug.exception import RDDLActionPreconditionNotSatisfiedError
    domain = FLUENTS.replace('(and (at ?p ?a) (road ?a ?b) (> (fuel) (dist ?a ?b)))',
                             '(and (at ?p ?a) (road ?a ?b))')
    (env, _) = make_envs(translate_pddl(domain, FLUENTS_PROBLEM), tmp_path)
    env.reset(seed=0)
    env.step({'move___b__x__y': True})  # dist(x, y) is defined
    with pytest.raises(RDDLActionPreconditionNotSatisfiedError):
        env.step({'move___b__y__home': True})  # dist(y, home) is not



# =============================================================================
# goal, reward and translate
# =============================================================================

def test_goal_becomes_termination():
    b = translate_pddl(FLUENTS, FLUENTS_PROBLEM).builder
    assert b.termination_defs == [
        '(exists_{?b : pkg, ?home : loc}[OBJ-b(?b) ^ OBJ-home(?home) ^ at(?b, ?home)])']


def test_reward_from_total_cost_and_noop():
    b = translate_pddl(FLUENTS, FLUENTS_PROBLEM).builder
    assert b.reward_def == (
        '((-1.0) * (sum_{?p : pkg, ?a : loc, ?b : loc}[(move(?p, ?a, ?b)) * 1.0])) - '
        '(1.0 * ~((exists_{?p : pkg, ?a : loc, ?b : loc}[move(?p, ?a, ?b)])))')


def test_reward_without_metric_counts_actions():
    problem = FLUENTS_PROBLEM.replace(' (:metric minimize (total-cost))', '')
    b = translate_pddl(FLUENTS, problem).builder
    assert b.reward_def.startswith('-((sum_{?p : pkg, ?a : loc, ?b : loc}[move(?p, ?a, ?b)]))')


def test_linear_metric():
    domain = FLUENTS.replace('(fuel) (total-cost)', '(fuel) (total-cost) (used)').replace(
        '(increase (total-cost) 1)', '(increase (total-cost) 1) (increase (used) (dist ?a ?b))')
    problem = FLUENTS_PROBLEM.replace(
        '(:metric minimize (total-cost))',
        '(:metric maximize (- 10 (+ (* 2 (total-cost)) (/ (used) 4))))')
    tr = translate_pddl(domain, problem)
    assert tr._linear(tr.problem.metric[1]) == {'total-cost': -2.0, 'used': -0.25}
    assert tr.builder.reward_def.startswith('((-2.0) * (sum_')


@pytest.mark.parametrize('metric, message', [
    ('minimize (fuel)', 'read by the domain'),
    ('minimize (* (total-cost) (total-cost))', 'not linear'),
])
def test_unsupported_metrics(metric, message):
    with pytest.raises(PDDLUnsupportedError, match=message):
        translate_pddl(FLUENTS, FLUENTS_PROBLEM.replace('(:metric minimize (total-cost))',
                                                        f'(:metric {metric})'))


def test_metric_function_must_only_increase():
    domain = FLUENTS.replace('(increase (total-cost) 1)', '(assign (total-cost) 1)')
    with pytest.raises(PDDLUnsupportedError, match='changed by assign'):
        translate_pddl(domain, FLUENTS_PROBLEM)


def test_translate_end_to_end(tmp_path):
    (dom_path, prob_path) = (tmp_path / 'domain.pddl', tmp_path / 'problem.pddl')
    dom_path.write_text(FLUENTS)
    prob_path.write_text(FLUENTS_PROBLEM)
    (domain, instance) = PDDLToRDDLTranslator.translate(str(dom_path), FLUENTS_PROBLEM, horizon=20)
    assert domain.startswith('domain d {') and 'instance p {' in instance
    assert 'horizon = 20;' in instance and 'discount = 1.0;' in instance
    (tmp_path / 'd.rddl').write_text(domain)
    (tmp_path / 'i.rddl').write_text(instance)
    for vec in (False, True):
        env = RDDLEnv(str(tmp_path / 'd.rddl'), str(tmp_path / 'i.rddl'), vectorized=vec)
        env.reset(seed=0)
        rewards, done = [], False
        for action in [{'move___b__x__y': True}, {}]:
            (_, r, done, _, _) = env.step(vector_action(env, action) if vec else action)
            rewards.append(r)
        assert rewards == [-1.0, -1.0] and not done  # one move, one no-op


def test_translate_reaches_goal_and_combined_file(tmp_path):
    problem = FLUENTS_PROBLEM.replace('(= (dist x y) 2)', '(= (dist x y) 2) (= (dist y home) 3)')
    (domain, instance) = PDDLToRDDLTranslator.translate(FLUENTS + problem, FLUENTS + problem,
                                                        horizon=5, noop_cost=0.0)
    (tmp_path / 'd.rddl').write_text(domain)
    (tmp_path / 'i.rddl').write_text(instance)
    for vec in (False, True):
        env = RDDLEnv(str(tmp_path / 'd.rddl'), str(tmp_path / 'i.rddl'), vectorized=vec)
        env.reset(seed=0)
        total, done = 0.0, False
        for action in [{'move___b__x__y': True}, {'move___b__y__home': True}]:
            (_, r, done, _, _) = env.step(vector_action(env, action) if vec else action)
            total += r
        assert (total, done) == (-2.0, True)  # plan cost 2, goal reached on the last step


def test_translate_validates_horizon():
    with pytest.raises(ValueError):
        PDDLToRDDLTranslator.translate(FLUENTS, FLUENTS_PROBLEM, horizon=0)


# =============================================================================
# rejections point at the construct
# =============================================================================

def line_of(text, marker):
    return next(i for (i, line) in enumerate(text.splitlines(), start=1) if marker in line)


@pytest.mark.parametrize('dom_old, dom_new, prob_old, prob_new, where, marker, message', [
    ('(at ?p ?b) (decrease', '(at ?p ?b)\n(probabilistic (/ 1 (dist ?a ?b)) (at ?p ?a)) (decrease',
     '', '', 'domain', '(probabilistic (/', 'partially initialized function'),
    ('(at ?p ?b) (decrease', '(at ?p ?b)\n(near ?p) (decrease', '', '',
     'domain', '(near ?p) (decrease', 'changes the derived predicate'),
    ('(:derived (far ?p - pkg) (not (near ?p)))',
     '(:derived (near ?p - pkg) (far ?p))\n(:derived (far ?p - pkg) (not (near ?p)))', '', '',
     'domain', '(:derived (far', 'recursive'),
    ('', '', '(at b x)', '(at b x)\n(near b)', 'problem', '(near b)', 'is in :init'),
    ('(decrease (fuel) (dist ?a ?b))', '\n(assign (dist ?a ?b) 1)', '', '',
     'domain', '(assign (dist', 'assigned by effects'),
    ('(increase (total-cost) 1)', '\n(assign (total-cost) 1)', '', '',
     'domain', '(assign (total-cost)', 'changed by assign'),
    ('', '', '(:metric minimize (total-cost))', '(:metric\nminimize (fuel))',
     'problem', 'minimize (fuel)', 'read by the domain'),
    ('', '', '(:metric minimize (total-cost))', '(:metric minimize\n(* (total-cost) (total-cost)))',
     'problem', '(* (total-cost)', 'not linear'),
])
def test_rejections_point_at_the_construct(dom_old, dom_new, prob_old, prob_new, where,
                                           marker, message):
    domain = FLUENTS.replace(dom_old, dom_new) if dom_old else FLUENTS
    problem = FLUENTS_PROBLEM.replace(prob_old, prob_new) if prob_old else FLUENTS_PROBLEM
    with pytest.raises(PDDLUnsupportedError, match=message) as e:
        translate_pddl(domain, problem)
    assert e.value.line == line_of(domain if where == 'domain' else problem, marker)


# =============================================================================
# PPDDL
# =============================================================================

PPDDL = """(define (domain r) (:requirements :typing :probabilistic-effects :rewards :fluents)
  (:types cell)
  (:constants c1 - cell)
  (:predicates (a ?c - cell) (b ?c - cell) (p) (done))
  (:functions (height ?c - cell))
  (:action roll :parameters (?c - cell)
    :effect (probabilistic 0.2 (and (a ?c) (b ?c)) 0.5 (and (a ?c) (increase (height ?c) 1))))
  (:action spread :parameters () :effect (forall (?c - cell) (probabilistic 0.5 (b ?c))))
  (:action toss :parameters () :effect (probabilistic (/ (height c1) 4) (p)))
  (:action finish :parameters () :effect (and (done) (increase (reward) 3))))"""

PPDDL_PROBLEM = """(define (problem q) (:domain r) (:objects c2 - cell)
  (:init (= (height c1) 2) (= (height c2) 0))
  (:goal (done)) (:goal-reward 10) (:metric maximize (reward)))"""

RANDOM_INIT_PROBLEM = """(define (problem q) (:domain r) (:objects c2 - cell)
  (:init (p) (= (height c1) 0) (= (height c2) 5)
    (probabilistic 0.3 (and (a c1) (b c1) (not (p)))
                   0.4 (probabilistic 0.5 (a c2)))
    (probabilistic 0.5 (and (a c1) (= (height c1) 2)) 0.5 (= (height c2) 7)))
  (:goal (done)))"""

CRITICAL = {1: 10.83, 3: 16.27, 2: 13.82, 5: 20.52}  # chi-square, p = 0.001, by df


def assert_frequencies(samples, expected):
    counts = Counter(samples)
    assert set(counts) <= set(expected), set(counts) - set(expected)
    n = len(samples)
    stat = sum((counts[k] - n * p) ** 2 / (n * p) for (k, p) in expected.items())
    assert stat < CRITICAL[len(expected) - 1], (counts, stat)


def test_outcome_fluents():
    b = translate_pddl(PPDDL, PPDDL_PROBLEM).builder
    assert b.enum_types == {'Outcome': ['@O0', '@O1', '@O2']}
    assert b.pvariable_defs['OUT-roll-1'] == (('cell',), 'interm-fluent', 'Outcome', None)
    assert b.cpf_defs['OUT-roll-1'] == (
        ('?c',), 'Discrete(Outcome, @O0 : 0.3, @O1 : 0.2, @O2 : 0.5)')
    assert b.cpf_defs['OUT-spread-1'] == (('?c',), 'Bernoulli(0.5)')
    assert b.cpf_defs['b'] == (('?c',), '(roll(?c) ^ (OUT-roll-1(?c) == @O1)) | '
                                        '(spread ^ OUT-spread-1(?c)) | b(?c)')


def test_probabilistic_effects_frequencies(tmp_path):
    tr = translate_pddl(PPDDL, PPDDL_PROBLEM)
    cases = [
        ({'roll___c1': True}, ('a___c1', 'b___c1', 'height___c1'),  # correlated effects
         {(True, True, 2.0): 0.2, (True, False, 3.0): 0.5, (False, False, 2.0): 0.3}),
        ({'spread': True}, ('b___c1', 'b___c2'),  # one sample per forall grounding
         {(x, y): 0.25 for x in (False, True) for y in (False, True)}),
        ({'toss': True}, ('p',), {(True,): 0.5, (False,): 0.5}),  # probability (height c1) / 4
    ]
    for env in make_envs(tr, tmp_path):
        for (action, keys, expected) in cases:
            samples = []
            for seed in range(2000):
                env.reset(seed=seed)
                obs = env.step(vector_action(env, action) if env.vectorized else action)[0]
                state = grounded_state(env, obs)
                samples.append(tuple(state[k] for k in keys))
            assert_frequencies(samples, expected)


def test_goal_reward(tmp_path):
    for env in make_envs(translate_pddl(PPDDL, PPDDL_PROBLEM), tmp_path):
        env.reset(seed=0)
        (_, r1, done1, *_) = env.step({})
        (_, r2, done2, *_) = env.step(vector_action(env, {'finish': True})
                                      if env.vectorized else {'finish': True})
        assert (r1, done1, r2, done2) == (-1.0, False, 13.0, True)  # 3 + goal reward 10


def test_goal_probability_metric():
    problem = PPDDL_PROBLEM.replace('(reward)', '(goal-probability)')
    b = translate_pddl(PPDDL, problem).builder
    assert b.reward_def.startswith("(1.0 * done') - ")  # (reward) is no longer counted


def test_random_init_expressions():
    b = translate_pddl(PPDDL, RANDOM_INIT_PROBLEM).builder
    assert b.init_states['INIT-OUT-1'] == 'Discrete(Outcome, @O0 : 0.3, @O1 : 0.3, @O2 : 0.4)'
    assert b.init_states['INIT-OUT-2'] == 'Bernoulli(0.5)'
    assert b.init_states['a(c1)'] == '((INIT-OUT-1 == @O1)) | ((INIT-OUT-3 == @O1))'
    assert b.init_states['a(c2)'] == '((INIT-OUT-1 == @O2) ^ INIT-OUT-2)'
    assert b.init_states['p'] == '~(((INIT-OUT-1 == @O1)))'
    assert b.init_states['height(c2)'] == 'if ((INIT-OUT-3 == @O2)) then 7.0 else 5.0'


def test_random_init_frequencies(tmp_path):
    keys = ('a___c1', 'b___c1', 'p', 'a___c2', 'height___c1', 'height___c2')
    expected = {  # block 1 x nested block x block 2, outcomes with equal states merged
        (True, True, False, False, 2.0, 5.0): 0.15,
        (True, True, False, False, 0.0, 7.0): 0.15,
        (True, False, True, True, 2.0, 5.0): 0.1,
        (False, False, True, True, 0.0, 7.0): 0.1,
        (True, False, True, False, 2.0, 5.0): 0.25,
        (False, False, True, False, 0.0, 7.0): 0.25,
    }
    for env in make_envs(translate_pddl(PPDDL, RANDOM_INIT_PROBLEM), tmp_path):
        samples = []
        for seed in range(3000):
            state = grounded_state(env, env.reset(seed=seed)[0])
            samples.append(tuple(state[k] for k in keys))
        assert_frequencies(samples, expected)
