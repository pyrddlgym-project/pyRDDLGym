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
                                 'IS-pkg(b1)': True}
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
                     'fuel': 'state-fluent', 'OBJ-home': 'non-fluent'}
    assert b.pvariable_defs['dist'] == (('loc', 'loc'), 'non-fluent', 'real', 0.0)
    assert b.init_states['at(b, x)'] is True and b.init_states['fuel'] == 10.0
    assert b.nonfluent_inits['road(x, y)'] is True
    assert b.nonfluent_inits['dist(x, y)'] == 2.0
    assert b.nonfluent_inits['DEFINED-dist(x, y)'] is True
    assert b.cpf_defs['at'] == (('?p', '?l'), 'at(?p, ?l)')


def test_derived_predicates_are_state_fluents_with_next_state_cpfs():
    b = translate_pddl(FLUENTS, FLUENTS_PROBLEM).builder
    near = ('exists_{?l : loc}[(at%s(?p, ?l) ^ '
            'exists_{?home : loc}[OBJ-home(?home) ^ road(?l, ?home)])]')
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
        'exists_{?b : pkg, ?home : loc}[OBJ-b(?b) ^ OBJ-home(?home) ^ at(?b, ?home)]')


def test_quantifier_guards_and_fresh_variables():
    domain = TYPED.replace('(big ?t - truck)', '(big ?t - truck) (any)').replace(
        '(:predicates', '(:predicates (q ?x - vehicle)') + ''
    domain = domain[:-1] + '''
      (:derived (any) (exists (?x - truck) (forall (?x - car) (and (q ?x) (= ?x ?x))))))'''
    tr = translate_pddl(domain, '''(define (problem p) (:domain d)
      (:objects t1 - truck c1 - car b1 - pkg x - place) (:init) (:goal (any)))''')
    assert tr.builder.init_states['any'] == (
        'exists_{?x : vehicle}[IS-truck(?x) ^ forall_{?x-2 : vehicle}[(IS-car(?x-2)) => '
        '(q(?x-2) ^ (?x-2 == ?x-2))]]')


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
    ('(at ?p ?b) (decrease', '(probabilistic 0.5 (at ?p ?b)) (decrease', 'probabilistic effects'),
])
def test_unsupported_fluents(old, new, message):
    with pytest.raises(PDDLUnsupportedError, match=message):
        translate_pddl(FLUENTS.replace(old, new), FLUENTS_PROBLEM)


def test_derived_atom_in_init_rejected():
    with pytest.raises(PDDLUnsupportedError, match='in :init'):
        translate_pddl(FLUENTS, FLUENTS_PROBLEM.replace('(at b x)', '(at b x) (near b)'))
