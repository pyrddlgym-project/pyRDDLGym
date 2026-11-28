'''Translates the local benchmark corpus in tests/pddl/benchmarks/ and loads the RDDL.

The corpus is not committed (see PDDL_PLAN.md); these tests skip when it is missing.
Each case is a folder with problem.pddl and, unless the domain is inside
problem.pddl, domain.pddl. The slowest case, logistics00 adl-98-prob02, takes about 25
seconds.
'''

import os

import numpy as np
import pytest

from pyRDDLGym.core.env import RDDLEnv
from pyRDDLGym.pddl.errors import PDDLParseError, PDDLUnsupportedError
from pyRDDLGym.pddl.translator import PDDLToRDDLTranslator

ROOT = os.path.join(os.path.dirname(__file__), 'benchmarks')
CASES = sorted(os.path.relpath(d, ROOT).replace(os.sep, '/')
               for (d, _, files) in os.walk(ROOT) if 'problem.pddl' in files)

# malformed source files that the parser deliberately does not tolerate
MALFORMED = {
    'ippc2008/rectangle-tireworld/': 'an effect uses the atom "dead" without parentheses',
}


def _param(case):
    for (prefix, reason) in MALFORMED.items():
        if case.startswith(prefix):
            return pytest.param(case, marks=pytest.mark.xfail(
                reason=reason, raises=PDDLParseError, strict=True))
    return case


@pytest.mark.skipif(not CASES, reason='the local benchmark corpus is not present')
@pytest.mark.parametrize('case', [_param(c) for c in CASES] or ['none'])
def test_benchmark_translates_or_is_unsupported(case, tmp_path):
    '''Each case is rejected with PDDLUnsupportedError, or its RDDL loads in both modes
    and a step without actions keeps the state and costs noop_cost (1).'''
    folder = os.path.join(ROOT, case)
    problem = os.path.join(folder, 'problem.pddl')
    domain = os.path.join(folder, 'domain.pddl')
    if not os.path.exists(domain):
        domain = problem
    try:
        (domain_rddl, instance_rddl) = PDDLToRDDLTranslator.translate(domain, problem, horizon=20)
    except PDDLUnsupportedError:
        return
    (tmp_path / 'domain.rddl').write_text(domain_rddl)
    (tmp_path / 'instance.rddl').write_text(instance_rddl)
    for vectorized in (False, True):
        env = RDDLEnv(str(tmp_path / 'domain.rddl'), str(tmp_path / 'instance.rddl'),
                      vectorized=vectorized)
        (before, _) = env.reset(seed=0)
        (after, reward, *_) = env.step({})
        assert all(np.array_equal(before[k], after[k]) for k in before)
        assert reward == -1.0
