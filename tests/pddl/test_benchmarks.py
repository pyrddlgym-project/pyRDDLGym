'''Runs the parser over the local benchmark corpus in tests/pddl/benchmarks/.

The corpus is not committed (see PDDL_PLAN.md); these tests skip when it is missing.
Each case is a folder with problem.pddl and, unless the domain is inside
problem.pddl, domain.pddl.
'''

import os

import pytest

from pyRDDLGym.pddl.errors import PDDLParseError, PDDLUnsupportedError
from pyRDDLGym.pddl.parser import PDDLParser

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
def test_benchmark_parses_or_is_unsupported(case):
    folder = os.path.join(ROOT, case)
    with open(os.path.join(folder, 'problem.pddl'), encoding='latin-1') as f:
        problem = f.read()
    domain = problem
    if os.path.exists(os.path.join(folder, 'domain.pddl')):
        with open(os.path.join(folder, 'domain.pddl'), encoding='latin-1') as f:
            domain = f.read()
    try:
        PDDLParser.parse_problem(problem, PDDLParser.parse_domain(domain))
    except PDDLUnsupportedError:
        pass
