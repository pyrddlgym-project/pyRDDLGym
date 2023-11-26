"""An example VI run."""

import argparse

from pyRDDLGym.Core.Grounder.RDDLGrounder import RDDLGrounder
from pyRDDLGym.Core.Parser.parser import RDDLParser
from pyRDDLGym.Core.Parser.RDDLReader import RDDLReader
from pyRDDLGym.Examples.ExampleManager import ExampleManager
from pyRDDLGym.Solvers.SDP.helper import MDP, Parser
from pyRDDLGym.Solvers.SDP.vi import ValueIteration
from pyRDDLGym.XADD.RDDLModelXADD import RDDLModelWXADD


def run_vi(args: argparse.Namespace):
    """Runs VI."""
    env_info = ExampleManager.GetEnvInfo(args.env)
    domain = env_info.get_domain()
    instance = env_info.get_instance(args.inst)

    # Read and parse domain and instance.
    reader = RDDLReader(domain, instance)
    domain = reader.rddltxt
    rddl_parser = RDDLParser(None, False)
    rddl_parser.build()

    # Parse RDDL file.
    rddl_ast = rddl_parser.parse(domain)

    # Ground domain.
    grounder = RDDLGrounder(rddl_ast)
    model = grounder.Ground()

    # XADD compilation.
    xadd_model = RDDLModelWXADD(model, simulation=False)
    xadd_model.compile()

    mdp_parser = Parser()
    mdp = mdp_parser.parse(
        xadd_model,
        xadd_model.discount,
        concurrency=rddl_ast.instance.max_nondef_actions,
        is_linear=args.is_linear,
    )

    vi = ValueIteration(
        mdp=mdp,
        max_iter=args.max_iter,
        enable_early_convergence=args.enable_early_convergence,
    )
    value_dd = vi.solve()

    # Export the solution to a file.

    # Visualize the solution XADD.


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, default='marsrover',
                        help='The name of the RDDL environment')
    parser.add_argument('--inst', type=str, default='1c',
                        help='The instance number of the RDDL environment')
    parser.add_argument('--max_iter', type=int, default=100,
                        help='The maximum number of iterations')
    parser.add_argument('--enable_early_convergence', action='store_true',
                        help='Whether to enable early convergence')
    parser.add_argument('--is_linear', action='store_true',
                        help='Whether the MDP is linear or not')
    args = parser.parse_args()

    run_vi(args)
