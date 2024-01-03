"""An example VI run."""

import argparse
import os

from pyRDDLGym.Core.Grounder.RDDLGrounder import RDDLGrounder
from pyRDDLGym.Core.Parser.parser import RDDLParser
from pyRDDLGym.Core.Parser.RDDLReader import RDDLReader
from pyRDDLGym.Examples.ExampleManager import ExampleManager
from pyRDDLGym.Solvers.SDP.helper import MDPParser, PolicyParser, Policy
from pyRDDLGym.Solvers.SDP.pe import PolicyEvaluation
from pyRDDLGym.XADD.RDDLModelXADD import RDDLModelWXADD


def run_pe(args: argparse.Namespace):
    """Runs PE."""
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

    mdp_parser = MDPParser()
    mdp = mdp_parser.parse(
        xadd_model,
        xadd_model.discount,
        concurrency=rddl_ast.instance.max_nondef_actions,
        is_linear=args.is_linear,
        is_vi=False,
    )

    policy_parser = PolicyParser()
    policy = policy_parser.parse(
        mdp=mdp,
        policy_fname=args.policy_fpath,
        assert_concurrency=args.assert_concurrency,
        concurrency=mdp.max_allowed_actions,
    )

    pe_solver = PolicyEvaluation(
        policy=policy,
        mdp=mdp,
        max_iter=args.max_iter,
        enable_early_convergence=args.enable_early_convergence,
        perform_reduce_lp=args.reduce_lp,
    )
    res = pe_solver.solve()

    # Export the solution to a file.
    env_path = env_info.path_to_env
    sol_dir = os.path.join(env_path, 'sdp', 'pe')
    os.makedirs(sol_dir, exist_ok=True)
    for i in range(args.max_iter):
        sol_fpath = os.path.join(sol_dir, f'value_dd_iter_{i+1}.xadd')
        value_dd = res['value_dd'][i]
        mdp.context.export_xadd(value_dd, fname=sol_fpath)

        # Visualize the solution XADD.
        if args.save_graph:
            graph_fpath = os.path.join(os.path.dirname(sol_fpath),
                                    f'value_dd_iter_{i+1}.pdf')
            mdp.context.save_graph(value_dd, file_name=graph_fpath)
    print(f'Times per iterations: {res["time"]}')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, default='RobotLinear_1D',
                        help='The name of the RDDL environment')
    parser.add_argument('--inst', type=str, default='0',
                        help='The instance number of the RDDL environment')
    parser.add_argument('--policy_fpath', type=str,
                        help='The file path to the policy')
    parser.add_argument('--max_iter', type=int, default=100,
                        help='The maximum number of iterations')
    parser.add_argument('--enable_early_convergence', action='store_true',
                        help='Whether to enable early convergence')
    parser.add_argument('--is_linear', action='store_true',
                        help='Whether the MDP is linear or not')
    parser.add_argument('--reduce_lp', action='store_true',
                        help='Whether to perform the reduce LP function.')
    parser.add_argument('--assert_concurrency', action='store_true',
                        help='Whether to assert concurrency or not')
    parser.add_argument('--save_graph', action='store_true',
                        help='Whether to save the XADD graph to a file')
    args = parser.parse_args()

    run_pe(args)
