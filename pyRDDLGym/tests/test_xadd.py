from pathlib import Path
from typing import Optional
from pyRDDLGym.Core.Grounder.RDDLGrounder import RDDLGrounder
from pyRDDLGym.XADD.RDDLModelXADD import RDDLModelWXADD
from pyRDDLGym.Core.Parser.RDDLReader import RDDLReader
from pyRDDLGym.Core.Parser.parser import RDDLParser
from pyRDDLGym.Examples.ExampleManager import ExampleManager


def test_xadd(
        env_name: str = 'wildfire',
        cpf: Optional[str] = None,
        save_graph: bool = False,
        simulation: bool = False,
):
    env_info = ExampleManager.GetEnvInfo(env_name)
    domain = env_info.get_domain()
    instance = env_info.get_instance(0)
    
    # Read and parse domain and instance
    reader = RDDLReader(domain, instance)
    domain = reader.rddltxt
    parser = RDDLParser(None, False)
    parser.build()

    # Parse RDDL file
    rddl_ast = parser.parse(domain)

    # Ground domain
    grounder = RDDLGrounder(rddl_ast)
    model = grounder.Ground()

    # XADD compilation
    xadd_model = RDDLModelWXADD(model, simulation=simulation)
    xadd_model.compile()
    context = xadd_model._context
    
    if save_graph:
        Path(f'tmp/{env_name}').mkdir(parents=True, exist_ok=True)
        
    if cpf is not None or cpf == 'reward':
        if cpf == 'reward':
            expr = xadd_model.reward
        else:
            expr = xadd_model.cpfs.get(f"{cpf}'")
        if expr is None:
            raise AttributeError(f"Cannot retrieve {cpf}' from 'model.cpfs'")
        print(f"cpf {cpf}':", end='\n')
        xadd_model.print(expr)
        if save_graph:
            f_path = f"{env_name}/{env_name}_inst0_{cpf}"
            context.save_graph(expr, f_path)
    else:
        for cpf_, expr in xadd_model.cpfs.items():
            print(f"cpf {cpf_}:", end='\n')
            xadd_model.print(expr)
            if save_graph:
                cpf = cpf_.strip("'")
                f_path = f"{env_name}/{env_name}_inst0_{cpf}"
                context.save_graph(expr, f_path)
    # xadd_model.print(xadd_model.reward)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, default='wildfire',
                        help='The name of the RDDL environment')
    parser.add_argument('--cpf', type=str, default=None,
                        help='If specified, only print out this CPF')
    parser.add_argument('--save_graph', action='store_true',
                        help='Save the graph as pdf file')
    parser.add_argument('--simulation', action='store_true',
                        help='Use simulation mode for XADD compilation')
    args = parser.parse_args()
    test_xadd(env_name=args.env,
              cpf=args.cpf,
              save_graph=args.save_graph,
              simulation=args.simulation)
