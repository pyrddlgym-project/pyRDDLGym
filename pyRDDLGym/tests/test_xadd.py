from pyRDDLGym.Core.Grounder.RDDLGrounder import RDDLGrounder
from pyRDDLGym.XADD.RDDLModelXADD import RDDLModelWXADD
from pyRDDLGym.Core.Parser.RDDLReader import RDDLReader
from pyRDDLGym.Core.Parser.parser import RDDLParser
from pyRDDLGym.Examples.ExampleManager import ExampleManager


def test_xadd(env_name: str = 'Wildfire'):
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
    xadd_model = RDDLModelWXADD(model)
    xadd_model.compile()

    for cpf, expr in xadd_model.cpfs.items():
        print(f"cpf {cpf}:", end='\n')
        xadd_model.print(expr)
    
    # xadd_model.print(xadd_model.reward)


if __name__ == "__main__":
    test_xadd()
