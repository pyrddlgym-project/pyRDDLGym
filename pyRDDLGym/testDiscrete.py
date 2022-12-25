from pyRDDLGym import ExampleManager
from pyRDDLGym.Core.Parser.RDDLReader import RDDLReader
from pyRDDLGym.Core.Parser.parser import RDDLParser

ENV = 'PropDBN'

def main():
    # get the environment info
    EnvInfo = ExampleManager.GetEnvInfo(ENV)

    # read and parse domain and instance
    reader = RDDLReader(EnvInfo.get_domain(), EnvInfo.get_instance(0))
    domain = reader.rddltxt

    # parse RDDL file
    parser = RDDLParser(lexer=None, verbose=False)
    parser.build()
    rddl = parser.parse(domain)
    print(rddl)






if __name__ == "__main__":
    main()
