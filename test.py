import RDDLGenerator
from Parser import parser2 as parser
from Parser import RDDLReader as RDDLReader
import sys

# DOMAIN = 'power_unit_commitment.rddl'

# DOMAIN = 'ThiagosReservoir.rddl'
# DOMAIN = 'Thiagos_Mars_Rover.rddl'
DOMAIN = 'Thiagos_HVAC.rddl'


def main():

    MyReader = RDDLReader.RDDLReader('RDDL/'+DOMAIN)
    # MyReader = RDDLReader.RDDLReader('RDDL/power_unit_commitment_domain.rddl',
    #                                  'RDDL/power_unit_commitment_instance.rddl')
    domain = MyReader.rddltxt

    MyLexer = parser.RDDLlex()
    MyLexer.build()
    MyLexer.input(domain)
    # [token for token in MyLexer._lexer]


    # build parser
    MyRDDLParser = parser.RDDLParser(None, False)
    MyRDDLParser.build()

    # parse RDDL file
    rddl_ast = MyRDDLParser.parse(domain)

    generator = RDDLGenerator.RDDLGenerator(rddl_ast)
    rddl = generator.GenerateRDDL()
    print(rddl)

    # print(rddl_ast.domain.cpfs[1])
    # print("e")



if __name__ == "__main__":
    main()



