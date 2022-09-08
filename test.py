import RDDLGenerator
from Parser import parser2 as parser
from Parser import RDDLReader as RDDLReader
from Parser import RDDLGrounder as RDDLGrounder
from xaddpy.Translator.Translator import XADDTranslator
import sys

# DOMAIN = 'power_unit_commitment.rddl'

# DOMAIN = 'ThiagosReservoir.rddl'
# DOMAIN = 'Thiagos_Mars_Rover.rddl'
#DOMAIN = 'Thiagos_HVAC.rddl'
# DOMAIN = 'dbn_prop.rddl'
DOMAIN = 'wildfire_mdp.rddl'

def main():

    MyReader = RDDLReader.RDDLReader('RDDL/'+DOMAIN)
    # MyReader = RDDLReader.RDDLReader('RDDL/power_unit_commitment_domain.rddl',
    #                                  'RDDL/power_unit_commitment_instance.rddl')
    domain = MyReader.rddltxt

    MyLexer = parser.RDDLlex()
    MyLexer.build()
    MyLexer.input(domain)
    # [token for token in MyLexer._lexer]


    # build parser - built in lexer, non verbose
    MyRDDLParser = parser.RDDLParser(None, False)
    MyRDDLParser.build()

    # parse RDDL file
    rddl_ast = MyRDDLParser.parse(domain)

    # MyXADDTranslator = XADDTranslator(rddl_ast)
    # MyXADDTranslator.Translate()

    grounder = RDDLGrounder.RDDLGrounder(rddl_ast)
    grounder.Ground()
    grounder.InitGround()
    # generator = RDDLGenerator.RDDLGenerator(rddl_ast)
    # rddl = generator.GenerateRDDL()
    # print(rddl)


if __name__ == "__main__":
    main()



