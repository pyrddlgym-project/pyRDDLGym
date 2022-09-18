from Parser import parser as parser
from Parser import RDDLReader as RDDLReader
import Grounder.RDDLGrounder as RDDLGrounder

# DOMAIN = 'power_unit_commitment.rddl'

# DOMAIN = 'ThiagosReservoir.rddl'
# DOMAIN = 'Thiagos_Mars_Rover.rddl'
# DOMAIN = 'dbn_prop.rddl'
# DOMAIN = 'RamMod_Thiagos_HVAC.rddl'
DOMAIN = 'RamMod_Thiagos_HVAC_grounded.rddl'

# DOMAIN = 'Thiagos_HVAC.rddl'
# DOMAIN = 'Thiagos_HVAC_grounded.rddl'
# DOMAIN = 'wildfire_mdp.rddl'

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

    grounder = RDDLGrounder.RDDLGroundedGrounder(rddl_ast)
    # grounder = RDDLGrounder(rddl_ast)
    model = grounder.Ground()
    # grounder.InitGround()

    grounder2 = RDDLGrounder.RDDLGrounder(rddl_ast)
    model2 = grounder2.Ground()


    # generator = RDDLGenerator.RDDLGenerator(rddl_ast)
    # rddl = generator.GenerateRDDL()
    print(rddl)


if __name__ == "__main__":
    main()



