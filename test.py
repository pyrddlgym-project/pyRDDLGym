from Parser import parser as parser
from Parser import RDDLReader as RDDLReader
import Grounder.RDDLGrounder as RDDLGrounder
from Simulator.RDDLSimulator import RDDLSimulator

DOMAIN = 'power_unit_commitment.rddl'

# DOMAIN = 'ThiagosReservoir.rddl'
# DOMAIN = 'Thiagos_Mars_Rover.rddl'
# DOMAIN = 'Thiagos_HVAC.rddl'
# DOMAIN = 'dbn_prop.rddl'
DOMAIN = 'Thiagos_HVAC_grounded.rddl'
# DOMAIN = 'wildfire_mdp.rddl'


def main():

    MyReader = RDDLReader.RDDLReader('RDDL/' + DOMAIN)
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

    from pprint import pprint
    grounder = RDDLGrounder.RDDLGroundedGrounder(rddl_ast)
    model = grounder.Ground()
    pprint(vars(model))
    
    good_policy = True
    sampler = RDDLSimulator(model)
    state = sampler.reset_state()
    for _ in range(50):
        sampler.check_state_invariants()
        actions = {'AIR_r1': 0., 'AIR_r2': 0., 'AIR_r3': 0.}
        if good_policy:
            if state['TEMP_r1'] < 20.5:
                actions['AIR_r1'] = 5.
            if state['TEMP_r2'] < 20.5:
                actions['AIR_r2'] = 5.
            if state['TEMP_r3'] < 20.5:
                actions['AIR_r3'] = 5.
            if state['TEMP_r1'] > 23.:
                actions['AIR_r1'] = 0.
            if state['TEMP_r2'] > 23.:
                actions['AIR_r2'] = 0.
            if state['TEMP_r3'] > 23.:
                actions['AIR_r3'] = 0.
        state = sampler.sample_next_state(actions)
        reward = sampler.sample_reward()
        sampler.update_state()
        print(state)
        print(reward)
    
    IS_ROOM_r1 = True
    AIR_r1 = 0.
    COST_AIR = 1
    TEMP_r1 = 10.
    TEMP_LOW_r1 = 20.
    TEMP_UP_r1 = 23.5
    PENALTY = 20000
    IS_ROOM_r2 = True
    AIR_r2 = 0.
    TEMP_r2 = 10.
    TEMP_LOW_r2 = 20.
    TEMP_LOW_r3 = 20.
    TEMP_UP_r2 = 23.5
    TEMP_UP_r3 = 23.5
    IS_ROOM_r3 = True
    AIR_r3 = 0.
    TEMP_r3 = 10.
            
    # grounder = RDDLGrounder.RDDLGrounder(rddl_ast)
    # grounder.Ground()
    # pprint(vars(grounder))
    
    # grounder.InitGround()

    # generator = RDDLGenerator.RDDLGenerator(rddl_ast)
    # rddl = generator.GenerateRDDL()
    

if __name__ == "__main__":
    main()

