from pyRDDLGym.Core.Parser import parser as parser
from pyRDDLGym.Core.Parser import RDDLReader as RDDLReader
from pyRDDLGym.Core import Grounder as RDDLGrounder
from pyRDDLGym.Core.Simulator.RDDLSimulator import RDDLSimulator

DOMAIN = 'rddl_termination_test.rddl'

MyReader = RDDLReader.RDDLReader('RDDL/' + DOMAIN)
domain = MyReader.rddltxt

MyLexer = parser.RDDLlex()
MyLexer.build()
MyLexer.input(domain)
token_list = [token for token in MyLexer._lexer]
# print(token_list)

# build parser - built in lexer, non verbose
MyRDDLParser = parser.RDDLParser(None, False)
MyRDDLParser.build()

# parse RDDL file
rddl_ast = MyRDDLParser.parse(domain)
grounder = RDDLGrounder.RDDLGrounder(rddl_ast)
model = grounder.Ground()
sampler = RDDLSimulator(model)

loops = 2
for h in range(loops):
    state = sampler.reset_state()
    print(state)
    action = {
        'curProd_p1': 1,
        'curProd_p2': 2,
        'curProd_p3': 3,
    }
    state = sampler.sample_next_state(action)
    sampler.check_terminal_states()
    print(state)

print ('hello world')