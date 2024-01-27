import json
import numpy as np
import os
import sys

from pyRDDLGym.Core.Policies.RDDLSimAgent import RDDLSimAgent

# read the command line args
args = sys.argv[1:]
if len(args) != 1:
    print('')
    print('    Usage: python rddlsim.py <rounds>')
    print(f'    Given: {args}')
    print('')
    sys.exit(1)
rounds = int(args[0])

# load the RDDL files
RDDL_path = os.environ.get('RDDL')
domain_path = os.path.join(RDDL_path, 'domain.rddl')
instance_path = os.path.join(RDDL_path, 'instance.rddl')

# launch the RDDL server
agent = RDDLSimAgent(domain_path, instance_path, rounds, 99999)
agent.run()

# aggregate results
round_returns = [float(round_data[-1]['reward']) for round_data in agent.logs]
stats = {
    'mean': np.mean(round_returns),
    'median': np.median(round_returns),
    'min': np.min(round_returns),
    'max': np.max(round_returns),
    'std': np.std(round_returns)
}

# save outputs
out_path = os.environ.get('PROST_OUT')
dom_name = agent.env.model._AST.domain.name
inst_name = agent.env.model._AST.domain.name
with open(os.path.join(out_path, f'summary_{dom_name}_{inst_name}.json'), 'w') as f:
    json.dump(stats, f)    
agent.dump_data(os.path.join(out_path, f'data_{dom_name}_{inst_name}.json'))
