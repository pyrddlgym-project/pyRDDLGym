'''In this example, compute lower and upper value bounds on the random policy.

The syntax for running this example is:

    python run_intervals.py <domain> <instance>
    
where:
    <domain> is the name of a domain located in the /Examples directory
    <instance> is the instance number
'''
import sys
import numpy as np

import pyRDDLGym
from pyRDDLGym.core.intervals import RDDLIntervalAnalysis


def main(domain, instance):
    
    # create the environment
    env = pyRDDLGym.make(domain, instance, enforce_action_constraints=False)
    
    # evaluate lower and upper bounds on accumulated reward of random policy
    analysis = RDDLIntervalAnalysis(env.model)
    reward_lower, reward_upper = analysis.bound(per_epoch=True)['reward']
    print(f'value lower bound = {np.sum(reward_lower)}, '
          f'value upper bound = {np.sum(reward_upper)}')
    
    env.close()


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 2:
        print('python run_intervals.py <domain> <instance>')
        exit(1)
    kwargs = {'domain': args[0], 'instance': args[1]}
    main(**kwargs)
