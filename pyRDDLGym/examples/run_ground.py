'''In this example, the grounder is run and the grounded domain is printed.

The syntax for running this example is:

    python run_ground.py <domain> <instance>
    
where:
    <domain> is the name of a domain located in the /Examples directory
    <instance> is the instance number
'''
import sys

import pyRDDLGym
from pyRDDLGym.core.debug.decompiler import RDDLDecompiler
from pyRDDLGym.core.grounder import RDDLGrounder


def main(domain, instance):
    
    # create the environment
    env = pyRDDLGym.make(domain, instance, enforce_action_constraints=False)
    
    # ground the model
    grounder = RDDLGrounder(env.model.ast)
    grounded_model = grounder.ground()
    
    # decompile and print model
    decompiler = RDDLDecompiler()
    print(decompiler.decompile_domain(grounded_model))
    
    env.close()


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 2:
        print('python run_ground.py <domain> <instance>')
        exit(1)
    kwargs = {'domain': args[0], 'instance': args[1]}
    main(**kwargs)
