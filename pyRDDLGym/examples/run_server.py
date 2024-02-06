import sys

import pyRDDLGym
from pyRDDLGym.core.server import RDDLSimServer


def main(domain, instance):
    env = pyRDDLGym.make(domain, instance)    
    agent = RDDLSimServer(env.domain_text, env.instance_text, 30, 300)
    agent.run()


if __name__ == "__main__":
    args = sys.argv[1:]
    domain, instance = "Wildfire", 0
    if len(args) == 2:
        domain, instance = args
    main(domain, instance)
