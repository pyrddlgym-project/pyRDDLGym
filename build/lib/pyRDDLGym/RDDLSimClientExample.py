import sys

from pyRDDLGym import ExampleManager
from pyRDDLGym.Core.Policies.RDDLSimAgent import RDDLSimAgent

def main(domain):
    EnvInfo = ExampleManager.GetEnvInfo(domain)
    agent = RDDLSimAgent(EnvInfo.get_domain(), EnvInfo.get_instance(1), 30, 300)
    agent.run()

if __name__ == "__main__":
    domain = "recon2018"
    if len(sys.argv) == 2:
        domain = sys.argv[1]
    main(domain)
