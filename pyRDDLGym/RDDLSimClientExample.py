
import sys

from pyRDDLGym import RDDLEnv
from pyRDDLGym import ExampleManager
from pyRDDLGym.Policies.RDDLSimAgent import RDDLSimAgent

def main(domain):
    EnvInfo = ExampleManager.GetEnvInfo(domain)
    agent = RDDLSimAgent(EnvInfo.get_domain(), EnvInfo.get_instance(0), 30, 300)
    agent.run()

if __name__ == "__main__":
    domain = "Wildfire"
    if len(sys.argv) == 2:
        domain = sys.argv[1]
    main(domain)


