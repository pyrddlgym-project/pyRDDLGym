import sys

from pyRDDLGym.core.server import RDDLSimServer
from pyRDDLGym.envs.registration import get_path_to_instance


def main(domain, instance):
    instance_path, domain_path, *_ = get_path_to_instance(domain, instance)
    agent = RDDLSimServer(domain_path, instance_path, 30, 300)
    agent.run()


if __name__ == "__main__":
    args = sys.argv[1:]
    domain, instance = "Wildfire", 0
    if len(args) == 2:
        domain, instance = args
    main(domain, instance)
