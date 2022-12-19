from pyRDDLGym import RDDLEnv
from pyRDDLGym import ExampleManager
from pyRDDLGym.Policies.RDDLSimAgent import RDDLSimAgent

def main():
    EnvInfo = ExampleManager.GetEnvInfo('Wildfire')
    agent = RDDLSimAgent(EnvInfo.get_domain(), EnvInfo.get_instance(0), 30, 300)
    agent.run()

if __name__ == "__main__":
    main()


