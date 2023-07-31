import sys

from pyRDDLGym.GurobiExperiment import GurobiExperiment


class GurobiReservoirExperiment(GurobiExperiment):
    
    def get_action_bounds(self, model):
        return {'release___t1': (0, 100),
                'release___t2': (0, 200),
                'release___t3': (0, 400)}
    
    def get_state_bounds(self, model):
        return {'rlevel___t1': (0, 100),
                'rlevel___t2': (0, 200),
                'rlevel___t3': (0, 400)}
    
    def get_state_init_bounds(self, model):
        return {'rlevel___t1': (50, 100),
                'rlevel___t2': (100, 200),
                'rlevel___t3': (250, 350)}
    
    def get_state_dependencies_S(self, model):
        return {'release___t1': ['rlevel___t1'],
                'release___t2': ['rlevel___t2'],
                'release___t3': ['rlevel___t3']}
    
if __name__ == "__main__":
    if len(sys.argv) < 7:
        inst, horizon, constr, value, cases, chance = 1, 10, 'S', 'C', 0, 0.995
    else:
        inst, horizon, constr, value, cases, chance = sys.argv[1:7]
        horizon, cases, chance = int(horizon), int(cases), float(chance)
        
    dom = 'Reservoir linear'
    dom_test = dom
    
    for _ in range(5):
        experiment = GurobiReservoirExperiment(
            constr=constr, value=value, cases=cases, chance=chance,
            iters=15)
        experiment.run(dom, inst, horizon, dom_test)
