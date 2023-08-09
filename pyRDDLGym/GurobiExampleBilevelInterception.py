import sys

from pyRDDLGym.GurobiExperiment import GurobiExperiment


class GurobiInterceptionExperiment(GurobiExperiment):
    
    def get_action_bounds(self, model):
        return {'fire': (0, 1)}
    
    def get_state_bounds(self, model):
        return {'missile-x': (0, 1),
                'missile-y': (0, 5),
                'intercept-y': (0, 10)}
    
    def get_state_init_bounds(self, model):
        return {'missile-x': (0, 0),
                'missile-y': (5, 5),
                'intercept-y': (0, 0)}
    
    def get_state_dependencies_S(self, model):
        return {'fire': ['missile-x', 'missile-y']}
    
    def get_state_dependencies_L(self, model):
        return {'fire': ['missile-x', 'missile-y']}

    
if __name__ == "__main__":
    if len(sys.argv) < 6:
        horizon, constr, value, cases, chance = 10, 'L', 'C', 1, 0.995
    else:
        horizon, constr, value, cases, chance = sys.argv[1:6]
        horizon, cases, chance = int(horizon), int(cases), float(chance)
        
    dom = 'Interception'
    dom_test = dom

    for _ in range(1):
        experiment = GurobiInterceptionExperiment(
            constr=constr, value=value, cases=cases, chance=chance,
            iters=10, epsilon=1e-3)
        experiment.run(dom, 0, horizon, dom_test)
