import matplotlib.pyplot as plt
import numpy as np
import os
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

    @staticmethod
    def prepare_policy_plot():
        
        def policy(x, y):
            #fire = 0.0 if -0.13604 <= -0.10 * x + -0.0204 * y <= 0.0 else 1.0
            fire = 0.0 if -0.13604 <= -0.10 * x + -0.0204 * y <= -0.0 else 0.0 if 0.001 <= -1.0 * x + 0.0 * y <= 1.0 else 1.0
            return fire
        
        n = 200
        xs = np.linspace(0, 1, n)
        ys = np.linspace(0, 5, n)
        data = np.zeros((n, n))
        for iy in range(n):
            for ix in range(n):
                data[iy, ix] = policy(xs[ix], ys[iy])
        data = data[::-1,:]
        plt.figure(figsize=(6.4, 4.8))
        im = plt.imshow(data, extent=[0, 1, 0, 5], aspect=0.14, cmap='seismic')
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.colorbar(im, pad=0.03, fraction=0.09)
        plt.tight_layout()
        plt.savefig(os.path.join('gurobi_results', 'interception_policy_2.pdf'))
        plt.clf()
        plt.close()

    
if __name__ == "__main__":
    if len(sys.argv) < 6:
        horizon, constr, value, cases, chance = 10, 'L', 'C', 1, 0.995
    else:
        horizon, constr, value, cases, chance = sys.argv[1:6]
        horizon, cases, chance = int(horizon), int(cases), float(chance)
        
    dom = 'Interception'
    dom_test = dom
    
    GurobiInterceptionExperiment.prepare_policy_plot()
    
    for _ in range(1):
        experiment = GurobiInterceptionExperiment(
            constr=constr, value=value, cases=cases, chance=chance,
            iters=10, epsilon=1e-3)
        experiment.run(dom, 0, horizon, dom_test)
