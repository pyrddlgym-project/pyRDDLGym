import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from pyRDDLGym.GurobiExperiment import GurobiExperiment


class GurobiVTOLExperiment(GurobiExperiment):
    
    def __init__(self, *args, **kwargs):
        super(GurobiVTOLExperiment, self).__init__(
            *args, iters=5, rollouts=1, **kwargs)
        self.model_params['NonConvex'] = 2
        
    def get_action_bounds(self, model):
        return {'F': (-1, 1)}
    
    def get_state_bounds(self, model):
        return {'theta': (-0.38, 0.71),
                'omega': (-20., 20.)}  
    
    def get_state_init_bounds(self, model):
        return {'theta': (0.1, 0.1),
                'omega': (0.0, 0.0)}
    
    def get_state_dependencies_S(self, model):
        return {'F': ['theta', 'omega']} 
    
    @staticmethod
    def prepare_policy_plot():
        
        def policy(theta, omega):
            return 0.6 - 15.4 * theta - 2.3 * omega + 100 * theta ** 2 - 1.5 * omega ** 2
         
        n = 200
        thetas = np.linspace(-0.2, 0.2, n)
        omegas = np.linspace(-1, 1, n)
        data = np.zeros((n, n))
        for io in range(n):
            for it in range(n):
                data[io, it] = policy(thetas[it], omegas[io])
        data = np.clip(data, -1, 1)
        data = data[::-1,:]
        plt.figure(figsize=(6.4, 3.2))
        im = plt.imshow(data, extent=[-0.2, 0.2, -1, 1], aspect=0.1, cmap='seismic')
        plt.xlabel('$\\theta$')
        plt.ylabel('$\\omega$')
        plt.colorbar(im, pad=0.03, fraction=0.09)
        plt.tight_layout()
        plt.savefig(os.path.join('gurobi_results', 'VTOL_policy.pdf'))
        plt.clf()
        plt.close()
    
        
if __name__ == "__main__":
    if len(sys.argv) < 2:
        horizon = 6
    else:
        horizon = sys.argv[1]
        horizon = int(horizon)
          
    dom = 'VTOL'
    dom_test = dom
    
    for _ in range(5): 
        experiment = GurobiVTOLExperiment(value='Q', cases=0)
        experiment.run(dom, 0, horizon, dom_test)
