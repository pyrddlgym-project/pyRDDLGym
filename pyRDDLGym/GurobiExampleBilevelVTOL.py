import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from pyRDDLGym.GurobiExperiment import GurobiExperiment


class GurobiVTOLExperiment(GurobiExperiment):
    
    def __init__(self, *args, **kwargs):
        super(GurobiVTOLExperiment, self).__init__(
            *args, value='Q', cases=0, iters=5, rollouts=1, **kwargs)
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
        
    def prepare_policy_plot(self):
        
        def policy(theta, omega):
            return -0.38739016377776403 + -0.4742443203026949 * theta + \
                -19.807301089794326 * omega + 100.0 * theta * theta + \
                -100.0 * theta * omega + -81.09237718477412 * omega * omega
         
        n = 200
        thetas = np.linspace(-0.1, 0.1, n)
        omegas = np.linspace(-0.5, 0.5, n)
        data = np.zeros((n, n))
        for io in range(n):
            for it in range(n):
                data[io, it] = policy(thetas[it], omegas[io])
        data = np.clip(data, -1, 1)
        data = data[::-1,:]
        plt.figure(figsize=(6.4, 3.2))
        im = plt.imshow(data, extent=[-0.1, 0.1, -0.5, 0.5], aspect=0.1, cmap='seismic')
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
        experiment = GurobiVTOLExperiment(log=True)
        experiment.run(dom, 0, horizon, dom_test)
