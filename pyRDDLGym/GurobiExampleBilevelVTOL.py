import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from pyRDDLGym.Core.Gurobi.GurobiRDDLPlan import GurobiQuadraticPolicy
from pyRDDLGym.GurobiExperiment import GurobiExperiment

# settings for pyplot
SMALL_SIZE = 20
MEDIUM_SIZE = 24
BIGGER_SIZE = 28

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams['text.usetex'] = True


class GurobiVTOLExperiment(GurobiExperiment):
    
    def __init__(self, *args, **kwargs):
        super(GurobiVTOLExperiment, self).__init__(*args, iters=5, rollouts=1, **kwargs)
        self.model_params['NonConvex'] = 2
        self.model_params['MIPGap'] = 0.02
        self._chance = kwargs['chance']
        
    def get_policy(self, model):
        action_bounds = {'F': (-1, 1)}        
        policy = GurobiQuadraticPolicy(
            action_bounds=action_bounds
        )
        return policy
    
    def get_state_init_bounds(self, model):
        state_bounds_init = {'theta': (0.1, 0.1),
                             'omega': (0.0, 0.0)}
        return state_bounds_init
    
    def get_experiment_id_str(self):
        return f'{self._chance}'
    
    def prepare_policy_plot(self, domain, inst=0, horizon=6):
        
        def policy(theta, omega):
            return 0.6206685409847967 + -8.836488823820638 * theta + \
                -1.041761277800478 * omega + 34.76625526044426 * theta * theta + \
                66.07315031611493 * theta * omega + 8.64200631418066 * omega * omega
         
        n = 200
        thetas = np.linspace(-0.38, 0.71, n)
        omegas = np.linspace(-2.0, 2.0, n)
        data = np.zeros((n, n))
        for io in range(n):
            for it in range(n):
                data[io, it] = policy(thetas[it], omegas[io])
        data = np.clip(data, -1, 1)
        data = data[::-1,:]
        plt.figure(figsize=(6.4, 3.2))
        im = plt.imshow(data, extent=[-0.38, 0.71, -2, 2], aspect=0.12, cmap='seismic')
        plt.xlabel('$\\theta$')
        plt.ylabel('$\\omega$')
        plt.colorbar(im, pad=0.03, fraction=0.09)
        plt.tight_layout()
        plt.savefig(os.path.join(
            'gurobi_results', f'{domain}_{inst}_{horizon}_policy.pdf'))
        plt.clf()
        plt.close()


if __name__ == "__main__":
    dom = 'VTOL'
    if len(sys.argv) < 3:
        horizon, chance = 6, 0.995
    else:
        horizon, chance = sys.argv[1:3]
        horizon, chance = int(horizon), float(chance)    
    for _ in range(5): 
        experiment = GurobiVTOLExperiment(chance=chance)
        experiment.run(dom, 0, horizon)
    
