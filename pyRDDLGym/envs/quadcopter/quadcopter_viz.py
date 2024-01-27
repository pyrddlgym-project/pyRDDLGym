import io
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pygame

from pyRDDLGym.Core.Compiler.RDDLModel import PlanningModel
from pyRDDLGym.Visualizer.StateViz import StateViz


class QuadcopterVisualizer(StateViz):

    def __init__(self, model: PlanningModel,
                 bound=4,
                 figure_size=(6, 6),
                 wait_time=10) -> None:
        self._model = model
        self._bound = bound
        self._figure_size = figure_size
        self._wait_time = wait_time
        
        self._objects = model.objects
        self._nonfluents = model.groundnonfluents()
        
        self.fig = plt.figure(figsize=self._figure_size)
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.xs, self.ys, self.zs = {}, {}, {}
        for dk in self._objects['drone']:
            self.xs[dk] = []
            self.ys[dk] = []
            self.zs[dk] = []
            
    def convert2img(self):
        img_buf = io.BytesIO()
        self.fig.savefig(img_buf, format='png')
        im = Image.open(img_buf)
        return im
    
    def render(self, state):
        
        for dk in self._objects['drone']:
            pos = np.array([state[f'x___{dk}'], state[f'y___{dk}'], state[f'z___{dk}']])
            phi, theta, psi = state[f'phi___{dk}'], state[f'theta___{dk}'], state[f'psi___{dk}']
            cph, cth, cps = np.cos(phi), np.cos(theta), np.cos(psi)
            sph, sth, sps = np.sin(phi), np.sin(theta), np.sin(psi)        
            R = np.array([
                [cph * cps - cth * sth * sps, -cps * sph - cph * cth * sps, sth * sps],
                [cth * cps * sph + cph * sps, cph * cth * cps - sph * sps, -cps * sth],
                [sph * sth, cph * sth, cth]
            ])
            l = self._nonfluents['L']
            motor1 = (R @ np.array([[l], [0], [0]])).reshape((-1,)) + pos
            motor2 = (R @ np.array([[-l], [0], [0]])).reshape((-1,)) + pos
            motor3 = (R @ np.array([[0], [l], [0]])).reshape((-1,)) + pos
            motor4 = (R @ np.array([[0], [-l], [0]])).reshape((-1,)) + pos
            perp = (R @ np.array([[0], [0], [2 * l]])).reshape((-1,)) + pos
            
            tpos = np.array([self._nonfluents[f'TX___{dk}'],
                             self._nonfluents[f'TY___{dk}'],
                             self._nonfluents[f'TZ___{dk}']])
            
            self.ax.scatter(*pos, c='black', s=(40 * self._nonfluents['R']) ** 2)
            self.ax.scatter(*motor1, c='black', s=(20 * self._nonfluents['R']) ** 2)
            self.ax.scatter(*motor2, c='black', s=(20 * self._nonfluents['R']) ** 2)
            self.ax.scatter(*motor3, c='black', s=(20 * self._nonfluents['R']) ** 2)
            self.ax.scatter(*motor4, c='black', s=(20 * self._nonfluents['R']) ** 2)
            self.ax.plot(*zip(pos, motor1), c='black')
            self.ax.plot(*zip(pos, motor2), c='black')
            self.ax.plot(*zip(pos, motor3), c='black')
            self.ax.plot(*zip(pos, motor4), c='black')
            self.ax.plot(*zip(pos, perp), c='red', alpha=0.4)
            self.ax.scatter(*tpos, c='red', marker='>', s=40)
            
            self.xs[dk].append(pos[0])
            self.ys[dk].append(pos[1])
            self.zs[dk].append(pos[2])
            
            self.ax.plot(self.xs[dk], self.ys[dk], self.zs[dk], 
                         color='blue', alpha=0.1, linewidth=1.0)
        
        self.ax.set_xlim3d([-self._bound, self._bound])
        self.ax.set_ylim3d([-self._bound, self._bound])
        self.ax.set_zlim3d([-self._bound, self._bound])
        self.ax.xaxis.set_pane_color((0.0, 0.0, 1.0, 0.0))
        self.ax.yaxis.set_pane_color((0.0, 0.0, 1.0, 0.0))
        self.ax.zaxis.set_pane_color((0.0, 0.0, 1.0, 0.0))
        self.ax.tick_params(axis='x', colors='dimgray')
        self.ax.tick_params(axis='y', colors='dimgray')
        self.ax.tick_params(axis='z', colors='dimgray')
        self.ax.xaxis._axinfo['grid']['linewidth'] = 0.1
        self.ax.yaxis._axinfo['grid']['linewidth'] = 0.1
        self.ax.zaxis._axinfo['grid']['linewidth'] = 0.1

        # self.ax.grid(False)
        plt.rcParams.update({'font.size': 12, 'font.family': 'Consolas'})
        plt.tight_layout()
        
        pygame.time.wait(self._wait_time)
        
        img = self.convert2img()
        
        self.ax.cla()
        
        return img

