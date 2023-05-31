import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
from math import ceil
from datetime import datetime

class TimeSpaceDiagram:
    def __init__(self, env, num_warmup_steps=0, agent_id=None):
        self.env = env
        self.nonfluents = env.non_fluents
        self.N = env.numConcurrentActions
        self.linklen = self.nonfluents['Dl___l0']
        self.simstep = self.nonfluents['Ts']
        self.vehlen = self.nonfluents['Lv']
        self.horizon = env.horizon
        self.proptime = round(self.linklen/(13.6*self.simstep))+1

        self.link_keys = tuple(tuple(f'flow-on-link___l{2*n+1}__t{t}' for t in range(self.proptime)) for n in range(self.N+1))
        self.Q_keys = tuple(f'q___l{2*n+1}__l{2*(n+1)+1}' for n in range(self.N))
        self.signal_keys = tuple(f'signal___i{n}' for n in range(self.N))

        self.T = 0
        self.segments = []
        self.phases = [[(0,0)] for _ in range(self.N)]

        self.num_warmup_steps = num_warmup_steps
        self.agent_id = agent_id

    def step(self, state):
        prevT = self.T
        self.T += self.simstep

        for i in range(self.N):
            d = self.linklen * (i+1)
            dprev = d - self.linklen
            Qcnt = state[self.Q_keys[i]]
            for Qi in range(ceil(Qcnt/2)):
                Qd = d - Qi*self.vehlen
                self.segments.append(((prevT,Qd),(self.T,Qd)))

            Qline = Qcnt*self.vehlen/2
            dline = d - Qline
            Qprop = round((self.linklen-Qline)/(13.6*self.simstep))
            for t in range(Qprop):
                inflow = state[self.link_keys[i][t]]
                for fi in range(ceil(inflow/2)):
                    fr = max(dline - (t+1)*13.6 - self.vehlen*fi, dprev+10)
                    to = dline - t*13.6 - self.vehlen*fi
                    self.segments.append(((self.T, fr), (self.T+self.simstep, to)))

            signal = state[self.signal_keys[i]]
            if self.phases[i][-1][0] != signal:
                self.phases[i].append((signal,self.T))

        d = self.linklen * (self.N+1)
        dprev = d - self.linklen
        for t in range(self.proptime):
             inflow = state[self.link_keys[self.N][t]]
             for fi in range(ceil(inflow/2)):
                 fr = max(d - (t+1)*13.6 - self.vehlen*fi, dprev+10)
                 to = d - t*13.6 - self.vehlen*fi
                 self.segments.append(((self.T, fr), (self.T+self.simstep, to)))

    def plot(self, cmltrew=None, filepath=None):
        fig, ax = plt.subplots()
        fig.set_size_inches((10,10))

        ax.add_collection(LineCollection(self.segments))

        for i in range(self.N):
            # Traffic lights
            for ph0,ph1 in zip(self.phases[i], self.phases[i][1:]):
                color = 'green' if ph0[0] == 1 else 'red'
                ax.add_patch(Rectangle(
                                 (ph0[1],self.linklen*(i+1)+5), ph1[1]-ph0[1], 10,
                                 facecolor=color))
            ph = self.phases[i][-1]
            color = 'green' if ph[0] == 1 else 'red'
            ax.add_patch(Rectangle(
                             (ph[1],self.linklen*(i+1)+5), self.horizon-ph[1], 10,
                             facecolor=color))

        ax.set_xlim(0,(self.horizon - self.num_warmup_steps))
        ax.set_ylim(0,self.linklen*(self.N+1))
        title_str = ''
        if self.agent_id is not None:
            title_str += f'Agent {self.agent_id}'
        if cmltrew is not None:
            title_str += f', Cmlt. rew.={round(cmltrew)}'
        if title_str != '':
            ax.set_title(title_str)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Space (m)')

        if filepath is None:
            now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            filepath = f'img/timespace_{now}.png'
        plt.savefig(filepath)
