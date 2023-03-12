import math
import numpy as np
import random

from pyRDDLGym.Core.Compiler.RDDLLiftedModel import RDDLLiftedModel
from pyRDDLGym.Examples.ExampleManager import ExampleManager
from pyRDDLGym.Core.Env.RDDLEnv import RDDLEnv
from pyRDDLGym.Core.Jax.JaxRDDLSimulator import JaxRDDLSimulator


class Node:
    
    def __init__(self, num_actions: int, action, parent=None):
        self.n = 0
        self.q = 0.0
        self.action = action
        self.parent = parent
        self.children = []
        self.untried = list(range(num_actions))
        random.shuffle(self.untried)
    
    def best_child(self, c: float=1.0 / math.sqrt(2.0)):
        scores = [
            child.q + 2.0 * c * math.sqrt((2.0 * math.log(self.n) / child.n))
            for child in self.children
        ]
        best_index = np.argmax(scores)
        best_child = self.children[best_index]
        return best_child
    
    def is_fully_expanded(self):
        return not self.untried
    

class JaxRDDLBackpropPlannerUCT:
    
    def __init__(self, rddl: RDDLLiftedModel, horizon, c=1.0 / math.sqrt(2.0)):
        self.T = horizon
        self.c = c
        self.rewards = np.zeros((horizon,))
        
        # self.actions = [{}]
        # for outer in ['put-out', 'cut-out']:
        #     for x in range(1, 11):
        #         for y in range(1, 4):
        #             self.actions.append({f'{outer}___x{x}__y{y}': True})
        self.actions = [{'force-side': 0}, {'force-side': 1}]
        self.num_actions = len(self.actions)
        self.sim = JaxRDDLSimulator(rddl)
        
    def search(self, s0, epochs: int, steps: int=1, root: Node=None):
        if root is None:
            root = Node(self.num_actions, action=None, parent=None)
        for it in range(epochs):
            R_train = self._iterate(s0, root)
            if it % steps == 0:
                best = root.best_child(c=0.0)
                action = best.action
                yield (R_train, action, it, best)
        best = root.best_child(c=0.0)
        action = best.action
        return (R_train, action, it, best)
            
    def test(self, s0, root: Node):
        step_fn = self.sim.step
        self._reset_state(s0)
        
        # start of rollout...
        node = root
        R = 0.0
        for _ in range(self.T):
            if node is not None:
                if node.children:
                    node = node.best_child(c=0.0)
                    action = node.action
                else:
                    node = None
            if node is None:
                action = random.choice(self.actions)
            _, reward, done = step_fn(action)
            R += reward
            if done:
                break
        return R
            
    def _reset_state(self, s0):
        self.sim.subs = s0.copy()
        
    def _iterate(self, s0, root: Node) -> float:
        step_fn = self.sim.step
        self._reset_state(s0)
        
        # start of rollout...
        node = root
        rollout = False
        node_depth = 1
        for t in range(self.T):
                
            # select   
            if not rollout and node.is_fully_expanded():
                child = node.best_child(self.c)
                action = child.action
                node = child
                node_depth += 1
                
            # expand
            elif not rollout:
                *node.untried, action = node.untried
                action = self.actions[action]
                child = Node(self.num_actions, action, parent=node)
                node.children.append(child)
                node = child
                rollout = True
                node_depth += 1
            
            # rollout
            else:
                action = random.choice(self.actions)
            
            # make a step in the environment with current action
            _, reward, done = step_fn(action)
            self.rewards[t] = reward
            if done:
                break
        
        # backup
        R = np.sum(self.rewards[node_depth:t + 1])
        for _ in range(node_depth - 1, -1, -1):
            R += self.rewards[t]
            node.n += 1
            node.q += (R - node.q) / node.n
            node = node.parent
        assert node is None
        return R            


def get_rddl(domain, inst):
    try:
        from rddlrepository.Manager.RDDLRepoManager import RDDLRepoManager
        manager = RDDLRepoManager()
        EnvInfo = manager.get_problem(domain)
        print('reading domain from rddlrepository...')
    except:
        EnvInfo = ExampleManager.GetEnvInfo(domain)
        print('reading domain from Examples...')
    domain = EnvInfo.get_domain()
    inst = EnvInfo.get_instance(inst)
    return RDDLEnv(domain, inst).model

rddl = get_rddl('CartPole discrete', 0)
world = JaxRDDLSimulator(rddl)
planner = JaxRDDLBackpropPlannerUCT(rddl, 5)

print('starting MCTS...')
world.reset()
root = None
for step in range(200):
    
    print('\n' + f'iteration {step}...')
    s0 = world.subs.copy()
    for (R, action, it, best) in planner.search(s0, 700, 70, root=None):
        print(f'iter={it}, reward={R}, action={action}')
    print(f'final action {action}')
    
    _, reward, done = world.step(action)
    #root = best
    #root.parent = None
    if done:
        break
    
