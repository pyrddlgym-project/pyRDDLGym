import jax
import math
import numpy as np
import random
from typing import Dict, List

from pyRDDLGym.Core.Compiler.RDDLLiftedModel import RDDLLiftedModel
from pyRDDLGym.Core.Env.RDDLEnv import RDDLEnv
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxRDDLBackpropPlanner
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxStraightLinePlan
from pyRDDLGym.Core.Jax.JaxRDDLLogic import FuzzyLogic
from pyRDDLGym.Core.Jax.JaxRDDLSimulator import JaxRDDLSimulator

from pyRDDLGym.Examples.ExampleManager import ExampleManager


class ChanceNode:
    '''A chance node represents an action/decision in the search tree.'''
    
    def __init__(self, action: object) -> None:
        self.action = action
        self.children = []
    
    def add(self, vD: 'DecisionNode') -> None:
        self.children.append(vD)
        
    def lvn(self) -> 'DecisionNode':
        least, leastvisit = None, math.inf
        for c in self.children:
            score = c.N
            if score < leastvisit:
                least, leastvisit = c, score
        return least
                

class DecisionNode:
    '''A decision node represents a transition following some decision made.'''
    
    def __init__(self, transition: object, actions: List[object]=None) -> None:
        self.transition = transition
        self.N = 0
        self.Nc = []
        self.Qc = []
        self.children = []    
        if actions is not None:
            self.unvisited = actions.copy()
            random.shuffle(self.unvisited)
    
    def add(self, vC: 'ChanceNode') -> None:
        self.Nc.append(0)
        self.Qc.append(0.0)
        self.children.append(vC)
        
    def uct(self, c: float) -> int:
        logN = math.log(self.N)
        Qc, Nc = self.Qc, self.Nc
        best, bestscore = None, -math.inf
        for i in range(len(self.children)):
            score = Qc[i] + c * math.sqrt(logN / Nc[i])
            if score > bestscore:
                best, bestscore = i, score
        return best
    
    def update(self, i: int, R: float) -> None:
        self.N += 1
        self.Nc[i] += 1
        self.Qc[i] += (R - self.Qc[i]) / self.Nc[i]


class JaxRDDLHybridBackpropUCTPlanner:
    '''Represents a hybrid approach between UCT-MCTS and planning by backprop.
    '''
    
    def __init__(self, rddl: RDDLLiftedModel,
                 action_bounds: Dict,
                 rollout_horizon: int,
                 alpha: float,
                 beta: float,
                 delta: float,
                 c: float=1.0 / math.sqrt(2.0),
                 max_sgd_updates: int=1,
                 policy_hyperparams: Dict={},
                 **planner_kwargs) -> None:
        '''Creates a new hybrid backprop + UCT-MCTS planner.
        
        :param rddl: the rddl domain to optimize
        :param actions: the set of possible actions in the domain: can be None, 
        in which case progressive widening and the action_sample distribution are
        used instead
        :param rollout_horizon: how many steps to plan ahead
        :param alpha: the growth factor for chance nodes
        :param beta: the growth factor for decision nodes
        :param delta: how much can backprop updates change the MCTS plan
        :param c: scaling factor for UCT: note that this is adapted during learning
        so the parameter here specifies only the initial guess
        :param max_sgd_updates: max number of SGD updates for backprop planner
        :param policy_hyperparams: same as what would normally be passed to
        planner.optimize()
        :param **planner_kwargs: keywords arguments to initialize backprop planner:
        will not use backprop if none are specified 
        '''
        self.rddl = rddl
        self.sim = JaxRDDLSimulator(rddl, keep_tensors=True)  # TODO (mike): no need to compile twice
        self.T = rollout_horizon
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.c = c
        self.action_sample = self.action_bounds_to_samplers(rddl, action_bounds)     
        self.max_sgd_updates = max_sgd_updates
        
        # use progressive widening for action nodes if action space not specified
        self.use_pw_action = True
        
        # optionally use a backprop planner to refine the search tree
        if planner_kwargs is not None and planner_kwargs:
            self.sgd = JaxRDDLBackpropPlanner(
                rddl=rddl, 
                rollout_horizon=self.T, 
                action_bounds=action_bounds,
                **planner_kwargs)
        else:
            self.sgd = None
        self.policy_hyperparams = policy_hyperparams
    
    @staticmethod
    def action_bounds_to_samplers(rddl, action_bounds):
        samplers = {}
        for (action, prange) in rddl.actionsranges.items():
            ptypes = rddl.param_types[action]
            shape = rddl.object_counts(ptypes) if ptypes else None
            if prange == 'bool':
                samplers[action] = (lambda _: np.random.uniform(size=shape) < 0.5)
            else:
                bounds = action_bounds[action]
                samplers[action] = (lambda _: np.random.uniform(*bounds, size=shape))
        return samplers
        
    def _select_action(self, s, vD, t, c):
        
        # determine whether a new chance (action) node needs to be created
        # if so, proceed with roll-out stage after
        if self.use_pw_action:
            rollout = len(vD.children) <= vD.N ** self.alpha
        else:
            rollout = bool(vD.unvisited)
            
        # create a new chance node
        if rollout:
            
            # if progressive widening is used, sample action from the sampling
            # distribution, otherwise sample a random unvisited action
            if self.use_pw_action:
                a = {var: self.action_sample[var](s) for var in self.rddl.actions}
            else:
                a = vD.unvisited.pop()
                
            # append the new chance node to the decision node's children
            vC = ChanceNode(a)
            vD.add(vC)
            ivC = len(vD.children) - 1
        else:
            
            # sample an action from the UCT bound for the vD children
            ivC = vD.uct(c)
            vC = vD.children[ivC]
            a = vC.action     
            
        return (vC, ivC, a, rollout)
    
    def _sample_noise(self, vD, vC, ivC):
        
        # determine if a new decision (state) node needs to be created
        # for now assume continuous state space and use progressive widening
        if len(vC.children) <= vD.Nc[ivC] ** self.beta:
            
            # a new decision node must be created
            # here a new transition will be sampled from the environment
            vD1 = DecisionNode(None, None)
            vC.add(vD1)
        else:
            
            # just return the existing transition
            vD1 = vC.lvn()
        return vD1
    
    def _rollout(self, s, t):
        step_fn = self.sim.step
        policy = self.action_sample
        R = 0.0
        for _ in range(t, self.T):
            a = {var: policy[var](s) for var in self.rddl.actions}
            s, r, done = step_fn(a)
            R += r
            if done:
                break
        return R
    
    def _simulate(self, s, vD, t, c, vCs):
        
        # the end of the planning horizon
        if t >= self.T:
            return 0.0
        
        # select an action
        vC, ivC, a, rollout = self._select_action(s, vD, t, c)
        
        # perform a transition
        vD1 = self._sample_noise(vD, vC, ivC)
        tr = vD1.transition
        if tr is None:
            vD1.transition = tr = self.sim.step(a)
        s1, r, done = tr        
        vCs.append(vC)
        
        # termination early
        if done:
            t = self.T - 1
        
        # continue with rollout or simulation
        if rollout:
            R1 = self._rollout(s1, t + 1)
        else:
            R1 = self._simulate(s1, vD1, t + 1, c, vCs)
        
        # update decision node statistics
        R = r + R1
        vD.update(ivC, R)        
        return R
        
    def _jax_sgd(self, key, train_subs, test_subs, vCs): 
        
        # generate initial guess for plan from search tree
        sgd = self.sgd
        dtype = sgd.compiled.REAL
        action_names = vCs[0].action.keys()
        guess = {var: np.asarray([vC.action[var] for vC in vCs], dtype=dtype) 
                 for var in action_names}
        
        # run SGD updates
        model_params = sgd.compiled.model_params
        hyper_params = self.policy_hyperparams
        params = guess
        opt_state = sgd.optimizer.init(params)        
        for epoch in range(self.max_sgd_updates):
            key, subkey = jax.random.split(key)
            params, _, opt_state, _ = sgd.update(
                subkey, params, hyper_params, train_subs, model_params, opt_state)
            
            # stop once the actions diverge by delta
            max_delta = 0.0
            for var in action_names:
                delta = np.linalg.norm(params[var] - guess[var])
                max_delta = max(max_delta, delta)
            if max_delta >= self.delta:
                break
            
        # store update in the search tree
        num_steps = len(vCs)
        for step in range(num_steps):
            
            # get testing action
            action = sgd.test_policy(key, params, hyper_params, step, test_subs)
            
            # project action within a trust region around current estimate
            for var in action_names:
                if rddl.actionsranges[var] == 'real':
                    action[var] = np.clip(
                        action[var], guess[var] - self.delta, guess[var] + self.delta)
            vCs[step].action = action 
                   
        return max_delta, epoch + 1
        
    def search(self, key: jax.random.PRNGKey,
               s0: Dict,
               epochs: int,
               steps: int=1):
        '''Performs a search using the current planner.
        
        :param key: Jax PRNG key for random sampling
        :param s0: initial pvariables (e.g. state, non-fluents) to begin from
        :param epochs: how many iterations of MCTS to perform
        :param steps: how often to return a callback
        '''
        root = DecisionNode(None)
        R0 = self.c   
             
        sgd = self.sgd
        if sgd is not None:
            train_subs, test_subs = sgd._batched_init_subs(s0)
            
        for it in range(epochs):
            
            # regular MCTS with double-progressive widening
            self.sim.subs = s0.copy()
            vCs = []
            R = self._simulate(s0, root, 0, R0, vCs)
            R0 += (R - R0) / (it + 1)
            
            # backprop stage
            max_delta = np.nan
            if sgd is not None:
                key, subkey = jax.random.split(key)
                max_delta, num_updates = self._jax_sgd(
                    subkey, train_subs, test_subs, vCs)                
            
            # callback
            if it % steps == 0 or it == epochs - 1:
                best = root.uct(0.0)
                action = root.children[best].action
                yield {
                    'return': R,
                    'action': action,
                    'best': best,
                    'iter': it,
                    'c': R0,
                    'max_delta': max_delta,
                    'sgd_updates': num_updates,
                }


if __name__ == '__main__':
    EnvInfo = ExampleManager.GetEnvInfo('Wildfire')
    rddl = RDDLEnv(EnvInfo.get_domain(), EnvInfo.get_instance(0)).model
    
    world = JaxRDDLSimulator(rddl, keep_tensors=True)
    planner = JaxRDDLHybridBackpropUCTPlanner(
        rddl, 
        action_bounds={}, 
        rollout_horizon=5, 
        alpha=0.5, 
        beta=0.5, 
        delta=1.0,
        plan=JaxStraightLinePlan(),
        batch_size_train=32,
        policy_hyperparams={'cut-out': 5.0, 'put-out': 5.0},
        logic=FuzzyLogic(weight=100.),
        optimizer_kwargs={'learning_rate': 0.1})
    
    print('starting MCTS...')
    world.reset()
    key = jax.random.PRNGKey(42)
    total_reward = 0
    for step in range(100):        
        print('\n' + f'iteration {step}...')
        s0 = world.subs.copy()
        for callback in planner.search(key, s0, 200, 20):
            print(f'iter={callback["iter"]}, '
                  f'reward={callback["return"]}, '
                  f'c={callback["c"]}, '
                  f'delta={callback["max_delta"]}, '
                  f'updates={callback["sgd_updates"]}')
        _, reward, done = world.step(callback["action"])
        print(f'reward = {reward}')
        total_reward += reward
        if done:
            break
    print(f'total reward {total_reward}')
    