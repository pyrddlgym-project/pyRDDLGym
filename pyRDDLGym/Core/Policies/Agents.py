from abc import ABCMeta, abstractmethod
import numpy as np
import random
import gym
from typing import Dict

from pyRDDLGym.Core.Env.RDDLEnv import RDDLEnv
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLRandPolicyVecNotImplemented

class BaseAgent(metaclass=ABCMeta):
    '''Base class for policies.'''
    use_tensor_obs = False  # uses internal tensor representation of state
    
    @abstractmethod
    def sample_action(self, state: object) -> object:
        '''Samples an action from the current policy evaluated at the given state.
        
        :param state: the current state
        '''
        pass
    
    def reset(self) -> None:
        '''Resets the policy and prepares it for the next episode.'''
        pass
    
    def evaluate(self, env: RDDLEnv, episodes: int=1, 
                 verbose: bool=False, render: bool=False, 
                 seed: int=None) -> Dict[str, float]:
        '''Evaluates the current agent on the specified environment by simulating
        roll-outs. Returns a dictionary of summary statistics of the returns
        accumulated on the roll-outs.
        
        :param env: the environment
        :param episodes: how many episodes (trials) to perform
        :param verbose: whether to print the transition information to console
        at each step of the simulation
        :param render: visualize the domain using the env internal visualizer
        :param seed: optional RNG seed for the environment
        '''
        
        # check compatibility with environment
        if env.vectorized != self.use_tensor_obs:
            raise Exception(f'RDDLEnv vectorized flag must match use_tensor_obs '
                            f'of current policy, got {env.vectorized} and '
                            f'{self.use_tensor_obs}, respectively.')
        
        gamma = env.discount
        
        history = np.zeros((episodes,))
        for episode in range(episodes):
            
            # restart episode
            total_reward, cuml_gamma = 0.0, 1.0
            self.reset()
            if env.new_gym_api:
                state, _ = env.reset(seed=seed)
            else:
                state = env.reset(seed=seed)
            
            # simulate to end of horizon
            for step in range(env.horizon):
                if render:
                    env.render()
                
                # take a step in the environment
                action = self.sample_action(state)   
                if env.new_gym_api: 
                    next_state, reward, done, _, _ = env.step(action)
                else:
                    next_state, reward, done, _ = env.step(action)
                total_reward += reward * cuml_gamma
                cuml_gamma *= gamma
                
                # printing
                if verbose: 
                    print(f'step       = {step}\n'
                          f'state      = {state}\n'
                          f'action     = {action}\n'
                          f'next state = {next_state}\n'
                          f'reward     = {reward}\n')
                state = next_state
                if done:
                    break
            
            if verbose:
                print(f'episode {episode + 1} ended with return {total_reward}')
            history[episode] = total_reward
        
        # summary statistics
        return {
            'mean': np.mean(history),
            'median': np.median(history),
            'min': np.min(history),
            'max': np.max(history),
            'std': np.std(history)
        }


class RandomAgent(BaseAgent):
   '''Uniformly pseudo-random policy.'''
   def __init__(self, action_space, num_actions=1, seed=None, vectorized=False, noop_values=None):
       '''Creates a new uniformly pseudo-random policy.
       :param action_space: the set of actions from which to sample uniformly
       :param num_actions: the number of samples to produce
       '''
       self.action_space = action_space
       self.noop_values = noop_values
       self.action_buckets = []
       self.full_action_space = True

       self.vectorized = vectorized
       for key, value in self.action_space.items():
           if len(value.shape) >= 1:
               self.action_buckets.append(value.shape[0])
           else:
               self.action_buckets.append(1)
       if sum(self.action_buckets) - num_actions > 0:
           self.full_action_space = False

       if self.vectorized and not self.full_action_space:
           raise RDDLRandPolicyVecNotImplemented(
               f'Random Agent does not support vectorized spaces for partial action specification, '
               f'number of required actions is {num_actions}, where {sum(self.action_buckets)} is expected.')

       self.num_actions = num_actions
       self.rng = random.Random(seed)
       if seed is not None:
           self.action_space.seed(seed)

   def sample_action(self, state=None):
       s = self.action_space.sample()
       action = {}
       if not self.vectorized:
           selected_actions = self.rng.sample(list(s), self.num_actions)
           for sample in selected_actions:
               if isinstance(self.action_space[sample], gym.spaces.Box):
                   action[sample] = s[sample][0].item()
               elif isinstance(self.action_space[sample], gym.spaces.Discrete):
                   action[sample] = s[sample]
       else:
           action = s
       return action


class NoOpAgent(BaseAgent):
    '''No-op policy.'''
    
    def __init__(self, action_space, num_actions=0):
        '''Creates a new no-op policy.
            
        :param action_space: the set of actions (currently unused)
        :param num_actions: the number of samples to produce (currently unused)
        '''
        self.action_space = action_space
        self.num_actions = num_actions

    def sample_action(self, state=None):
        action = {}
        return action

