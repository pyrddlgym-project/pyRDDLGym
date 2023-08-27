from abc import ABCMeta, abstractmethod
import numpy as np
import random
import gym
from typing import Dict

from pyRDDLGym.Core.Env.RDDLEnv import RDDLEnv


class BaseAgent(metaclass=ABCMeta):
    '''Base class for policies.'''

    @abstractmethod
    def sample_action(self, state: object) -> object:
        '''Samples an action from the current policy evaluated at the given state.
        
        :param state: the current state
        '''
        pass
    
    def reset(self) -> None:
        '''Resets the policy and prepares it for the next episode.'''
        pass
    
    def evaluate(self, env: RDDLEnv, ground_state: bool=True,
                 episodes: int=1, verbose: bool=False, render: bool=False) -> Dict[str, float]:
        '''Evaluates the current agent on the specified environment by simulating
        roll-outs. Returns a dictionary of summary statistics of the returns
        accumulated on the roll-outs.
        
        :param env: the environment
        :param ground_state: whether to use the grounded state from the env, 
        or the internal tensor form (which also contains other fluent information)
        :param episodes: how many episodes (trials) to perform
        :param verbose: whether to print the transition information to console
        at each step of the simulation
        :param render: visualize the domain using the env internal visualizer
        '''
        gamma = env.discount
        
        history = np.zeros((episodes,))
        for episode in range(episodes):
            
            # restart episode
            total_reward, cuml_gamma = 0.0, 1.0
            self.reset()
            state = env.reset()
            for step in range(env.horizon):
                if render:
                    env.render()
                
                # take a step in the environment
                if ground_state:
                    policy_input = state
                else:
                    policy_input = env.sampler.subs
                action = self.sample_action(policy_input)    
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

    def __init__(self, action_space, num_actions=1, seed=None):
        self.action_space = action_space
        self.num_actions = num_actions
        self.rng = random.Random(seed)
        if seed is not None:
            self.action_space.seed(seed)

    def sample_action(self, state=None):
        s = self.action_space.sample()
        action = {}
        selected_actions = self.rng.sample(list(s), self.num_actions)
        for sample in selected_actions:
            if isinstance(self.action_space[sample], gym.spaces.Box):
                action[sample] = s[sample][0].item()
            elif isinstance(self.action_space[sample], gym.spaces.Discrete):
                action[sample] = s[sample]
                # if str(self.action_space[sample]) == 'Discrete(2)':
                #     action[sample] = bool(s[sample])
                # else:
                #     action[sample] = s[sample]
        return action


class NoOpAgent(BaseAgent):

    def __init__(self, action_space, num_actions=0):
        self.action_space = action_space
        self.num_actions = num_actions

    def sample_action(self, state=None):
        action = {}
        return action

