from abc import ABCMeta, abstractmethod
import numpy as np
import random
import gymnasium as gym
import shutil
from typing import Any, Dict, Optional
import sys

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.env import RDDLEnv
from pyRDDLGym.core.debug.exception import RDDLRandPolicyVecNotImplemented


class BaseAgent(metaclass=ABCMeta):
    '''Base class for policies.'''
    
    use_tensor_obs = False  # uses internal tensor representation of state
    
    @abstractmethod
    def sample_action(self, state: Any) -> Any:
        '''Samples an action from the current policy evaluated at the given state.
        
        :param state: the current state
        '''
        pass
    
    def reset(self) -> None:
        '''Resets the policy and prepares it for the next episode.'''
        pass
    
    @staticmethod
    def _format_key(key):
        '''Shows grounded fluents in RDDL notation, e.g. burning___x1__y1 -> burning(x1, y1).'''
        try:
            name, objects = RDDLPlanningModel.parse_grounded(str(key))
        except Exception:
            return str(key)
        return f'{name}({", ".join(objects)})' if objects else name

    @staticmethod
    def _format_value(value):
        if isinstance(value, np.ndarray) and value.ndim > 0:
            return np.array2string(value, precision=4, floatmode='fixed', separator=', ',
                                   sign=' ')
        elif isinstance(value, (float, np.floating)):
            if value != 0 and not 1e-4 <= abs(value) < 1e9:
                return f'{value: .4e}'
            else:
                return f'{value: .4f}'
        else:
            return str(value)
    
    @staticmethod
    def _format(state, width=80, indent=4):
        if len(state) == 0:
            return ' ' * indent + '{}'
        pad = ' ' * indent
        values = {key: BaseAgent._format_value(value) for (key, value) in state.items()}

        # arrays (vectorized) are one per row, aligned under the value
        if any('\n' in value or value.startswith('[') for value in values.values()):
            klen = max(len(str(key)) for key in values)
            cont = '\n' + ' ' * (indent + klen + 2)
            text = '\n'.join(f'{pad}{str(key).rjust(klen)}  {value.replace(chr(10), cont)}'
                             for (key, value) in values.items())
            return '\n'.join(line.rstrip() for line in text.split('\n'))

        # each fluent is a row of its values by objects
        rows = {}
        for (key, value) in values.items():
            try:
                name, objects = RDDLPlanningModel.parse_grounded(str(key))
            except Exception:
                name, objects = str(key), []
            rows.setdefault(name, {})[', '.join(objects)] = value
        tables = {}
        for (name, row) in rows.items():
            tables.setdefault(tuple(row), []).append((name, row))

        # print each table, split into several if its columns do not fit the width,
        # with fluent names right-aligned so that all tables start at the same column
        nlen = max(len(name) for name in rows)
        lines = []
        for (columns, table) in tables.items():
            widths = [max(len(column), *(len(row[column]) for (_, row) in table))
                      for column in columns]
            start = 0
            while start < len(columns):
                end, used = start + 1, indent + nlen + 2 + widths[start]
                while end < len(columns) and used + 2 + widths[end] <= width:
                    used += 2 + widths[end]
                    end += 1
                if any(columns):
                    lines.append(pad + ' ' * nlen + ''.join(
                        '  ' + columns[c].rjust(widths[c]) for c in range(start, end)))
                for (name, row) in table:

                    # a single fluent continued from the previous rows is not named again
                    if start and len(table) == 1:
                        name = ''
                    lines.append(pad + name.rjust(nlen) + ''.join(
                        '  ' + row[columns[c]].rjust(widths[c]) for c in range(start, end)))
                start = end
        return '\n'.join(line.rstrip() for line in lines)
        
    def evaluate(self, env: RDDLEnv, episodes: int=1, 
                 verbose: bool=False, render: bool=False,
                 seed: Optional[int]=None) -> Dict[str, float]:
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
            raise ValueError(f'RDDLEnv vectorized flag must match use_tensor_obs '
                             f'of current policy, got {env.vectorized} and '
                             f'{self.use_tensor_obs}, respectively.')
        
        gamma = env.discount
        
        # get terminal width
        if verbose:
            width = shutil.get_terminal_size().columns
            sep_bar = '-' * width
        
        # start simulation
        history = np.zeros((episodes,))
        for episode in range(episodes):
            
            # restart episode
            total_reward, cuml_gamma = 0.0, 1.0
            self.reset()
            state, _ = env.reset(seed=seed)
            
            # simulate to end of horizon
            for step in range(env.horizon):
                if render:
                    env.render()
                
                # take a step in the environment
                action = self.sample_action(state)   
                next_state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward * cuml_gamma
                cuml_gamma *= gamma
                done = terminated or truncated
                
                # printing
                if verbose: 
                    print(f'{sep_bar}\n'
                          f'step {step}   reward {self._format_value(reward).strip()}   '
                          f'done {done}\n'
                          f'state\n{self._format(state, width)}\n'
                          f'action\n{self._format(action, width)}')
                state = next_state
                if done:
                    break
            
            if render:
                env.render()
            if verbose:
                print(f'{sep_bar}\n'
                      f'final state\n{self._format(state, width)}\n\n'
                      f'episode {episode + 1} ended with return '
                      f'{self._format_value(total_reward).strip()}\n'
                      f'{"=" * width}')
            history[episode] = total_reward
            
            # set the seed on the first episode only
            seed = None
        
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

    def __init__(self, action_space: Any, 
                 num_actions: int=1, seed: Optional[int]=None) -> None:
        '''Creates a new uniformly pseudo-random policy.
        
        :param action_space: the set of actions from which to sample uniformly
        :param num_actions: the number of samples to produce
        '''
        self.action_space = action_space
        self.action_buckets = []
        self.full_action_space = True

        self.vectorized = False
        for key, value in self.action_space.items():
            if len(value.shape) >= 1:
                if value.shape[0] > 1:
                    self.vectorized = True
                self.action_buckets.append(value.shape[0])
            else:
                self.action_buckets.append(1)
        if sum(self.action_buckets) - num_actions > 0:
            self.full_action_space = False

        if self.vectorized and not self.full_action_space:
            raise RDDLRandPolicyVecNotImplemented(
                f'Random Agent does not support vectorized spaces '
                f'for partial action specification, '
                f'number of required actions is {num_actions}, '
                f'where {sum(self.action_buckets)} is expected.')

        self.num_actions = num_actions
        self.rng = random.Random(seed)
        if seed is not None:
            self.action_space.seed(seed)

    def sample_action(self, state: Any=None) -> Any:
        s = self.action_space.sample()
        action = {}
        if not self.vectorized:
            selected_actions = set(self.rng.sample(list(s), self.num_actions))
            for sample in (key for key in s if key in selected_actions):
                if isinstance(self.action_space[sample], gym.spaces.Box):
                    action[sample] = s[sample].item()
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
