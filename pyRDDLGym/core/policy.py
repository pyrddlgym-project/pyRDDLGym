from abc import ABCMeta, abstractmethod
import numpy as np
import random
import gymnasium as gym
import shutil
from typing import Any, Dict, Optional
import sys

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.env import RDDLEnv
from pyRDDLGym.core.debug.exception import RDDLActionPreconditionNotSatisfiedError
from pyRDDLGym.core.simulator import RDDLSimulator


class BaseAgent(metaclass=ABCMeta):
    '''Base class for policies.'''
    
    use_tensor_obs = False  # uses internal tensor representation of state (None: either)
    
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
        if self.use_tensor_obs is not None and env.vectorized != self.use_tensor_obs:
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
    '''Random policy that only samples actions satisfying the action preconditions in
    the current state of the environment, which must be the state being acted on.

    Each try sets a random number of action elements to non-default values, or none with
    the same probability as setting any one element. Actions that are legal are not
    equally likely, and an exception is raised if none is found in max_tries tries.'''

    use_tensor_obs = None

    def __init__(self, env: RDDLEnv, num_actions: Optional[int]=None,
                 seed: Optional[int]=None, max_tries: int=1000) -> None:
        '''Creates a new random policy.

        :param env: the environment whose action preconditions are checked
        :param num_actions: the most action elements to set per action (defaults to
        env.max_allowed_actions)
        :param seed: optional RNG seed
        :param max_tries: how many actions to try per sample
        '''
        self.requested_actions = num_actions
        self.seed = seed
        self.max_tries = max_tries
        self.rng = random.Random(seed)
        self._set_env(env)

    def _set_env(self, env: RDDLEnv) -> None:
        sampler = env.sampler
        self.env = env
        self.noop_actions = (sampler.noop_actions if env.vectorized
                             else sampler.grounded_noop_actions)

        # an element whose only possible value is its default cannot be set
        self.elements = [(key, index) for (key, value) in self.noop_actions.items()
                         for index in range(np.size(value))
                         if self._has_choice(env.action_space[key], index)]
        num_actions = self.requested_actions
        if num_actions is None:
            num_actions = env.max_allowed_actions
        self.num_actions = min(num_actions, env.max_allowed_actions, len(self.elements))
        if self.seed is not None:
            env.action_space.seed(self.seed)

    @staticmethod
    def _has_choice(space: Any, index: int) -> bool:
        if isinstance(space, gym.spaces.Discrete):
            return space.n > 1
        else:
            return np.ravel(space.low)[index] < np.ravel(space.high)[index]

    def _sample_non_default(self, key: str, index: int) -> Any:
        default = np.ravel(self.noop_actions[key])[index]
        if isinstance(default, np.bool_):
            return not default
        while True:
            value = np.ravel(self.env.action_space[key].sample())[index]
            if value != default:
                return value

    def _sample_elements(self) -> Dict[str, Any]:
        if self.num_actions < 1 or self.rng.randrange(len(self.elements) + 1) == 0:
            return {}
        count = self.rng.randint(1, self.num_actions)
        action = {}
        for (key, index) in self.rng.sample(self.elements, count):
            values = action.setdefault(key, np.array(self.noop_actions[key]))
            values.flat[index] = self._sample_non_default(key, index)
        return {key: values[()] for (key, values) in action.items()}

    def sample_action(self, state: Any=None) -> Any:
        sampler = self.env.sampler
        for _ in range(self.max_tries):
            action = self._sample_elements()
            sim_actions = sampler.prepare_actions_for_sim(action)
            if sampler.check_action_preconditions(sim_actions, silent=True):
                return action
        raise RDDLActionPreconditionNotSatisfiedError(
            f'No action satisfying the action preconditions was found in '
            f'{self.max_tries} tries: num_actions may be too small or too large.')

    def evaluate(self, env: RDDLEnv, *args, **kwargs) -> Dict[str, float]:
        if env is not self.env:
            self._set_env(env)
        return super().evaluate(env, *args, **kwargs)


class RolloutAgent(RandomAgent):
    '''Monte Carlo rollout policy, which samples legal actions with the random policy and
    returns the one with the highest return, averaged over simulated rollouts that follow
    the random policy after the first step. Sequential halving decides how many rollouts
    each candidate gets.'''

    def __init__(self, env: RDDLEnv, num_candidates: int, num_rollouts: int,
                 horizon: Optional[int]=None, num_actions: Optional[int]=None,
                 seed: Optional[int]=None, max_tries: int=1000) -> None:
        '''Creates a new rollout policy.

        :param env: the environment to simulate, using the default simulator backend
        :param num_candidates: how many legal actions to compare per sample
        :param num_rollouts: how many rollouts per candidate action on average, with
        more given to the better candidates
        :param horizon: the most steps per rollout (defaults to the env horizon)
        :param num_actions: the most action elements to set per action (defaults to
        env.max_allowed_actions)
        :param seed: optional RNG seed
        :param max_tries: how many actions to try per sample of the random policy
        '''
        self.num_candidates = num_candidates
        self.num_rollouts = num_rollouts
        self.horizon = horizon
        self.rollout_rng = np.random.default_rng(seed)
        super().__init__(env, num_actions, seed, max_tries)

    def _set_env(self, env: RDDLEnv) -> None:
        if type(env.sampler) is not RDDLSimulator:
            raise ValueError('RolloutAgent requires the default RDDLSimulator backend.')
        elif env.sampler.is_pomdp:
            raise ValueError('RolloutAgent does not support POMDPs.')
        super()._set_env(env)

    def _rollout(self, action: Any) -> float:
        env = self.env
        sampler = env.sampler
        steps = env.horizon - env.timestep
        if self.horizon is not None:
            steps = min(steps, self.horizon)

        total_reward, cuml_gamma = 0.0, 1.0
        for _ in range(steps):
            _, reward, done = sampler.step(sampler.prepare_actions_for_sim(action))
            total_reward += reward * cuml_gamma
            cuml_gamma *= env.discount
            if done or not sampler.check_state_invariants(silent=True):
                break
            try:
                action = super().sample_action()
            except RDDLActionPreconditionNotSatisfiedError:
                break
        return total_reward

    def sample_action(self, state: Any=None) -> Any:
        sampler = self.env.sampler
        subs, sim_state, rng = sampler.subs, sampler.state, sampler.rng
        sampler.rng = self.rollout_rng
        try:
            actions = []
            for _ in range(self.num_candidates):
                sampler.subs = subs.copy()
                actions.append(super().sample_action())

            # sequential halving
            alive = np.arange(self.num_candidates)
            values = np.zeros(self.num_candidates)
            budget = self.num_candidates * self.num_rollouts
            for rounds in range(int(np.ceil(np.log2(alive.size))), 0, -1):
                rollouts = budget // (alive.size * rounds)
                rollouts = max(rollouts, int(alive.size == self.num_candidates))
                budget -= rollouts * alive.size
                for index in alive:
                    for _ in range(rollouts):
                        sampler.subs = subs.copy()
                        values[index] += self._rollout(actions[index])
                order = np.argsort(-values[alive], kind='stable')
                alive = alive[order[:(alive.size + 1) // 2]]
            return actions[alive[0]]
        finally:
            sampler.subs, sampler.state, sampler.rng = subs, sim_state, rng


class RandomShootingAgent(RolloutAgent):
    '''Random shooting policy, which simulates one random action sequence per candidate
    and returns the first action of the sequence with the highest return.'''

    def __init__(self, env: RDDLEnv, num_candidates: int,
                 horizon: Optional[int]=None, num_actions: Optional[int]=None,
                 seed: Optional[int]=None, max_tries: int=1000) -> None:
        '''Creates a new random shooting policy.

        :param env: the environment to simulate, using the default simulator backend
        :param num_candidates: how many action sequences to compare per sample
        :param horizon: the most steps per sequence (defaults to the env horizon)
        :param num_actions: the most action elements to set per action (defaults to
        env.max_allowed_actions)
        :param seed: optional RNG seed
        :param max_tries: how many actions to try per sample of the random policy
        '''
        super().__init__(env, num_candidates, 1, horizon, num_actions, seed, max_tries)


class NoOpAgent(BaseAgent):
    '''No-op policy.'''

    use_tensor_obs = None

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
