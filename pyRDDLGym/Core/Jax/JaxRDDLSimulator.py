import jax
from typing import Dict

from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLActionPreconditionNotSatisfiedError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidExpressionError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLStateInvariantNotSatisfiedError

from pyRDDLGym.Core.Compiler.RDDLLiftedModel import RDDLLiftedModel
from pyRDDLGym.Core.Jax.JaxRDDLCompiler import JaxRDDLCompiler
from pyRDDLGym.Core.Parser.expr import Value
from pyRDDLGym.Core.Simulator.RDDLSimulator import RDDLSimulator

Args = Dict[str, Value]


class JaxRDDLSimulator(RDDLSimulator):
        
    def __init__(self, rddl: RDDLLiftedModel,
                 key: jax.random.PRNGKey,
                 raise_error: bool=False,
                 **compiler_args) -> None:
        self.rddl = rddl
        self.key = key
        self.raise_error = raise_error
        
        # static analysis and compilation
        compiled = JaxRDDLCompiler(rddl, **compiler_args)
        compiled.compile()
        self.compiled = compiled
        self.static = compiled.static
        self.levels = compiled.levels
        self.tensors = compiled.tensors
        
        self.invariants = jax.tree_map(jax.jit, compiled.invariants)
        self.preconds = jax.tree_map(jax.jit, compiled.preconditions)
        self.terminals = jax.tree_map(jax.jit, compiled.termination)
        self.reward = jax.jit(compiled.reward)
        self.cpfs = jax.tree_map(jax.jit, compiled.cpfs)
        
        # initialize all fluent and non-fluent values
        self.init_values, self.noop_actions = \
            compiled.init_values, compiled.noop_actions
        self.subs = self.init_values.copy()
        self.next_states = compiled.next_states
        self.state = None
        
        # is a POMDP
        self.observ_fluents = [var 
                               for var, ftype in rddl.variable_types.items()
                               if ftype == 'observ-fluent']
        self._pomdp = bool(self.observ_fluents)
        
    def handle_error_code(self, error, msg):
        if self.raise_error:
            errors = JaxRDDLCompiler.get_error_messages(error)
            if errors:
                message = f'Internal error in evaluation of {msg}:\n'
                errors = '\n'.join(f'{i + 1}. {s}' for i, s in enumerate(errors))
                raise RDDLInvalidExpressionError(message + errors)
    
    def _check_state_invariants(self) -> None:
        '''Throws an exception if the state invariants are not satisfied.'''
        for i, invariant in enumerate(self.invariants):
            sample, self.key, error = invariant(self.subs, self.key)
            self.handle_error_code(error, f'invariant {i + 1}')
            
            if not bool(sample):
                raise RDDLStateInvariantNotSatisfiedError(
                    f'Invariant {i + 1} is not satisfied.')
    
    def check_action_preconditions(self, actions: Args) -> None:
        '''Throws an exception if the action preconditions are not satisfied.'''
        actions = self._actions_to_tensors(actions)
        subs = self.subs
        subs.update(actions)
        
        for i, precond in enumerate(self.preconds):
            sample, self.key, error = precond(self.subs, self.key)
            self.handle_error_code(error, f'precondition {i + 1}')
            
            if not bool(sample):
                raise RDDLActionPreconditionNotSatisfiedError(
                    f'Precondition {i + 1} is not satisfied.')
    
    def check_terminal_states(self) -> bool:
        '''return True if a terminal state has been reached.'''
        for i, terminal in enumerate(self.terminals):
            sample, self.key, error = terminal(self.subs, self.key)
            self.handle_error_code(error, f'termination {i + 1}')           
             
            if bool(sample):
                return True
        return False
    
    def sample_reward(self) -> float:
        '''Samples the current reward given the current state and action.'''
        reward, self.key, error = self.reward(self.subs, self.key)
        self.handle_error_code(error, 'reward function')
        return float(reward)
    
    def step(self, actions: Args) -> Args:
        '''Samples and returns the next state from the cpfs.
        
        :param actions: a dict mapping current action fluents to their values
        '''
        actions = self._actions_to_tensors(actions)
        subs = self.subs
        subs.update(actions)
        
        for _, cpfs in self.levels.items():
            for cpf in cpfs:
                subs[cpf], self.key, error = self.cpfs[cpf](subs, self.key)
                self.handle_error_code(error, f'CPF <{cpf}>')            
        reward = self.sample_reward()
        
        for next_state, state in self.next_states.items():
            subs[state] = subs[next_state]
        
        self.state = {}
        for var in self.next_states.values():
            self.state.update(self.tensors.expand(var, subs[var]))
        
        if self._pomdp: 
            obs = {}
            for var in self.observ_fluents:
                obs.update(self.tensors.expand(var, subs[var]))
            obs = {o: None for o in obs.keys()}
        else:
            obs = self.state
        
        done = self.check_terminal_states()        
        return obs, reward, done
        
