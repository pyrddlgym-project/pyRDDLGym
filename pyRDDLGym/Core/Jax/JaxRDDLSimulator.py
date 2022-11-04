import jax
import jax.random as jrng
from typing import Dict

from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLActionPreconditionNotSatisfiedError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLStateInvariantNotSatisfiedError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLUndefinedCPFError
from pyRDDLGym.Core.Jax.JaxRDDLCompiler import JaxRDDLCompiler
from pyRDDLGym.Core.Parser.expr import Value
from pyRDDLGym.Core.Simulator.RDDLSimulator import RDDLSimulator
import sys

Args = Dict[str, Value]


class JaxRDDLSimulator(RDDLSimulator):
    
    def __init__(self, jax_model: JaxRDDLCompiler, key: jrng.PRNGKey) -> None:
        self._jax_model = jax_model
        self._key = key
        
        self._model = self._jax_model._model
        self._init_actions = self._model.actions.copy()
        self._action_fluents = set(self._init_actions.keys())
        self._subs = self._model.nonfluents.copy()
        self._pomdp = bool(self._model.observ)
    
    @staticmethod
    def _print_stack_trace(expr, subs, key):
        return str(jax.make_jaxpr(expr)(subs, key))
    
    def check_state_invariants(self) -> None:
        '''Throws an exception if the state invariants are not satisfied.'''
        for idx, invariant in enumerate(self._jax_model.invariants):
            sample, self._key = invariant(self._subs, self._key)
            sample = bool(sample)
            if not sample:
                raise RDDLStateInvariantNotSatisfiedError(
                    'State invariant {} is not satisfied.'.format(idx + 1) + 
                    '\n' + JaxRDDLSimulator._print_stack_trace(invariant, self._subs, self._key))
    
    def check_action_preconditions(self) -> None:
        '''Throws an exception if the action preconditions are not satisfied.'''
        for idx, precondition in enumerate(self._jax_model.preconditions):
            sample, self._key = precondition(self._subs, self._key)
            sample = bool(sample)
            if not sample:
                raise RDDLActionPreconditionNotSatisfiedError(
                    'Action precondition {} is not satisfied.'.format(idx + 1) + 
                    '\n' + JaxRDDLSimulator._print_stack_trace(precondition, self._subs, self._key))
    
    def check_terminal_states(self) -> bool:
        '''return True if a terminal state has been reached.'''
        for _, terminal in enumerate(self._jax_model.termination):
            sample, self._key = terminal(self._subs, self._key)
            sample = bool(sample)
            if sample:
                return True
        return False
    
    def sample_reward(self) -> float:
        '''Samples the current reward given the current state and action.'''
        reward, self._key = self._jax_model.jit_reward(self._subs, self._key)
        return float(reward)
    
    def reset(self) -> Args:
        '''Resets the state variables to their initial values.'''
        return super(JaxRDDLSimulator, self).reset()
            
    def step(self, actions: Args) -> Args:
        '''Samples and returns the next state from the cpfs.
        
        :param actions: a dict mapping current action fluents to their values
        '''
        
        # if actions are missing use their default values
        self._model.actions = {var: actions.get(var, default_value) 
                               for var, default_value in self._init_actions.items()}
        subs = self._subs 
        subs.update(self._model.actions)  
        
        # evaluate all CPFs, record next state fluents, update sub table 
        next_states, next_obs = {}, {}
        for order in self._jax_model.order_cpfs:
            for cpf in self._jax_model.cpforder[order]: 
                if cpf in self._model.next_state:
                    primed_cpf = self._model.next_state[cpf]
                    jaxpr = self._jax_model.jit_cpfs[primed_cpf]
                    sample, self._key = jaxpr(subs, self._key)
                    if primed_cpf not in self._model.prev_state:
                        raise KeyError(
                            'Internal error: variable <{}> is not in prev_state.'.format(primed_cpf))
                    next_states[cpf] = sample   
                    subs[primed_cpf] = sample   
                elif cpf in self._model.derived:
                    jaxpr = self._jax_model.jit_cpfs[cpf]
                    sample, self._key = jaxpr(subs, self._key)
                    self._model.derived[cpf] = sample
                    subs[cpf] = sample   
                elif cpf in self._model.interm:
                    jaxpr = self._jax_model.jit_cpfs[cpf]
                    sample, self._key = jaxpr(subs, self._key)
                    self._model.interm[cpf] = sample
                    subs[cpf] = sample
                elif cpf in self._model.observ:
                    jaxpr = self._jax_model.jit_cpfs[cpf]
                    sample, self._key = jaxpr(subs, self._key)
                    self._model.observ[cpf] = sample
                    subs[cpf] = sample
                    next_obs[cpf] = sample
                else:
                    raise RDDLUndefinedCPFError('CPF <{}> is not defined.'.format(cpf))     
        
        # evaluate the immediate reward
        reward = self.sample_reward()
        
        # update the internal model state and the sub table
        self._model.states = {}
        for cpf, value in next_states.items():
            self._model.states[cpf] = value
            subs[cpf] = value
            primed_cpf = self._model.next_state[cpf]
            subs[primed_cpf] = None
        
        # check the termination condition
        done = self.check_terminal_states()
        
        if self._pomdp:
            return next_obs, reward, done
        else:
            return next_states, reward, done
    
