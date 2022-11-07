import jax
import jax.random as random
import numpy as np
from typing import Dict

from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLActionPreconditionNotSatisfiedError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLStateInvariantNotSatisfiedError
from pyRDDLGym.Core.Jax.JaxRDDLCompiler import JaxRDDLCompiler
from pyRDDLGym.Core.Parser.rddl import RDDL
from pyRDDLGym.Core.Parser.expr import Value
from pyRDDLGym.Core.Simulator.RDDLSimulator import RDDLSimulator

Args = Dict[str, Value]


class JaxRDDLSimulator(RDDLSimulator):
    
    def __init__(self, rddl: RDDL, key: random.PRNGKey) -> None:
        self.rddl = rddl
        self.key = key
        
        self.compiler = JaxRDDLCompiler(rddl)
        data = self.compiler.compile()
        self.invariants = data['invariants']
        self.preconds = data['preconds']
        self.terminals = data['terminals']
        _, self.reward = data['reward']
        _, self.cpfs = data['cpfs']
        
        # initialization of scalar and pvariables tensors
        rev_objects = {}          
        self.init_values = {}
        for obj, values in rddl.non_fluents.objects:
            rev_objects.update(zip(values, [obj] * len(values)))   
        for pvar in rddl.domain.pvariables:
            name = pvar.name
            params = pvar.param_types
            if params is None:
                self.init_values[name] = pvar.default
            else:
                self.init_values[name] = np.zeros(
                    shape=tuple(len(self.compiler.objects[p]) for p in params))    
        if hasattr(rddl.non_fluents, 'init_non_fluent'): 
            for (name, params), value in rddl.non_fluents.init_non_fluent:
                if params is not None:
                    coords = tuple(self.compiler.objects[rev_objects[p]][p] for p in params)
                    self.init_values[name][coords] = value                
        self.subs = self.init_values.copy()
        
    @staticmethod
    def _print_stack_trace(expr, subs, key):
        return str(jax.make_jaxpr(expr)(subs, key))
    
    def _check_state_invariants(self) -> None:
        '''Throws an exception if the state invariants are not satisfied.'''
        for idx, invariant in enumerate(self.invariants):
            sample, self.key = invariant(self.subs, self.key)
            if not bool(sample):
                raise RDDLStateInvariantNotSatisfiedError(
                    'State invariant {} is not satisfied.'.format(idx + 1) + 
                    '\n' + JaxRDDLSimulator._print_stack_trace(invariant, self.subs, self.key))
    
    def check_action_preconditions(self) -> None:
        '''Throws an exception if the action preconditions are not satisfied.'''
        for idx, precond in enumerate(self.preconds):
            sample, self.key = precond(self.subs, self.key)
            if not bool(sample):
                raise RDDLActionPreconditionNotSatisfiedError(
                    'Action precondition {} is not satisfied.'.format(idx + 1) + 
                    '\n' + JaxRDDLSimulator._print_stack_trace(precond, self.subs, self.key))
    
    def check_terminal_states(self) -> bool:
        '''return True if a terminal state has been reached.'''
        for terminal in self.terminals:
            sample, self.key = terminal(self.subs, self.key)
            if bool(sample):
                return True
        return False
    
    def sample_reward(self) -> float:
        '''Samples the current reward given the current state and action.'''
        reward, self.key = self.reward(self.subs, self.key)
        return float(reward)
    
    def reset(self) -> Args:
        '''Resets the state variables to their initial values.'''
        self.subs = self.init_values.copy()  
        obs = {var: self.subs[var] for var in self.compiler.states.values()}
        done = self.check_terminal_states()    
        return obs, done
    
    def step(self, actions: Args) -> Args:
        '''Samples and returns the next state from the cpfs.
        
        :param actions: a dict mapping current action fluents to their values
        '''
        subs = self.subs 
        subs.update(actions)
        
        for cpf, expr in self.cpfs.items():
            subs[cpf], self.key = expr(subs, self.key)
        next_subs = subs.copy()
        for cpf in self.cpfs.keys():
            unprimed = self.compiler.states.get(cpf, cpf)
            next_subs[unprimed] = subs[cpf]

        obs = {var: next_subs[var] for var in self.compiler.states.values()}
        reward = self.sample_reward()
        self.subs = next_subs
        done = self.check_terminal_states()
        return obs, reward, done
        
