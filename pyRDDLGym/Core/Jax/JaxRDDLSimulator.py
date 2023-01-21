import jax
import numpy as np
from typing import Dict

from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLActionPreconditionNotSatisfiedError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidExpressionError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLStateInvariantNotSatisfiedError

from pyRDDLGym.Core.Compiler.RDDLLiftedModel import RDDLLiftedModel
from pyRDDLGym.Core.Jax.JaxRDDLCompiler import JaxRDDLCompiler
from pyRDDLGym.Core.Simulator.RDDLSimulator import RDDLSimulator

from pyRDDLGym.Core.Debug.Logger import Logger
from pyRDDLGym.Core.Parser.expr import Value

Args = Dict[str, Value]


class JaxRDDLSimulator(RDDLSimulator):
        
    def __init__(self, rddl: RDDLLiftedModel,
                 key: jax.random.PRNGKey=None,
                 raise_error: bool=True,
                 logger: Logger=None,
                 **compiler_args) -> None:
        '''Creates a new simulator for the given RDDL model with Jax as a backend.
        
        :param rddl: the RDDL model
        :param key: the Jax PRNG key for sampling random variables
        :param raise_error: whether errors are raised as they are encountered in
        the Jax program: unlike the numpy sim, errors cannot be caught in the 
        middle of evaluating a Jax expression; instead they are accumulated and
        returned to the user upon complete evaluation of the expression
        :param logger: to log information about compilation to file
        :param **compiler_args: keyword arguments to pass to the Jax compiler
        '''
        if key is None:
            seed = np.random.randint(0, 2 ** 31 - 1)
            key = jax.random.PRNGKey(seed)
            
        self.rddl = rddl
        self.key = key
        self.raise_error = raise_error
        self.logger = logger
        
        # compilation
        compiled = JaxRDDLCompiler(rddl, logger=logger, **compiler_args)
        compiled.compile(log_jax_expr=True)
        self.init_values = compiled.init_values
        self.levels = compiled.levels
        self.traced = compiled.traced
        
        self.invariants = jax.tree_map(jax.jit, compiled.invariants)
        self.preconds = jax.tree_map(jax.jit, compiled.preconditions)
        self.terminals = jax.tree_map(jax.jit, compiled.termination)
        self.reward = jax.jit(compiled.reward)
        jax_cpfs = jax.tree_map(jax.jit, compiled.cpfs)
        
        # level analysis
        self.cpfs = []  
        for cpfs in self.levels.values():
            for cpf in cpfs:
                expr = jax_cpfs[cpf]
                prange = rddl.variable_ranges[cpf]
                dtype = JaxRDDLCompiler.JAX_TYPES.get(prange, JaxRDDLCompiler.INT)
                self.cpfs.append((cpf, expr, dtype))
        
        # initialize all fluent and non-fluent values    
        self.subs = self.init_values.copy() 
        self.state = None 
        self.noop_actions = {var: values 
                             for (var, values) in self.init_values.items() 
                             if rddl.variable_types[var] == 'action-fluent'}
        self._pomdp = bool(self.rddl.observ)
        
    def handle_error_code(self, error, msg) -> None:
        if self.raise_error:
            errors = JaxRDDLCompiler.get_error_messages(error)
            if errors:
                message = f'Internal error in evaluation of {msg}:\n'
                errors = '\n'.join(f'{i + 1}. {s}' for i, s in enumerate(errors))
                raise RDDLInvalidExpressionError(message + errors)
    
    def check_state_invariants(self) -> None:
        '''Throws an exception if the state invariants are not satisfied.'''
        for (i, invariant) in enumerate(self.invariants):
            sample, self.key, error = invariant(self.subs, self.key)
            self.handle_error_code(error, f'invariant {i + 1}')            
            if not bool(sample):
                raise RDDLStateInvariantNotSatisfiedError(
                    f'Invariant {i + 1} is not satisfied.')
    
    def check_action_preconditions(self, actions: Args) -> None:
        '''Throws an exception if the action preconditions are not satisfied.'''
        actions = self._process_actions(actions)
        subs = self.subs
        subs.update(actions)
        
        for (i, precond) in enumerate(self.preconds):
            sample, self.key, error = precond(subs, self.key)
            self.handle_error_code(error, f'precondition {i + 1}')            
            if not bool(sample):
                raise RDDLActionPreconditionNotSatisfiedError(
                    f'Precondition {i + 1} is not satisfied.')
    
    def check_terminal_states(self) -> bool:
        '''return True if a terminal state has been reached.'''
        for (i, terminal) in enumerate(self.terminals):
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
        actions = self._process_actions(actions)
        subs = self.subs
        subs.update(actions)
        
        # compute CPFs in topological order
        for (cpf, expr, _) in self.cpfs:
            subs[cpf], self.key, error = expr(subs, self.key)
            self.handle_error_code(error, f'CPF <{cpf}>')            
                
        # sample reward
        reward = self.sample_reward()
        
        # update state
        rddl = self.rddl
        self.state = {}
        for (state, next_state) in rddl.next_state.items():
            subs[state] = subs[next_state]
            self.state.update(rddl.ground_values(state, subs[state]))
        
        # update observation
        if self._pomdp: 
            obs = {}
            for var in rddl.observ:
                obs.update(rddl.ground_values(var, subs[var]))
        else:
            obs = self.state
        
        done = self.check_terminal_states()        
        return obs, reward, done
        
