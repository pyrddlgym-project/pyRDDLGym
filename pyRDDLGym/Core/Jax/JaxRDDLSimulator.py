import jax
import time
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
                 keep_tensors: bool=False,
                 **compiler_args) -> None:
        '''Creates a new simulator for the given RDDL model with Jax as a backend.
        
        :param rddl: the RDDL model
        :param key: the Jax PRNG key for sampling random variables
        :param raise_error: whether errors are raised as they are encountered in
        the Jax program: unlike the numpy sim, errors cannot be caught in the 
        middle of evaluating a Jax expression; instead they are accumulated and
        returned to the user upon complete evaluation of the expression
        :param logger: to log information about compilation to file
        :param keep_tensors: whether the sampler takes actions and
        returns state in numpy array form
        :param **compiler_args: keyword arguments to pass to the Jax compiler
        '''
        if key is None:
            key = jax.random.PRNGKey(round(time.time() * 1000))
        self.key = key
        self.raise_error = raise_error
        self.compiler_args = compiler_args
        
        # generate direct sampling with default numpy RNG and operations
        super(JaxRDDLSimulator, self).__init__(
            rddl, logger=logger, keep_tensors=keep_tensors)
    
    def seed(self, seed: int) -> None:
        super(JaxRDDLSimulator, self).seed(seed)
        self.key = jax.random.PRNGKey(seed)
        
    def _compile(self):
        rddl = self.rddl
        
        # compilation
        if self.logger is not None:
            self.logger.clear()
        compiled = JaxRDDLCompiler(rddl, logger=self.logger, **self.compiler_args)
        compiled.compile(log_jax_expr=True)
        self.init_values = compiled.init_values
        self.levels = compiled.levels
        self.traced = compiled.traced
        
        self.invariants = jax.tree_map(jax.jit, compiled.invariants)
        self.preconds = jax.tree_map(jax.jit, compiled.preconditions)
        self.terminals = jax.tree_map(jax.jit, compiled.termination)
        self.reward = jax.jit(compiled.reward)
        jax_cpfs = jax.tree_map(jax.jit, compiled.cpfs)
        self.model_params = compiled.model_params
        
        # level analysis
        self.cpfs = []  
        for cpfs in self.levels.values():
            for cpf in cpfs:
                expr = jax_cpfs[cpf]
                prange = rddl.variable_ranges[cpf]
                dtype = compiled.JAX_TYPES.get(prange, compiled.INT)
                self.cpfs.append((cpf, expr, dtype))
        
        # initialize all fluent and non-fluent values    
        self.subs = self.init_values.copy() 
        self.state = None 
        self.noop_actions = {var: values 
                             for (var, values) in self.init_values.items() 
                             if rddl.variable_types[var] == 'action-fluent'}
        self._pomdp = bool(rddl.observ)
        
        # cached for performance
        self.invariant_names = [f'Invariant {i}' for i in range(len(rddl.invariants))]        
        self.precond_names = [f'Precondition {i}' for i in range(len(rddl.preconditions))]
        self.terminal_names = [f'Termination {i}' for i in range(len(rddl.terminals))]
        
        self.grounded_actionsranges = rddl.groundactionsranges()
        self.grounded_noop_actions = rddl.ground_values_from_dict(self.noop_actions)
        
    def handle_error_code(self, error, msg) -> None:
        if self.raise_error:
            errors = JaxRDDLCompiler.get_error_messages(error)
            if errors:
                message = f'Internal error in evaluation of {msg}:\n'
                errors = '\n'.join(f'{i + 1}. {s}' for (i, s) in enumerate(errors))
                raise RDDLInvalidExpressionError(message + errors)
    
    def check_state_invariants(self, silent: bool=False) -> bool:
        '''Throws an exception if the state invariants are not satisfied.'''
        for (i, invariant) in enumerate(self.invariants):
            loc = self.invariant_names[i]
            sample, self.key, error = invariant(
                self.subs, self.model_params, self.key)
            self.handle_error_code(error, loc)            
            if not bool(sample):
                if not silent:
                    raise RDDLStateInvariantNotSatisfiedError(
                        f'{loc} is not satisfied.')
                return False
        return True
    
    def check_action_preconditions(self, actions: Args, silent: bool=False) -> bool:
        '''Throws an exception if the action preconditions are not satisfied.'''
        actions = self._process_actions(actions)
        subs = self.subs
        subs.update(actions)
        
        for (i, precond) in enumerate(self.preconds):
            loc = self.precond_names[i]
            sample, self.key, error = precond(subs, self.model_params, self.key)
            self.handle_error_code(error, loc)            
            if not bool(sample):
                if not silent:
                    raise RDDLActionPreconditionNotSatisfiedError(
                        f'{loc} is not satisfied for actions {actions}.')
                return False
        return True
    
    def check_terminal_states(self) -> bool:
        '''return True if a terminal state has been reached.'''
        for (i, terminal) in enumerate(self.terminals):
            loc = self.terminal_names[i]
            sample, self.key, error = terminal(
                self.subs, self.model_params, self.key)
            self.handle_error_code(error, loc)
            if bool(sample):
                return True
        return False
    
    def sample_reward(self) -> float:
        '''Samples the current reward given the current state and action.'''
        reward, self.key, error = self.reward(
            self.subs, self.model_params, self.key)
        self.handle_error_code(error, 'reward function')
        return float(reward)
    
    def step(self, actions: Args) -> Args:
        '''Samples and returns the next state from the cpfs.
        
        :param actions: a dict mapping current action fluents to their values
        '''
        rddl = self.rddl
        keep_tensors = self.keep_tensors
        actions = self._process_actions(actions)
        subs = self.subs
        subs.update(actions)
        
        # compute CPFs in topological order
        for (cpf, expr, _) in self.cpfs:
            subs[cpf], self.key, error = expr(subs, self.model_params, self.key)
            self.handle_error_code(error, f'CPF <{cpf}>')            
                
        # sample reward
        reward = self.sample_reward()
        
        # update state
        self.state = {}
        for (state, next_state) in rddl.next_state.items():
            subs[state] = subs[next_state]
            if keep_tensors:
                self.state[state] = subs[state]
            else:
                self.state.update(rddl.ground_values(state, subs[state]))
        
        # update observation
        if self._pomdp: 
            obs = {}
            for var in rddl.observ:
                if keep_tensors:
                    obs[var] = subs[var]
                else:
                    obs.update(rddl.ground_values(var, subs[var]))
        else:
            obs = self.state
        
        done = self.check_terminal_states()        
        return obs, reward, done
        
