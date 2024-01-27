import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, Dict, Tuple

from pyRDDLGym.Core.Compiler.RDDLLiftedModel import RDDLLiftedModel
from pyRDDLGym.Core.Jax.JaxRDDLCompiler import JaxRDDLCompiler
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxRDDLCompilerWithGrad
from pyRDDLGym.Core.Jax.JaxRDDLLogic import FuzzyLogic


class Subplots:
    
    def __init__(self, num_plots: int,
                 figsize: Tuple[int, int]) -> None:
        self.num_plots = num_plots
        
        # sub-plots arranged in a square grid
        self.dim = dim = np.floor(np.sqrt(num_plots)).astype(int) + 1
        w, h = figsize
        self.figure, self.axes = plt.subplots(
            nrows=dim, ncols=dim, figsize=(w * dim, h * dim))
        self.ix, self.iy = 0, 0
    
    def current_axis(self) -> matplotlib.axis.Axis:
        return self.axes[self.iy, self.ix]
    
    def next_axis(self) -> None:
        self.ix += 1
        if self.ix >= self.dim:
            self.ix = 0
            self.iy += 1
        
    def blank_unused_axes(self) -> None:
        while self.iy < self.dim:
            self.current_axis().set_axis_off()
            self.next_axis()
            
    def close(self) -> None:
        plt.clf()
        plt.close(self.figure)
        del self.axes
        del self.figure
    
    
class JaxRDDLModelError:
    '''Assists in the analysis of model error that arises from replacing exact
    operations with fuzzy logic.
    '''

    def __init__(self, rddl: RDDLLiftedModel,
                 test_policy: Callable,
                 batch_size: int,
                 rollout_horizon: int=None,
                 logic: FuzzyLogic=FuzzyLogic()) -> None:
        '''Creates a new instance to empirically estimate the model error.
        
        :param rddl: the RDDL domain to optimize
        :param test_policy: policy to generate rollouts
        :param batch_size: batch size
        :param rollout_horizon: lookahead planning horizon: None uses the
        horizon parameter in the RDDL instance
        :param logic: a subclass of FuzzyLogic for mapping exact mathematical
        operations to their differentiable counterparts
        '''
        self.rddl = rddl
        self.test_policy = test_policy
        self.batch_size = batch_size
        if rollout_horizon is None:
            rollout_horizon = rddl.horizon
        self.horizon = rollout_horizon
        self.logic = logic
        
        self._jax_compile()
    
    def _jax_compile(self):
        rddl = self.rddl
        n_batch = self.batch_size
        test_policy = self.test_policy
        
        # Jax compilation of the differentiable RDDL for training
        self.train_compiled = JaxRDDLCompilerWithGrad(
            rddl=rddl,
            logic=self.logic)
        self.train_compiled.compile()
        
        # Jax compilation of the exact RDDL for testing
        self.test_compiled = JaxRDDLCompiler(rddl=rddl)
        self.test_compiled.compile()
        
        # roll-outs
        def _jax_wrapped_train_policy(key, policy_params, step, subs):
            test_actions = test_policy(key, policy_params, step, subs)
            train_actions = jax.tree_map(
                lambda x: x.astype(self.test_compiled.REAL),
                test_actions)
            return train_actions
        
        train_rollouts = self.train_compiled.compile_rollouts(
            policy=_jax_wrapped_train_policy,
            n_steps=self.horizon,
            n_batch=n_batch)
        
        test_rollouts = self.test_compiled.compile_rollouts(
            policy=test_policy,
            n_steps=self.horizon,
            n_batch=n_batch)
        
        # batched initial subs
        def _jax_wrapped_batched_subs(subs): 
            train_subs, test_subs = {}, {}
            for (name, value) in subs.items():
                value = jnp.asarray(value)[jnp.newaxis, ...]
                train_value = jnp.repeat(value, repeats=n_batch, axis=0)
                train_value = train_value.astype(self.test_compiled.REAL)
                train_subs[name] = train_value
                test_subs[name] = jnp.repeat(value, repeats=n_batch, axis=0)
            for (state, next_state) in rddl.next_state.items():
                train_subs[next_state] = train_subs[state]
                test_subs[next_state] = test_subs[state]
            return train_subs, test_subs
        
        # forward simulation
        def _jax_wrapped_simulate_particles(
                key, policy_params, subs, params_train, params_test):
            
            # generate rollouts from train and test model for the same inputs
            train_subs, test_subs = _jax_wrapped_batched_subs(subs)
            train_log = train_rollouts(key, policy_params, train_subs, params_train)
            test_log = test_rollouts(key, policy_params, test_subs, params_test)
            
            # extract CPF and reward trajectories
            train_fluents, test_fluents = {}, {}
            train_fluents['reward'] = train_log['reward']
            test_fluents['reward'] = test_log['reward']
            for cpf in self.train_compiled.cpfs:
                train_fluents[cpf] = train_log['pvar'][cpf]
                test_fluents[cpf] = test_log['pvar'][cpf]
            return train_fluents, test_fluents
        
        self.simulation = jax.jit(_jax_wrapped_simulate_particles)
            
    def sensitivity(self, key: random.PRNGKey,
                    policy_params: Dict,
                    param_ranges=[1., 5., 10., 50., 100., 500., 1000.],
                    bars=100,
                    epochs=5,
                    subs: Dict=None,
                    figsize: Tuple[int, int]=(4, 2)) -> None:
        if subs is None:
            subs = self.test_compiled.init_values
        test_params = self.test_compiled.model_params
        
        # subplots
        param_ranges = list(param_ranges)
        num_rows = len(param_ranges)
        num_cols = epochs
        w, h = figsize
        
        # collect data
        train_dicts, test_dicts = [], []
        for weight in param_ranges:
            model_params = {name: weight 
                            for name in self.train_compiled.model_params}
            key, subkey = random.split(key)
            train_fluents, test_fluents = self.simulation(
                subkey, policy_params, subs, model_params, test_params)  
            train_fluents = {k: np.asarray(v, dtype=np.float64).reshape(
                                (v.shape[0], v.shape[1], -1))
                             for (k, v) in train_fluents.items()}
            test_fluents = {k: np.asarray(v, dtype=np.float64).reshape(
                                (v.shape[0], v.shape[1], -1))
                            for (k, v) in test_fluents.items()}
            train_dicts.append(train_fluents)
            test_dicts.append(test_fluents)
        
        # select epochs as equally spaced as possible including 0 and T - 1
        num_epochs = train_fluents['reward'].shape[1]
        iepochs = np.round(np.linspace(0, num_epochs - 1, epochs)).astype(int)
        
        # plot for cumulative reward
        fig, axes = plt.subplots(
            nrows=num_rows, ncols=1, figsize=(w * 1.5, h * num_rows),
            sharex='col')      
        for (row, weight) in enumerate(param_ranges):
            train_data = train_dicts[row]['reward'][:,:, ...]
            test_data = test_dicts[row]['reward'][:,:, ...]
            train = np.ravel(np.sum(train_data, axis=1))
            test = np.ravel(np.sum(test_data, axis=1))
            xmin = min(np.min(train), np.min(test))
            xmax = max(np.max(train), np.max(test))
            bins = np.linspace(xmin, xmax, bars)
            axis = axes[row]
            axis.hist(train, bins=bins, alpha=0.5, label='train', histtype='step')
            axis.hist(test, bins=bins, alpha=0.5, label='test', histtype='step')
            axis.legend()
            axis.set_xlabel(f'cuml reward')
            axis.set_ylabel(f'weight {weight}')
        plt.tight_layout()
        fig.savefig(f'sensitivity_cuml_reward.pdf')
        plt.clf()
        plt.close(fig)
        
        # each plot per grounded fluent, rows weights, columns epochs
        for name in train_fluents:
            grounded_names = list(self.rddl.ground_names(
                name, self.rddl.param_types.get(name, [])))
            for (iname, gname) in enumerate(grounded_names):
                fig, axes = plt.subplots(
                    nrows=num_rows, ncols=num_cols,
                    figsize=(w * num_cols, h * num_rows), sharex='col')            
                for (row, weight) in enumerate(param_ranges):
                    train_data = train_dicts[row][name][:,:, iname]
                    test_data = test_dicts[row][name][:,:, iname]
                    for (col, epoch) in enumerate(iepochs):
                        train = train_data[:, epoch]
                        test = test_data[:, epoch]
                        xmin = min(np.min(train), np.min(test))
                        xmax = max(np.max(train), np.max(test))
                        bins = np.linspace(xmin, xmax, bars)
                        axis = axes[row, col]
                        axis.hist(train, bins=bins, 
                                  alpha=0.5, label='train', histtype='step')
                        axis.hist(test, bins=bins, 
                                  alpha=0.5, label='test', histtype='step')
                        axis.set_xlabel(f'epoch {epoch}')
                        axis.set_ylabel(f'weight {weight}')
                        if row == 0 and col == 0:
                            axis.legend()
                print(f'ploting fluent {gname}...')
                plt.tight_layout()
                fig.savefig(f'sensitivity_{gname}.pdf')
                plt.clf()
                plt.close(fig)
    
    def summarize(self, key: random.PRNGKey,
                  policy_params: Dict,
                  subs: Dict=None,
                  figsize: Tuple[int, int]=(6, 3)) -> None:
        
        # get train and test particles
        if subs is None:
            subs = self.test_compiled.init_values
        train_fluents, test_fluents = self.simulation(
            key, policy_params, subs,
            self.train_compiled.model_params,
            self.test_compiled.model_params)
    
        # figure out # of plots required to plot each fluent component separately
        sizes = {name: np.prod(value.shape[2:], dtype=int)
                 for (name, value) in train_fluents.items()}
        
        # plot each fluent component
        subplots = Subplots(sum(sizes.values()), figsize)
        for (name, train_value) in train_fluents.items():
            train_value = np.asarray(train_value, dtype=np.float64)
            test_value = np.asarray(test_fluents[name], dtype=np.float64)
            
            # calculate statistics for fluent
            train_value = train_value.reshape(*train_value.shape[:2], -1, order='C')
            test_value = test_value.reshape(*test_value.shape[:2], -1, order='C')
            train_mean = np.mean(train_value, axis=0)
            test_mean = np.mean(test_value, axis=0)
            train_std = np.std(train_value, axis=0)
            test_std = np.std(test_value, axis=0)     
            
            grounded_names = list(self.rddl.ground_names(
                name, self.rddl.param_types.get(name, [])))
            assert len(grounded_names) == sizes[name]
            
            # plot each component j of fluent on a separate plot
            for j in range(sizes[name]):
                axis = subplots.current_axis()
                
                # show mean
                xs = np.arange(train_mean.shape[0])
                axis.plot(xs, train_mean[:, j], label='approx mean', color='blue')
                axis.plot(xs, test_mean[:, j], label='true mean',
                          linestyle='dashed', color='black')
                
                # show two standard deviations spread
                axis.fill_between(xs, train_mean[:, j] - 2 * train_std[:, j],
                                  train_mean[:, j] + 2 * train_std[:, j],
                                  color='blue', alpha=0.15)
                axis.fill_between(xs, test_mean[:, j] - 2 * test_std[:, j],
                                  test_mean[:, j] + 2 * test_std[:, j],
                                  color='black', alpha=0.15)
                
                axis.legend()
                axis.set_xlabel('epoch')
                axis.set_ylabel(grounded_names[j])
                subplots.next_axis()                
        subplots.blank_unused_axes()        
        plt.tight_layout()
        subplots.figure.savefig('summary.pdf')
        subplots.close()
