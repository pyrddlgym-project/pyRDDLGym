"""RDDL SumOfHalfSpaces models for quick iteration"""
import jax.numpy as jnp
import os

from pyRDDLGym import ExampleManager
from pyRDDLGym import RDDLEnv

from pyRDDLGym.Core.Jax.JaxRDDLCompiler import JaxRDDLCompiler
from pyRDDLGym.Core.Jax.JaxRDDLLogic import FuzzyLogic
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxRDDLCompilerWithGrad


instance_dir = os.path.dirname(os.path.abspath(__file__))


class SumOfHalfSpacesModel:
    def __init__(self,
                 key,
                 action_dim,
                 n_summands,
                 instance_idx,
                 is_relaxed,
                 compiler_kwargs):
        """.
        """
        # check the requested directory and file exist
        dirpath = os.path.join(instance_dir, f'dim{action_dim}_sum{n_summands}')
        if not os.path.isdir(dirpath):
            raise RuntimeError('[SumOfHalfSpacesModel] Please check that the instance '
                               f'directory parameters dim={action_dim}, sum={n_summands} '
                               'has been generated to Examples/SumOfHalfSpaces/instances')
        filepath = os.path.join(dirpath, f'instance{instance_idx}.rddl')
        if not os.path.isfile(filepath):
            raise RuntimeError('[SumOfHalfSpacesModel] Please check that the instance '
                              f'file instance{instance_idx}.rddl has been generated in '
                              f'Examples/SumOfHalfSpaces/instances/dim{action_dim}_sum{n_summands}')

        self.instance_path = filepath
        EnvInfo = ExampleManager.GetEnvInfo('SumOfHalfSpaces')
        self.rddl_env = RDDLEnv.RDDLEnv(domain=EnvInfo.get_domain(),
                                        instance=self.instance_path)
        self.model = self.rddl_env.model
        self.action_dim = action_dim
        self.n_summands = n_summands
        assert self.action_dim == self.rddl_env.numConcurrentActions

        if is_relaxed:
            self.compile_relaxed(compiler_kwargs)
        else:
            self.compile(compiler_kwargs)


    def compile(self, compiler_kwargs):
        n_rollouts = compiler_kwargs['n_rollouts']
        use64bit = compiler_kwargs.get('use64bit', True)

        self.n_rollouts = n_rollouts

        self.compiler = JaxRDDLCompiler(
            rddl=self.model,
            use64bit=use64bit)
        self.compiler.compile()

        init_state_subs = self.rddl_env.sampler.subs
        rollout_horizon = self.rddl_env.horizon

        def policy_fn(key, policy_params, hyperparams, step, states):
            # policy_params should have the type {'a': action batch}
            return None

        self.sampler = self.compiler.compile_rollouts(
            policy=policy_fn,
            n_steps=rollout_horizon,
            n_batch=n_rollouts)

        # repeat subs over the batch
        subs = {}
        for (name, value) in init_state_subs.items():
            value = jnp.array(value)[jnp.newaxis, ...]
            value_repeated = jnp.repeat(value, repeats=n_rollouts, axis=0)
            subs[name] = value_repeated
        for (state, next_state) in self.model.next_state.items():
            subs[next_state] = subs[state]
        self.subs = subs

    def compile_relaxed(self, compiler_kwargs):
        n_rollouts = compiler_kwargs['n_rollouts']
        weight = compiler_kwargs.get('weight', 15)
        use64bit = compiler_kwargs.get('use64bit', True)

        self.n_rollouts = n_rollouts

        self.compiler = JaxRDDLCompilerWithGrad(
            rddl=self.model,
            logic=FuzzyLogic(weight=weight),
            use64bit=use64bit)
        self.compiler.compile()

        init_state_subs = self.rddl_env.sampler.subs
        rollout_horizon = self.rddl_env.horizon

        def policy_fn(key, policy_params, hyperparams, step, states):
            # policy_params should have the type {'a': action batch}
            return None

        self.sampler = self.compiler.compile_rollouts(
            policy=policy_fn,
            n_steps=rollout_horizon,
            n_batch=n_rollouts)

        # repeat subs over the batch and convert to real
        subs = {}
        for (name, value) in init_state_subs.items():
            value = value.astype(self.compiler.REAL)
            value = jnp.array(value)[jnp.newaxis, ...]
            value_repeated = jnp.repeat(value, repeats=n_rollouts, axis=0)
            subs[name] = value_repeated
        for (state, next_state) in self.model.next_state.items():
            subs[next_state] = subs[state]
        self.subs = subs

    def compute_loss(self, key, actions):
        print(actions.shape)
        self.subs.update({'a': actions})
        rollouts = self.sampler(
            key,
            policy_params=None,
            hyperparams=None,
            subs=self.subs,
            model_params=self.compiler.model_params)
        return rollouts['reward'][..., 0]
