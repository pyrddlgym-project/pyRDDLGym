"""RDDL QTM/BLX models with an interface for setting the inflow rates at the network boundary"""
import jax.numpy as jnp
import os

from pyRDDLGym import ExampleManager
from pyRDDLGym import RDDLEnv

from pyRDDLGym.Core.Jax.JaxRDDLCompiler import JaxRDDLCompiler
from pyRDDLGym.Core.Jax.JaxRDDLLogic import FuzzyLogic
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxRDDLCompilerWithGrad


def set_inflow_rates(model, subs, rates):
    subs['SOURCE-ARRIVAL-RATE'] = subs['SOURCE-ARRIVAL-RATE'].at[:,model.s0:model.s1].set(rates)
    if model.s1 < model.s2:
        subs['SOURCE-ARRIVAL-RATE'] = subs['SOURCE-ARRIVAL-RATE'].at[:,model.s1:model.s2].set(0.)
    return subs

class InflowCalibrationModel:
    def __init__(self,
                 key,
                 instance_path,
                 action_dim,
                 source_index_range,
                 true_rates,
                 is_relaxed,
                 compiler_kwargs):
        """A RDDL QTM/BLX traffic model with an interface for setting
        the inflow rates at the network boundary.

        In order to test the scaling of REINFORCE and ImpSmp methods on
        a single network, we allow only a subset of the sources to be controlled.
        The number of controlled sources is set by the 'action_dim' parameter.
        The uncontrolled sources get assigned an inflow rate of 0, both in
        ground truth and in training/testing.

        The following notation is used:
            The controlled sources have indices s0, s0 + 1, ..., s1 - 1
            The uncontrolled sources have indices s1, s1 + 1, ..., s2 - 1
            For example, for a 2x2 network, the source indices are (16, ..., 23)
            If three sources are controlled, we have
                s0 = 16,  s1 = 19,  s2 = 24
        """
        self.instance_path = instance_path
        EnvInfo = ExampleManager.GetEnvInfo('traffic4phase')
        self.rddl_env = RDDLEnv.RDDLEnv(domain=EnvInfo.get_domain(),
                                        instance=self.instance_path)
        self.model = self.rddl_env.model
        self.n_traffic_lights = self.rddl_env.numConcurrentActions

        self.source_index_range = source_index_range
        n_sources = len(self.source_index_range)
        assert 1 <= action_dim <= n_sources
        assert len(true_rates) >= action_dim

        self.s0 = self.source_index_range[0]
        self.s1 = self.source_index_range[0] + action_dim
        self.s2 = self.source_index_range[-1] + 1

        if is_relaxed:
            self.compile_relaxed(compiler_kwargs)
        else:
            self.compile(compiler_kwargs)

        true_rates = jnp.array(true_rates)
        true_rates_repeated = jnp.broadcast_to(
            true_rates[jnp.newaxis, :action_dim],
            shape=(self.n_rollouts, action_dim))
        self.subs = set_inflow_rates(self, self.subs, true_rates_repeated)
        rollouts = self.sampler(
            key,
            policy_params=None,
            hyperparams=None,
            subs=self.subs,
            model_params=self.compiler.model_params)
        self.ground_truth = rollouts['pvar']['flow-into-link'][:,:,:]


    def compile(self, compiler_kwargs):
        n_rollouts = compiler_kwargs['n_rollouts']
        policy_left = compiler_kwargs.get('policy_left', 20)
        policy_through = compiler_kwargs.get('policy_through', 60)
        use64bit = compiler_kwargs.get('use64bit', True)

        self.n_rollouts = n_rollouts

        self.compiler = JaxRDDLCompiler(
            rddl=self.model,
            use64bit=use64bit)
        self.compiler.compile()

        init_state_subs = self.rddl_env.sampler.subs
        rollout_horizon = self.rddl_env.horizon

        # define the policy fn
        all_red_stretch = [True, False, False, False]
        protected_left_stretch = [True] + [False]*(policy_left-1)
        through_stretch = [True] + [False]*(policy_through-1)
        full_cycle = (all_red_stretch + protected_left_stretch + through_stretch) * 2
        BASE_PHASING = jnp.array(full_cycle*10, dtype=bool)
        FIXED_TIME_PLAN = jnp.broadcast_to(
            BASE_PHASING[..., jnp.newaxis],
            shape=(BASE_PHASING.shape[0], self.n_traffic_lights))

        def policy_fn(key, policy_params, hyperparams, step, states):
            return {'advance': FIXED_TIME_PLAN[step]}


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
        policy_left = compiler_kwargs.get('policy_left', 20)
        policy_through = compiler_kwargs.get('policy_through', 60)
        use64bit = compiler_kwargs.get('use64bit', True)

        self.n_rollouts = n_rollouts

        self.compiler = JaxRDDLCompilerWithGrad(
            rddl=self.model,
            logic=FuzzyLogic(weight=weight),
            use64bit=use64bit)
        self.compiler.compile()

        init_state_subs = self.rddl_env.sampler.subs
        rollout_horizon = self.rddl_env.horizon

        # define the policy fn
        all_red_stretch = [True, False, False, False]
        protected_left_stretch = [True] + [False]*(policy_left-1)
        through_stretch = [True] + [False]*(policy_through-1)
        full_cycle = (all_red_stretch + protected_left_stretch + through_stretch) * 2
        BASE_PHASING = jnp.array(full_cycle*10, dtype=self.compiler.REAL)
        FIXED_TIME_PLAN = jnp.broadcast_to(
            BASE_PHASING[..., jnp.newaxis],
            shape=(BASE_PHASING.shape[0], self.n_traffic_lights))

        def policy_fn(key, policy_params, hyperparams, step, states):
            return {'advance': FIXED_TIME_PLAN[step]}

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
        self.subs = set_inflow_rates(self, self.subs, actions)
        rollouts = self.sampler(
            key,
            policy_params=None,
            hyperparams=None,
            subs=self.subs,
            model_params=self.compiler.model_params)
        delta = rollouts['pvar']['flow-into-link'][:,:,:] - self.ground_truth
        return jnp.sum(delta*delta, axis=(1,2))



# ===== Particular instances =====
instance_dir = os.path.dirname(os.path.abspath(__file__))


class InflowCalibration2x2GridModel(InflowCalibrationModel):
    """A network of 2x2 intersections (with 8 sources)"""
    def __init__(self, key, action_dim, true_rates, is_relaxed, compiler_kwargs):
        super().__init__(
            key=key,
            action_dim=action_dim,
            true_rates=true_rates,
            is_relaxed=is_relaxed,
            compiler_kwargs=compiler_kwargs,
            instance_path=os.path.join(instance_dir, 'network_2x2.rddl'),
            source_index_range=tuple(range(16, 24)))


class InflowCalibration3x3GridModel(InflowCalibrationModel):
    """ A network of 3x3 intersections (with 12 sources)"""
    def __init__(self, action_dim):
        super().__init__(
            action_dim=action_dim,
            instance_path=os.path.join(instance_dir, 'network_3x3.rddl'),
            source_index_range=tuple(range(36, 48)))

class InflowCalibration4x4GridModel(InflowCalibrationModel):
    """A network of 4x4 intersections (with 16 sources)"""
    def __init__(self, action_dim):
        super().__init__(
            action_dim=action_dim,
            instance_path=os.path.join(instance_dir, 'network_4x4.rddl'),
            source_index_range=tuple(range(64, 80)))

class InflowCalibration6x6GridModel(InflowCalibrationModel):
    """A network of 6x6 intersections (with 24 sources)"""
    def __init__(self, action_dim):
        super().__init__(
            action_dim=action_dim,
            instance_path=os.path.join(instance_dir, 'network_6x6.rddl'),
            source_index_range=tuple(range(144, 168)))
