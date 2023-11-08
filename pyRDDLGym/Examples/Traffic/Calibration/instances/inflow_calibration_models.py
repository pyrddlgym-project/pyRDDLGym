"""RDDL QTM/BLX models with an interface for setting the inflow rates at the network boundary"""
import jax.numpy as jnp

from pyRDDLGym import ExampleManager
from pyRDDLGym import RDDLEnv

from pyRDDLGym.Core.Jax.JaxRDDLCompiler import JaxRDDLCompiler
from pyRDDLGym.Core.Jax.JaxRDDLLogic import FuzzyLogic
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxRDDLCompilerWithGrad

class InflowCalibrationModel:
    def __init__(self,
                 action_dim,
                 instance_path,
                 source_index_range):
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

        self.s0 = self.source_index_range[0]
        self.s1 = self.source_index_range[0] + action_dim
        self.s2 = self.source_index_range[-1] + 1

    def compile(self,
                n_rollouts,
                policy_left=20,
                policy_through=60,
                use64bit=True):
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
        self.subs = {}
        for (name, value) in init_state_subs.items():
            value = jnp.array(value)[jnp.newaxis, ...]
            value_repeated = jnp.repeat(value, repeats=n_rollouts, axis=0)
            self.subs[name] = value_repeated
        for (state, next_state) in self.model.next_state.items():
            self.subs[next_state] = self.subs[state]

    def compile_relaxed(self,
                        n_rollouts,
                        weight=15,
                        policy_left=20,
                        policy_through=60,
                        use64bit=True):
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
        self.subs = {}
        for (name, value) in init_state_subs.items():
            value = value.astype(self.compiler.REAL)
            value = jnp.array(value)[jnp.newaxis, ...]
            value_repeated = jnp.repeat(value, repeats=n_rollouts, axis=0)
            self.subs[name] = value_repeated
        for (state, next_state) in self.model.next_state.items():
            self.subs[next_state] = self.subs[state]

    def set_inflow_rates(self, rates):
        self.subs['SOURCE-ARRIVAL-RATE'] = self.subs['SOURCE-ARRIVAL-RATE'].at[:,self.s0:self.s1].set(rates)
        if self.s1 < self.s2:
            self.subs['SOURCE-ARRIVAL-RATE'] = self.subs['SOURCE-ARRIVAL-RATE'].at[:,self.s1:self.s2].set(0.)

    def compute_ground_truth(self,
                             key,
                             true_rates):
        action_dim = self.s1 - self.s0
        assert len(true_rates) >= action_dim

        true_rates_repeated = jnp.broadcast_to(
            true_rates[jnp.newaxis, :action_dim],
            shape=(self.n_rollouts, action_dim))
        self.set_inflow_rates(true_rates_repeated)
        rollouts = self.sampler(
            key,
            policy_params=None,
            hyperparams=None,
            subs=self.subs,
            model_params=self.compiler.model_params)
        self.ground_truth = rollouts['pvar']['flow-into-link'][:,:,:]

    def compute_loss(self,
                     key,
                     actions):
        self.set_inflow_rates(actions)
        rollouts = self.sampler(
            key,
            policy_params=None,
            hyperparams=None,
            subs=self.subs,
            model_params=self.compiler.model_params)
        delta = rollouts['pvar']['flow-into-link'][:,:,:] - self.ground_truth
        return jnp.sum(delta*delta, axis=(1,2))



# ===== Particular instances =====
class InflowCalibration2x2Model(InflowCalibrationModel):
    """A network of 2x2 intersections (with 8 sources)"""
    def __init__(self, action_dim):
        super().__init__(
            action_dim=action_dim,
            instance_path='instances/network_2x2.rddl',
            source_index_range=tuple(range(16, 24)))


class InflowCalibration3x3Model(InflowCalibrationModel):
    """ A network of 3x3 intersections (with 12 sources)"""
    def __init__(self, action_dim):
        super().__init__(
            action_dim=action_dim,
            instance_path='instances/network_3x3.rddl',
            source_index_range=tuple(range(36, 48)))

class InflowCalibration4x4Model(InflowCalibrationModel):
    """A network of 4x4 intersections (with 16 sources)"""
    def __init__(self, action_dim):
        super().__init__(
            action_dim=action_dim,
            instance_path='instances/network_4x4.rddl',
            source_index_range=tuple(range(64, 80)))

class InflowCalibration6x6Model(InflowCalibrationModel):
    """A network of 6x6 intersections (with 24 sources)"""
    def __init__(self, action_dim):
        super().__init__(
            action_dim=action_dim,
            instance_path='instances/network_6x6.rddl',
            source_index_range=tuple(range(144, 168)))
