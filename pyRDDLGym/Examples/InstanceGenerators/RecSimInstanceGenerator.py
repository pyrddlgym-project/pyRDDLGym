from typing import Dict

import numpy as np
from pyRDDLGym.Examples.InstanceGenerator import InstanceGenerator
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class RecSimInstanceGenerator(InstanceGenerator):

  def get_env_path(self) -> str:
    return 'RecSim'

  def get_domain_name(self) -> str:
    return 'recsim_ecosystem_welfare'

  def init_random_provider_clusters(
      self,
      provider_disp: float,
      provider_fan_out: int,
      num_provider_clusters: int,
      num_topics: int,
  ) -> np.ndarray:
    """Initializes a set of providers over different topics for mixtures."""
    provider_cluster_seed = np.float32(
        (
            (provider_disp**0.5)
            * np.random.randn(num_provider_clusters, num_topics)
        )
    )
    providers = None
    for provider_cluster_center in range(num_provider_clusters):
      provider_cluster = np.float32(
          ((0.5**0.5) * np.random.randn(provider_fan_out, num_topics))
          + provider_cluster_seed[provider_cluster_center, :]
      )
      if providers is None:
        providers = provider_cluster
      else:
        providers = np.vstack((providers, provider_cluster))
    return np.array(providers)

  def init_random_user_points(self, provider_means, num_users, user_stddev):
    provider_logits = -tf.math.log(1.0 + tf.norm(provider_means, axis=1))
    batch_provider_logits = tf.broadcast_to(
        tf.expand_dims(provider_logits, axis=0),
        [num_users] + provider_logits.shape,
    )

    dist = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(logits=batch_provider_logits),
        components_distribution=tfd.MultivariateNormalDiag(
            loc=provider_means, scale_identity_multiplier=user_stddev
        ),
    )
    return dist.sample()

  def sample_instance(self, params: Dict[str, object]) -> Dict[str, object]:
    provider_disp = params['provider_dispersion']
    provider_fan_out = params['provider_fan_out']
    num_provider_clusters = params['num_provider_clusters']
    num_topics = 2  # These problems are always 2-dimensional.
    num_users = params['num_users']
    num_docs = num_provider_clusters * params['docs_per_cluster']
    user_stddev = params['user_stddev']
    provider_clusters = self.init_random_provider_clusters(
        provider_disp, provider_fan_out, num_provider_clusters, num_topics
    )
    user_locs = self.init_random_user_points(
        provider_clusters, num_users, user_stddev
    )

    # Begin instance generation.
    # Objects.
    providers = [f'p{i+1}' for i in range(num_provider_clusters)]
    users = [f'c{i+1}' for i in range(num_users)]
    features = ['f1', 'f2']
    items = [f'i{i+1}' for i in range(num_docs)]
    # Non-fluents.
    nonfluents = {}
    for i, user in enumerate(users):
      nonfluents[f'CONSUMER-AFFINITY({user}, f1)'] = user_locs[i][0].numpy()
      nonfluents[f'CONSUMER-AFFINITY({user}, f2)'] = user_locs[i][1].numpy()
    for i, provider in enumerate(providers):
      nonfluents[f'PROVIDER-COMPETENCE({provider}, f1)'] = provider_clusters[i][
          0
      ]
      nonfluents[f'PROVIDER-COMPETENCE({provider}, f2)'] = provider_clusters[i][
          1
      ]
    providers = [
        'pn'
    ] + providers  # Add dummy provider for total ordering purposes.
    for i, provider in enumerate(providers[:-1]):
      nonfluents[f'NEXT-PROVIDER({provider}, {providers[i+1]})'] = True
    for i, provider in enumerate(providers[:-1]):
      for next_provider in providers[i + 1 :]:
        nonfluents[f'LESS({provider}, {next_provider})'] = True
    nonfluents['MAX-AFFINITY'] = 50.0
    # Initial states.
    states = {'provider-satisfaction(pn)': 0.0}
    return {
        'objects': {
            'feature': features,
            'item': items,
            'consumer': users,
            'provider': providers,
        },
        'non-fluents': nonfluents,
        'init-states': states,
        'horizon': params['horizon'],
        'discount': params['discount'],
        'max-nondef-actions': 'pos-inf',
    }


