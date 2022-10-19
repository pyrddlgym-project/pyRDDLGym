import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

def init_random_provider_clusters(
    provider_disp: float,
    provider_fan_out: int,
    num_provider_clusters: int,
    num_topics: int,
) -> np.ndarray:
  """Initializes a set of providers over different topics for mixtures."""
  provider_cluster_seed = np.float32(
      ((provider_disp**.5) *
       np.random.randn(num_provider_clusters, num_topics)))
  providers = None
  for provider_cluster_center in range(num_provider_clusters):
    provider_cluster = np.float32(
        ((0.5**.5) * np.random.randn(provider_fan_out, num_topics)) +
        provider_cluster_seed[provider_cluster_center, :])
    if providers is None:
      providers = provider_cluster
    else:
      providers = np.vstack((providers, provider_cluster))
  return np.array(providers)

def init_random_user_points(provider_means, num_users, user_stddev):
  provider_logits = -tf.math.log(1.0 + tf.norm(provider_means, axis=1))
  batch_provider_logits = tf.broadcast_to(
        tf.expand_dims(provider_logits, axis=0),
        [num_users] + provider_logits.shape)

  # self._interest_model = static.GMMVector(
  #       batch_ndims=1,
  #       mixture_logits=batch_provider_logits,
  #       component_means=batch_provider_means,
  #       component_scales=tf.constant(self._user_stddev),
  #       linear_operator_ctor=lop_ctor)
  dist = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(logits=batch_provider_logits),
    components_distribution=tfd.MultivariateNormalDiag(loc=provider_means, scale_identity_multiplier=user_stddev))
  return dist.sample()


def main(*args, **kwargs):
  provider_disp = 64. 
  provider_fan_out = 2
  num_provider_clusters = 15
  num_topics = 2
  provider_clusters = init_random_provider_clusters(provider_disp, provider_fan_out, num_provider_clusters, num_topics)
  num_users = 100
  num_docs = num_provider_clusters * 5
  user_stddev = 20.5 ** 0.5
  user_locs = init_random_user_points(provider_clusters, num_users, user_stddev)
  providers = [f'p{i+1}' for i in range(num_provider_clusters)]
  users = [f'c{i+1}' for i in range(num_users)]
  features = ', '.join(['f1', 'f2'])
  items = [f'i{i+1}' for i in range(num_docs)]
  item_str = ', '.join(items) 
  user_str = ', '.join(users)
  provider_str = ', '.join(['pn'] + providers) 
  inst_str = 'non-fluents nf_recsim_ecosystem_welfare__2 {\n'
  inst_str += '\tdomain = recsim_ecosystem_welfare;\n\tobjects {\n'
  inst_str += f'\t\tfeature: {{{features}}};\n'
  inst_str += f'\t\titem: {{{item_str}}};\n'
  inst_str += f'\t\tconsumer: {{{user_str}}};\n'
  inst_str += f'\t\tprovider: {{{provider_str}}};\n'
  inst_str += '};\n\tnon-fluents {\n'
  for i, user in enumerate(users):
    inst_str += f'\t\tCONSUMER-AFFINITY({user}, f1) = {user_locs[i][0].numpy().round(4)};\n'
    inst_str += f'\t\tCONSUMER-AFFINITY({user}, f2) = {user_locs[i][1].numpy().round(4)};\n'
  for i, provider in enumerate(providers):
    inst_str += f'\t\tPROVIDER-COMPETENCE({provider}, f1) = {provider_clusters[i][0].round(4)};\n'
    inst_str += f'\t\tPROVIDER-COMPETENCE({provider}, f2) = {provider_clusters[i][1].round(4)};\n'
  providers = ['pn'] + providers
  for i, provider in enumerate(providers[:-1]):
    inst_str += f'\t\tNEXT-PROVIDER({provider}, {providers[i+1]});\n'
  for i, provider in enumerate(providers[:-1]):
    for next_provider in providers[i+1:]:
      inst_str += f'\t\tLESS({provider}, {next_provider});\n'
  inst_str += '\t\tMAX-AFFINITY = 50.0;\n\t};\n}\n'
  inst_str += 'instance recsim_ecosystem_welfare__2 {\n'
  inst_str += '\tdomain = recsim_ecosystem_welfare;\n'
  inst_str += '\tnon-fluents = nf_recsim_ecosystem_welfare__2;\n'
  inst_str += '\tinit-state {\n'
  inst_str += '\t\tprovider-satisfaction(pn) = 0.0;\n'
  inst_str += '\t};\n'
  inst_str += '\tmax-nondef-actions = 1;\n\thorizon  = 200;\n\tdiscount = 1.0;\n}'
  print(inst_str)
 

if __name__ == "__main__":
  main()