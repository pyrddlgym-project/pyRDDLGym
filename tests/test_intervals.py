import numpy as np

import pyRDDLGym
from pyRDDLGym.core.intervals import RDDLIntervalAnalysis

TEST_DOMAIN = './tests/data/intervalanalysis/domain.rddl'
TEST_INSTANCE = './tests/data/intervalanalysis/instance.rddl'

def get_action_bounds(env, policy):
    action_bounds = None
    
    if policy != 'random':
        return action_bounds
    
    action_bounds = {}
    for action, prange in env.model.action_ranges.items():
        lower, upper = env._bounds[action]
        if prange == 'bool':
            lower = np.full(np.shape(lower), fill_value=0, dtype=int)
            upper = np.full(np.shape(upper), fill_value=1, dtype=int)
        action_bounds[action] = (lower, upper)
        
    return action_bounds

def perform_interval_analysis(domain, instance, policy):
    env = pyRDDLGym.make(domain, instance, vectorized=True)
    
    action_bounds = get_action_bounds(env, policy)
    
    analysis = RDDLIntervalAnalysis(env.model)
    bounds = analysis.bound(action_bounds=action_bounds, per_epoch=True)
    
    env.close()
    
    return bounds


def test_interval_analysis_simple_case():
    bounds = perform_interval_analysis(TEST_DOMAIN, TEST_INSTANCE, 'random')   
    
    reward_lower, reward_upper = bounds['reward']
    np.testing.assert_array_equal(reward_lower, [0.0, -1.0])
    np.testing.assert_array_equal(reward_upper, [0.0, 1.0])
    
    realstatefluent_lower, realstatefluent_upper = bounds['realstatefluent']
        
    np.testing.assert_array_equal(realstatefluent_lower.flatten(), [-1.0, -2.0])
    np.testing.assert_array_equal(realstatefluent_upper.flatten(), [1.0, 2.0])