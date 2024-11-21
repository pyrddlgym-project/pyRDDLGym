import os
import numpy as np

import pyRDDLGym
from pyRDDLGym.core.intervals import RDDLIntervalAnalysis

ROOT_DIR = os.path.dirname(__file__)

TEST_DOMAIN = f'{ROOT_DIR}/data/intervalanalysis/domain.rddl'
TEST_INSTANCE = f'{ROOT_DIR}/data/intervalanalysis/instance.rddl'

##################################################################################
# Helper functions
##################################################################################

def get_action_bounds_from_env(env):    
    action_bounds = {}
    for action, prange in env.model.action_ranges.items():
        lower, upper = env._bounds[action]
        if prange == 'bool':
            lower = np.full(np.shape(lower), fill_value=0, dtype=int)
            upper = np.full(np.shape(upper), fill_value=1, dtype=int)
        action_bounds[action] = (lower, upper)
        
    return action_bounds

def perform_interval_analysis(domain, instance, action_bounds = None, state_bounds = None):
    env = pyRDDLGym.make(domain, instance, vectorized=True)
    
    if action_bounds is None:
        action_bounds = get_action_bounds_from_env(env)
    
    analysis = RDDLIntervalAnalysis(env.model)
    bounds = analysis.bound(action_bounds=action_bounds, state_bounds=state_bounds, per_epoch=True)
    
    env.close()
    
    return bounds


##################################################################################
# Test definitions
##################################################################################

def test_simple_case():
    ''' Evaluate if the interval propagation works well to a simple use case,
        with a real-valued state fluent and the reward function,
        without setting up the action and state bounds.
    '''
    bounds = perform_interval_analysis(TEST_DOMAIN, TEST_INSTANCE)   
    
    reward_lower, reward_upper = bounds['reward']
    np.testing.assert_array_equal(reward_lower, [0.0, -1.0])
    np.testing.assert_array_equal(reward_upper, [0.0, 1.0])
    
    realstatefluent_lower, realstatefluent_upper = bounds['realstatefluent']
        
    np.testing.assert_array_equal(realstatefluent_lower.flatten(), [-1.0, -2.0])
    np.testing.assert_array_equal(realstatefluent_upper.flatten(), [1.0, 2.0])
    
def test_action_bounds():
    ''' Evaluate if the interval propagation works well to a simple use case,
        with a real-valued state fluent and the reward function,
        setting up the action bounds.
    '''
    action_bounds = {
        'realactionfluent': ( np.array([ -0.5 ]), np.array([ 0.5 ]) )
    }
    
    bounds = perform_interval_analysis(TEST_DOMAIN, TEST_INSTANCE, action_bounds)   
    
    reward_lower, reward_upper = bounds['reward']
    np.testing.assert_array_equal(reward_lower, [0.0, -0.5])
    np.testing.assert_array_equal(reward_upper, [0.0, 0.5])
    
    realstatefluent_lower, realstatefluent_upper = bounds['realstatefluent']
        
    np.testing.assert_array_equal(realstatefluent_lower.flatten(), [-0.5, -1.0])
    np.testing.assert_array_equal(realstatefluent_upper.flatten(), [0.5, 1.0])
    
def test_state_bounds():
    ''' Evaluate if the interval propagation works well to a simple use case,
        with a real-valued state fluent and the reward function,
        setting up the state bounds.
    '''
    state_bounds = {
        'realstatefluent': ( np.array([ -0.5 ]), np.array([ 0.5 ]) )
    }
    
    action_bounds = {
        'realactionfluent': ( np.array([ -0.1 ]), np.array([ 0.1 ]) )
    }
    
    bounds = perform_interval_analysis(TEST_DOMAIN, TEST_INSTANCE, action_bounds, state_bounds)   
    
    reward_lower, reward_upper = bounds['reward']
    np.testing.assert_array_equal(reward_lower, [-0.5, -0.6])
    np.testing.assert_array_equal(reward_upper, [0.5, 0.6])
    
    realstatefluent_lower, realstatefluent_upper = bounds['realstatefluent']
        
    np.testing.assert_array_equal(realstatefluent_lower.flatten(), [-0.6, -0.7])
    np.testing.assert_array_equal(realstatefluent_upper.flatten(), [0.6, 0.7])