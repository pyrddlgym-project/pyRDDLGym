import os
import numpy as np

import pyRDDLGym
from pyRDDLGym.core.intervals import RDDLIntervalAnalysis, IntervalAnalysisStrategy

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

def perform_interval_analysis(domain, instance, action_bounds = None, state_bounds = None, strategy = IntervalAnalysisStrategy.SUPPORT, percentiles = None):
    env = pyRDDLGym.make(domain, instance, vectorized=True)
    
    if action_bounds is None:
        action_bounds = get_action_bounds_from_env(env)
    
    analysis = RDDLIntervalAnalysis(env.model, strategy=strategy, percentiles=percentiles)
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
    
def test_diracdelta_propagation():
    ''' Evaluate if the interval propagation works well to a state fluent
        that has its next value sampled by a Dirac Delta distribution.
    '''
    fluent_name = 'diracdeltastatefluent'
    
    ## Support strategy
    bounds = perform_interval_analysis(TEST_DOMAIN, TEST_INSTANCE, strategy = IntervalAnalysisStrategy.SUPPORT)   
    
    fluent_lower, fluent_upper = bounds[fluent_name]
    np.testing.assert_array_equal(fluent_lower.flatten(), [1.0, 1.0])
    np.testing.assert_array_equal(fluent_upper.flatten(), [1.0, 1.0])
    
    ## Mean strategy
    bounds = perform_interval_analysis(TEST_DOMAIN, TEST_INSTANCE, strategy = IntervalAnalysisStrategy.MEAN)   
    
    fluent_lower, fluent_upper = bounds[fluent_name]
    np.testing.assert_array_equal(fluent_lower.flatten(), [1.0, 1.0])
    np.testing.assert_array_equal(fluent_upper.flatten(), [1.0, 1.0])
    
    ## Percentiles strategy
    bounds = perform_interval_analysis(TEST_DOMAIN, TEST_INSTANCE, strategy = IntervalAnalysisStrategy.PERCENTILE, percentiles=[0.1, 0.9])   
    
    fluent_lower, fluent_upper = bounds[fluent_name]
    np.testing.assert_array_equal(fluent_lower.flatten(), [1.0, 1.0])
    np.testing.assert_array_equal(fluent_upper.flatten(), [1.0, 1.0])
    
def test_bernoulli_propagation():
    ''' Evaluate if the interval propagation works well to a state fluent
        that has its next value sampled by a Bernoulli distribution.
    '''
    
    fluent_name = 'bernoullistatefluent'
    
    ## Support strategy
    bounds = perform_interval_analysis(TEST_DOMAIN, TEST_INSTANCE, strategy = IntervalAnalysisStrategy.SUPPORT)   
    
    fluent_lower, fluent_upper = bounds[fluent_name]
    np.testing.assert_array_equal(fluent_lower.flatten(), [0.0, 0.0])
    np.testing.assert_array_equal(fluent_upper.flatten(), [1.0, 1.0])
    
    ## Mean strategy
    bounds = perform_interval_analysis(TEST_DOMAIN, TEST_INSTANCE, strategy = IntervalAnalysisStrategy.MEAN)   
    
    fluent_lower, fluent_upper = bounds[fluent_name]
    np.testing.assert_array_equal(fluent_lower.flatten(), [0.5, 0.5])
    np.testing.assert_array_equal(fluent_upper.flatten(), [0.5, 0.5])
    
    ## Percentiles strategy
    bounds = perform_interval_analysis(TEST_DOMAIN, TEST_INSTANCE, strategy = IntervalAnalysisStrategy.PERCENTILE, percentiles=[0.1, 0.9])   
    
    fluent_lower, fluent_upper = bounds[fluent_name]
    np.testing.assert_array_equal(fluent_lower.flatten(), [0.0, 0.0])
    np.testing.assert_array_equal(fluent_upper.flatten(), [1.0, 1.0])
    
def test_normal_propagation():
    ''' Evaluate if the interval propagation works well to a state fluent
        that has its next value sampled by a Normal distribution.
    '''
    
    fluent_name = 'normalstatefluent'
    
    ## Support strategy
    bounds = perform_interval_analysis(TEST_DOMAIN, TEST_INSTANCE, strategy = IntervalAnalysisStrategy.SUPPORT)   
    
    fluent_lower, fluent_upper = bounds[fluent_name]
    np.testing.assert_array_equal(fluent_lower.flatten(), [-np.inf, -np.inf])
    np.testing.assert_array_equal(fluent_upper.flatten(), [np.inf, np.inf])
    
    ## Mean strategy
    bounds = perform_interval_analysis(TEST_DOMAIN, TEST_INSTANCE, strategy = IntervalAnalysisStrategy.MEAN)   
    
    fluent_lower, fluent_upper = bounds[fluent_name]
    np.testing.assert_array_equal(fluent_lower.flatten(), [0.0, 0.0])
    np.testing.assert_array_equal(fluent_upper.flatten(), [0.0, 0.0])
    
    ## Percentiles strategy
    bounds = perform_interval_analysis(TEST_DOMAIN, TEST_INSTANCE, strategy = IntervalAnalysisStrategy.PERCENTILE, percentiles=[0.1, 0.9])   
    
    fluent_lower, fluent_upper = bounds[fluent_name]
    np.testing.assert_array_almost_equal(fluent_lower.flatten(), [-1.281552, -1.281552], decimal=5)
    np.testing.assert_array_almost_equal(fluent_upper.flatten(), [1.281552, 1.281552], decimal=5)
    
def test_weibull_propagation():
    ''' Evaluate if the interval propagation works well to a state fluent
        that has its next value sampled by a Weibull distribution.
    '''
    
    fluent_name = 'weibullstatefluent'
    
    ## Support strategy
    bounds = perform_interval_analysis(TEST_DOMAIN, TEST_INSTANCE, strategy = IntervalAnalysisStrategy.SUPPORT)   
    
    fluent_lower, fluent_upper = bounds[fluent_name]
    np.testing.assert_array_equal(fluent_lower.flatten(), [0.0, 0.0])
    np.testing.assert_array_equal(fluent_upper.flatten(), [np.inf, np.inf])
    
    ## Mean strategy
    bounds = perform_interval_analysis(TEST_DOMAIN, TEST_INSTANCE, strategy = IntervalAnalysisStrategy.MEAN)   
    
    fluent_lower, fluent_upper = bounds[fluent_name]
    np.testing.assert_array_equal(fluent_lower.flatten(), [5.0, 5.0])
    np.testing.assert_array_equal(fluent_upper.flatten(), [5.0, 5.0])
    
    ## Percentiles strategy
    bounds = perform_interval_analysis(TEST_DOMAIN, TEST_INSTANCE, strategy = IntervalAnalysisStrategy.PERCENTILE, percentiles=[0.1, 0.9])   
    
    fluent_lower, fluent_upper = bounds[fluent_name] # TODO: instead of using precalculated numbers, we could use other libs to evaluate this
    np.testing.assert_array_almost_equal(fluent_lower.flatten(), [0.5268, 0.5268], decimal=5)
    np.testing.assert_array_almost_equal(fluent_upper.flatten(), [11.51293, 11.51293], decimal=5)