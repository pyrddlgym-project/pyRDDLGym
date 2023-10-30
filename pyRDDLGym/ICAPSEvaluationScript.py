from datetime import datetime
import jax
import json
import os
import sys

from pyRDDLGym.Core.Env.RDDLEnv import RDDLEnv
from pyRDDLGym.Core.Policies.Agents import NoOpAgent, RandomAgent

from pyRDDLGym.Core.Gurobi.GurobiParameterTuning import GurobiParameterTuningReplan
from pyRDDLGym.Core.Gurobi.GurobiRDDLPlanner import GurobiStraightLinePlan
from pyRDDLGym.Core.Gurobi.GurobiRDDLPlanner import GurobiOnlineController

from pyRDDLGym.Core.Jax.JaxParameterTuning import JaxParameterTuningSLPReplan
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import _parse_config_file, _load_config
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxRDDLBackpropPlanner
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxOnlineController

from rddlrepository.Manager.RDDLRepoManager import RDDLRepoManager as RDDLRepoManager

CPUS = 6
GP_ITERS = 5
EPISODES = 5

 
def main(method, domain, instance, do_tune):
    assert method in ['jax', 'gurobi', 'noop', 'random']
    
    # create the environment
    filename = f'{method}_{domain}_{instance}'
    manager = RDDLRepoManager()

    EnvInfo = manager.get_problem(domain)    
    env = RDDLEnv(domain=EnvInfo.get_domain(),
                  instance=EnvInfo.get_instance(instance),
                  enforce_action_constraints=True,
                  log=True, simlogname=f'{filename}.log')
    
    # load the config file with planner settings
    abs_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(abs_path, 'ICAPSConfigs', f'{domain}.cfg') 
    config, args = _parse_config_file(config_path)
    planner_args, plan_args, train_args = _load_config(config, args)
    
    # create the planning algorithm
    tuning_key = int(datetime.now().timestamp())
    if method == 'jax':
        
        # run hyper-parameter tuning
        if do_tune:
            tuning = JaxParameterTuningSLPReplan(
                env=env,
                train_epochs=train_args['epochs'],
                timeout_training=train_args['train_seconds'],
                planner_kwargs=planner_args,
                plan_kwargs=plan_args,
                num_workers=CPUS,
                gp_iters=GP_ITERS,
                eval_trials=EPISODES)
            params = tuning.tune(key=jax.random.PRNGKey(tuning_key),
                                 filename=f'gp_{filename}')
                
            # update config with optimal hyper-parameters
            args['method_kwargs']['initializer'] = 'normal'
            args['method_kwargs']['initializer_kwargs'] = {'stddev': params['std']}
            args['optimizer_kwargs']['learning_rate'] = params['lr']
            args['logic_kwargs']['weight'] = params['w']
            args['policy_hyperparams'] = {k: params['wa'] for k in args['policy_hyperparams']}
            args['rollout_horizon'] = params['T']     
            planner_args, plan_args, train_args = _load_config(config, args)
        
        # solve planning problem with new optimal parameters
        planner = JaxRDDLBackpropPlanner(rddl=env.model, **planner_args)
        policy = JaxOnlineController(planner, **train_args)
        ground_state = False
        
    elif method == 'gurobi':
        
        # run hyper-parameter tuning
        if do_tune:
            tuning = GurobiParameterTuningReplan(
                env, 
                timeout_training=train_args['train_seconds'], 
                num_workers=CPUS,  
                gp_iters=GP_ITERS,
                eval_trials=EPISODES)
            params = tuning.tune(key=tuning_key, filename=f'{filename}')
        else:
            params = {'T': args['rollout_horizon']}
        
        # solve planning problem with new optimal parameters
        policy = GurobiOnlineController(
            rddl=env.model,
            plan=GurobiStraightLinePlan(),
            rollout_horizon=params['T'],
            model_params={'OutputFlag': 0, 'NonConvex': 2, 
                          'TimeLimit': train_args['train_seconds']})
        ground_state = True
    
    elif method == 'noop':
        policy = NoOpAgent(action_space=env.action_space, 
                           num_actions=env.numConcurrentActions)
        ground_state = True
    
    elif method == 'random':
        policy = RandomAgent(action_space=env.action_space, 
                             num_actions=env.numConcurrentActions)
        ground_state = True
        
    # evaluation
    result = policy.evaluate(
        env, verbose=True, episodes=EPISODES, ground_state=ground_state)
    
    # dump all history to files
    with open(f'{filename}.json', 'w') as fp:
        json.dump(result, fp, indent=4)
    env.close()


if __name__ == '__main__':
    method, domain, instance, do_tune, = 'gurobi', 'Wildfire_MDP_ippc2014', '5', True
    # method, domain, instance, do_tune = sys.argv[1:5]
    do_tune = do_tune in {'True', 'true', True, '1', 1}
    main(method, domain, instance, do_tune)
    
