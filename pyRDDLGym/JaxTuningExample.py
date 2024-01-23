'''This example runs hyper-parameter tuning on the Jax planner. The tuning
is performed using a batched parallelized Bayesian optimization.

The syntax is:

    python JaxTuningExample.py <domain> <instance> <method> [<trials>] [<iters>] [<workers>]
    
where:
    <domain> is the name of a domain located in the /Examples directory
    <instance> is the instance number
    <method> is either slp, drp or replan, as described in JaxExample.py
    <trials> is the number of trials to simulate when estimating the meta loss
    (defaults to 5)
    <iters> is the number of iterations of Bayesian optimization to perform
    (defaults to 20)
    <workers> is the number of parallel workers (i.e. batch size), which must
    not exceed the number of cores available on the machine (defaults to 4)
'''
import os
import sys

from pyRDDLGym.Core.Env.RDDLEnv import RDDLEnv
from pyRDDLGym.Core.Jax.JaxParameterTuning import JaxParameterTuningDRP
from pyRDDLGym.Core.Jax.JaxParameterTuning import JaxParameterTuningSLP
from pyRDDLGym.Core.Jax.JaxParameterTuning import JaxParameterTuningSLPReplan
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import load_config
from pyRDDLGym.Examples.ExampleManager import ExampleManager


def main(domain, instance, method, trials=5, iters=20, workers=4):
    
    # set up the environment
    info = ExampleManager.GetEnvInfo(domain)    
    env = RDDLEnv.build(info, instance, vectorized=True, enforce_action_constraints=True)
    
    # load the config file with planner settings
    abs_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(abs_path, 'JaxPlanConfigs', f'{domain}_{method}.cfg')
    planner_args, plan_args, train_args = load_config(config_path)
    
    # define algorithm to perform tuning
    if method == 'slp':
        tuning_class = JaxParameterTuningSLP        
    elif method == 'drp':
        tuning_class = JaxParameterTuningDRP    
    elif method == 'replan':
        tuning_class = JaxParameterTuningSLPReplan
    
    tuning = tuning_class(env=env,
                          train_epochs=train_args['epochs'],
                          timeout_training=train_args['train_seconds'],
                          eval_trials=trials,
                          planner_kwargs=planner_args,
                          plan_kwargs=plan_args,
                          num_workers=workers,
                          gp_iters=iters)
    
    best = tuning.tune(key=train_args['key'], filename=f'gp_{method}', 
                       save_plot=True)
    print(f'best parameters found: {best}')


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 3:
        print('python JaxTuningExample.py <domain> <instance> <method> [<trials>] [<iters>] [<workers>]')
        exit(1)
    if args[2] not in ['drp', 'slp', 'replan']:
        print('<method> in [drp, slp, replan]')
        exit(1)
    kwargs = {'domain': args[0], 'instance': args[1], 'method': args[2]}
    if len(args) >= 4: kwargs['trials'] = int(args[3])
    if len(args) >= 5: kwargs['iters'] = int(args[4])
    if len(args) >= 6: kwargs['workers'] = int(args[5])
    main(**kwargs) 
    
