import os
import sys

from pyRDDLGym.Core.Env.RDDLEnv import RDDLEnv
from pyRDDLGym.Core.Jax.JaxParameterTuning import JaxParameterTuningSLP
from pyRDDLGym.Core.Jax.JaxParameterTuning import JaxParameterTuningSLPReplan
from pyRDDLGym.Core.Jax.JaxParameterTuning import JaxParameterTuningDRP
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import load_config
from pyRDDLGym.Examples.ExampleManager import ExampleManager


def tune(config_path, env, method, trials, iters, workers):
    planner_args, plan_args, train_args = load_config(config_path)
    
    if method == 'slp':
        tuning = JaxParameterTuningSLP(
            env=env,
            train_epochs=train_args['epochs'],
            timeout_training=train_args['train_seconds'],
            planner_kwargs=planner_args,
            plan_kwargs=plan_args,
            num_workers=workers,
            gp_iters=iters)
        
    elif method == 'drp':
        tuning = JaxParameterTuningDRP(
            env=env,
            train_epochs=train_args['epochs'],
            timeout_training=train_args['train_seconds'],
            planner_kwargs=planner_args,
            plan_kwargs=plan_args,
            num_workers=workers,
            gp_iters=iters)
    
    elif method == 'replan':
        tuning = JaxParameterTuningSLPReplan(
            env=env,
            train_epochs=train_args['epochs'],
            timeout_training=train_args['train_seconds'],
            eval_trials=trials,
            planner_kwargs=planner_args,
            plan_kwargs=plan_args,
            num_workers=workers,
            gp_iters=iters)

    best = tuning.tune(key=train_args['key'], filename='gp_' + method)
    print(f'best parameters found = {best}')


if __name__ == "__main__":
    dom, inst, method, trials, iters, workers = 'Wildfire', 0, 'drp', 1, 20, 4
    if len(sys.argv) >= 7:
        dom, inst, method, trials, iters, workers = sys.argv[1:7]
        trials = int(trials)
        iters = int(iters)
        workers = int(workers)
    
    EnvInfo = ExampleManager.GetEnvInfo(dom)    
    env = RDDLEnv(domain=EnvInfo.get_domain(),
                  instance=EnvInfo.get_instance(inst),
                  enforce_action_constraints=True)
    
    abs_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(abs_path, 'JaxPlanConfigs', f'{dom}_{method}.cfg')
        
    tune(config_path, env, method, trials, iters, workers) 
    
