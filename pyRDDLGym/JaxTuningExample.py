import sys

from pyRDDLGym.Core.Jax.JaxParameterTuning import JaxParameterTuningSLP
from pyRDDLGym.Core.Jax.JaxParameterTuning import JaxParameterTuningSLPReplan
from pyRDDLGym.Planner import JaxConfigManager


def tune(env, replan, trials, timeout, timeout_ps, iters, workers):
    myEnv, planner, opt_args, train_args, _ = JaxConfigManager.get(f'{env}.cfg')
    key = train_args['key']    
    
    opt_args.pop('rddl', None)
    opt_args.pop('plan', None)
    opt_args.pop('optimizer_kwargs', None)
    opt_args.pop('logic', None)
    opt_args.pop('rollout_horizon', None)
    
    if replan:
        tuning = JaxParameterTuningSLPReplan(
            env=myEnv,
            max_train_epochs=train_args['epochs'],
            timeout_episode=timeout,
            timeout_epoch=timeout_ps,
            eval_trials=trials,
            verbose=True,
            print_step=train_args['step'],
            planner_kwargs=opt_args,
            num_workers=workers,
            gp_iters=iters,
            wrap_sigmoid=planner.plan._wrap_sigmoid)
        tuning.tune(key, 'gp_replan')
    else:
        tuning = JaxParameterTuningSLP(
            env=myEnv,
            max_train_epochs=train_args['epochs'],
            timeout_episode=timeout,
            verbose=True,
            print_step=train_args['step'],
            planner_kwargs=opt_args,
            num_workers=workers,
            gp_iters=iters,
            wrap_sigmoid=planner.plan._wrap_sigmoid)
        tuning.tune(key, 'gp_slp')


if __name__ == "__main__":
    if len(sys.argv) < 7:
        env, trials, timeout, timeout_ps, iters, workers = 'MountainCar', 1, 10, 1, 20, 4
    else:
        env, trials, timeout, timeout_ps, iters, workers = sys.argv[1:7]
        trials = int(trials)
        timeout = int(timeout)
        timeout_ps = int(timeout_ps)
        iters = int(iters)
        workers = int(workers)
    replan = env.endswith('replan')
    tune(env, replan, trials, timeout, timeout_ps, iters, workers) 
    
