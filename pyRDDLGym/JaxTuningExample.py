import sys

from pyRDDLGym.Core.Jax.JaxParameterTuning import JaxParameterTuning
from pyRDDLGym.Planner import JaxConfigManager


def tune(env, replan, trials, timeout, timeout_ps, iters, workers):
    myEnv, planner, train_args, _ = JaxConfigManager.get(f'{env}.cfg')
    key = train_args['key']    
    
    tuning = JaxParameterTuning(
        num_workers=workers,
        gp_iters=iters,
        env=myEnv,
        action_bounds=planner.plan.bounds,
        max_train_epochs=train_args['epochs'],
        timeout_episode=timeout,
        timeout_epoch=timeout_ps,
        eval_trials=trials,
        print_step=train_args['step'],
        wrap_sigmoid=planner.plan._wrap_sigmoid,
        planner_kwargs={'batch_size_train': planner.batch_size_train,
                        'batch_size_test': planner.batch_size_test,
                        'use64bit': planner.use64bit,
                        'clip_grad': planner.clip_grad,
                        'use_symlog_reward': planner.use_symlog_reward,
                        'utility': planner.utility,
                        'cpfs_without_grad': planner.cpfs_without_grad})
    if replan:
        tuning.tune_mpc(key)
    else:
        tuning.tune_slp(key)


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
    
