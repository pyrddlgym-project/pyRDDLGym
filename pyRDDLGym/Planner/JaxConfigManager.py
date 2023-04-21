from ast import literal_eval
import configparser
import jax
import jax.nn.initializers as initializers
import optax
import os
from typing import Dict
import warnings

from pyRDDLGym.Core.Env.RDDLEnv import RDDLEnv
from pyRDDLGym.Core.Jax import JaxRDDLBackpropPlanner
from pyRDDLGym.Core.Jax import JaxRDDLLogic
from pyRDDLGym.Examples.ExampleManager import ExampleManager


def get(path: str, **optional_args) -> Dict[str, object]:
    
    # read the config file
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    config = configparser.RawConfigParser()
    config.optionxform = str 
    config.read(path)
    args = {k: literal_eval(v) 
            for section in config.sections()
                for (k, v) in config.items(section)}
    
    # read the environment settings
    env_args = {k: args[k] for (k, v) in config.items('Environment')}
    use_repo = env_args.get('check_rddlrepository', False)
    if 'check_rddlrepository' in env_args:
        del env_args['check_rddlrepository']
    
    # try to read from rddlrepository
    domain_name = env_args['domain']
    inst_name = env_args['instance']
    try:
        if not use_repo:
            raise Exception
        warnings.warn(f'reading domain {domain_name} from rddlrepository...',
                      stacklevel=2)
        from rddlrepository.Manager.RDDLRepoManager import RDDLRepoManager
        manager = RDDLRepoManager()
        EnvInfo = manager.get_problem(domain_name)
    except:
        warnings.warn(f'failed to read from rddlrepository, '
                      f'reading domain {domain_name} from Examples...',
                      stacklevel=2)
        EnvInfo = ExampleManager.GetEnvInfo(domain_name)
    
    env_args['domain'] = EnvInfo.get_domain()
    env_args['instance'] = EnvInfo.get_instance(inst_name)
        
    myEnv = RDDLEnv(**env_args)
    # myEnv.set_visualizer(EnvInfo.get_visualizer())
    
    # read the model settings
    model_args = {k: args[k] for (k, v) in config.items('Model')}
    tnorm = getattr(JaxRDDLLogic, model_args['tnorm'])(
        **model_args['tnorm_kwargs'])
    logic = getattr(JaxRDDLLogic, model_args['logic'])(
        tnorm=tnorm, **model_args['logic_kwargs'])
    
    # read the optimizer settings
    opt_args = {k: args[k] for (k, v) in config.items('Optimizer')}
    opt_args['rddl'] = myEnv.model
    if 'method_kwargs' in opt_args and 'initializer' in opt_args['method_kwargs']:
        init_method = opt_args['method_kwargs']['initializer']
        initializer = getattr(initializers, init_method)
        try:       
            initializer = initializer(
                **opt_args['method_kwargs']['initializer_kwargs'])
        except:
            warnings.warn(f'warning: initializer <{init_method}> '
                          f'cannot take arguments, ignoring them.', stacklevel=2)
        opt_args['method_kwargs']['initializer'] = initializer
        if 'initializer_kwargs' in opt_args['method_kwargs']:
            del opt_args['method_kwargs']['initializer_kwargs']
            
    if 'method_kwargs' in opt_args and 'activation' in opt_args['method_kwargs']:
        opt_args['method_kwargs']['activation'] = getattr(
            jax.nn, opt_args['method_kwargs']['activation'])
        
    opt_args['plan'] = getattr(JaxRDDLBackpropPlanner, opt_args['method'])(
        **opt_args['method_kwargs'])
    opt_args['optimizer'] = getattr(optax, opt_args['optimizer'])
    opt_args['logic'] = logic
    
    del opt_args['method']
    del opt_args['method_kwargs']
    if optional_args is not None:
        for name, value in optional_args.items():
            opt_args[name] = value
    optimizer = JaxRDDLBackpropPlanner.JaxRDDLBackpropPlanner(**opt_args)
    
    # read the train/test arguments
    train_args = {k: args[k] for (k, v) in config.items('Training')}
    train_args['key'] = jax.random.PRNGKey(train_args['key'])
    
    return myEnv, optimizer, train_args, (domain_name, inst_name)

    
