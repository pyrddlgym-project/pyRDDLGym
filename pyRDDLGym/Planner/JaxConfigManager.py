from ast import literal_eval
import configparser
import jax
import optax
import os
from typing import Dict

from pyRDDLGym.Core.Env.RDDLEnv import RDDLEnv
from pyRDDLGym.Core.Jax import JaxRDDLBackpropPlanner
from pyRDDLGym.Core.Jax import JaxRDDLLogic
from pyRDDLGym.Examples.ExampleManager import ExampleManager


def get(path: str) -> Dict[str, object]:
    
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
    EnvInfo = ExampleManager.GetEnvInfo(env_args['domain'])
    env_args['domain'] = EnvInfo.get_domain()
    env_args['instance'] = EnvInfo.get_instance(env_args['instance'])
    myEnv = RDDLEnv(**env_args)
    myEnv.set_visualizer(EnvInfo.get_visualizer())
    
    # read the model settings
    model_args = {k: args[k] for (k, v) in config.items('Model')}
    tnorm = getattr(JaxRDDLLogic, model_args['tnorm'])(**model_args['tnorm_kwargs'])
    logic = getattr(JaxRDDLLogic, model_args['logic'])(tnorm=tnorm, **model_args['logic_kwargs'])
    
    # read the optimizer settings
    opt_args = {k: args[k] for (k, v) in config.items('Optimizer')}
    opt_args['rddl'] = myEnv.model
    opt_args['plan'] = getattr(JaxRDDLBackpropPlanner, opt_args['method'])(**opt_args['method_kwargs'])
    opt_args['optimizer'] = getattr(optax, opt_args['optimizer'])(**opt_args['optimizer_kwargs'])
    opt_args['logic'] = logic
    del opt_args['method']
    del opt_args['method_kwargs']
    del opt_args['optimizer_kwargs']
    optimizer = JaxRDDLBackpropPlanner.JaxRDDLBackpropPlanner(**opt_args)
    
    # read the train/test arguments
    train_args = {k: args[k] for (k, v) in config.items('Training')}
    train_args['key'] = jax.random.PRNGKey(train_args['key'])
    
    return myEnv, optimizer, train_args

    
