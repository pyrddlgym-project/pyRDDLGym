from ast import literal_eval
import configparser
import jax
import optax
from typing import Dict

from pyRDDLGym.Core.Env.RDDLEnv import RDDLEnv
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxRDDLBackpropPlanner
from pyRDDLGym.Core.Jax import JaxRDDLLogic
from pyRDDLGym.Examples.ExampleManager import ExampleManager


def get(path: str) -> Dict[str, object]:
    
    # read the config file
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
    
    # read the optimizer settings
    opt_args = {k: args[k] for (k, v) in config.items('Optimizer')}
    opt_args['rddl'] = myEnv.model
    opt_args['key'] = jax.random.PRNGKey(opt_args['key'])
    opt_args['optimizer'] = getattr(optax, opt_args['optimizer'])(**opt_args['optimizer_kwargs'])
    opt_args['logic'] = getattr(JaxRDDLLogic, opt_args['logic'])(**opt_args['logic_kwargs'])
    del opt_args['optimizer_kwargs']
    del opt_args['logic_kwargs']
    optimizer = JaxRDDLBackpropPlanner(**opt_args)
    
    # read the train/test arguments
    train_args = {k: args[k] for (k, v) in config.items('Training')}
    
    return myEnv, optimizer, train_args

    