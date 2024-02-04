import configparser
import os
from typing import Tuple

from pyRDDLGym.core.debug.exception import (
    RDDLEnvironmentNotExistError,
    RDDLInstanceNotExistError
)
from pyRDDLGym.core.env import RDDLEnv

INFO_NAME = 'domain.info'
DOMAIN_NAME = 'domain.rddl'
VIZ_PACKAGE_ROOT = 'pyRDDLGym.envs'


def _get_instance_name(num):
    return f'instance{num}.rddl'

    
def get_path_to_domain_and_viz(domain: str) -> Tuple[str, str, str]:
    path = os.path.dirname(os.path.abspath(__file__))
    
    valid_names = set()
    for domain_folder, _, domain_files in os.walk(path):
        if INFO_NAME in domain_files:
            
            # parse the configuration file to get the domain name
            info_path = os.path.join(domain_folder, INFO_NAME)
            config = configparser.RawConfigParser()
            config.optionxform = str 
            config.read(info_path)
            general = dict(config.items('General'))
            name = general.get('name', None)
            
            # check the name of the domain requested matches            
            valid_names.add(name)            
            if name == domain:
                domain_path = os.path.join(domain_folder, DOMAIN_NAME)
                viz_path = general.get('viz', None)
                return (domain_path, viz_path, domain_folder)
    
    # could not find the exact match
    raise RDDLEnvironmentNotExistError(
        f'Domain <{domain}> does not exist, must be one of {valid_names}.')         


def get_path_to_instance(domain: str, instance: str) -> Tuple[str, str, str, str]:
    
    # find the domain path first
    domain_path, viz_path, domain_folder = get_path_to_domain_and_viz(domain)
    
    # search the domain folder for the instance
    instance_name = _get_instance_name(instance)
    for (_, _, filenames) in os.walk(domain_folder):
        if instance_name in filenames:
            instance_path = os.path.join(domain_folder, instance_name)
            return (instance_path, domain_path, viz_path, domain_folder)
    
    # could not find the exact match
    raise RDDLInstanceNotExistError(
            f'Instance <{instance}> does not exist for domain <{domain}>.')
    

def make(domain: str, instance: str, 
         custom_viz: bool=True, base_class=RDDLEnv, **env_kwargs) -> RDDLEnv:
    '''Creates a new RDDLEnv gym environment from the specified domain and 
    instance.
    
    :param domain: the domain name identifier
    :param instance: the instance name identifier
    :param custom_viz: whether to use the custom visualizer defined for this domain 
    (if False, defaults to a generic domain-independent visualizer)
    :param base_class: a subclass of RDDLEnv to load
    :param **env_kwargs: other arguments to pass to the RDDLEnv.
    '''
    instance_path, domain_path, viz_path, _ = get_path_to_instance(domain, instance)
    env = base_class(domain=domain_path, instance=instance_path, **env_kwargs)
    
    # load visualizer
    if custom_viz and viz_path is not None and viz_path:
        *modules, viz_class_name = viz_path.strip().split('.')
        module = '.'.join(modules)
        viz_package_name = VIZ_PACKAGE_ROOT + '.' + module
        viz_package = __import__(viz_package_name, {}, {}, viz_class_name)
        viz = getattr(viz_package, viz_class_name)
        env.set_visualizer(viz)
    
    return env


def make_from_rddlrepository(info: object, instance: str, 
                             base_class=RDDLEnv, **env_kwargs) -> RDDLEnv:
    '''Creates a new RDDLEnv gym environment from the specified rddlrepository
    ProblemInfo object. Requires rddlrepository to be installed, e.g.
    
    pip install rddlrepository.
    
    :param problem_info: the ProblemInfo object containing the domain info
    :param instance: the instance name identifier
    :param base_class: a subclass of RDDLEnv to load
    :param **env_kwargs: other arguments to pass to the RDDLEnv. 
    '''
    domain_path = info.get_domain()
    instance_path = info.get_instance(instance)
    env = base_class(domain=domain_path, instance=instance_path, **env_kwargs)
    viz = info.get_visualizer()
    if viz is not None:
        env.set_visualizer(viz)
    return env
