import importlib
import os
from typing import Type

from pyRDDLGym.core.env import RDDLEnv

VALID_EXT = '.rddl'
REPO_MANAGER_MODULE = 'rddlrepository'  # uses alias
REPO_MANAGER_CLASS = 'RDDLRepoManager'


def make(domain: str, instance: str, base_class: Type[RDDLEnv]=RDDLEnv, **env_kwargs) -> RDDLEnv:
    '''Creates a new RDDLEnv gym environment from domain and instance identifier
    or local file paths. 
    
    If the domain and instance arguments are paths to rddl files, then this 
    creates a RDDLEnv from the rddl files. If the domain and instance 
    arguments are general strings, then creates a RDDLEnv from rddlrepository.
    Note, that the second option requires rddlrepository to be installed and
    on the Python path, i.e. pip install rddlrepository.
    
    :param domain: the domain identifier, or path to domain rddl
    :param instance: the instance identifier, or path to instance rddl
    :param base_class: a subclass of RDDLEnv to load
    :param **env_kwargs: other arguments to pass to the RDDLEnv. 
    '''
    
    # check if arguments are file paths
    domain_is_file = os.path.isfile(domain)
    instance_is_file = os.path.isfile(instance)
    if domain_is_file != instance_is_file:
        raise ValueError(f'Domain and instance must either be both valid file '
                         f'paths or neither, but got {domain} and {instance}.')
    
    # if they are files, check they are RDDL files
    if domain_is_file:
        domain_is_rddl = str(os.path.splitext(domain)[1]).lower() == VALID_EXT
        instance_is_rddl = str(os.path.splitext(instance)[1]).lower() == VALID_EXT
        if not domain_is_rddl or not instance_is_rddl:
            raise ValueError(f'Domain and instance paths {domain} and {instance} '
                             f'are not valid RDDL ({VALID_EXT}) files.')
        
        # extract environment
        env = base_class(domain=domain, instance=instance, **env_kwargs)
        return env
    
    # check the repository exists
    spec = importlib.util.find_spec(REPO_MANAGER_MODULE)
    if spec is None:
        raise ImportError('rddlrepository is not installed: '
                          'can be installed with \'pip install rddlrepository\'.')
    
    # load the repository manager
    module = importlib.import_module(REPO_MANAGER_MODULE)
    manager = getattr(module, REPO_MANAGER_CLASS)()
    info = manager.get_problem(domain)
        
    # extract environment
    domain_path = info.get_domain()
    instance_path = info.get_instance(instance)
    env = base_class(domain=domain_path, instance=instance_path, **env_kwargs)
    viz = info.get_visualizer()
    if viz is not None:
        env.set_visualizer(viz)
    return env
