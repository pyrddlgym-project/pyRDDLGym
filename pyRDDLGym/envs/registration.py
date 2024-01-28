import configparser
import os

from pyRDDLGym.core.debug.exception import (
    RDDLEnvironmentNotExist,
    RDDLInstanceNotExist
)
from pyRDDLGym.core.env import RDDLEnv

INFO_NAME = 'domain.info'
VIZ_PACKAGE_ROOT = 'pyRDDLGym.envs'


def get_paths_to_example(domain: str, instance: str):
    path = os.path.dirname(os.path.abspath(__file__))
    valid_names = set()
    
    for domain_path, _, domain_files in os.walk(path):
        if INFO_NAME in domain_files:
            
            # parse the configuration file
            info_path = os.path.join(domain_path, INFO_NAME)
            config = configparser.RawConfigParser()
            config.optionxform = str 
            config.read(info_path)
            
            # check the name of the domain requested matches
            general = dict(config.items('General'))
            name = general.get('name', None)
            valid_names.add(name)
            if name == domain:
                
                # check the name of the instance requested matches
                for fname in domain_files:
                    if fname.startswith('instance') and fname.endswith('.rddl') \
                    and fname[8:-5] == str(instance):
                        return (os.path.join(domain_path, 'domain.rddl'),
                                os.path.join(domain_path, fname),
                                general.get('viz', None))
                
                raise RDDLInstanceNotExist(
                    f'Instance <{instance}> does not exist for '
                    f'example domain <{domain}>.')
    
    raise RDDLEnvironmentNotExist(
        f'Domain <{domain}> does not exist, must be one of {valid_names}.')                        


def make(domain: str, instance: str, custom_viz: bool=True, **env_kwargs):
    domain_path, inst_path, viz_path = get_paths_to_example(domain, instance)
    env = RDDLEnv(domain=domain_path, instance=inst_path, **env_kwargs)
    
    # load visualizer
    if custom_viz and viz_path is not None and viz_path:
        *modules, viz_class_name = viz_path.strip().split('.')
        module = '.'.join(modules)
        viz_package_name = VIZ_PACKAGE_ROOT + '.' + module
        viz_package = __import__(viz_package_name, {}, {}, viz_class_name)
        viz = getattr(viz_package, viz_class_name)
        env.set_visualizer(viz)
    
    return env
