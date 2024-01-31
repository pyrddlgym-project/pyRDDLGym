import ast
import configparser
import os
from pathlib import Path
import shutil
from typing import Tuple

from pyRDDLGym.core.debug.exception import (
    RDDLEnvironmentNotExistError,
    RDDLInstanceNotExistError
)
from pyRDDLGym.core.env import RDDLEnv

INFO_NAME = 'domain.info'
DOMAIN_NAME = 'domain.rddl'
VIZ_PACKAGE_ROOT = 'pyRDDLGym.envs'


def _get_instance_name(instance_num):
    return f'instance{instance_num}.rddl'

    
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


def register_domain(domain_name: str, domain_path: str,
                    instance_name: str, instance_path: str,
                    viz_path: str=None,
                    description: str=None) -> None:
    '''Registers the specified domain and instance, if it does not already exist.
    
    :param domain_name: the domain name to assign
    :param domain_path: the path pointing to the domain.rddl file
    :param instance_name: the instance name to assign
    :param instance_path: the path pointing to the instance.rddl file
    :param viz_path: the path pointing to the visualizer .py file
    :param description: the description meta data to write about the domain.
    '''
    # try to make the directory for the new environment
    path = os.path.dirname(os.path.abspath(__file__))
    folder_name = domain_name.lower()
    dest_path = os.path.join(path, folder_name)
    os.mkdir(dest_path)
    
    # for a visualizer get the name of the class
    if viz_path is not None:
        with open(viz_path, 'r') as viz_file:
            viz_source = viz_file.read()
        viz_ast = ast.parse(viz_source)
        classes = [node.name 
                   for node in ast.walk(viz_ast) 
                   if isinstance(node, ast.ClassDef)]
        if not classes:
            raise SyntaxError(f'Visualizer file {viz_path} '
                              f'does not contain any classes to instantiate.')
        if len(classes) > 1:
            raise SyntaxError(f'Visualizer file {viz_path} '
                              f'contains multiple class definitions.')
        viz_class_name = classes[0]
        viz_module_name = Path(viz_path).stem
        viz_import = folder_name + '.' + viz_module_name + '.' + viz_class_name
    else:
        viz_import = ''
        
    # make the info
    if description is None:
        description = f'Custom domain {domain_name} imported by the user.'
    info_content = (
        f'[General]\n'
        f'name={domain_name}\n'
        f'description={description}\n'
    )    
    if viz_path is not None:
        info_content += f'viz={viz_import}'
    with open(os.path.join(dest_path, INFO_NAME), 'w') as info: 
        info.write(info_content)
    
    # copy the RDDL contents
    shutil.copy2(domain_path, 
                 os.path.join(dest_path, DOMAIN_NAME))
    shutil.copy2(instance_path, 
                 os.path.join(dest_path, _get_instance_name(instance_name)))
    if viz_path is not None:
        shutil.copy2(viz_path, dest_path)
        
    print(f'Environment {domain_name} was successfully registered.')


def register_instance(domain: str, instance_name: str, instance_path: str):
    '''Registers the specified instance for an existing domain.
    
    :param domain: the domain name identifier
    :param instance_name: the instance name to assign
    :param instance_path: the path pointing to the instance.rddl file
    '''    
    _, _, domain_folder = get_path_to_domain_and_viz(domain)
    shutil.copy2(instance_path,
                 os.path.join(domain_folder, _get_instance_name(instance_name)))
    
    print(f'Instance {instance_name} was successfully registered for domain {domain}.')
