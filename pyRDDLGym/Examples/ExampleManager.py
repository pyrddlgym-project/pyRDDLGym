import configparser
import csv
import os
import re

from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLEnvironmentNotExist, RDDLInstanceNotExist

HEADER = ['name', 'description', 'location', 'instances', 'viz']


def rebuild():
    path = os.path.dirname(os.path.abspath(__file__))    
    # path_to_manifest = os.path.join(path, '../Domains/manifest.csv')
    path_to_manifest = os.path.join(path, 'manifest.csv')
    print (path, path_to_manifest)
    
    with open(path_to_manifest, 'w', newline='') as file:
        
        # write the header for the manifest
        writer = csv.writer(file, delimiter=',')
        writer.writerow(HEADER)
        
        # walk through current folder to find valid domains
        for dirpath, _, filenames in os.walk(path):
            if 'domain.info' in filenames:
                infopath = os.path.join(dirpath, 'domain.info')
                config = configparser.RawConfigParser()
                config.optionxform = str 
                config.read(infopath)
                general = dict(config.items('General'))
                name = general.get('name', None)
                desc = general.get('description', None)
                viz = general.get('viz', None)
                loc = dirpath[len(path):]
                loc = loc.replace('\\', '/') + '/'
                instances = [fname[8:-5] for fname in filenames 
                             if fname.startswith('instance') and fname.endswith('.rddl')]
                instances = ','.join(instances)
                writer.writerow([name, desc, loc, instances, viz])


def load():
    path = os.path.dirname(os.path.abspath(__file__))
    # path_to_manifest = os.path.join(path, '../Domains/manifest.csv')
    path_to_manifest = os.path.join(path, 'manifest.csv')
    if not os.path.isfile(path_to_manifest):
        return {}
        
    EXP_DICT = {}
    with open(path_to_manifest) as file:
        reader = csv.reader(file, delimiter=',')
        for i, row in enumerate(reader):
            if i > 0:
                name, *entries = row
                EXP_DICT[name] = dict(zip(HEADER[1:], entries))
    return EXP_DICT


if __name__ == '__main__':
    rebuild()


class ExampleManager:
    
    EXP_DICT = load()
    
    def __init__(self, env: str):
        self.env = env
        
        if not ExampleManager.EXP_DICT:
            ExampleManager.RebuildExamples()
            
        if env not in ExampleManager.EXP_DICT:
            raise RDDLEnvironmentNotExist(
                f'Environment <{env}> does not exist, '
                f'must be one of {set(ExampleManager.EXP_DICT.keys())}.')

        self.path_to_env = os.path.dirname(os.path.abspath(__file__)) + \
                            ExampleManager.EXP_DICT[env]['location']

    def get_domain(self):
        return self.path_to_env + 'domain.rddl'

    def list_instances(self):
        files = os.listdir(self.path_to_env)
        instances = []
        for file in files:
            x = re.search("instance\d+.*", file)
            if x is not None:
                instances.append(file)
        return instances

    def get_instance(self, num: int):
        instance = f'instance{num}.rddl'
        if not os.path.exists(self.path_to_env + instance):
            raise RDDLInstanceNotExist(
                f'instance {instance} does not exist for '
                f'example environment <{self.env}>.')
        return self.path_to_env + instance

    def get_visualizer(self):
        viz = None
        viz_info = ExampleManager.EXP_DICT[self.env]['viz']
        if viz_info:
            module, viz_class_name = viz_info.strip().split('.')
            viz_package_name = 'pyRDDLGym.Visualizer.' + module
            viz_package = __import__(viz_package_name, {}, {}, viz_class_name)
            viz = getattr(viz_package, viz_class_name)
        return viz

    @staticmethod
    def ListExamples():
        print('Available example environment(s):')
        for key, values in ExampleManager.EXP_DICT.items():
            print(key + ' -> ' + values['description'])
    
    @staticmethod
    def RebuildExamples():
        rebuild()
        ExampleManager.EXP_DICT = load()
        ExampleManager.ListExamples()
        
    @staticmethod
    def GetEnvInfo(env):
        return ExampleManager(env)
    
