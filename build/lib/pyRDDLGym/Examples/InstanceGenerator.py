from abc import ABCMeta, abstractmethod
import os
from typing import Dict


class InstanceGenerator(metaclass=ABCMeta):
    
    @abstractmethod
    def get_env_path(self) -> str:
        pass

    @abstractmethod
    def get_domain_name(self) -> str:
        pass

    @abstractmethod
    def sample_instance(self, params: Dict[str, object]) -> Dict[str, object]:
        pass
    
    def generate_instance(self, inst_name: str, params: Dict[str, object]) -> str:
        instance = self.sample_instance(params)
        objects = instance['objects']
        nonfluents = instance['non-fluents']
        states = instance['init-states']
        horizon = instance['horizon']
        discount = instance['discount']
        actions = instance['max-nondef-actions']
        
        domain_name = self.get_domain_name()
        
        # object lists
        objects_lines = [f'{key} : {{{", ".join(val)}}};' 
                         for (key, val) in objects.items()]
        
        # non-fluents list
        nonfluent_lines = []
        for (key, val) in nonfluents.items():
            if isinstance(val, bool):
                if val == True:
                    nonfluent_lines.append(f'{key};')
                else:
                    nonfluent_lines.append(f'~{key};')
            else:
                valstr = f'{val}'
                if 'e' in valstr or 'E' in valstr:
                    valstr = '{:.15f}'.format(val)
                nonfluent_lines.append(f'{key} = {valstr};')
        
        # init-state list
        states_lines = []
        for (key, val) in states.items():
            if isinstance(val, bool):
                if val == True:
                    states_lines.append(f'{key};')
                else:
                    states_lines.append(f'~{key};')
            else: 
                valstr = f'{val}'
                if 'e' in valstr or 'E' in valstr:
                    valstr = '{:.15f}'.format(val)
                states_lines.append(f'{key} = {valstr};')
        
        # generate non-fluents block
        value = f'non-fluents nf_{domain_name}_{inst_name}' + ' {'
        value += '\n\t' + f'domain = {domain_name};'
        if objects_lines:
            value += '\n\t' + 'objects {' + '\n\t\t'
            value += '\n\t\t'.join(objects_lines)
            value += '\n\t' + '};'
        if nonfluent_lines:
            value += '\n\t' + 'non-fluents {' + '\n\t\t'
            value += '\n\t\t'.join(nonfluent_lines)
            value += '\n\t' + '};'
        value += '\n' + '}'
        
        # generate instance block
        value += '\n' + f'instance inst_{domain_name}_{inst_name}' + ' {'
        value += '\n\t' + f'domain = {domain_name};'
        value += '\n\t' + f'non-fluents = nf_{domain_name}_{inst_name};'
        if states_lines:
            value += '\n\t' + 'init-state {' + '\n\t\t'
            value += '\n\t\t'.join(states_lines)
            value += '\n\t' + '};'
        value += '\n\t' + f'max-nondef-actions = {actions};'
        value += '\n\t' + f'horizon = {horizon};'
        value += '\n\t' + f'discount = {discount};'
        value += '\n' + '}'
        return value
        
    def save_instance(self, instance: int, params: Dict[str, object], path=None) -> None:
        if path == None:
            path = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(path, self.get_env_path())

        dir = path
        path = os.path.join(path, f'instance{instance}.rddl')
        rddl = self.generate_instance(instance, params)
        isExist = os.path.exists(dir)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(dir)
        with open(path, 'w') as text_file:
            text_file.write(rddl)
        print(f'saved RDDL file to {path}.')
        