from abc import ABCMeta, abstractmethod
import os
from typing import Dict, Iterable


class InstanceGenerator(metaclass=ABCMeta):
    
    @abstractmethod
    def get_env_path(self) -> str:
        pass
    
    @abstractmethod
    def get_domain_name(self) -> str:
        pass
    
    @abstractmethod
    def generate_rddl_variables(self, params: Dict[str, object]) -> Dict[str, object]:
        pass
    
    def generate_instance(self, instance: int, params: Dict[str, object]) -> str:
        name = self.get_domain_name()
        params = self.generate_rddl_variables(params)
        objects = params['objects']
        nonfluents = params['non-fluents']
        states = params['init-states']
        horizon = params['horizon']
        discount = params['discount']
        actions = params['max-nondef-actions']
        
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
        value = f'non-fluents nf_{name}_{instance}' + ' {'
        value += '\n\t' + f'domain = {name};'
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
        value += '\n' + f'instance inst_{name}_{instance}' + ' {'
        value += '\n\t' + f'domain = {name};'
        value += '\n\t' + f'non-fluents = nf_{name}_{instance};'
        if states_lines:
            value += '\n\t' + 'init-state {' + '\n\t\t'
            value += '\n\t\t'.join(states_lines)
            value += '\n\t' + '};'
        value += '\n\t' + f'max-nondef-actions = {actions};'
        value += '\n\t' + f'horizon = {horizon};'
        value += '\n\t' + f'discount = {discount};'
        value += '\n' + '}'
        return value
        
    def save_instance(self, instance: int, params: Dict[str, object]) -> None:
        path = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(path, self.get_env_path())
        path = os.path.join(path, f'instance{instance}.rddl')
        rddl = self.generate_instance(instance, params)
        with open(path, 'w') as text_file:
            text_file.write(rddl)
        print(f'saved RDDL file to {path}.')
        
    def save_instances(self, params: Iterable[Dict[str, object]]) -> None:
        for (instance, param) in enumerate(params):
            self.save_instance(instance + 1, param)
