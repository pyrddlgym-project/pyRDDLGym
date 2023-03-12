from abc import ABCMeta, abstractmethod
import os

class InstanceGenerator(metaclass=ABCMeta):
    
    @abstractmethod
    def generate_instance(self, instance: int) -> str:
        pass
    
    @abstractmethod
    def get_env_name(self) -> str:
        pass
    
    def save_instance(self, instance: int) -> None:
        path = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(path, self.get_env_name())
        path = os.path.join(path, f'instance{instance}.rddl')
        rddl = self.generate_instance(instance)
        with open(path, 'w') as text_file:
            text_file.write(rddl)
        print(f'saved RDDL file to {path}.')