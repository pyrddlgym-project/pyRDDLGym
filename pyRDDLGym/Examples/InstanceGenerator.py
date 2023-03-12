from abc import ABCMeta, abstractmethod


class InstanceGenerator(metaclass=ABCMeta):
    
    @abstractmethod
    def generate_instance(self, instance: int) -> str:
        pass
    
    @abstractmethod
    def get_env_name(self) -> str:
        pass
    