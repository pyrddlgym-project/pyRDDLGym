from abc import ABCMeta, abstractmethod


class BaseViz(metaclass=ABCMeta):
    
    @abstractmethod
    def render(self, state):
        pass