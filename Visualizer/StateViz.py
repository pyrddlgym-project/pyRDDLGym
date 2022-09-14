from abc import ABCMeta, abstractmethod

class StateViz(metaclass=ABCMeta):
    @abstractmethod
    def render(self):
        pass