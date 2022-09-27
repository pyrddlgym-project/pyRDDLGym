from abc import ABCMeta, abstractmethod

class StateViz(metaclass=ABCMeta):
    @abstractmethod
    def render(self):
        pass
    
    # @abstractmethod
    # def build_object_layout(self):
    #     pass

    @abstractmethod
    def display_img(self):
        pass
    
    @abstractmethod
    def save_img(self):
        pass