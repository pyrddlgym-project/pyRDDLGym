from abc import ABCMeta, abstractmethod

class StateViz(metaclass=ABCMeta):
    @abstractmethod
    def render(self):
        pass
    
    # @abstractmethod
    # def build_nonfluents_layout(self):
    #     pass
    
    # @abstractmethod
    # def build_states_layout(self):
    #     pass

    # @abstractmethod
    # def init_canvas(self):
    #     pass
    
    # @abstractmethod
    # def build_object_layout(self):
    #     pass
    
    # @abstractmethod
    # def convert2img(self):
    #     pass
    

