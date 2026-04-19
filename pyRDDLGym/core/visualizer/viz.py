from abc import ABCMeta, abstractmethod


class BaseViz(metaclass=ABCMeta):
    '''Base class for visualizers.'''
    
    @abstractmethod
    def render(self, state):
        '''Renders the given state and returns an image.'''
        pass