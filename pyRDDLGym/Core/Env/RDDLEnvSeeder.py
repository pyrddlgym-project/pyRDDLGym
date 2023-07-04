import abc

from itertools import cycle


class RDDLEnvSeeder(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def Next(self) -> float:
        pass

class RDDLEnvSeederFibonacci(RDDLEnvSeeder):
    def __init__(self, a1=None):
        self.an = 0
        if a1 is None:
            a1 = 1
        self.anext = a1

    def Next(self) -> float:
        next = self.anext
        self.anext = self.an + next
        self.an = next
        return next


class RDDLEnvSeederCyclic(RDDLEnvSeeder):
    def __init__(self, seed_list: list=None):
        self.list = seed_list
        self.iterator = None
        if self.list is not None:
            if len(self.list) > 0:
                self.iterator = cycle(self.list)

    def Next(self):
        if self.iterator is not None:
            return next(self.iterator)
        else:
            return None

