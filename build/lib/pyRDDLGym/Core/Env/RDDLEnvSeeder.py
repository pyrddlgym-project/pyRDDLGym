import abc
import time

MAX_INT_32 = 2 ** 31 - 1


class RDDLEnvSeeder(metaclass=abc.ABCMeta):
    pass


class RDDLEnvSeederFibonacci(RDDLEnvSeeder):
    
    def __iter__(self):
        self.a = 1
        self.b = 2
        return self
    
    def __next__(self) -> int:
        v = self.a
        self.a, self.b = self.b, self.a + self.b
        if self.b > MAX_INT_32:
            self.a = self.a % MAX_INT_32
            self.b = self.b % MAX_INT_32
        return v


class RDDLEnvSeederTimestamp(RDDLEnvSeeder):
    
    def __iter__(self):
        return self
    
    def __next__(self) -> int:
        time_millis = int(time.time_ns())
        return time_millis % MAX_INT_32
