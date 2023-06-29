from itertools import cycle


class RDDLEnvSeeder():
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

