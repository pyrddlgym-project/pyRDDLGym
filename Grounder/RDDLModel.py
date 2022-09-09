from abc import ABCMeta

class PlanningModel(metaclass=ABCMeta):
    def __init__(self):
        self._AST = None
        self._nonfluents = None
        self._states = None
        self._nextstates = None
        self._prevstates = None
        self._initstate = None
        self._actions = None
        self._derived = None
        self._interm = None
        self._cpfs = None
        self._cpforder = None
        self._reward = None
        self._preconditions = None
        self._invariants = None

    def SetAST(self, AST):
        self._AST = AST

    @property
    def nonfluents(self):
        return self._nonfluents

    @nonfluents.setter
    def nonfluents(self, val):
        self._nonfluents = val

    @property
    def states(self):
        return self._states

    @states.setter
    def states(self, val):
        self._states = val

    @property
    def next_state(self):
        return self._nextstates

    @next_state.setter
    def next_state(self, val):
        self._nextstates = val

    @property
    def prev_state(self):
        return self._prevstates

    @prev_state.setter
    def prev_state(self, val):
        self._prevstates = val

    @property
    def init_state(self):
        return self._initstate

    @init_state.setter
    def init_state(self, val):
        self._initstate = val

    @property
    def actions(self):
        return self._actions

    @actions.setter
    def actions(self, val):
        self._actions = val

    @property
    def derived(self):
        return self._derived

    @derived.setter
    def derived(self, val):
        self._derived = val

    @property
    def interm(self):
        return self._interm

    @interm.setter
    def interm(self, val):
        self._interm = val

    @property
    def cpfs(self):
        return self._cpfs

    @cpfs.setter
    def cpfs(self, val):
        self._cpfs = val

    @property
    def cpforder(self):
        return self._cpforder

    @cpforder.setter
    def cpforder(self, val):
        self._cpforder = val

    @property
    def reward(self):
        return self._reward

    @reward.setter
    def reward(self, val):
        self._reward = val

    @property
    def preconditions(self):
        return self._preconditions

    @preconditions.setter
    def preconditions(self, val):
        self._preconditions = val

    @property
    def invariants(self):
        return self._invariants

    @invariants.setter
    def invariants(self, val):
        self._invariants = val



class RDDLModel(PlanningModel):
    def __init__(self):
        super().__init__()


def main():
    AST = "dfdf"
    M = RDDLModel()
    M.SetAST(AST)
    print(M.objects)
    M.objects["2"]=3
    print(M.objects)
    M.objects = {1 : 3, 4 : 5}
    print(M.objects)
    print("hello")




if __name__ == "__main__":
    main()
