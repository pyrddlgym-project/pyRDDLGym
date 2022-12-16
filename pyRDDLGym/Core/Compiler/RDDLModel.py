from abc import ABCMeta


class PlanningModel(metaclass=ABCMeta):

    def __init__(self):
        self._AST = None
        self._nonfluents = None
        self._states = None
        self._statesranges = None
        self._nextstates = None
        self._prevstates = None
        self._initstate = None
        self._actions = None
        self._objects = None
        self._actionsranges = None
        self._derived = None
        self._interm = None
        self._observ = None
        self._observranges = None
        self._cpfs = None
        self._cpforder = None
        self._gvar_to_cpforder = None
        self._reward = None
        self._terminals = None
        self._preconditions = None
        self._invariants = None
        self._gvar_to_type = None

        # new definitions
        self._max_allowed_actions = None
        self._horizon = None
        self._discount = None

    def SetAST(self, AST):
        self._AST = AST

    @property
    def objects(self):
        return self._objects

    @objects.setter
    def objects(self, value):
        self._objects = value

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
    def statesranges(self):
        return self._statesranges

    @statesranges.setter
    def statesranges(self, value):
        self._statesranges = value

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
    def actionsranges(self):
        return self._actionsranges

    @actionsranges.setter
    def actionsranges(self, value):
        self._actionsranges = value

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
    def observ(self):
        return self._observ

    @observ.setter
    def observ(self, value):
        self._observ = value

    @property
    def observranges(self):
        return self._observranges

    @observranges.setter
    def observranges(self, value):
        self._observranges = value

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
    def gvar_to_cpforder(self):
        return self._gvar_to_cpforder

    @gvar_to_cpforder.setter
    def gvar_to_cpforder(self, val):
        self._gvar_to_cpforder = val
    
    @property
    def reward(self):
        return self._reward

    @reward.setter
    def reward(self, val):
        self._reward = val

    @property
    def terminals(self):
        return self._terminals

    @terminals.setter
    def terminals(self, value):
        self._terminals = value

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
    
    @property
    def gvar_to_type(self):
        return self._gvar_to_type

    @gvar_to_type.setter
    def gvar_to_type(self, val):
        self._gvar_to_type = val
    
    @property
    def pvar_to_type(self):
        return self._pvar_to_type

    @pvar_to_type.setter
    def pvar_to_type(self, val):
        self._pvar_to_type = val

    @property
    def gvar_to_pvar(self):
        return self._gvar_to_pvar

    @gvar_to_pvar.setter
    def gvar_to_pvar(self, val):
        self._gvar_to_pvar = val

    @property
    def discount(self):
        return self._discount

    @discount.setter
    def discount(self, val):
        self._discount = val

    @property
    def horizon(self):
        return self._horizon

    @horizon.setter
    def horizon(self, val):
        self._horizon = val

    @property
    def max_allowed_actions(self):
        return self._max_allowed_actions

    @max_allowed_actions.setter
    def max_allowed_actions(self, val):
        self._max_allowed_actions = val


class RDDLModel(PlanningModel):

    def __init__(self):
        super().__init__()

