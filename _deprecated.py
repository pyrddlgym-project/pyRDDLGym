# class RDDLGroundedGrounder(Grounder):
#
#   def __init__(self, RDDL_AST):
#     super(RDDLGroundedGrounder, self).__init__()
#     self._AST = RDDL_AST
#     # self._Model = None
#     self._actions = {}
#     self._nonfluents = {}
#     self._states = {}
#     self._nextstates = {}
#     self._prevstates = {}
#     self._init_state = {}
#     self._cpfs = {}
#     self._cpforder = {0: []}
#     self._derived = {}
#     self._interm = {}
#     self._objects = {}
#
#     self._reward = None
#     self._preconditions = []
#     self._invariants = []
#
#     self._actionsranges = {}
#     self._statesranges = {}
#
#   def Ground(self):
#     # there are no objects or types in grounded domains
#
#     # initialize the Model object
#     model = RDDLModel()
#
#     self._getObjects()
#
#     # ground pvariables and appropriate cpfs if applicable
#     # update pvariables
#     self._groundPvariables()
#
#     # update non_fluents values in case the default values were overridden in the instance
#     self._groundNonfluents()
#
#     # ground init_state
#     self._groundInitState()
#
#     # ground reward
#     self._groundReward()
#
#     # ground constraints
#     self._groundPreConstraints()
#
#     # update model object
#     model.states = self._states
#     model.actions = self._actions
#     model.nonfluents = self._nonfluents
#     model.next_state = self._nextstates
#     model.prev_state = self._prevstates
#     model.init_state = self._init_state
#     model.cpfs = self._cpfs
#     model.cpforder = self._cpforder
#     model.reward = self._reward
#     model.preconditions = self._preconditions
#     model.invariants = self._invariants
#     model.derived = self._derived
#     model.interm = self._interm
#
#     # new properties
#     model.max_allowed_actions = self._groundMaxActions()
#     model.horizon = self._groundHorizon()
#     model.discount = self._groundDiscount()
#     model.actionsranges = self._actionsranges
#     model.statesranges = self._statesranges
#     model.objects = self._objects
#     # new properties
#
#     return model
#
#   def _getObjects(self):
#     self._objects = {}
#     try:
#       for type in self._AST.non_fluents.objects:
#         self._objects[type[0]] = type[1]
#     except:
#       return
#
#   def _groundHorizon(self):
#     return self._AST.instance.horizon
#
#   def _groundMaxActions(self):
#     numactions = self._AST.instance.max_nondef_actions
#     if numactions == 'pos-inf':
#       return len(self._actions)
#     else:
#       return int(numactions)
#
#   def _groundDiscount(self):
#     return self._AST.instance.discount
#
#   def _groundPvariables(self):
#     for pvariable in self._AST.domain.pvariables:
#       name = pvariable.name
#       if pvariable.fluent_type == 'non-fluent':
#         self._nonfluents[name] = pvariable.default
#       elif pvariable.fluent_type == 'action-fluent':
#         self._actions[name] = pvariable.default
#         self._actionsranges[name] = pvariable.range
#       elif pvariable.fluent_type == 'state-fluent':
#         cpf = None
#         next_state = name + '\''
#         for cpfs in self._AST.domain.cpfs[1]:
#           if cpfs.pvar[1][0] == next_state:
#             cpf = cpfs
#         if cpf is not None:
#           self._states[name] = pvariable.default
#           self._statesranges[name] = pvariable.range
#           self._nextstates[name] = next_state
#           self._prevstates[next_state] = name
#           self._cpfs[next_state] = cpf.expr
#           self._cpforder[0].append(name)
#       elif pvariable.fluent_type == 'derived-fluent':
#         cpf = None
#         for cpfs in self._AST.domain.derived_cpfs:
#           if cpfs.pvar[1][0] == name:
#             cpf = cpfs
#         if cpf is not None:
#           self._derived[name] = pvariable.default
#           self._cpfs[name] = cpf.expr  #sim expects expression here.
#           level = pvariable.level
#           if level is None:
#             level = 1
#           if level in self._cpforder:
#             self._cpforder[level].append(name)
#           else:
#             self._cpforder[level] = [name]
#       elif pvariable.fluent_type == 'interm-fluent':
#         cpf = None
#         for cpfs in self._AST.domain.intermediate_cpfs:
#           if cpfs.pvar[1][0] == name:
#             cpf = cpfs
#         if cpf is not None:
#           self._interm[name] = pvariable.default
#           self._cpfs[name] = cpf.expr
#           level = pvariable.level
#           if level is None:
#             level = 1
#           if level in self._cpforder:
#             self._cpforder[level].append(name)
#           else:
#             self._cpforder[level] = [name]
#
#   def _groundNonfluents(self):
#     if hasattr(self._AST.non_fluents, 'init_non_fluent'):
#       for init_vals in self._AST.non_fluents.init_non_fluent:
#         key = init_vals[0][0]
#         val = init_vals[1]
#         self._nonfluents[key] = val
#
#   def _groundInitState(self):
#     self._init_state = self._states.copy()
#     if hasattr(self._AST.instance, 'init_state'):
#       for init_vals in self._AST.instance.init_state:
#         key = init_vals[0][0]
#         val = init_vals[1]
#         self._init_state[key] = val
#
#   def _groundReward(self):
#     self._reward = self._AST.domain.reward
#
#   def _groundPreConstraints(self):
#     if hasattr(self._AST.domain, 'preconds'):
#       for precond in self._AST.domain.preconds:
#         self._preconditions.append(precond)
#
#     if hasattr(self._AST.domain, 'invariants'):
#       for inv in self._AST.domain.invariants:
#         self._invariants.append(inv)
