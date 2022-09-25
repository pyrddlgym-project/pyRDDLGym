import sys
import copy
from Parser.expr import Expression
from abc import ABCMeta, abstractmethod
from Grounder.RDDLModel import RDDLModel
import warnings
import itertools
# import RDDLModel

# AGGREG_OPERATION_STRING_LIST = ["sum","prod","max","min","avg"]
# AGGREG_RECURSIVE_OPERATION_STRING_MAPPED_LIST = ["+","*","max","min","+"]
# QUANTIFIER_OPERATION_STRING_LIST = ["forall","exists"]
# QUANTIFIER_RECURSIVE_OPERATION_STRING_MAPPED_LIST = ["&","|"]

AGGREG_OPERATION_LIST = [
    'prod', 'sum', 'avg', 'minimum', 'maximum', 'forall', 'exists'
]
AGGREG_RECURSIVE_OPERATION_INDEX_MAPPED_LIST = [
    '*', '+', '+', '<', '>', '&', '|'
]
AGGREG_OP_TO_STRING_DICT = dict(
    zip(AGGREG_OPERATION_LIST, AGGREG_RECURSIVE_OPERATION_INDEX_MAPPED_LIST))


class Grounder(metaclass=ABCMeta):

  @abstractmethod
  def Ground(self):
    pass


class RDDLGroundedGrounder(Grounder):

  def __init__(self, RDDL_AST):
    super(RDDLGroundedGrounder, self).__init__()
    self._AST = RDDL_AST
    # self._Model = None
    self._actions = {}
    self._nonfluents = {}
    self._states = {}
    self._nextstates = {}
    self._prevstates = {}
    self._init_state = {}
    self._cpfs = {}
    self._cpforder = {0: []}
    self._derived = {}
    self._interm = {}
    self._objects = {}

    self._reward = None
    self._preconditions = []
    self._invariants = []

    self._actionsranges = {}
    self._statesranges = {}

  def Ground(self):
    # there are no objects or types in grounded domains

    # initialize the Model object
    model = RDDLModel()

    self._getObjects()

    # ground pvariables and appropriate cpfs if applicable
    # update pvariables
    self._groundPvariables()

    # update non_fluents values in case the default values were overridden in the instance
    self._groundNonfluents()

    # ground init_state
    self._groundInitState()

    # ground reward
    self._groundReward()

    # ground constraints
    self._groundPreConstraints()

    # update model object
    model.states = self._states
    model.actions = self._actions
    model.nonfluents = self._nonfluents
    model.next_state = self._nextstates
    model.prev_state = self._prevstates
    model.init_state = self._init_state
    model.cpfs = self._cpfs
    model.cpforder = self._cpforder
    model.reward = self._reward
    model.preconditions = self._preconditions
    model.invariants = self._invariants
    model.derived = self._derived
    model.interm = self._interm

    # new properties
    model.max_allowed_actions = self._groundMaxActions()
    model.horizon = self._groundHorizon()
    model.discount = self._groundDiscount()
    model.actionsranges = self._actionsranges
    model.statesranges = self._statesranges
    model.objects = self._objects
    # new properties

    return model

  def _getObjects(self):
    self._objects = {}
    try:
      for type in self._AST.non_fluents.objects:
        self._objects[type[0]] = type[1]
    except:
      return

  def _groundHorizon(self):
    return self._AST.instance.horizon

  def _groundMaxActions(self):
    numactions = self._AST.instance.max_nondef_actions
    if numactions == 'pos-inf':
      return len(self._actions)
    else:
      return int(numactions)

  def _groundDiscount(self):
    return self._AST.instance.discount

  def _groundPvariables(self):
    for pvariable in self._AST.domain.pvariables:
      name = pvariable.name
      if pvariable.fluent_type == 'non-fluent':
        self._nonfluents[name] = pvariable.default
      elif pvariable.fluent_type == 'action-fluent':
        self._actions[name] = pvariable.default
        self._actionsranges[name] = pvariable.range
      elif pvariable.fluent_type == 'state-fluent':
        cpf = None
        next_state = name + '\''
        for cpfs in self._AST.domain.cpfs[1]:
          if cpfs.pvar[1][0] == next_state:
            cpf = cpfs
        if cpf is not None:
          self._states[name] = pvariable.default
          self._statesranges[name] = pvariable.range
          self._nextstates[name] = next_state
          self._prevstates[next_state] = name
          self._cpfs[next_state] = cpf
          self._cpforder[0].append(name)
      elif pvariable.fluent_type == 'derived-fluent':
        cpf = None
        for cpfs in self._AST.domain.derived_cpfs:
          if cpfs.pvar[1][0] == name:
            cpf = cpfs
        if cpf is not None:
          self._derived[name] = pvariable.default
          self._cpfs[name] = cpf
          level = pvariable.level
          if level is None:
            level = 1
          if level in self._cpforder:
            self._cpforder[level].append(name)
          else:
            self._cpforder[level] = [name]
      elif pvariable.fluent_type == 'interm-fluent':
        cpf = None
        for cpfs in self._AST.domain.intermediate_cpfs:
          if cpfs.pvar[1][0] == name:
            cpf = cpfs
        if cpf is not None:
          self._interm[name] = pvariable.default
          self._cpfs[name] = cpf
          level = pvariable.level
          if level is None:
            level = 1
          if level in self._cpforder:
            self._cpforder[level].append(name)
          else:
            self._cpforder[level] = [name]

  def _groundNonfluents(self):
    if hasattr(self._AST.non_fluents, 'init_non_fluent'):
      for init_vals in self._AST.non_fluents.init_non_fluent:
        key = init_vals[0][0]
        val = init_vals[1]
        self._nonfluents[key] = val

  def _groundInitState(self):
    self._init_state = self._states.copy()
    if hasattr(self._AST.instance, 'init_state'):
      for init_vals in self._AST.instance.init_state:
        key = init_vals[0][0]
        val = init_vals[1]
        self._init_state[key] = val

  def _groundReward(self):
    self._reward = self._AST.domain.reward

  def _groundPreConstraints(self):
    if hasattr(self._AST.domain, 'preconds'):
      for precond in self._AST.domain.preconds:
        self._preconditions.append(precond)

    if hasattr(self._AST.domain, 'invariants'):
      for inv in self._AST.domain.invariants:
        self._invariants.append(inv)


class RDDLGrounder(Grounder):
    def __init__(self, RDDL_AST):
        super(RDDLGrounder, self).__init__()
        self.AST = RDDL_AST
        self.objects = {}
        self.objects_rev = {}
        self.nonfluents = {}
        self.states = {}
        self.statesranges = {}
        self.nextstates = {}
        self.prevstates = {}
        self.initstates = {}
        self.dynamicstate = {}
        self.actions = {}
        self.actionsranges = {}
        self.cpfs = {}
        self.cpforder = {0: []}
        self.derived = {}
        self.interm = {}
        self.reward = None
        self.preconditions = []
        self.invariants = []

  def __init__(self, rddl_ast):
    super(RDDLGrounder, self).__init__()
    self.AST = rddl_ast
    self.objects = {}
    self.objects_rev = {}
    self.nonfluents = {}
    self.states = {}
    self.nextstates = {}
    self.dynamicstate = {}
    self.actions = {}
    self.derived = {}
    self.interm = {}

    #===============================================

    def Ground(self):
        # get all the objects is the problem
        self._extract_objects()
        self._groundNonfluents()
        self._groundInitState()
        # ground pvariables
        # self._groundPvariables()
        self._groundPvariablesAndCPF()
        self.AST.domain.reward = self._scan_expr_tree(self.AST.domain.reward,{})#empty dictionary at this level
        self._groundConstraints()

        model = RDDLModel()
        # update model object
        model.states = self.states
        model.actions = self.actions
        model.nonfluents = self.nonfluents
        model.nextstate = self.nextstates
        model.prevstate = self.prevstates
        model.initstate = self.initstate
        model.cpfs = self.cpfs
        model.cpforder = self.cpforder
        model.reward = self.reward
        model.preconditions = self.preconditions
        model.invariants = self.invariants
        model.derived = self.derived
        model.interm = self.interm
        model.objects = self.objects
        model.maxallowedactions = self._get_num_max_actions()
        model.horizon = self._get_horizon()
        model.discount = self._get_discount()

        # model.actionsranges = self.actionsranges
        # model.statesranges = self.statesranges

        # new properties

        return model
    #===============================================
    def _extract_objects(self):
        """

        """
        self.objects = {}
        self.objects_rev = {}
        if self.AST.non_fluents.objects[0] is None:
            self.objects = None
            self.objects_rev = None
        else:
            for type in self.AST.non_fluents.objects:
                self.objects[type[0]] = type[1]
                for obj in type[1]:
                    self.objects_rev[obj] = type[0]
    #===============================================
    def _groundObjects(self, args):
        """

        """

        list = []
        new_list = []
        for type in args:
            objs = self.objects[type]

      if not obj_list:
        new_list = [[obj] for obj in objs]
      else:
        for l in obj_list:
          for obj in objs:
            new_l = l.copy()
            new_l.append(obj)
            new_list.append(new_l)
      obj_list = new_list
      new_list = []
    return obj_list

    #======================================================
    def _generateName(self, name, list):
        names = []
        for variation in list:
            names.append(name + '(' +  ','.join(variation) + ')')
        return names
    #======================================================
    def _groundNonfluents(self):
        if hasattr(self.AST.non_fluents, "init_non_fluent"):
            for init_vals in self.AST.non_fluents.init_non_fluent:
                key = init_vals[0][0]
                val = init_vals[1]
                self.nonfluents[key] = val
    #======================================================
    def _groundPvariables(self):
        """
        :summary: as the name implies
        """
        for pvariable in self.AST.domain.pvariables:
            name = pvariable.name
            if pvariable.fluent_type == 'non-fluent':
                self.nonfluents[name] = pvariable.default
            elif pvariable.fluent_type == 'action-fluent':
                self.actions[name] = pvariable.default
                self.actionsranges[name] = pvariable.range
            elif pvariable.fluent_type == 'state-fluent':
                cpf = None
                next_state = name + '\''
                for cpfs in self.AST.domain.cpfs[1]:
                    if cpfs.pvar[1][0] == next_state:
                        cpf = cpfs
                if cpf is not None:
                    self.states[name] = pvariable.default
                    self.statesranges[name] = pvariable.range
                    self.nextstates[name] = next_state
                    self.prevstates[next_state] = name
                    self.cpfs[next_state] = cpf
                    self.cpforder[0].append(name)
            elif pvariable.fluent_type == 'derived-fluent':
                cpf = None
                for cpfs in self.AST.domain.derived_cpfs:
                    if cpfs.pvar[1][0] == name:
                        cpf = cpfs
                if cpf is not None:
                    self.derived[name] = pvariable.default
                    self.cpfs[name] = cpf
                    level = pvariable.level
                    if level is None:
                        level = 1
                    if level in self.cpforder:
                        self.cpforder[level].append(name)
                    else:
                        self.cpforder[level] = [name]
            elif pvariable.fluent_type == 'interm-fluent':
                cpf = None
                for cpfs in self.AST.domain.intermediate_cpfs:
                    if cpfs.pvar[1][0] == name:
                        cpf = cpfs
                if cpf is not None:
                    self.interm[name] = pvariable.default
                    self.cpfs[name] = cpf
                    level = pvariable.level
                    if level is None:
                        level = 1
                    if level in self.cpforder:
                        self.cpforder[level].append(name)
                    else:
                        self.cpforder[level] = [name]


    #======================================================
    def InitGround(self):
        # init non fluents
        # print(self.AST.non_fluent_size)
        if hasattr(self.AST.non_fluents, "init_non_fluent"):
            for init_vals in self.AST.non_fluents.init_non_fluent:
                key = init_vals[0][0] + '(' + ','.join(init_vals[0][1]) + ')'
                val = init_vals[1]
                self.nonfluents[key] = val

    # init state
    self.dynamicstate = self.states.copy()
    for init_vals in self.AST.instance.init_state:
      if init_vals[0][1] is not None:
        key = init_vals[0][0] + '(' + ','.join(init_vals[0][1]) + ')'
      else:
        key = init_vals[0][0]
      val = init_vals[1]
      self.dynamicstate[key] = val

    #=================================================
    def _groundPvariablesAndCPF(self):
        """

        """

        all_grounded_cpfs = []
        for pvariable in self.AST.domain.pvariables:
            name = pvariable.name
            if pvariable.arity > 0:
                variations = self._groundObjects(pvariable.param_types)
                grounded = self._generateName(name, variations)
            else:
                grounded = [name]
                #todo merge martin's code check for abuse of arity
            if pvariable.fluent_type == 'non-fluent':
                for g in grounded:
                    self.nonfluents[g] = pvariable.default
            elif pvariable.fluent_type == 'action-fluent':
                for g in grounded:
                    self.actions[g] = pvariable.default
                    self.actionsranges[g] = pvariable.range
            elif pvariable.fluent_type == 'state-fluent':
                cpf = None
                next_state = name + '\''
                for cpfs in self.AST.domain.cpfs[1]:
                    if cpfs.pvar[1][0] == next_state:
                        cpf = cpfs
                        break;
                if cpf == None:
                    warnings.warn("No conditional prob func found for " + name)
                for g in grounded:
                    grounded_cpf = self._ground_single_CPF(name, cpf, g)
                    all_grounded_cpfs.append(grounded_cpf)
                    next_state = g + '\'' #update to grounded version, satisfied single-variables too (i.e. not a type)
                    self.states[g] = pvariable.default
                    self.statesranges[g] = pvariable.range
                    self.nextstates[g] = next_state
                    self.prevstates[next_state] = g
                    self.cpfs[next_state] = grounded_cpf
                    self.cpforder[0].append(g)
            elif pvariable.fluent_type == 'derived-fluent':
                cpf = None
                for cpfs in self.AST.domain.derived_cpfs:
                    if cpfs.pvar[1][0] == name:
                        cpf = cpfs
                        break
                if cpf == None:
                    warnings.warn("No conditional prob func found for " + name)
                for g in grounded:
                    grounded_cpf = self._ground_single_CPF(name, cpf, g)
                    all_grounded_cpfs.append(grounded_cpf)
                    self.derived[g] = pvariable.default
                    self.cpfs[g] = grounded_cpf
                    level = pvariable.level
                    if level is None:
                        level = 1
                    if level in self.cpforder:
                        self.cpforder[level].append(g)
                    else:
                        self.cpforder[level] = [g]

            elif pvariable.fluent_type == 'interm-fluent':
                cpf = None
                for cpfs in self.AST.domain.intermediate_cpfs:
                    if cpfs.pvar[1][0] == name:
                        cpf = cpfs
                        break
                if cpf == None:
                    warnings.warn("No conditional prob func found for " + name)
                for g in grounded:
                    grounded_cpf = self._ground_single_CPF(name, cpf, g)
                    all_grounded_cpfs.append(grounded_cpf)
                    self.interm[g] = pvariable.default
                    self.cpfs[g] = grounded_cpf
                    level = pvariable.level
                    if level is None:
                        level = 1
                    if level in self.cpforder:
                        self.cpforder[level].append(g)
                    else:
                        self.cpforder[level] = [g]
        #---end loop through pvars
        self.AST.domain.cpfs = (self.AST.domain.cpfs[0], all_grounded_cpfs)  # replacing the previous lifted entries

    #===========================================================
    def _ground_single_CPF(self, name, cpf, variable):
        """

        Args:
            name:
            cpf:
            variable:

        Returns:

        """
        # map arguments to actual objects
        args = cpf.pvar[1][1]
        if args is None:
            return cpf
        variable_args = variable[len(name)+1:-1].split(',')
        args = cpf.pvar[1][1]
        args_dic = {}
        for i in range(len(args)):
            args_dic[args[i]] = variable_args[i]
        # parse cpf w.r.t cpf args and variables
        # print(cpf)
        new_cpf = copy.deepcopy(cpf)
        # fix name
        new_name = new_cpf.pvar[1][0] + "("
        for arg in new_cpf.pvar[1][1]:
            new_name = new_name + args_dic[arg]+','
        new_name = new_name[:-1] + ')'

  #===================================================
  def do_aggregate_expression_nesting(self, original_dict, new_variables_list,
                                      instances_list, operation_string,
                                      expression):
    """
        Args:
            original_dict:
            new_variables_list:
            instances_list:
            operation_string:
            expression:

        Returns:
        Summary: expands the dictionary with the object instances for the
        variables passed in;
        NOTE that the order of variables, and order of elements in each entry of
        instances_set should line up.
        With the expanded dictionary, creates an expression of the type
        specified in "operation_String"
        the lhs (first) argument would be an instance from the set, and rhs
        would be from
        recursively calling this function with a reduced set, and the original
        dictionary.
        """

    # TODO: create expression with the type passed in.
    # The first argument (lhs) will have an updated dictionary based on the
    # objects spec'd in the set definition (gets one instance added to the
    # dictionary). We can scan_expression on it for the second arg 
    # (righ hand side), we will recursively call this func with a reduced
    # set of objects.
    if len(instances_list) == 1:
      # This case CAN happen, if there is only one object of the type specified
      # which can be due to misspec or due to a difficult constraint
      # satisfaction in set definition that varies over instances.
      updated_dict = copy.deepcopy(original_dict)
      updated_dict.update(dict(zip(new_variables_list, instances_list[0])))
      new_expr = self._scan_expr_tree(expression, updated_dict)
    elif len(instances_list) == 2:  # normal base case
      new_children = []
      lhs_updated_dict = copy.deepcopy(original_dict)
      lhs_updated_dict.update(dict(zip(new_variables_list, instances_list[0])))
      new_children.append(self._scan_expr_tree(expression, lhs_updated_dict))
      rhs_updated_dict = copy.deepcopy(original_dict)
      rhs_updated_dict.update(dict(zip(new_variables_list, instances_list[1])))
      new_children.append(self._scan_expr_tree(expression, rhs_updated_dict))
      # Even if it is a max or min operation, when there are only 2 left,
      # we just do "> or <".
      new_expr = Expression((operation_string, tuple(new_children)))
    else:  # recursive case
      if operation_string in [
          '+', '*', '&', '|'
      ]:  # Those ">,<" are for the min and max respectively.
        new_children = []
        lhs_updated_dict = copy.deepcopy(original_dict)
        lhs_updated_dict.update(
            dict(zip(new_variables_list, instances_list[0])))
        instances_list = instances_list[
            1:]  # Remove the first element before the recursion, simple.
        new_children.append(self._scan_expr_tree(expression, lhs_updated_dict))
        new_children.append(
            self.do_aggregate_expression_nesting(original_dict,
                                                 new_variables_list,
                                                 instances_list,
                                                 operation_string, expression))
        new_expr = Expression((operation_string, tuple(new_children)))
      else:
        # Handling the min and max case:
        # this is done by comparing two at a time, and recursively calling with
        # the remainder of the list, keeping the one that "wins" the comparison.
        # Eg: if (a > b) then (>, [a,rest of list]) else (>, [b,rest of list]),
        # I assume if they are equal, we have a default behavior in the
        # simulator (eg: take first arg).
        lhs_updated_dict = copy.deepcopy(original_dict)
        lhs_updated_dict.update(
            dict(zip(new_variables_list, instances_list[0])))
        lhs_comparison_arg = self._scan_expr_tree(expression, lhs_updated_dict)
        rhs_updated_dict = copy.deepcopy(original_dict)
        rhs_updated_dict.update(
            dict(zip(new_variables_list, instances_list[1])))
        rhs_comparison_arg = self._scan_expr_tree(expression, rhs_updated_dict)
        comparison_expression = Expression(
            (operation_string, (lhs_comparison_arg, rhs_comparison_arg)))
        new_expr = Expression(
            ('if', (comparison_expression,
                    self.do_aggregate_expression_nesting(
                        original_dict, new_variables_list,
                        [instances_list[0]] + instances_list[2:],
                        operation_string, expression),
                    self.do_aggregate_expression_nesting(
                        original_dict, new_variables_list,
                        [instances_list[1]] + instances_list[2:],
                        operation_string, expression))))
    return new_expr

  # ===================================================

  def _scan_expr_tree_pvar(self, expr, dic):
    """Ground out a pvar expression."""
    if expr.args[1] is None:  # should really be etype = constant in parsed tree
      # This is a constant.
      pass
    elif expr.args[1]:
      new_name = expr.args[0] + '_'
      for arg in expr.args[1]:
        if arg in dic:
          new_name = new_name + dic[arg] + '_'
        else:
          new_name = new_name + arg + '_'
      new_name = new_name[:-1]
      expr = Expression(('pvar_expr', (new_name, None)))
    else:
      raise ValueError(f'Malformed expression {str(expr)}.')
    return expr

  def _scan_expr_tree_abr(self, expr, dic):
    new_children = []
    for child in expr.args:
      new_children.append(self._scan_expr_tree(child, dic))
    return Expression((expr.etype[1], tuple(new_children)))

  def _scan_expr_tree_control(self, expr, dic):
    children_list = [
        self._scan_expr_tree(expr.args[0], dic),
        self._scan_expr_tree(expr.args[1], dic),
        self._scan_expr_tree(expr.args[2], dic)
    ]
    # TODO: verify the elif statements are in args[2] and what 
    # happens if no "else".
    return Expression(('if', tuple(children_list)))

  def _scan_expr_tree_func(self, expr, dic):
    new_children = []
    for child in expr.args:
      new_children.append(self._scan_expr_tree(child, dic))
    return Expression(
        (expr.etype[0], (expr.etype[1], new_children)))  # Only one arg for abs.

  # def _scan_expr_tree_randomvar(self, expr, dic):
  #   raise NotImplementedError('Random variable grounding not implemented.')


    #===============================================
    def _groundConstraints(self):
        if hasattr(self.AST.domain, "preconds"):
            for precond in self.AST.domain.preconds:
                self.preconditions.append(self._scan_expr_tree(precond,{}))

        if hasattr(self.AST.domain, "constraints"):
            for constraint in self.AST.domain.preconds:
                self.preconditions.append(self._scan_expr_tree(constraint,{}))

        if hasattr(self.AST.domain, "invariants"):
            for inv in self.AST.domain.invariants:
                self.invariants.append(self._scan_expr_tree(inv,{}))
    #===============================================
    def _groundInitState(self):
        self.initstate = self.states.copy()
        if hasattr(self.AST.instance, "init_state"):
            for init_vals in self.AST.instance.init_state:
                key = init_vals[0][0]
                val = init_vals[1]
                self.initstate[key] = val
    #===============================================
    def _get_horizon(self):
        return self.AST.instance.horizon
    #===============================================

    def _get_num_max_actions(self):
        try:
            numactions = self.AST.instance.max_nondef_actions
        except AttributeError: #was not set
            numactions = "pos-inf"
        #--end try-catch
        if numactions == "pos-inf":
            return len(self.actions)
        else:
            return int(numactions)
    #===============================================

    def _get_discount(self):
        return self.AST.instance.discount
    #===============================================

