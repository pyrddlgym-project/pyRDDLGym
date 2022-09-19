import sys
import copy
from Parser.expr import Expression
from abc import ABCMeta, abstractmethod
from Grounder.RDDLModel import RDDLModel
# import RDDLModel


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
        self._cpforder = {0 : []}
        self._derived = {}
        self._interm = {}

        self._reward = None
        self._preconditions = []
        self._invariants = []

        self._actionsranges = {}
        self._statesranges = {}


    def Ground(self):
        # there are no objects or types in grounded domains

        # initialize the Model object
        model = RDDLModel()

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
        # new properties

        return model

    def _groundHorizon(self):
        return self._AST.instance.horizon

    def _groundMaxActions(self):
        numactions = self._AST.instance.max_nondef_actions
        if numactions == "pos-inf":
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
        if hasattr(self._AST.non_fluents, "init_non_fluent"):
            for init_vals in self._AST.non_fluents.init_non_fluent:
                key = init_vals[0][0]
                val = init_vals[1]
                self._nonfluents[key] = val

    def _groundInitState(self):
        self._init_state = self._states.copy()
        if hasattr(self._AST.instance, "init_state"):
            for init_vals in self._AST.instance.init_state:
                key = init_vals[0][0]
                val = init_vals[1]
                self._init_state[key] = val

    def _groundReward(self):
        self._reward = self._AST.domain.reward

    def _groundPreConstraints(self):
        if hasattr(self._AST.domain, "preconds"):
            for precond in self._AST.domain.preconds:
                self._preconditions.append(precond)

        if hasattr(self._AST.domain, "invariants"):
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
        self.nextstates = {}
        self.dynamicstate = {}
        self.actions = {}
        self.derived = {}
        self.interm = {}


    def Ground(self):
        # get all the objects is the problem
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

        # ground pvariables
        for pvariable in self.AST.domain.pvariables:
            name = pvariable.name
            if pvariable.arity > 0:
                variations = self._groundObjects(pvariable.param_types)
                grounded = self._generateName(name, variations)
            else:
                grounded = [name]
            if pvariable.fluent_type == 'non-fluent':
                for g in grounded:
                    self.nonfluents[g] = pvariable.default
            elif pvariable.fluent_type == 'state-fluent':
                # find cpf
                for cpfs in self.AST.domain.cpfs[1]:
                    if cpfs.pvar[1][0] == name +'\'':
                        cpf = cpfs
                # ground state, init state and cpf
                for g in grounded:
                    self.states[g] = pvariable.default
                    l = len(name)
                    next_state = g[:l] + '\'' + g[l:]
                    self.nextstates[next_state] = g
                    self._groundCPF(name, cpf, g)
            elif pvariable.fluent_type == 'action-fluent':
                for g in grounded:
                    self.actions[g] = pvariable.default
            elif pvariable.fluent_type == 'derived-fluent':
                for g in grounded:
                    self.derived[g] = pvariable.default
            elif pvariable.fluent_type == 'interm-fluent':
                for g in grounded:
                    self.interm[g] = pvariable.default

        return

    def _groundObjects(self, args):
        list = []
        new_list = []
        for type in args:
            objs = self.objects[type]

            if len(list) == 0:
                new_list = [[obj] for obj in objs]
            else:
                for l in list:
                    for obj in objs:
                        new_l = l.copy()
                        new_l.append(obj)
                        new_list.append(new_l)
            list = new_list
            new_list = []
        return list

    def _generateName(self, name, list):
        names = []
        for variation in list:
            names.append(name + '(' +  ','.join(variation) + ')')
        return names

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


    def _groundCPF(self, name, cpf, variable):
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

        new_pvar = ('pvar_expr', (new_name, None))
        new_cpf.pvar = new_pvar
        self._scan_expr_tree(new_cpf.expr, args_dic)
        #print(new_cpf)

        return new_cpf


    def _scan_expr_tree(self, expr, dic):
        if isinstance(expr, tuple):
            return
        if expr.etype[0] == 'pvar':
            if len(expr.args[1]) > 0:
                new_name = expr.args[0] + '('
                for arg in expr.args[1]:
                    if arg in dic:
                        new_name = new_name + dic[arg] + ','
                    else:
                        new_name = new_name + arg + ','
                new_name = new_name[:-1] + ')'
                expr = Expression(('pvar_expr', (new_name, None)))
        elif expr.etype[0] == 'constant':
            return
        else:
            for child in expr.args:
                self._scan_expr_tree(child, dic)

    def _groundReward(self):
        return

    def _groundConstraints(self):
        return

