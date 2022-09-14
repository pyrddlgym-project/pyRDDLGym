import sys
import copy
from Parser.expr import Expression
from abc import ABCMeta, abstractmethod
from Grounder.RDDLModel import RDDLModel
# import RDDLModel

AGGREG_OPERATION_STRING_LIST = ["sum","prod","max","min","avg"]
AGGREG_RECURSIVE_OPERATION_STRING_MAPPED_LIST = ["+","*","max","min","+"]


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

        return model

    def _groundPvariables(self):
        for pvariable in self._AST.domain.pvariables:
            name = pvariable.name
            if pvariable.fluent_type == 'non-fluent':
                self._nonfluents[name] = pvariable.default
            elif pvariable.fluent_type == 'action-fluent':
                self._actions[name] = pvariable.default
            elif pvariable.fluent_type == 'state-fluent':
                cpf = None
                next_state = name + '\''
                for cpfs in self._AST.domain.cpfs[1]:
                    if cpfs.pvar[1][0] == next_state:
                        cpf = cpfs
                if cpf is not None:
                    self._states[name] = pvariable.default
                    self._nextstates[name] = next_state
                    self._prevstates[next_state] = name
                    self._cpfs[next_state] = cpf
                    self._cpforder[0].append(name)
            elif pvariable.fluent_type == 'derived-fluent':
                cpf = None
                for cpfs in self._AST.domain.dervied_cpfs[1]:
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
                for cpfs in self._AST.domain.intermediate_cpfs[1]:
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
                for g in grounded:
                    self.states[g] = pvariable.default
                    l = len(name)
                    next_state = g[:l] + '\'' + g[l:]
                    self.nextstates[next_state] = g
            elif pvariable.fluent_type == 'action-fluent':
                for g in grounded:
                    self.actions[g] = pvariable.default
            elif pvariable.fluent_type == 'derived-fluent':
                for g in grounded:
                    self.derived[g] = pvariable.default
            elif pvariable.fluent_type == 'interm-fluent':
                for g in grounded:
                    self.interm[g] = pvariable.default
        #----NOW LOOP AGAIN, with all the state variable and property grounding options done
        #---this lets us ground the cpfs, rewards and constraints more easily
        for pvariable in self.AST.domain.pvariables:
            name = pvariable.name
            if pvariable.arity > 0:
                variations = self._groundObjects(pvariable.param_types)
                grounded = self._generateName(name, variations)
            else:
                grounded = [name]
            if pvariable.fluent_type == 'state-fluent':
                # find cpf
                for cpfs in self.AST.domain.cpfs[1]:
                    if cpfs.pvar[1][0] == name +'\'':
                        cpf = cpfs
                        break #added to avoid going over all cpfs, as soon as we have the target one, we stop the loop
                # ground state, init state and cpf
                for g in grounded:
                    l = len(name)
                    next_state = g[:l] + '\'' + g[l:]
                    self._groundCPF(name, cpf, g)
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
        new_cpf = self._scan_expr_tree(new_cpf.expr, args_dic)
        print(new_cpf)

        return new_cpf


    def _scan_expr_tree(self, expr, dic):
        """
        Args:
            expr:
            list_dic:
        Returns:
        :Summary:
        """

        if isinstance(expr, tuple):
            pass # return is at end of function
        elif expr.etype[0] == 'pvar':
            if expr.args[1] == None: #should really be etype = constant in parsed tree
                #then it is a constant
                pass #return statement is at end of func as per coding conventions
            elif len(expr.args[1]) > 0:
                new_name = expr.args[0] + '('
                for arg in expr.args[1]:
                    if arg in dic:
                        new_name = new_name + dic[arg] + ','
                    else:
                        new_name = new_name + arg + ','
                new_name = new_name[:-1] + ')'
                expr = Expression(('pvar_expr', (new_name, None)))
        elif expr.etype[0] == 'constant':
            pass #the return statement is at the end
        elif expr.etype[0] in ['arithmetic','boolean','relational']:
            new_children = []
            for child in expr.args:
                new_children.append(self._scan_expr_tree(child, dic))
            #if we reached here the expression is either a +,*, or comparator (>,<) or aggregator (sum, product)
            expr = Expression((expr.etype[1], tuple(new_children)))
        elif expr.etype[0] == "aggregation":
            if expr.etype[1] in AGGREG_OPERATION_STRING_LIST:
                aggreg_type_idx = AGGREG_OPERATION_STRING_LIST.index(expr.etype[1])
                #determine what the recursive op is for the aggreg type. Eg: sum = "+"
                aggreg_recursive_operation_string = AGGREG_RECURSIVE_OPERATION_STRING_MAPPED_LIST[aggreg_type_idx]

                #need to do a for loop through the objects, and pass in
                # a NEW dictionary (keep the old copy to recover state)
                #--only for average, we need to first "/ n ", for all others
                # we need to decide the symbol "+" or "*" and iterative
                # build a tree over the objects specd in the dict, using a for loop
                # need to ground ofcourse. We update the dictionary, and call the
                # the expression on the other argument, which can require further nesting handling
                #---first let's collect the instances like (?x,?y) = (x1,y3) that satisfy the set definition passed in
                object_instances_set = set()
                instances_def_args = expr.args[0]
                if instances_def_args[0] == 'typed_var': #then we iterate over the objects of this type
                    var_key_string = instances_def_args[1][0]#like ?x
                    object_type = instances_def_args[1][1]
                    object_instances_set.update(self.objects[object_type])
                #FIRST if this is an average, we need a "/ |set|" on the second arg
                #todo ...could create another "sum" expression and recursively call :-)

                pass
            elif expr.etype[1] in ['forall', 'exists']:
                # I think this is also a for loop and a logical "&"(forall) or "|"(exists)
                # need to chain expressions...ugh this can be a deep tree
                pass
        elif expr.etype[0] == "control": #if statements and such
            #https://en.wikipedia.org/wiki/Abstract_syntax_tree
            # condition, if, else... NOTE WE DO Support multiple else ifs, which I guess
            # is just many if conditions, and one else condition. I assume we DO require an else condition?
            # or do we manually have to insert an else with "default" value for cpfs ??
            pass
        else:
            new_children = []
            for child in expr.args:
                new_children.append(self._scan_expr_tree(child, dic))
            #if we reached here the expression is either a +,*, or comparator (>,<) or aggregator (sum, product)
            expr = Expression((expr.etype[1], tuple(new_children)))
        #--end else
        return expr

    def _groundReward(self):
        return

    def _groundConstraints(self):
        return

