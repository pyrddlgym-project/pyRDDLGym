import abc
import copy
import itertools
import warnings

from Grounder.RDDLModel import RDDLModel
from Parser.expr import Expression

# import RDDLModel

# AGGREG_OPERATION_STRING_LIST = ["sum","prod","max","min","avg"]
# AGGREG_RECURSIVE_OPERATION_STRING_MAPPED_LIST = ["+","*","max","min","+"]
# QUANTIFIER_OPERATION_STRING_LIST = ["forall","exists"]
# QUANTIFIER_RECURSIVE_OPERATION_STRING_MAPPED_LIST = ["&","|"]

AGGREG_OPERATION_LIST = [
    'prod', 'sum', 'avg', 'minimum', 'maximum', 'forall', 'exists'
]
AGGREG_RECURSIVE_OPERATION_INDEX_MAPPED_LIST = [
    '*', '+', '+', '<', '>', '^', '|'
]
AGGREG_OP_TO_STRING_DICT = dict(
    zip(AGGREG_OPERATION_LIST, AGGREG_RECURSIVE_OPERATION_INDEX_MAPPED_LIST))


class RDDLUndefinedVariableError(SyntaxError):
    pass


class Grounder(metaclass=abc.ABCMeta):

  @abc.abstractmethod
  def Ground(self):
    pass


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

    def Ground(self):
        self._extract_objects()
        self._ground_non_fluents()
        self._ground_pvariables_and_cpf()
        self._ground_init_state()
        # ground pvariables
        # self._groundPvariables()
        self.reward = self._scan_expr_tree(self.AST.domain.reward, {})  # empty args dictionary
        self._ground_constraints()

        # update model object
        model = RDDLModel()
        model.states = self.states
        model.actions = self.actions
        model.nonfluents = self.nonfluents
        model.next_state = self.nextstates
        model.prev_state = self.prevstates
        model.init_state = self.initstate
        model.cpfs = self.cpfs
        model.cpforder = self.cpforder
        model.reward = self.reward
        model.preconditions = self.preconditions
        model.invariants = self.invariants
        model.derived = self.derived
        model.interm = self.interm
        model.objects = self.objects
        model.actionsranges = self.actionsranges
        model.statesranges = self.statesranges
        model.max_allowed_actions = self._ground_max_actions()
        model.horizon = self._ground_horizon()
        model.discount = self._ground_discount()
    
        return model

    def _ground_horizon(self):
        return self.AST.instance.horizon

    def _ground_max_actions(self):
        numactions = self.AST.instance.max_nondef_actions
        if numactions == 'pos-inf':
            return len(self.actions)
        else:
            return int(numactions)

    def _ground_discount(self):
        return self.AST.instance.discount

    def _extract_objects(self):
        if self.AST.non_fluents.objects[0] is None:
            self.objects = None
            self.objects_rev = None
        else:
            self.objects = {}
            self.objects_rev = {}
            for type in self.AST.non_fluents.objects:
                self.objects[type[0]] = type[1]
                for obj in type[1]:
                    self.objects_rev[obj] = type[0]

    def _ground_objects(self, args):
        objects_by_type = [self.objects[obj_type] for obj_type in args]
        return itertools.product(*objects_by_type)

    def _generate_grounded_names(self,
                                 base_name,
                                 variation_list,
                                 return_grounding_param_dict=False):
        all_grounded_names = []
        prime_var = "\'" in base_name
        base_name = base_name.strip("\'")
        grounded_name_to_params_dict = {}
        for variation in variation_list:
            grounded_name = base_name + '_' + '_'.join(variation)
            if prime_var: 
                grounded_name += "\'"
            all_grounded_names.append(grounded_name)
            grounded_name_to_params_dict[grounded_name] = variation
        if return_grounding_param_dict:
            return all_grounded_names, grounded_name_to_params_dict
        else:  # in some calls to _generate_name, we do not care about the param dict, hence this
            return all_grounded_names

    def _ground_non_fluents(self):
        if not hasattr(self.AST.non_fluents, 'init_non_fluent'):
            return
        for init_vals in self.AST.non_fluents.init_non_fluent:
            name = init_vals[0][0]
            variations_list = [init_vals[0][1]]
            val = init_vals[1]
            if variations_list[0] is not None:
                name = self._generate_grounded_names(
                    name, variations_list, return_grounding_param_dict=False)[0]    
            self.nonfluents[name] = val

    def _ground_interm(self, pvariable, cpf_set):
        cpf = None
        for cpfs in cpf_set:
            if cpfs.pvar[1][0] == pvariable.name:
                cpf = cpfs
        if cpf is not None:
            self.derived[pvariable.name] = pvariable.default
            self.cpfs[pvariable.name] = cpf.expr
            level = pvariable.level
            if level is None:
                level = 1
            if level in self.cpforder:
                self.cpforder[level].append(pvariable.name)
            else:
                self.cpforder[level] = [pvariable.name]

    def _ground_pvariables_and_cpf(self):
        all_grounded_state_cpfs = []
        all_grounded_interim_cpfs = []
        all_grounded_derived_cpfs = []
        for pvariable in self.AST.domain.pvariables:
            name = pvariable.name
            if pvariable.arity > 0:
                # In line below, if we leave as an iterator object, will be empty after one iteration. Hence "list(.)"
                variations = list(self._ground_objects(pvariable.param_types))
                grounded, grounded_name_to_params_dict = self._generate_grounded_names(
                    name, variations, return_grounding_param_dict=True)
            else:
                grounded = [name]
                grounded_name_to_params_dict = {name: []}
            
            # todo merge martin's code check for abuse of arity                
            if pvariable.fluent_type == 'non-fluent':
                for g in grounded:
                    if g not in self.nonfluents:
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
                        break
                if cpf is None:
                    warnings.warn('No conditional prob func found for ' + name)
    
                for g in grounded:
                    grounded_cpf = self._ground_single_cpf(
                        cpf, g, grounded_name_to_params_dict[g])
                    all_grounded_state_cpfs.append(grounded_cpf)
                    next_state = g + '\''  # update to grounded version, satisfied single-variables too (i.e. not a type)
                    self.states[g] = pvariable.default
                    self.statesranges[g] = pvariable.range
                    self.nextstates[g] = next_state
                    self.prevstates[next_state] = g
                    self.cpfs[next_state] = grounded_cpf.expr
                    self.cpforder[0].append(g)
              
            elif pvariable.fluent_type == 'derived-fluent':
                cpf = None
                for cpfs in self.AST.domain.derived_cpfs:
                    if cpfs.pvar[1][0] == name:
                        cpf = cpfs
                        break
                if cpf is None:
                    warnings.warn('No conditional prob func found for ' + name)
                for g in grounded:
                    grounded_cpf = self._ground_single_cpf(
                        cpf, g, grounded_name_to_params_dict[g])
                    all_grounded_derived_cpfs.append(grounded_cpf)
                    self.derived[g] = pvariable.default
                    self.cpfs[g] = grounded_cpf.expr
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
                if cpf is None:
                    warnings.warn('No conditional prob func found for ' + name)
                for g in grounded:
                    grounded_cpf = self._ground_single_cpf(
                        cpf, g, grounded_name_to_params_dict[g])
                    all_grounded_interim_cpfs.append(grounded_cpf)
                    self.interm[g] = pvariable.default
                    self.cpfs[g] = grounded_cpf.expr
                    level = pvariable.level
                    if level is None:
                        level = 1
                    if level in self.cpforder:
                        self.cpforder[level].append(g)
                    else:
                        self.cpforder[level] = [g]
        # self.AST.domain.cpfs = (self.AST.domain.cpfs[0], all_grounded_state_cpfs
        #                        )  # replacing the previous lifted entries
        # self.AST.domain.derived_cpfs = all_grounded_derived_cpfs
        # self.AST.domain.intermediate_cpfs = all_grounded_interim_cpfs

    def _ground_single_cpf(self, cpf, variable, variable_args):
        """Map arguments to actual objects."""
        args = cpf.pvar[1][1]
        new_cpf = copy.deepcopy(cpf)
        if args is None:
            new_cpf.expr = self._scan_expr_tree(new_cpf.expr, {})
            return new_cpf
        args_dic = {}
        if len(args) != len(variable_args):
            raise ValueError(
                f'Ground instance {variable} is of arity {len(variable_args)} but '
                f'was expected to be of arity {len(args)} according to declaration.')
        for arg, vararg in zip(args, variable_args):
            args_dic[arg] = vararg
            
        # Parse cpf w.r.t cpf args and variables.
        # Fix name.
        cpf_base_name = new_cpf.pvar[1][0]
        cpf_name_grounding_variation = [[args_dic[arg] for arg in new_cpf.pvar[1][1]]]  # must be a nested list for func call
        new_name = self._generate_grounded_names( cpf_base_name,cpf_name_grounding_variation)
        new_pvar = ('pvar_expr', (new_name, None))
        new_cpf.pvar = new_pvar
        new_cpf.expr = self._scan_expr_tree(new_cpf.expr, args_dic)
        return new_cpf

    def do_aggregate_expression_nesting(self, original_dict, new_variables_list,
                                        instances_list, operation_string,
                                        expression):
        """Args:
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
    
        # todo create expression with the type passed in
        # the first argument (lhs) will have an updated dictionary based on the objects spec'd
        # in the set definition (gets one instance added to the dictionary). We can scan_expression on it
        # for the second arg (righ hand side), we will recursively call this func with a reduced set of objects
        if len(instances_list) == 1:
            # this case CAN happen, if there is only one object of the type specified
            # which can be due to misspec or due to a difficult constraint satisfaction in set definition,
            # that varies over instances
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
            # even if it is a max or min operation, when there are only 2 left, we just do "> or <"
            new_expr = Expression((operation_string, tuple(new_children)))
        else:  # recursive case
            if operation_string in {'+', '*', '&', '|'}:  # those ">,<" are for the min and max respectively
                new_children = []
                lhs_updated_dict = copy.deepcopy(original_dict)
                lhs_updated_dict.update(dict(zip(new_variables_list, instances_list[0])))
                instances_list = instances_list[1:]  # remove the first element before the recursion, simple.
                new_children.append(self._scan_expr_tree(expression, lhs_updated_dict))
                new_children.append(
                    self.do_aggregate_expression_nesting(original_dict,
                                                         new_variables_list,
                                                         instances_list,
                                                         operation_string, expression))
                new_expr = Expression((operation_string, tuple(new_children)))
            else:  # Handling the min and max case.
                # this is done by comparing two at a time, and recursively calling with the remainder of the list
                # keeping the one that "wins" the comparison. Eg: if (a > b) then (>, [a, rest of list]) else (>, [b, rest of list])
                # I assume if they are equal, we have a default behavior in the simulator (eg: take first arg)
                lhs_updated_dict = copy.deepcopy(original_dict)
                lhs_updated_dict.update(dict(zip(new_variables_list, instances_list[0])))
                lhs_comparison_arg = self._scan_expr_tree(expression, lhs_updated_dict)
                rhs_updated_dict = copy.deepcopy(original_dict)
                rhs_updated_dict.update(dict(zip(new_variables_list, instances_list[1])))
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
    
        # ---end else
        return new_expr

    def _scan_expr_tree_pvar(self, expr: Expression, dic) -> Expression:
        """Ground out a pvar expression."""
        if expr.args[1] is None:
            # should really be etype = constant in parsed tree.
            # This is a constant.
            pass
        elif expr.args[1]:
            variation_list = [[dic[arg] for arg in expr.args[1]]]
            new_name = self._generate_grounded_names(expr.args[0], variation_list)[0]
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
        # TODO: add default case when no "else". For now, we are safe, as else is expected in rddl
        return Expression(('if', tuple(children_list)))

    def _scan_expr_tree_func(self, expr, dic):
        new_children = []
        for child in expr.args:
            new_children.append(self._scan_expr_tree(child, dic))
        return Expression((expr.etype[0], (expr.etype[1], new_children)))  # Only one arg for abs.

    def _scan_expr_tree_aggregation(self, expr, dic):
        """Ground out an aggregation expression."""
        # TODO: as of now the code assumes all the leaf variables/constants are
        # of the right types, or can be reasonably cast into the right type
        # (eg: bool->int or v.v.).
        # However, some type checking would be nice in subsequent versions,
        # and give feedback to the language writer for debugging.
        aggreg_type = expr.etype[1]
        if aggreg_type in AGGREG_OP_TO_STRING_DICT:
            # Determine what the recursive op is for the aggreg type. Eg: sum = "+".
            aggreg_recursive_operation_string = AGGREG_OP_TO_STRING_DICT[aggreg_type]
            # TODO: only for average operation, we need to first "/ n " for all others
            # we need to decide the recursive operation symbol "+" or "*" and iterate.
            # First let's collect the instances like (?x,?y) = (x1, y3) that satisfy,
            # the set definition passed in.
            object_instances_list = []
            instances_def_args = []
            for arg in expr.args:
                if arg[0] == 'typed_var':
                    instances_def_args += list(arg[1])
            if instances_def_args:  # Then we iterate over the objects specified.
                # All even indexes (incl 0) are variable names,
                # all odd indexes are object types.
                var_key_strings_list = [
                    instances_def_args[2 * x]
                    for x in range(int(len(instances_def_args) / 2))
                ]  # Like ?x.
                object_type_list = [
                    instances_def_args[2 * x + 1]
                    for x in range(int(len(instances_def_args) / 2))
                ]
                instance_tuples = [
                    tuple([x]) for x in self.objects[object_type_list[0]]
                ]
                for var_idx in range(1, len(var_key_strings_list)):
                    instance_tuples = [
                        tuple(list(instance_tuples[i]) + [self.objects[object_type_list[var_idx]][j]])
                        for i in range(len(instance_tuples))
                        for j in range(len(self.objects[object_type_list[var_idx]]))
                    ]
                object_instances_list = instance_tuples
                expr = self.do_aggregate_expression_nesting(
                    dic, var_key_strings_list, object_instances_list,
                    aggreg_recursive_operation_string, arg
                )  # Last arg is the expression with which the aggregation is done.
                if aggreg_type == 'avg':
                    num_instances = len(instance_tuples)  # Needed if this is an "Avg" operation.
                    # Then the 'expr' becomes lhs argument and
                    # we add a "\ |set_size|" operation.
                    children_list = [expr, Expression(('number', num_instances))]
                    # Note "expr" would have been an aggregate sum already,
                    # the "aggreg_recursive_operation_string" is set for that.
                    expr = Expression(('/', tuple(children_list)))
            return expr

    def _scan_expr_tree(self, expr: Expression, dic) -> Expression:
        """Main dispatch method for recursively grounding the expression tree."""
        scan_expr_tree_noop = lambda expr, _: expr
        dispatch_dict = {
            'noop': scan_expr_tree_noop,
            'pvar': self._scan_expr_tree_pvar,
            'constant': scan_expr_tree_noop,
            'arithmetic': self._scan_expr_tree_abr,
            'boolean': self._scan_expr_tree_abr,
            'relational': self._scan_expr_tree_abr,
            'aggregation': self._scan_expr_tree_aggregation,
            'control': self._scan_expr_tree_control,
            # Random vars can be ground in the same way as functions.
            'func': self._scan_expr_tree_func,
            'randomvar': self._scan_expr_tree_func
        }
        if isinstance(expr, tuple):
            expression_type = 'noop'
        else:
            expression_type = expr.etype[0]
        if expression_type in dispatch_dict.keys():
            return dispatch_dict[expression_type](expr, dic)
        else:
            new_children = []
            for child in expr.args:
                new_children.append(self._scan_expr_tree(child, dic))
            # If we reached here the expression is either a +,*, or comparator (>,<),
            # or aggregator (sum, product).
            return Expression((expr.etype[1], tuple(new_children)))

    def _ground_constraints(self) -> None:
        if hasattr(self.AST.domain, 'preconds'):
            for precond in self.AST.domain.preconds:
                self.preconditions.append(self._scan_expr_tree(precond, {}))
    
        if hasattr(self.AST.domain, 'constraints'):
            for constraint in self.AST.domain.preconds:
                self.preconditions.append(self._scan_expr_tree(constraint, {}))
    
        if hasattr(self.AST.domain, 'invariants'):
            for inv in self.AST.domain.invariants:
                self.invariants.append(self._scan_expr_tree(inv, {}))

    def _ground_init_state(self) -> None:
        self.initstate = self.states.copy()
        if hasattr(self.AST.instance, 'init_state'):
            for init_vals in self.AST.instance.init_state:
                (key, subs), val = init_vals
                if subs is not None and len(subs) > 0:
                    key = key + '_' + '_'.join(subs)
                if key not in self.initstate:
                    raise RDDLUndefinedVariableError(
                        'Variable {} defined in init-state is not a state fluent.'.format(key))
                self.initstate[key] = val


