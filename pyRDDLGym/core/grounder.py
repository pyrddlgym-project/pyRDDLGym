import abc
import copy
import itertools

from pyRDDLGym.core.compiler.model import RDDLGroundedModel
from pyRDDLGym.core.debug.exception import (
    raise_warning,
    RDDLInvalidExpressionError,
    RDDLInvalidNumberOfArgumentsError,
    RDDLInvalidObjectError,
    RDDLMissingCPFDefinitionError,
    RDDLRepeatedVariableError,
    RDDLTypeError,
    RDDLUndefinedVariableError,
    RDDLValueOutOfRangeError
)
from pyRDDLGym.core.parser.expr import Expression

AGGREG_OP_TO_STRING_DICT = {
    'prod': '*',
    'sum': '+',
    'avg': '+',
    'minimum': 'min',
    'maximum': 'max',
    'forall': '^',
    'exists': '|'
}


class BaseRDDLGrounder(metaclass=abc.ABCMeta):
    '''Base class for all grounder classes.
    '''

    @abc.abstractmethod
    def ground(self) -> RDDLGroundedModel:
        '''Produces a grounded representation of the current RDDL.
        '''
        pass


class RDDLGrounder(BaseRDDLGrounder):
    '''Standard class for grounding RDDL pvariables. Does not support new 
    languages features currently.
    '''

    def __init__(self, RDDL_AST) -> None:
        '''Creates a new grounder object for grounding the specified RDDL file.
        '''
        super(RDDLGrounder, self).__init__()
        self.AST = RDDL_AST
        
        self.fluent_sep = RDDLGroundedModel.FLUENT_SEP
        self.object_sep = RDDLGroundedModel.OBJECT_SEP
        
        self.objects = {}
        self.objects_rev = {}
        
        self.variable_types = {}
        self.variable_ranges = {}
        self.variable_params = {}
        self.variable_defaults = {}
        self.variable_groundings = {}
        self.variable_base_pvars = {}
        
        self.nonfluents = {}
        self.states = {}
        self.statesranges = {}
        self.prevstates = {}
        self.nextstates = {}
        self.actions = {}
        self.actionsranges = {}
        self.derived = {}
        self.interm = {}
        self.observ = {}
        self.observranges = {}
        
        self.cpfs = {}
        self.level_to_cpfs = {}
        self.cpf_to_level = {}
        self.reward = None
        
        self.terminations = []
        self.preconditions = []
        self.invariants = []

    def ground(self) -> RDDLGroundedModel:
        self._extract_objects()
        self._ground_pvariables_and_cpf()
        self._ground_init_state()
        self._ground_init_non_fluents()
        self.reward = self._scan_expr_tree(self.AST.domain.reward, {})
        self._ground_constraints()

        # update model object
        model = RDDLGroundedModel()        
        model.ast = self.AST
        
        model.type_to_objects = {}
        model.object_to_type = {}
        model.object_to_index = {}
        model.enum_types = set()
        
        model.variable_types = self.variable_types
        model.variable_ranges = self.variable_ranges
        model.variable_params = self.variable_params
        model.variable_defaults = self.variable_defaults
        model.variable_groundings = self.variable_groundings
        model.variable_base_pvars = self.variable_base_pvars
        
        model.non_fluents = self.nonfluents
        model.state_fluents = self.states
        model.state_ranges = self.statesranges
        model.prev_state = self.prevstates
        model.next_state = self.nextstates
        model.action_fluents = self.actions
        model.action_ranges = self.actionsranges
        model.derived_fluents = self.derived
        model.interm_fluents = self.interm
        model.observ_fluents = self.observ
        model.observ_ranges = self.observranges
        
        model.cpfs = self.cpfs
        model.level_to_cpfs = self.level_to_cpfs
        model.cpf_to_level = self.cpf_to_level
        model.reward = self.reward
        
        model.terminations = self.terminations
        model.preconditions = self.preconditions
        model.invariants = self.invariants
        
        model.max_allowed_actions = self._ground_max_actions()
        model.horizon = self._ground_horizon()
        model.discount = self._ground_discount()        
        
        return model

    def _extract_objects(self):
        self.objects = {}
        self.objects_rev = {}
        if self.AST.non_fluents.objects \
        and self.AST.non_fluents.objects[0] is not None:
            for obj_type in self.AST.non_fluents.objects:
                self.objects[obj_type[0]] = obj_type[1]
                for obj in obj_type[1]:
                    self.objects_rev[obj] = obj_type[0]

    def _ground_objects(self, args):
        objects_by_type = []
        for obj_type in args:
            if obj_type not in self.objects:
                raise RDDLTypeError(
                    f'Object type <{obj_type}> is not defined, '
                    f'should be one of <{set(self.objects.keys())}>.')
            objects_by_type.append(self.objects[obj_type])
        return itertools.product(*objects_by_type)
    
    def _append_variation_to_name(self, base_name, variation):
        objects = self.object_sep.join(variation)
        return base_name + self.fluent_sep + objects
        
    def _generate_grounded_names(self, base_name, variation_list,
                                 return_grounding_param_dict=False):
        PRIME = RDDLGroundedModel.NEXT_STATE_SYM
        all_grounded_names = []
        prime_var = PRIME in base_name
        base_name = base_name.strip(PRIME)
        grounded_name_to_params_dict = {}
        for variation in variation_list:
            grounded_name = self._append_variation_to_name(base_name, variation)
            if prime_var: 
                grounded_name += PRIME
            all_grounded_names.append(grounded_name)
            grounded_name_to_params_dict[grounded_name] = variation
        if return_grounding_param_dict:
            return all_grounded_names, grounded_name_to_params_dict
        else: 
            # in some calls to _generate_name, we do not care about the param dict
            return all_grounded_names

    def _ground_pvariables_and_cpf(self):
        PRIME = RDDLGroundedModel.NEXT_STATE_SYM
        all_grounded_state_cpfs = []
        all_grounded_interim_cpfs = []
        all_grounded_derived_cpfs = []
        all_grounded_observ_cpfs = []
        for pvariable in self.AST.domain.pvariables:
            name = pvariable.name
            primed_name = (name + PRIME) if pvariable.is_state_fluent() else name
            
            # make sure name does not contain separators
            for separator in (RDDLGroundedModel.FLUENT_SEP, RDDLGroundedModel.OBJECT_SEP):
                if separator in name:
                    raise RDDLInvalidObjectError(
                        f'Variable <{name}> contains illegal separator {separator}.')
            
            # variations of all parameter objects for the variable
            if pvariable.arity > 0:
                # In line below, if we leave as an iterator object
                # will be empty after one iteration. Hence "list(.)"
                variations = list(self._ground_objects(pvariable.param_types))
                grounded, grounded_name_to_params_dict = self._generate_grounded_names(
                    name, variations,
                    return_grounding_param_dict=True)
            else:
                grounded = [name]
                grounded_name_to_params_dict = {name: []}
            
            # fill in the basic variable information
            for g in grounded:
                if g in self.variable_types:
                    raise RDDLRepeatedVariableError(
                        f'{pvariable.fluent_type} <{g}> has the same name as '
                        f'another pvariable {self.variable_types[g]}.')
                primed_g = (g + PRIME) if pvariable.is_state_fluent() else g
                self.variable_types[g] = pvariable.fluent_type
                if pvariable.is_state_fluent():
                    self.variable_types[primed_g] = 'next-state-fluent'                
                self.variable_ranges[g] = pvariable.range
                self.variable_ranges[primed_g] = pvariable.range    
                self.variable_params[g] = []
                self.variable_params[primed_g] = []    
                self.variable_defaults[g] = pvariable.default
                self.variable_groundings[g] = [g]
                self.variable_groundings[primed_g] = [primed_g]
                self.variable_base_pvars[g] = name
                self.variable_base_pvars[primed_g] = primed_name
                
            # todo merge martin's code check for abuse of arity                
            if pvariable.fluent_type == 'non-fluent':
                for g in grounded:
                    self.nonfluents[g] = pvariable.default
                
            elif pvariable.fluent_type == 'action-fluent':
                for g in grounded:
                    self.actions[g] = pvariable.default
                    self.actionsranges[g] = pvariable.range
              
            elif pvariable.fluent_type == 'state-fluent':
                cpf = None
                next_state = name + PRIME
                for cpfs in self.AST.domain.cpfs[1]:
                    if cpfs.pvar[1][0] == next_state:
                        cpf = cpfs
                        break
                if cpf is None:
                    raise RDDLMissingCPFDefinitionError(
                        f'CPF <{name}> is missing a valid definition.')
    
                for g in grounded:
                    grounded_cpf = self._ground_single_cpf(
                        cpf, g, grounded_name_to_params_dict[g])
                    all_grounded_state_cpfs.append(grounded_cpf)
                    next_state = g + PRIME  # update to grounded version, satisfied single-variables too (i.e. not a type)
                    self.states[g] = pvariable.default
                    self.statesranges[g] = pvariable.range
                    self.nextstates[g] = next_state
                    self.prevstates[next_state] = g
                    self.cpfs[next_state] = ([], grounded_cpf.expr)
                    self.level_to_cpfs.setdefault(0, []).append(g)
                    self.cpf_to_level[g] = 0
                    
            elif pvariable.fluent_type == 'derived-fluent':
                cpf = None
                for cpfs in self.AST.domain.derived_cpfs:
                    if cpfs.pvar[1][0] == name:
                        cpf = cpfs
                        break
                if cpf is None:
                    raise RDDLMissingCPFDefinitionError(
                        f'CPF <{name}> is missing a valid definition.')
                    
                for g in grounded:
                    grounded_cpf = self._ground_single_cpf(
                        cpf, g, grounded_name_to_params_dict[g])
                    all_grounded_derived_cpfs.append(grounded_cpf)
                    self.derived[g] = pvariable.default
                    self.cpfs[g] = ([], grounded_cpf.expr)
                    level = pvariable.level
                    if level is None:
                        level = 1
                    self.level_to_cpfs.setdefault(level, []).append(g)
                    self.cpf_to_level[g] = level
    
            elif pvariable.fluent_type == 'interm-fluent':
                cpf = None
                for cpfs in self.AST.domain.intermediate_cpfs:
                    if cpfs.pvar[1][0] == name:
                        cpf = cpfs
                        break
                if cpf is None:
                    raise RDDLMissingCPFDefinitionError(
                        f'CPF <{name}> is missing a valid definition.')
                    
                for g in grounded:
                    grounded_cpf = self._ground_single_cpf(
                        cpf, g, grounded_name_to_params_dict[g])
                    all_grounded_interim_cpfs.append(grounded_cpf)
                    self.interm[g] = pvariable.default
                    self.cpfs[g] = ([], grounded_cpf.expr)
                    level = pvariable.level
                    if level is None:
                        level = 1
                    self.level_to_cpfs.setdefault(level, []).append(g)
                    self.cpf_to_level[g] = level
                    
            elif pvariable.fluent_type == 'observ-fluent':
                cpf = None
                for cpfs in self.AST.domain.observation_cpfs:
                    if cpfs.pvar[1][0] == name:
                        cpf = cpfs
                        break
                if cpf is None:
                    raise RDDLMissingCPFDefinitionError(
                        f'CPF <{name}> is missing a valid definition.')
                    
                for g in grounded:
                    grounded_cpf = self._ground_single_cpf(
                        cpf, g, grounded_name_to_params_dict[g])
                    all_grounded_observ_cpfs.append(grounded_cpf)
                    self.observ[g] = pvariable.default
                    self.observranges[g] = pvariable.range
                    self.cpfs[g] = ([], grounded_cpf.expr)
                    self.level_to_cpfs.setdefault(0, []).append(g)
                    self.cpf_to_level[g] = 0

    def _ground_single_cpf(self, cpf, variable, variable_args):
        """Map arguments to actual objects."""
        args = cpf.pvar[1][1]
        new_cpf = copy.deepcopy(cpf)
        if args is None:
            new_cpf.expr = self._scan_expr_tree(new_cpf.expr, {})
            return new_cpf
        if len(args) != len(variable_args):
            raise RDDLInvalidNumberOfArgumentsError(
                f'Ground instance <{variable}> is of arity {len(variable_args)}, '
                f'but expected to be of arity {len(args)} according to definition.')
        args_dic = dict(zip(args, variable_args))
            
        # Parse cpf w.r.t cpf args and variables.
        # Fix name.
        cpf_base_name = new_cpf.pvar[1][0]
        cpf_name_grounding_variation = [[args_dic[arg] for arg in new_cpf.pvar[1][1]]]  # must be a nested list for func call
        new_name = self._generate_grounded_names(
            cpf_base_name, cpf_name_grounding_variation)
        new_pvar = ('pvar_expr', (new_name, None))
        new_cpf.pvar = new_pvar
        new_cpf.expr = self._scan_expr_tree(new_cpf.expr, args_dic)
        return new_cpf

    def do_aggregate_expression_grounding(self, original_dict, new_variables_list,
                                          instances_list, operation_string,
                                          expression):
        """Args:
                original_dict:
                new_variables_list:
                instances_list:
                operation_string:
                expression:
            Returns: Grounded expression for aggregation operations (min, max, sum, prod, forall, exists)
        """

        new_children = []
        for instance in instances_list:
            updated_dict = copy.deepcopy(original_dict)
            updated_dict.update(zip(new_variables_list, instance))
            new_children.append(self._scan_expr_tree(expression, updated_dict))
        new_expr = Expression((operation_string, tuple(new_children)))
        return new_expr

    def _scan_expr_tree_pvar(self, expr: Expression, dic) -> Expression:
        """Ground out a pvar expression."""
        if expr.args[1] is None:
            # This is a constant: should really be etype = constant in parsed tree.
            pass
        
        elif expr.args[1]:
            variation_list = []
            for arg in expr.args[1]:
                if arg not in dic:
                    raise RDDLUndefinedVariableError(
                        f'Parameter <{arg}> is not defined in call to <{expr.args[0]}>.')
                variation_list.append(dic[arg])
            variation_list = [variation_list]
            new_name = self._generate_grounded_names(expr.args[0], variation_list)[0]
            expr = Expression(('pvar_expr', (new_name, None)))
            
        else:
            raise RDDLInvalidExpressionError(f'Malformed expression <{expr}>.')
        
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
        # TODO: add default case when no "else". 
        # For now, we are safe, as else is expected in rddl
        return Expression(('if', tuple(children_list)))

    def _scan_expr_tree_func(self, expr, dic):
        new_children = []
        for child in expr.args:
            new_children.append(self._scan_expr_tree(child, dic))
        return Expression((expr.etype[0], (expr.etype[1], new_children)))  # Only one arg for abs.

    def _scan_expr_tree_aggregation(self, expr, dic):
        """Ground out an aggregation expression."""
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
            if instances_def_args: 
                
                # Then we iterate over the objects specified.
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
                        tuple(list(instance) + [objects])
                        for instance in instance_tuples
                        for objects in self.objects[object_type_list[var_idx]]
                    ]
                object_instances_list = instance_tuples
                expr = self.do_aggregate_expression_grounding(
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
        if hasattr(self.AST.domain, 'terminals'):
            for terminal in self.AST.domain.terminals:
                self.terminations.append(self._scan_expr_tree(terminal, {}))

        if hasattr(self.AST.domain, 'preconds'):
            for precond in self.AST.domain.preconds:
                self.preconditions.append(self._scan_expr_tree(precond, {}))
    
        if hasattr(self.AST.domain, 'constraints'):
            if self.AST.domain.constraints:
                raise_warning(
                    f'State-action constraints are not implemented '
                    f'in this RDDL version and will be ignored.', 'red')
    
        if hasattr(self.AST.domain, 'invariants'):
            for inv in self.AST.domain.invariants:
                self.invariants.append(self._scan_expr_tree(inv, {}))

    def _ground_init_state(self) -> None:
        if hasattr(self.AST.instance, 'init_state'):
            for init_vals in self.AST.instance.init_state:
                (key, subs), val = init_vals
                if subs:
                    key = self._append_variation_to_name(key, subs)
                if key in self.states:
                    self.states[key] = val
                else:
                    raise_warning(
                        f'Init-state block initializes undefined state-fluent <{key}>.', 
                        'red')
    
    def _ground_init_non_fluents(self) -> None:
        if hasattr(self.AST.non_fluents, 'init_non_fluent'):
            for init_vals in self.AST.non_fluents.init_non_fluent:
                (key, variations_list), val = init_vals
                if variations_list is not None:
                    key = self._generate_grounded_names(
                        key, [variations_list], 
                        return_grounding_param_dict=False)[0]  
                if key in self.nonfluents:
                    self.nonfluents[key] = val
                else:
                    raise_warning(
                        f'Non-fluents block initializes undefined non-fluent <{key}>.', 
                        'red')

    def _ground_horizon(self):
        horizon = self.AST.instance.horizon
        if not (horizon >= 0):
            raise RDDLValueOutOfRangeError(
                f'Rollout horizon {horizon} in the instance is not >= 0.')
        return horizon

    def _ground_max_actions(self):
        numactions = self.AST.instance.max_nondef_actions
        if numactions == 'pos-inf':
            return len(self.actions)
        else:
            return int(numactions)

    def _ground_discount(self):
        discount = self.AST.instance.discount
        if not (0. <= discount):
            raise RDDLValueOutOfRangeError(
                f'Discount factor {discount} in the instance is not >= 0')
        return discount