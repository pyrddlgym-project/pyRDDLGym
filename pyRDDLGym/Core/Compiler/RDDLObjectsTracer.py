import numpy as np
from typing import Callable, List, Set, Tuple, Union

from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidNumberOfArgumentsError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidObjectError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLNotImplementedError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLTypeError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLUndefinedVariableError

from pyRDDLGym.Core.Compiler.RDDLDecompiler import RDDLDecompiler
from pyRDDLGym.Core.Compiler.RDDLModel import PlanningModel
from pyRDDLGym.Core.Debug.Logger import Logger
from pyRDDLGym.Core.Parser.expr import Expression


class RDDLTracedObjects:
    
    def __init__(self) -> None:
        self._current_id = 0
        self._cached_objects_in_scope = []
        self._cached_enum_type = []
        self._cached_sim_info = []
        
    def _append(self, expr, objects, enum_type, info) -> None:
        expr.id = self._current_id
        self._current_id += 1
                
        self._cached_objects_in_scope.append(objects)
        self._cached_enum_type.append(enum_type)
        self._cached_sim_info.append(info)
        
    def cached_objects_in_scope(self, expr: Expression):
        return self._cached_objects_in_scope[expr.id]
    
    def cached_enum_type(self, expr: Expression) -> str:
        return self._cached_enum_type[expr.id]
    
    def cached_sim_info(self, expr: Expression) -> object:
        return self._cached_sim_info[expr.id]
    

class RDDLObjectsTracer:
    '''Performs static/compile-time tracing of a RDDL AST representation and
    annotates nodes with useful information about objects and enums that appear
    inside expressions.
    '''
    
    VALID_SYMBOLS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    def __init__(self, rddl: PlanningModel,
                 logger: Logger=None) -> None:
        '''Creates a new objects tracer object for the given RDDL domain.
        
        :param rddl: the RDDL domain to trace
        :param logger: to log compilation information during tracing to file
        '''
        self.rddl = rddl
        self.logger = logger
            
    @staticmethod
    def _check_not_enum(arg, expr, out, msg):
        enum_type = out.cached_enum_type(arg)
        if enum_type is not None:
            raise RDDLTypeError(
                f'{msg} can not be an enum type <{enum_type}>.\n' + 
                RDDLObjectsTracer._print_stack_trace(expr)) 
                
    @staticmethod
    def _print_stack_trace(expr):
        if isinstance(expr, Expression):
            trace = RDDLDecompiler().decompile_expr(expr)
        else:
            trace = str(expr)
        return f'>> {trace}'
    
    def trace(self) -> RDDLTracedObjects:
        '''Traces all expressions in CPF block and all constraints and annotates
        AST nodes with object information.
        '''   
        rddl = self.rddl 
        out = RDDLTracedObjects()   
        
        # trace CPFs; for enum-valued check type matches
        for (cpf, (objects, expr)) in rddl.cpfs.items():
            self._trace(expr, objects, out)
            cpf_range = rddl.variable_ranges[cpf]
            expr_range = out.cached_enum_type(expr)
            if cpf_range in rddl.enum_types and expr_range != cpf_range:
                if expr_range is None:
                    raise RDDLTypeError(
                        f'CPF <{cpf}> expects enum value of type <{cpf_range}>, '
                        f'got non-enum value.')
                else:
                    raise RDDLTypeError(
                        f'CPF <{cpf}> expects enum value of type <{cpf_range}>, '
                        f'got enum value of type <{expr_range}>.')

        # trace reward; check not enum value
        self._trace(rddl.reward, [], out)
        RDDLObjectsTracer._check_not_enum(rddl.reward, rddl.reward, out, 'reward')
        
        # trace all constraints; check not enum value
        for (i, expr) in enumerate(rddl.invariants):
            self._trace(expr, [], out)
            RDDLObjectsTracer._check_not_enum(expr, expr, out, f'Invariant {i + 1}')
        for (i, expr) in enumerate(rddl.preconditions):
            self._trace(expr, [], out)
            RDDLObjectsTracer._check_not_enum(expr, expr, out, f'Precondition {i + 1}')
        for (i, expr) in enumerate(rddl.terminals):
            self._trace(expr, [], out)
            RDDLObjectsTracer._check_not_enum(expr, expr, out, f'Termination {i + 1}')
            
        return out
        
    # ===========================================================================
    # start of tracing subroutines
    # ===========================================================================
        
    def _trace(self, expr, objects, out):
        etype, _ = expr.etype
        if etype == 'constant':
            self._trace_constant(expr, objects, out)
        elif etype == 'pvar':
            self._trace_pvar(expr, objects, out)
        elif etype == 'arithmetic':
            self._trace_arithmetic(expr, objects, out)
        elif etype == 'relational':
            self._trace_relational(expr, objects, out)
        elif etype == 'boolean':
            self._trace_logical(expr, objects, out)
        elif etype == 'aggregation':
            self._trace_aggregation(expr, objects, out)
        elif etype == 'func':
            self._trace_func(expr, objects, out)
        elif etype == 'control':
            self._trace_control(expr, objects, out)
        elif etype == 'randomvar':
            self._trace_random(expr, objects, out)
        else:
            raise RDDLNotImplementedError(
                f'Internal error: expression type is not supported.\n' + 
                RDDLObjectsTracer._print_stack_trace(expr))
    
    # ===========================================================================
    # leaves
    # ===========================================================================
    
    def _trace_constant(self, expr, objects, out):
        if objects:
            ptypes = (ptype for (_, ptype) in objects)
            shape = self.rddl.object_counts(ptypes)
            cached_value = np.full(shape=shape, fill_value=expr.args)
        else:
            cached_value = expr.args
            
        out._append(expr, objects, None, cached_value)
        
    def _trace_pvar(self, expr, objects, out):
        var, pvars = expr.args   
            
        # enum literal value treated as int
        if not pvars and self.rddl.is_literal(var): 
            literal = self.rddl.literal_name(var)
            const = self.rddl.index_of_object[literal]
            if objects:
                ptypes = (ptype for (_, ptype) in objects)
                shape = self.rddl.object_counts(ptypes)
                cached_value = np.full(shape=shape, fill_value=const)
            else:
                cached_value = const
                
            enum_type = self.rddl.objects_rev[literal]            
            out._append(expr, objects, enum_type, cached_value)
        
        # if the pvar has free variables:
        # 1. enum literal args (e.g. @x) converted to ints and slice the array
        # 2. resulting array is reshaped and axes rearranged to match objects
        else:
            if objects:
                slices, literals = self._slice(var, pvars)
                transform = self._map(var, pvars, objects, literals)
                cached_sim_info = (slices, transform)
            else:
                cached_sim_info = None
            
            prange = self.rddl.variable_ranges.get(var, None)
            enum_type = prange if prange in self.rddl.enum_types else None            
            out._append(expr, objects, enum_type, cached_sim_info)
        
    def _slice(self, var: str,
               pvars: List[str]) -> Tuple[Set[int], Tuple[Union[int, slice], ...]]:
        '''Given a var and a list of pvariable arguments containing both free 
        and/or enum literals, produces a tuple of slice or int objects that, 
        when indexed into an array of the appropriate shape, produces an array 
        where all literals are evaluated in the array. Also returns a set of 
        indices of pvars that are literals.
        '''
        
        # check that the number of pvariable arguments matches definition
        if pvars is None:
            pvars = []
        types_in = self.rddl.param_types.get(var, [])
        if len(pvars) != len(types_in):
            raise RDDLInvalidNumberOfArgumentsError(
                f'Variable <{var}> requires {len(types_in)} parameter(s), '
                f'got {len(pvars)}.')
            
        cached_slices, literals = [], set()
        for (i, pvar) in enumerate(pvars):
            
            # nested fluents not yet supported
            if isinstance(pvar, tuple):
                raise RDDLNotImplementedError(
                    f'Nested variables are not yet supported.\n' + 
                    RDDLObjectsTracer._print_stack_trace(pvars))
            
            # is an enum literal
            if self.rddl.is_literal(pvar):
                literal = self.rddl.literal_name(pvar)
                
                # check that the enum type argument is correct
                enum_type = self.rddl.objects_rev[literal]
                if types_in[i] != enum_type: 
                    raise RDDLInvalidObjectError(
                        f'Argument {i + 1} of variable <{var}> expects type '
                        f'<{types_in[i]}>, got <{pvar}> of type <{enum_type}>.')
                
                # an enum literal argument (e.g., @x) corresponds to slicing
                # value tensor at this axis with canonical index of the literal  
                index = self.rddl.index_of_object[literal]
                cached_slices.append(index)
                literals.add(i)
            
            # is a proper type argument (e.g., ?x): in this case do NOT slice
            # value tensor at all (e.g., emulate numpy's :) at this axis
            else:
                colon = slice(None)
                cached_slices.append(colon)
                        
        return (tuple(cached_slices), literals)

    def _map(self, var: str,
             obj_in: List[str],
             sign_out: List[Tuple[str, str]],
             literals: Set[int]) -> Callable[[np.ndarray], np.ndarray]:
        '''Returns a function that transforms a pvariable value tensor to one
        whose shape matches a desired output signature. This operation is
        achieved by adding new dimensions to the value as needed, and performing
        a combination of transposition/reshape/einsum operations to coerce the 
        value to the desired shape. Also returns a string of additional info
        about the transformation.
        
        :param var: a string pvariable defined in the domain
        :param obj_in: a list of desired object quantifications, e.g. ?x, ?y at
        which var will be evaluated
        :param sign_out: a list of tuples (objecti, typei) representing the
            desired signature of the output pvariable tensor
        :param literals: which indices of obj_in are treated as
            literal (e.g. enum literal or another pvariable) parameters
        :param gnp: the library in which to perform tensor arithmetic 
        (either numpy or jax.numpy)
        :param msg: a stack trace to print for error handling
        '''
               
        # check that the number of input objects match fluent type definition
        if obj_in is None:
            obj_in = []
        types_in = self.rddl.param_types.get(var, [])
        if len(obj_in) != len(types_in):
            raise RDDLInvalidNumberOfArgumentsError(
                f'Variable <{var}> requires {len(types_in)} parameter(s), '
                f'got {len(obj_in)}.')
        
        # eliminate literals
        old_sign_in = tuple(zip(obj_in, types_in))
        types_in = [p for (i, p) in enumerate(types_in) if i not in literals]
        obj_in = [p for (i, p) in enumerate(obj_in) if i not in literals]
        
        # reached limit on number of valid dimensions
        valid_symbols = RDDLObjectsTracer.VALID_SYMBOLS
        n_out = len(sign_out)
        if n_out > len(valid_symbols):
            raise RDDLInvalidNumberOfArgumentsError(
                f'At most {len(valid_symbols)} parameter(s) are supported '
                f'but variable <{var}> has {n_out} argument(s).')
        
        # find a map permutation(a,b,c...) -> (a,b,c...) for correct transpose
        sign_in = tuple(zip(obj_in, types_in))
        permutation = [None] * len(obj_in)
        new_dims = []
        for (i_out, (o_out, t_out)) in enumerate(sign_out):
            new_dim = True
            for (i_in, (o_in, t_in)) in enumerate(sign_in):
                if o_in == o_out:
                    permutation[i_in] = i_out
                    new_dim = False
                    if t_out != t_in: 
                        raise RDDLInvalidObjectError(
                            f'Argument {i_in + 1} of variable <{var}> '
                            f'expects type <{t_in}>, got <{o_out}> of type <{t_out}>.')
            
            # need to expand the shape of the value array
            if new_dim:
                permutation.append(i_out)
                num_objects = len(self.rddl.objects[t_out])
                new_dims.append(num_objects)
                
        # safeguard against any free remaining variables not accounted for
        free_variables = {obj_in[i] 
                          for (i, p) in enumerate(permutation) 
                          if p is None}
        if free_variables:
            raise RDDLInvalidNumberOfArgumentsError(
                f'Variable <{var}> has unresolved parameter(s) {free_variables}.')
        
        # compute the mapping function as follows:
        # 1. append new axes to value tensor equal to # of missing variables
        # 2. broadcast new axes to the desired shape (# of objects of each type)
        # 3. rearrange the axes as needed to match the desired variables in order
        #     3a. in most cases, it suffices to use np.transform (cheaper)
        #     3b. in cases where we have a more complex contraction like 
        #         fluent(?x) = matrix(?x, ?x), we will use np.einsum
        in_shape = self.rddl.object_counts(types_in)
        out_shape = in_shape + tuple(new_dims)
        new_axis = tuple(range(len(in_shape), len(out_shape)))
         
        lhs = ''.join(valid_symbols[p] for p in permutation)        
        rhs = valid_symbols[:n_out]
        use_einsum = len(set(lhs)) != len(lhs)
        use_tr = lhs != rhs
        if use_einsum:
            subscripts = lhs + '->' + rhs
        elif use_tr:
            subscripts = tuple(np.argsort(permutation))  # inverse permutation
        else:
            subscripts = None
        
        # log information about the new transformation
        if self.logger is not None:
            operation = 'einsum' if use_einsum else (
                'transpose' if use_tr else 'None')
            message = (f'computing info for filling missing pvariable arguments:' 
                       f'\n\tvar           ={var}'
                       f'\n\toriginal args ={old_sign_in}'
                       f'\n\tliterals      ={literals}'
                       f'\n\tnew args      ={sign_in}'
                       f'\n\ttarget args   ={sign_out}'
                       f'\n\tnew axes      ={new_axis}'
                       f'\n\toperation     ={operation}, subscripts={subscripts}\n')
            self.logger.log(message)
            
        return (new_axis, out_shape, use_einsum, use_tr, subscripts)
    
    # ===========================================================================
    # compound expressions
    # ===========================================================================
    
    def _trace_arithmetic(self, expr, objects, out): 
        for (i, arg) in enumerate(expr.args):
            self._trace(arg, objects, out)
        
            # argument cannot be enum type
            RDDLObjectsTracer._check_not_enum(
                arg, expr, out, f'Argument {i + 1} of operator {expr.etype[1]}')  
              
        out._append(expr, objects, None, None)
        
    def _trace_relational(self, expr, objects, out):
        _, op = expr.etype
        for arg in expr.args:
            self._trace(arg, objects, out)            
        
        # can not mix different enum types or primitive and enum types
        enum_types = {out.cached_enum_type(arg) for arg in expr.args}
        if len(enum_types) != 1:
            raise RDDLTypeError(
                f'Relational operator {op} can not compare arguments '
                f'of different enum types or mix enum and non-enum types.\n' + 
                RDDLObjectsTracer._print_stack_trace(expr))
        
        # can not use operator besides == and ~= to compare enum types
        enum_type = next(iter(enum_types))
        if enum_type is not None and op != '==' and op != '~=':
            raise RDDLNotImplementedError(
                f'Relational operator {op} is not valid for comparing enum types, '
                f'must be either == or ~=.\n' + 
                RDDLObjectsTracer._print_stack_trace(expr))
        
        out._append(expr, objects, None, None)
    
    def _trace_logical(self, expr, objects, out):
        for (i, arg) in enumerate(expr.args):
            self._trace(arg, objects, out)
            
            # argument cannot be enum type
            RDDLObjectsTracer._check_not_enum(
                arg, expr, out, f'Argument {i + 1} of operator {expr.etype[1]}')
        
        out._append(expr, objects, None, None)
    
    def _trace_aggregation(self, expr, objects, out):
        _, op = expr.etype
        * pargs, arg = expr.args
        
        # cache and read reduced axes tensor info for the aggregation
        new_objects = objects + [ptype for (_, ptype) in pargs]
        reduced_axes = tuple(range(len(objects), len(new_objects)))   
        cached_sim_info = (new_objects, reduced_axes)
        
        # check for undefined types
        bad_types = {ptype 
                     for (_, ptype) in new_objects 
                     if ptype not in self.rddl.objects}
        if bad_types:
            raise RDDLInvalidObjectError(
                f'Type(s) {bad_types} are not defined, '
                f'must be one of {set(self.rddl.objects.keys())}.\n' + 
                RDDLObjectsTracer._print_stack_trace(expr))
            
        # check for valid type arguments
        outer_scope_vars = {var for (var, _) in objects}
        free_vars_seen = set()
        for (_, (free_new, _)) in pargs:
            
            # check that there is no duplicated iteration variable
            if free_new in free_vars_seen:
                raise RDDLInvalidObjectError(
                    f'Iteration variable <{free_new}> is repeated in aggregation.\n' + 
                    RDDLObjectsTracer._print_stack_trace(expr))             
            free_vars_seen.add(free_new)
             
            # check if iteration variable is same as one defined in outer scope
            # since there is ambiguity to which is referred I raise an error
            if free_new in outer_scope_vars:
                raise RDDLInvalidObjectError(
                    f'Iteration variable <{free_new}> is already defined '
                    f'in outer scope.\n' + 
                    RDDLObjectsTracer._print_stack_trace(expr))
        
        # trace the aggregated expression with the new objects
        self._trace(arg, new_objects, out)
        
        # argument cannot be enum type
        RDDLObjectsTracer._check_not_enum(
            arg, expr, out, f'Argument of aggregation {op}')     
        
        out._append(expr, objects, None, cached_sim_info)
        
        # log information about aggregation operation
        if self.logger is not None:
            message = (f'computing object info for aggregation:'
                       f'\n\toperator       ={op} {pargs}'
                       f'\n\tinput objects  ={new_objects}'
                       f'\n\toutput objects ={objects}'
                       f'\n\toperation axes ={reduced_axes}\n')
            self.logger.log(message)        
        
    def _trace_func(self, expr, objects, out):
        for (i, arg) in enumerate(expr.args):
            self._trace(arg, objects, out)
            
            # argument cannot be enum type
            RDDLObjectsTracer._check_not_enum(
                arg, expr, out, f'Argument {i + 1} of function {expr.etype[1]}') 
        
        out._append(expr, objects, None, None)
            
    # ===========================================================================
    # control flow
    # ===========================================================================
    
    def _trace_control(self, expr, objects, out):
        _, op = expr.etype
        if op == 'if':
            self._trace_if(expr, objects, out)
        elif op == 'switch':
            self._trace_switch(expr, objects, out)
            
    def _trace_if(self, expr, objects, out):
        pred, *cases = expr.args
        self._trace(pred, objects, out)
        for arg in cases:
            self._trace(arg, objects, out)
        
        # can not mix different enum types or primitive and enum types
        enum_types = {out.cached_enum_type(arg) for arg in cases}
        if len(enum_types) != 1:
            raise RDDLTypeError(
                f'Branches in if then else statement cannot produce values '
                f'of different enum types or mix enum and non-enum types.\n' + 
                RDDLObjectsTracer._print_stack_trace(expr))     
    
        enum_type = next(iter(enum_types))
        out._append(expr, objects, enum_type, None)
        
    def _trace_switch(self, expr, objects, out):
        pred, *cases = expr.args
        
        # must be a pvar
        if not pred.is_pvariable_expression():
            raise RDDLNotImplementedError(
                f'Switch predicate is not a pvariable.\n' + 
                RDDLObjectsTracer._print_stack_trace(expr))
            
        # type in pvariables scope must be an enum
        var, _ = pred.args
        enum_type = self.rddl.variable_ranges[var]
        if enum_type not in self.rddl.enum_types:
            raise RDDLTypeError(
                f'Type <{enum_type}> of switch predicate <{var}> is not an '
                f'enum type, must be one of {self.rddl.enum_types}.\n' + 
                RDDLObjectsTracer._print_stack_trace(expr))
            
        # default statement becomes ("default", expr)
        case_dict = dict((cvalue if ctype == 'case' else (ctype, cvalue)) 
                          for (ctype, cvalue) in cases)
        if len(case_dict) != len(cases):
            raise RDDLInvalidNumberOfArgumentsError(
                f'Duplicated literal or default case(s).\n' + 
                RDDLObjectsTracer._print_stack_trace(expr))
        
        # order enum cases by canonical ordering of literals
        cached_sim_info = self._order_cases(enum_type, case_dict, expr)
        
        # trace predicate and cases
        self._trace(pred, objects, out)
        for arg in case_dict.values():
            self._trace(arg, objects, out)
        
        # can not mix different enum types or primitive and enum types
        enum_types = {out.cached_enum_type(arg) for arg in case_dict.values()}
        if len(enum_types) != 1:
            raise RDDLTypeError(
                f'Case expressions in switch statement cannot produce values '
                f'of different enum types or mix enum and non-enum types.\n' + 
                RDDLObjectsTracer._print_stack_trace(expr))    
                                
        enum_type = next(iter(enum_types))
        out._append(expr, objects, enum_type, cached_sim_info)
        
    def _order_cases(self, enum_type, case_dict, expr): 
        enum_values = self.rddl.objects[enum_type]
        
        # check that all literals belong to enum_type
        for literal in case_dict:
            if literal != 'default' \
            and self.rddl.objects_rev.get(literal, None) != enum_type:
                raise RDDLUndefinedVariableError(
                    f'Literal <{literal}> does not belong to enum type '
                    f'<{enum_type}>, must be one of {set(enum_values)}.\n' + 
                    RDDLObjectsTracer._print_stack_trace(expr))
        
        # store expressions in order of canonical literal index
        expressions = [None] * len(enum_values)
        for literal in enum_values:
            arg = case_dict.get(literal, None)
            if arg is not None: 
                index = self.rddl.index_of_object[literal]
                expressions[index] = arg
        
        # if default statement is missing, cases must be comprehensive
        default_expr = case_dict.get('default', None)
        if default_expr is None:
            for (i, arg) in enumerate(expressions):
                if arg is None:
                    raise RDDLUndefinedVariableError(
                        f'Enum literal <{enum_values[i]}> of type <{enum_type}> '
                        f'is missing in case list.\n' + 
                        RDDLObjectsTracer._print_stack_trace(expr))
        
        # log cases ordering
        if self.logger is not None:
            active_expr = [i for (i, e) in enumerate(expressions) if e is not None]
            message = (f'computing case info for {expr.etype[1]}:'
                       f'\n\tenum type ={enum_type}'
                       f'\n\tcases     ={active_expr}'
                       f'\n\tdefault   ={default_expr is not None}\n')
            self.logger.log(message)     
        
        return (expressions, default_expr)
    
    # ===========================================================================
    # random variable
    # ===========================================================================
    
    def _trace_random(self, expr, objects, out):
        _, name = expr.etype
        if name == 'Discrete' or name == 'UnnormDiscrete':
            self._trace_discrete(expr, objects, out)
        else:
            self._trace_random_other(expr, objects, out)
                
    def _trace_discrete(self, expr, objects, out):
        (_, enum_type), *cases = expr.args
            
        # enum type must be a valid enum type
        if enum_type not in self.rddl.enum_types:
            raise RDDLTypeError(
                f'Type <{enum_type}> in Discrete distribution is not an '
                f'enum, must be one of {self.rddl.enum_types}.\n' + 
                RDDLObjectsTracer._print_stack_trace(expr))
            
        # no duplicate cases are allowed
        case_dict = dict(case_tup for (_, case_tup) in cases)
        if len(case_dict) != len(cases):
            raise RDDLInvalidNumberOfArgumentsError(
                f'Duplicated literal or default cases.\n' + 
                RDDLObjectsTracer._print_stack_trace(expr))
        
        # no default cases are allowed
        if 'default' in case_dict:
            raise RDDLNotImplementedError(
                f'Default case not allowed in Discrete distribution.\n' + 
                RDDLObjectsTracer._print_stack_trace(expr))
            
        # order enum cases by canonical ordering of literals
        cached_sim_info, _ = self._order_cases(enum_type, case_dict, expr)
    
        # trace each case expression
        for (i, arg) in enumerate(case_dict.values()):
            self._trace(arg, objects, out)
            
            # argument cannot be enum type
            RDDLObjectsTracer._check_not_enum(
                arg, expr, out, f'Expression in case {i + 1} of Discrete') 
        
        out._append(expr, objects, enum_type, cached_sim_info)
    
    def _trace_random_other(self, expr, objects, out):
        for (i, arg) in enumerate(expr.args):
            self._trace(arg, objects, out)
                
            # argument cannot be enum type
            RDDLObjectsTracer._check_not_enum(
                arg, expr, out, f'Argument {i + 1} of {expr.etype[1]}') 
        
        out._append(expr, objects, None, None)
        
