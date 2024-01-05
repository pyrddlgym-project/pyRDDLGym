import numpy as np
from typing import Dict, List, Set, Tuple, Union

from pyRDDLGym.Core.ErrorHandling.RDDLException import print_stack_trace_root as PST
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidNumberOfArgumentsError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidObjectError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLNotImplementedError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLRepeatedVariableError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLTypeError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLUndefinedVariableError

from pyRDDLGym.Core.Compiler.RDDLLevelAnalysis import RDDLLevelAnalysis
from pyRDDLGym.Core.Compiler.RDDLModel import PlanningModel
from pyRDDLGym.Core.Debug.Logger import Logger
from pyRDDLGym.Core.Parser.expr import Expression


class RDDLTracedObjects:
    '''A generic container for storing traced information for a RDDL file.'''
    
    def __init__(self) -> None:
        self._current_id = 0
        self._current_root = ''
        self._cached_objects_in_scope = []
        self._cached_object_type = []
        self._cached_is_fluent = []
        self._cached_is_fluent_cpf = {}
        self._cached_sim_info = []
        self._expr_from_id = {}
        
    def _append(self, expr, objects, obj_type, is_fluent, info) -> None:
        expr.id = self._current_id
        self._current_id += 1
                
        self._cached_objects_in_scope.append(objects)
        self._cached_object_type.append(obj_type)
        self._cached_is_fluent.append(is_fluent)
        self._cached_sim_info.append(info)
        self._expr_from_id[expr.id] = expr
        
    def cached_objects_in_scope(self, expr: Expression):
        '''Returns the free variables/parameters in the scope of expression.'''
        return self._cached_objects_in_scope[expr.id]
    
    def cached_object_type(self, expr: Expression) -> Union[str, None]:
        '''Returns the returned object type of expression or None if not an object.'''
        return self._cached_object_type[expr.id]
    
    def cached_is_fluent(self, expr: Expression) -> bool:
        '''Returns whether the expression is fluent or non-fluent.'''
        return self._cached_is_fluent[expr.id]
    
    def cached_is_fluent_cpf(self, name: str) -> bool:
        '''Returns whether the CPF given by name is a fluent expression.'''
        return self._cached_is_fluent_cpf[name]
    
    def cached_sim_info(self, expr: Expression) -> object:
        '''Returns compiled info that is specific to the expression.'''
        return self._cached_sim_info[expr.id]
    
    def lookup(self, identifier: int) -> Expression:
        '''Returns the expression with given identifier, or None if does not 
        exist.'''
        return self._expr_from_id.get(identifier, None)
    

def py_enum(**enums):
    return type('Enum', (), enums)

    
class RDDLObjectsTracer:
    '''Performs static/compile-time tracing of a RDDL AST representation and
    annotates nodes with info about objects that appear inside expressions.'''
    
    # for multivariate sampling: how many dimensions sample has
    REQUIRED_DIST_PVARS = {
        'MultivariateNormal': 1,
        'MultivariateStudent': 1,
        'Dirichlet': 1,
        'Multinomial': 1
    }
    
    # for multivariate sampling: how many _ identifiers each argument has
    REQUIRED_DIST_UNDERSCORES = {
        'MultivariateNormal': (1, 2),  # mean, covariance
        'MultivariateStudent': (1, 2, 0),  # mean, covariance, df
        'Dirichlet': (1,),  # weights
        'Multinomial': (0, 1)  # trials, probabilities
    }
    
    # operation codes on pvariable tensors
    NUMPY_OP_CODE = py_enum(
        NESTED_SLICE=-1,  # nested pvariables detected: slice required
        EINSUM=0,  # duplicated variables like fluent(?x, ?x) - reduction required
        TRANSPOSE=1,  # evaluation order differs from outerscope -- reorder required
        NOOP=2  # a scalar pvariable for example does not require any operation
    )
    
    def __init__(self, rddl: PlanningModel, logger: Logger=None,
                 cpf_levels: Dict[int, Set[str]]=None) -> None:
        '''Creates a new objects tracer object for the given RDDL domain.
        
        :param rddl: the RDDL domain to trace
        :param logger: to log compilation information during tracing to file
        :param cpf_levels: pre-computed CPF levels (will be computed internally
        if None)
        '''
        self.rddl = rddl
        self.logger = logger
        self.cpf_levels = cpf_levels
            
    @staticmethod
    def _check_not_object(arg, expr, out, msg):
        obj_type = out.cached_object_type(arg)
        if obj_type is not None:
            raise RDDLTypeError(
                f'{msg} can not be an object of type <{obj_type}>.\n' + 
                PST(expr, out._current_root)) 
            
    def trace(self) -> RDDLTracedObjects:
        '''Traces all expressions in the current RDDL file and annotates
        AST nodes with object information. 
        
        Important notes: 
        (1) tracing annotates intermediate expressions in the AST with unique 
        identifiers; if the shape or structure of the AST changes, it is required
        to call trace() again to re-populate these IDs
        (2) the identifiers are cached internally in each Expression node: care
        must therefore be taken if object references in two different AST point
        to the same underlying Expression object, since they could require 
        different IDs; this generally shouldn't happen in typical use cases, but
        any manual AST construction can simply ensure that all expressions to be
        shared across different AST instances are (deep) copied.'''   
        rddl = self.rddl 
        out = RDDLTracedObjects()   
        
        # compute trace order for CPFs
        levels = self.cpf_levels
        if levels is None:
            levels = RDDLLevelAnalysis(rddl).compute_levels()
            
        # trace CPFs
        for cpfs in levels.values():
            for cpf in cpfs:
                objects, expr = rddl.cpfs[cpf]
                
                # check that the parameters are unique
                pvars = [pvar for (pvar, _) in objects]
                if len(set(pvars)) != len(pvars):
                    raise RDDLRepeatedVariableError(
                        f'Repeated parameter(s) {pvars} in definition of CPF <{cpf}>.')
                
                # trace the expression
                out._current_root = f'CPF {cpf}'
                self._trace(expr, objects, out)
                out._cached_is_fluent_cpf[cpf] = out.cached_is_fluent(expr)

                # for domain-object valued check that type matches expression output
                cpf_range = rddl.variable_ranges[cpf]
                expr_range = out.cached_object_type(expr)
                if (cpf_range in rddl.enums and expr_range != cpf_range) \
                or (cpf_range not in rddl.objects and expr_range is not None):
                    raise RDDLTypeError(
                        f'CPF <{cpf}> expression expects type <{cpf_range}>, '
                        f'got expression of type <{expr_range}>.')

        # trace reward, check not object value
        out._current_root = 'reward'
        self._trace(rddl.reward, [], out)
        RDDLObjectsTracer._check_not_object(rddl.reward, rddl.reward, out, 'reward')
        
        # trace constraints, check not object
        for (i, expr) in enumerate(rddl.invariants):
            out._current_root = f'Invariant {i + 1}'
            self._trace(expr, [], out)
            RDDLObjectsTracer._check_not_object(expr, expr, out, out._current_root)
        for (i, expr) in enumerate(rddl.preconditions):
            out._current_root = f'Precondition {i + 1}'
            self._trace(expr, [], out)
            RDDLObjectsTracer._check_not_object(expr, expr, out, out._current_root)
        for (i, expr) in enumerate(rddl.terminals):
            out._current_root = f'Termination {i + 1}'
            self._trace(expr, [], out)
            RDDLObjectsTracer._check_not_object(expr, expr, out, out._current_root)
        
        # log the fluent types
        if self.logger is not None:
            message = f'[info] computed whether each CPF expression is fluent:\n'
            for cpfs in levels.values():
                for cpf in cpfs:
                    is_fluent = out._cached_is_fluent_cpf[cpf]
                    message += f'\t{cpf}: {is_fluent}\n'
            self.logger.log(message)
            
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
        elif etype == 'randomvector':
            self._trace_random_vector(expr, objects, out)
        elif etype == 'matrix':
            self._trace_matrix(expr, objects, out)
        else:
            raise RDDLNotImplementedError(
                f'Internal error: expression type {etype} is not supported.\n' + 
                PST(expr, out._current_root))
    
    # ===========================================================================
    # leaves
    # ===========================================================================
    
    def _trace_constant(self, expr, objects, out):
        if objects:
            shape = self.rddl.object_counts((ptype for (_, ptype) in objects))
            cached_value = np.full(shape=shape, fill_value=expr.args)
        else:
            cached_value = expr.args
            
        out._append(expr, objects, None, False, cached_value)
        
    def _trace_pvar(self, expr, objects, out):
        var, pvars = expr.args   
        rddl = self.rddl
        
        # free variable (e.g., ?x) treated as array
        # first element True indicates value is to be returned directly by sim
        if rddl.is_free_variable(var):
            obj_to_index = {pobj: i for (i, (pobj, _)) in enumerate(objects)}
            index_of_var = obj_to_index.get(var, None)
                         
            # check var is valid in scope "objects"
            if index_of_var is None:
                raise RDDLInvalidObjectError(
                    f'Free variable <{var}> is not defined in outer scope, '
                    f'must be one of {set(obj_to_index.keys())}. '
                    f'Please check expression for <{out._current_root}>.')
            
            # create an array whose shape matches objects
            # along axis equal to index_of_var_in_objects values are (0, 1, ...)
            ptypes = [ptype for (_, ptype) in objects]
            shape = rddl.object_counts(ptypes)            
            cached_value = np.arange(shape[index_of_var])
            cached_value = cached_value[(...,) + (np.newaxis,) * len(ptypes[1:])]
            cached_value = np.swapaxes(cached_value, axis1=0, axis2=index_of_var)
            cached_value = np.broadcast_to(cached_value, shape=shape)
            cached_value = (True, cached_value)
            
            prange = ptypes[index_of_var]
            obj_type = prange if (prange in rddl.enums or prange in rddl.objects) else None
            out._append(expr, objects, obj_type, False, cached_value)
        
        # object can only be defined in domain - map to canonical index
        # first element True indicates value is to be returned directly by sim
        elif not pvars and rddl.is_object(
            var, msg='\n' + PST(expr, out._current_root)): 
            
            # check var is a domain object
            literal = rddl.object_name(var)            
            enum_type = rddl.objects_rev[literal]  
            if enum_type not in rddl.enums:
                raise RDDLInvalidObjectError(
                    f'Object <{var}> must be of a domain-defined object type, '
                    f'got type <{enum_type}>. '
                    f'Please check expression for <{out._current_root}>.')
            
            # map to canonical index - for pvariable fill an array with it
            const = rddl.index_of_object[literal]
            if objects:
                shape = rddl.object_counts((ptype for (_, ptype) in objects))
                cached_value = np.full(shape=shape, fill_value=const)
            else:
                cached_value = const            
            cached_value = (True, cached_value)
            
            out._append(expr, objects, enum_type, False, cached_value)
        
        # if the pvar has free variables (e.g., ?x)...
        else:
            
            # check if pvar is fluent
            primed_var = rddl.next_state.get(var, var)
            if primed_var in rddl.cpfs:
                is_fluent = out._cached_is_fluent_cpf.get(primed_var, None)
                if is_fluent is None:
                    is_fluent = True
                    if self.logger is not None:
                        message = (
                            f'[warning] cannot establish whether CPF is fluent:'
                            f'\n\taddress of expression ={str(expr)}'
                            f'\n\tCPF to check          ={primed_var}\n'
                        )
                        self.logger.log(message)
            else:
                is_fluent = rddl.variable_types[var] != 'non-fluent'

            # recursively trace nested pvariables
            if pvars is not None: 
                for arg in pvars:
                    if isinstance(arg, Expression):
                        self._trace(arg, objects, out)
                        is_fluent = is_fluent or out.cached_is_fluent(arg)
            
            # find a way to map value tensor of expr to match objects
            cached_sim_info = (False, self._map(expr, objects, out))    
               
            prange = rddl.variable_ranges.get(var, None)
            obj_type = prange if (prange in rddl.enums or prange in rddl.objects) else None
            out._append(expr, objects, obj_type, is_fluent, cached_sim_info)
        
    def _map(self, expr: Expression,
             objects: Union[List[Tuple[str, str]], None],
             out: RDDLTracedObjects) -> Tuple[object, ...]:
        '''Returns information needed to reshape and transform value tensor for a
        parameterized variable to match desired output signature.
        
        :param expr: outputs of expression value tensor to transform
        :param objects: a list of tuples (objecti, typei) representing the
            desired signature of the output value tensor
        :param cache where child of expr properties have been stored
        '''
        var, args = expr.args
        rddl = self.rddl
        
        # check that the number of input objects match fluent type definition
        if args is None:
            args = []
        args_types = rddl.param_types.get(var, [])
        if len(args) != len(args_types):
            raise RDDLInvalidNumberOfArgumentsError(
                f'Variable <{var}> requires {len(args_types)} argument(s), '
                f'got {len(args)}.\n' + PST(expr, out._current_root))
        
        # for vector distributions, occurrences of _ are mapped to unique new 
        # objects ?_1, ?_2, ... because we have to take care not to einsum them;
        # they will correspond to the last axes of the resulting value array
        # e.g., values(?p, ?q, ... ?_1, ?_2, ...);
        # this way ?p, ?q... can be interpreted as "batch" dimensions and 
        # ?_1, ?_2, ... as the "image" dimensions in an equivalent ML problem;
        # this is the convention of most numpy and JAX batched subroutines anyway
        new_objects, underscore_names = [], {}
        for (i, arg) in enumerate(args):
            if arg == '_':
                new_pvar = f'?_{1 + len(new_objects)}'
                new_objects.append((new_pvar, args_types[i]))
                underscore_names[i] = new_pvar
        objects = objects + new_objects
        
        # test for nested expressions
        is_arg_expr = [isinstance(arg, Expression) for arg in args]
        nested = np.any(is_arg_expr)
                
        # literals are converted to canonical indices in their object list
        # and used to extract from the value tensor at the corresponding axis
        # 1. if there are nested variables, then they are left as None slices
        #    since their values are only known at run time
        # 2. if there are free variables ?x among nested variables, then they
        #    are reshaped to match objects
        object_shape = rddl.object_counts((ptype for (_, ptype) in objects))
        object_index = {obj: i for (i, (obj, _)) in enumerate(objects)}
        slices = [slice(None)] * len(args)
        do_slice = False
        permuted = []
        for (i, arg) in enumerate(args):
            
            # is a nested fluent (e.g., fluent(another-fluent(?x)) )
            if is_arg_expr[i]:
                
                # check that type of the inner fluent matches what var expects
                obj_type = out.cached_object_type(arg)
                if args_types[i] != obj_type: 
                    if obj_type is None:
                        obj_type = 'real, int or bool'
                    raise RDDLTypeError(
                        f'Argument {i + 1} of variable <{var}> expects object '
                        f'of type <{args_types[i]}>, got nested expression '
                        f'of type <{obj_type}>.\n' + PST(expr, out._current_root))
                
                # leave slice blank since it's filled at runtime
                slices[i] = None
                
            # is an object literal (e.g., @x)
            elif rddl.is_object(arg, msg='\n' + PST(expr, out._current_root)):
                
                # check that the type of the object is correct
                literal = rddl.object_name(arg)
                enum_type = rddl.objects_rev[literal]
                if args_types[i] != enum_type: 
                    raise RDDLTypeError(
                        f'Argument {i + 1} of variable <{var}> expects object '
                        f'of type <{args_types[i]}>, got <{arg}> '
                        f'of type <{enum_type}>.\n' + PST(expr, out._current_root))
                
                # extract value at current dimension at object's canonical index
                slices[i] = rddl.index_of_object[literal]
                do_slice = True
            
            # is a free object (e.g., ?x)
            else:
                
                # a sampling dimension is mapped to its unique quantifier ?_i
                arg = underscore_names.get(i, arg)
                
                # make sure argument is well defined
                index_of_arg = object_index.get(arg, None)
                if index_of_arg is None:
                    raise RDDLInvalidObjectError(
                        f'Undefined argument <{arg}> at position {i + 1} '
                        f'of variable <{var}>.\n' + PST(expr, out._current_root))
                
                # make sure type of argument is correct
                _, ptype = objects[index_of_arg]
                if ptype != args_types[i]:
                    raise RDDLTypeError(
                        f'Argument {i + 1} of variable <{var}> expects object '
                        f'of type <{args_types[i]}>, got <{arg}> '
                        f'of type <{ptype}>.\n' + PST(expr, out._current_root))
                                      
                # if nesting is found, then free variables like ?x are implicitly 
                # converted to arrays with shape of objects
                # this way, the slice value array has shape that matches objects
                if nested:
                    indices = np.arange(len(rddl.objects[ptype]))
                    newshape = [1] * len(objects)
                    newshape[index_of_arg] = indices.size
                    indices = np.reshape(indices, newshape=newshape)
                    indices = np.broadcast_to(indices, shape=object_shape)
                    slices[i] = indices 
                
                # if no nesting, then we use einsum or transpose operations
                else:
                    permuted.append(index_of_arg)
                                                       
        slices = tuple(slices) if do_slice or nested else ()
        len_after_slice = len(permuted)
        
        # compute the mapping function as follows:
        # 0. first assume all literals are "sliced out" of the value tensor
        #    if there is nesting then a slice is sufficient and do nothing else
        # 1. append new axes to value tensor equal to # of missing variables
        # 2. broadcast new axes to the desired shape (# of objects of each type)
        # 3. rearrange the axes as needed to match the desired variables in order
        #    3a. in most cases, it suffices to use np.transform (cheaper)
        #    3b. in cases where we have a more complex contraction like 
        #        fluent(?x) = matrix(?x, ?x), we will use np.einsum
        if nested:
            new_axis = None
            new_shape = None
            op_args = None
            op_code = RDDLObjectsTracer.NUMPY_OP_CODE.NESTED_SLICE
        else:
            
            # update permutation based on objects not in args
            covered = set(permuted)
            permuted += [i for i in range(len(objects)) if i not in covered]
            
            # store the arguments for each operation: 
            # 0 means einsum, 1 means transpose, and others (-1, 2) are no-op
            new_axis = tuple(range(len_after_slice, len(permuted)))  
            new_shape = tuple(object_shape[i] for i in permuted)
            objects_range = list(range(len(objects)))        
            if len(covered) != len_after_slice:
                op_args = (permuted, objects_range)
                op_code = RDDLObjectsTracer.NUMPY_OP_CODE.EINSUM
            elif permuted != objects_range:
                op_args = tuple(np.argsort(permuted))  # inverse permutation
                op_code = RDDLObjectsTracer.NUMPY_OP_CODE.TRANSPOSE
            else:
                op_args = None
                op_code = RDDLObjectsTracer.NUMPY_OP_CODE.NOOP
        
        # log information about the new transformation
        if self.logger is not None:
            message = (
                f'[info] computing info for pvariable tensor transformation:' 
                f'\n\taddress of expression   ={super(Expression, expr).__str__()}'
                f'\n\tvariable                ={var}'
                f'\n\tvariable evaluated at   ={list(zip(args, args_types))}'
                f'\n\tfree object(s) in scope ={objects}'
                f'\n\tslice                   ={slices}'
                f'\n\tnew axes to append      ={new_axis}'
                f'\n\tbroadcast shape         ={new_shape}'
                f'\n\ttransform operation     ={op_code} with argument(s) {op_args}\n'
            )
            self.logger.log(message)
            
        return (slices, new_axis, new_shape, op_code, op_args)
    
    # ===========================================================================
    # compound expressions
    # ===========================================================================
    
    def _trace_arithmetic(self, expr, objects, out): 
        _, op = expr.etype
        is_fluent = False
        for (i, arg) in enumerate(expr.args):
            self._trace(arg, objects, out)
            is_fluent = is_fluent or out.cached_is_fluent(arg)
        
            # cannot be object type
            RDDLObjectsTracer._check_not_object(
                arg, expr, out, f'Argument {i + 1} of operator {op}')
              
        out._append(expr, objects, None, is_fluent, None)
        
    def _trace_relational(self, expr, objects, out):
        _, op = expr.etype
        args = expr.args
        is_fluent = False
        for arg in args:
            self._trace(arg, objects, out)
            is_fluent = is_fluent or out.cached_is_fluent(arg)         
        
        # can not mix different object types or primitive and object types
        obj_types = set(map(out.cached_object_type, args))
        if len(obj_types) != 1:
            raise RDDLTypeError(
                f'Relational operator {op} can not compare arguments '
                f'of different object types {obj_types}.\n' + 
                PST(expr, out._current_root))
        
        # can not use operator besides == and ~= to compare object types
        obj_type, = obj_types
        if obj_type is not None and op != '==' and op != '~=':
            raise RDDLTypeError(
                f'Relational operator {op} is not valid '
                f'for comparing objects of type {obj_type}.\n' + 
                PST(expr, out._current_root))
        
        out._append(expr, objects, None, is_fluent, None)
    
    def _trace_logical(self, expr, objects, out):
        is_fluent = False
        for (i, arg) in enumerate(expr.args):
            self._trace(arg, objects, out)
            is_fluent = is_fluent or out.cached_is_fluent(arg)
            
            # argument cannot be object type
            RDDLObjectsTracer._check_not_object(
                arg, expr, out, f'Argument {i + 1} of operator {expr.etype[1]}')
        
        out._append(expr, objects, None, is_fluent, None)
    
    def _trace_func(self, expr, objects, out):
        is_fluent = False
        for (i, arg) in enumerate(expr.args):
            self._trace(arg, objects, out)
            is_fluent = is_fluent or out.cached_is_fluent(arg)
            
            # argument cannot be object type
            RDDLObjectsTracer._check_not_object(
                arg, expr, out, f'Argument {i + 1} of function {expr.etype[1]}') 
        
        out._append(expr, objects, None, is_fluent, None)
            
    # ===========================================================================
    # aggregation
    # ===========================================================================
    
    def _check_iteration_variables(self, objects, iter_objects, expr, out):
        
        # check for undefined types
        rddl = self.rddl
        bad_types = {ptype 
                     for (_, ptype) in iter_objects 
                     if ptype not in rddl.objects}
        if bad_types:
            raise RDDLTypeError(
                f'Type(s) {bad_types} are not defined, '
                f'must be one of {set(rddl.objects.keys())}.\n' + 
                PST(expr, out._current_root))
        
        # check for valid type arguments
        scope_vars = {var for (var, _) in objects}
        seen_vars = set()
        for (var, _) in iter_objects:
            
            # check that there is no duplicated iteration variable
            if var in seen_vars:
                raise RDDLRepeatedVariableError(
                    f'Iteration variable <{var}> is repeated.\n' + 
                    PST(expr, out._current_root))             
            seen_vars.add(var)
             
            # check if iteration variable is same as one defined in outer scope
            # since there is ambiguity to which is referred I raise an error
            if var in scope_vars:
                raise RDDLRepeatedVariableError(
                    f'Iteration variable <{var}> is already defined '
                    f'in outer scope.\n' + PST(expr, out._current_root))
        
    def _trace_aggregation(self, expr, objects, out):
        _, op = expr.etype
        * pvars, arg = expr.args
        
        # cache and read reduced axes tensor info for the aggregation
        # axes of new free variables in aggregation are appended to value array
        iter_objects = [ptype for (_, ptype) in pvars]
        self._check_iteration_variables(objects, iter_objects, expr, out)
        new_objects = objects + iter_objects
        reduced_axes = tuple(range(len(objects), len(new_objects)))   
        cached_sim_info = (new_objects, reduced_axes)
        
        # argmax/argmin require exactly one parameter argument
        enum_type = None
        if op == 'argmin' or op == 'argmax':
            if len(pvars) != 1:
                raise RDDLInvalidNumberOfArgumentsError(
                    f'Aggregation <{op}> requires one iteration variable, '
                    f'got {len(pvars)}.\n' + PST(expr, out._current_root))
            cached_sim_info = (new_objects, reduced_axes[0])
            (_, (_, enum_type)), = pvars
        
        # trace the aggregated expression with the new objects
        self._trace(arg, new_objects, out)
        is_fluent = out.cached_is_fluent(arg)
        
        # argument cannot be object type
        RDDLObjectsTracer._check_not_object(
            arg, expr, out, f'Argument of aggregation {op}')     
        
        out._append(expr, objects, enum_type, is_fluent, cached_sim_info)
        
        # log information about aggregation operation
        if self.logger is not None:
            message = (f'[info] computing object info for aggregation:'
                       f'\n\taggregation variables(s)      ={pvars}'
                       f'\n\tfree object(s) in outer scope ={objects}'
                       f'\n\tfree object(s) in inner scope ={new_objects}'
                       f'\n\taggregation operation         ={op}'
                       f'\n\taggregation axes              ={reduced_axes}\n')
            self.logger.log(message)        
        
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
        pred, *branches = expr.args
        
        # trace predicate
        self._trace(pred, objects, out)
        RDDLObjectsTracer._check_not_object(
            pred, expr, out, f'Predicate of if statement')  
        
        # trace branches
        is_fluent = out.cached_is_fluent(pred)
        for arg in branches:
            self._trace(arg, objects, out)
            is_fluent = is_fluent or out.cached_is_fluent(arg)
        
        # can not mix different object types or primitive and object types
        obj_types = set(map(out.cached_object_type, branches))
        if len(obj_types) != 1:
            raise RDDLTypeError(
                f'Different branches in if statement cannot produce values '
                f'of different object types {obj_types}.\n' + 
                PST(expr, out._current_root))     
    
        obj_type, = obj_types
        out._append(expr, objects, obj_type, is_fluent, None)
        
    def _trace_switch(self, expr, objects, out):
        rddl = self.rddl
        pred, *cases = expr.args
        
        # predicate must be a pvar
        if not pred.is_pvariable_expression():
            raise RDDLNotImplementedError(
                f'Switch predicate is not a pvariable.\n' + 
                PST(expr, out._current_root))
            
        # type in pvariables scope must be a domain object
        var, _ = pred.args
        enum_type = rddl.variable_ranges.get(var, None)
        if enum_type not in rddl.enums:
            raise RDDLTypeError(
                f'Type <{enum_type}> of switch predicate <{var}> is not a '
                f'domain-defined object, must be one of {rddl.enums}.\n' + 
                PST(expr, out._current_root))
            
        # default statement becomes ("default", expr)
        case_dict = dict(
            (case_value if case_type == 'case' else (case_type, case_value)) 
            for (case_type, case_value) in cases
        )    
        
        # strip @ from any cases
        case_dict = {rddl.object_name(key): value 
                     for (key, value) in case_dict.items()}
        
        # check for duplicated cases
        if len(case_dict) != len(cases):
            raise RDDLInvalidNumberOfArgumentsError(
                f'Duplicated literal or default case(s).\n' + 
                PST(expr, out._current_root))
        
        # order cases by canonical ordering of objects
        cached_sim_info = self._order_cases(enum_type, case_dict, expr, out)
        
        # trace predicate and cases
        self._trace(pred, objects, out)
        is_fluent = out.cached_is_fluent(pred)
        for arg in case_dict.values():
            self._trace(arg, objects, out)
            is_fluent = is_fluent or out.cached_is_fluent(arg)
        
        # can not mix different object types or primitive and object types
        obj_types = set(map(out.cached_object_type, case_dict.values()))
        if len(obj_types) != 1:
            raise RDDLTypeError(
                f'Case expressions in switch statement cannot produce values '
                f'of different object types {obj_types}\n' + 
                PST(expr, out._current_root))    
                                
        obj_type, = obj_types
        out._append(expr, objects, obj_type, is_fluent, cached_sim_info)
        
    def _order_cases(self, enum_type, case_dict, expr, out): 
        rddl = self.rddl
        enum_values = rddl.objects[enum_type]
        
        # check that all literals belong to enum_type
        for _case in case_dict:
            if _case != 'default' and rddl.objects_rev.get(_case, None) != enum_type:
                raise RDDLInvalidObjectError(
                    f'Object <{_case}> does not belong to type <{enum_type}>, '
                    f'must be one of {set(enum_values)}.\n' + 
                    PST(expr, out._current_root))
        
        # store expressions in order of canonical literal index
        expressions = [None] * len(enum_values)
        for obj in enum_values:
            arg = case_dict.get(obj, None)
            if arg is not None: 
                index = rddl.index_of_object[obj]
                expressions[index] = arg
        
        # if default statement is missing, cases must be comprehensive
        default_expr = case_dict.get('default', None)
        if default_expr is None:
            for (i, arg) in enumerate(expressions):
                if arg is None:
                    raise RDDLUndefinedVariableError(
                        f'Object <{enum_values[i]}> of type <{enum_type}> '
                        f'is missing in case list.\n' + PST(expr, out._current_root))
        
        # log cases ordering
        if self.logger is not None:
            active_expr = [i for (i, e) in enumerate(expressions) if e is not None]
            message = (f'[info] computing case info for {expr.etype[1]}:'
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
        elif name == 'Discrete(p)' or name == 'UnnormDiscrete(p)':
            self._trace_discrete_pvar(expr, objects, out)
        else:
            self._trace_random_other(expr, objects, out)
                
    def _trace_discrete(self, expr, objects, out):
        rddl = self.rddl
        (_, enum_type), *cases = expr.args
            
        # object type must be a valid domain object type
        if enum_type not in rddl.enums:
            raise RDDLTypeError(
                f'Type <{enum_type}> in Discrete distribution is not a '
                f'domain-defined object, must be one of {rddl.enums}.\n' + 
                PST(expr, out._current_root))        
        case_dict = dict(case_tup for (_, case_tup) in cases) 
        
        # strip @ from any cases       
        case_dict = {rddl.object_name(key): value 
                     for (key, value) in case_dict.items()}
        
        # no duplicate cases are allowed
        if len(case_dict) != len(cases):
            raise RDDLInvalidNumberOfArgumentsError(
                f'Duplicated literal or default case(s).\n' + 
                PST(expr, out._current_root))
        
        # no default cases are allowed
        if 'default' in case_dict:
            raise RDDLNotImplementedError(
                f'Default case not allowed in Discrete distribution.\n' + 
                PST(expr, out._current_root))
            
        # order enum cases by canonical ordering of literals
        cached_sim_info, _ = self._order_cases(enum_type, case_dict, expr, out)
    
        # trace each case expression
        for (i, arg) in enumerate(case_dict.values()):
            self._trace(arg, objects, out)
            
            # argument cannot be object type
            RDDLObjectsTracer._check_not_object(
                arg, expr, out, f'Expression in case {i + 1} of Discrete') 
        
        out._append(expr, objects, enum_type, True, cached_sim_info)
    
    def _trace_discrete_pvar(self, expr, objects, out):
        * pvars, args = expr.args
        
        # check number of iteration variables and arguments
        if len(pvars) != 1:
            raise RDDLInvalidNumberOfArgumentsError(
                f'Discrete requires one iteration variable, got {len(pvars)}.\n' + 
                PST(expr, out._current_root))
        elif len(args) != 1:
            raise RDDLInvalidNumberOfArgumentsError(
                f'Discrete requires one argument, got {len(args)}.\n' + 
                PST(expr, out._current_root))
        
        # sampling variables are appended to scope free variables        
        iter_objects = [pvar for (_, pvar) in pvars]     
        self._check_iteration_variables(objects, iter_objects, expr, out)
        new_objects = objects + iter_objects
        
        # trace the arguments
        for (i, arg) in enumerate(args):
            self._trace(arg, new_objects, out)
                
            # argument cannot be object type
            RDDLObjectsTracer._check_not_object(
                arg, expr, out, f'Expression in case {i + 1} of Discrete') 

        (_, (_, enum_type)), = pvars
        out._append(expr, objects, enum_type, True, None)

    def _trace_random_other(self, expr, objects, out):
        for (i, arg) in enumerate(expr.args):
            self._trace(arg, objects, out)
                
            # argument cannot be object type
            RDDLObjectsTracer._check_not_object(
                arg, expr, out, f'Argument {i + 1} of {expr.etype[1]}') 
        
        out._append(expr, objects, None, True, None)
        
    # ===========================================================================
    # random vector
    # ===========================================================================
    
    def _trace_random_vector(self, expr, objects, out):
        _, op = expr.etype
        sample_pvars, args = expr.args
        
        # determine how many instances of _ should appear in each argument
        required_sample_pvars = RDDLObjectsTracer.REQUIRED_DIST_PVARS.get(op, None)
        required_underscores = RDDLObjectsTracer.REQUIRED_DIST_UNDERSCORES.get(op, None)
        if required_sample_pvars is None or required_underscores is None:
            raise RDDLNotImplementedError(
                f'Internal error: distribution {op} is not supported.\n' + 
                PST(expr, out._current_root))
        
        # check the number of sample parameters is correct
        if len(sample_pvars) != required_sample_pvars:
            raise RDDLInvalidNumberOfArgumentsError(
                f'Distribution {op} requires {required_sample_pvars} sampling '
                f'parameter(s), got {len(sample_pvars)}.\n' + 
                PST(expr, out._current_root))
        
        # check that all sample_pvars are defined in the outer scope
        scope_pvars = {pvar: i for (i, (pvar, _)) in enumerate(objects)}
        bad_pvars = {pvar for pvar in sample_pvars if pvar not in scope_pvars}
        if bad_pvars:
            raise RDDLInvalidObjectError(
                f'Sampling parameter(s) {bad_pvars} of {op} are not defined in '
                f'outer scope, must be one of {set(scope_pvars.keys())}.\n' + 
                PST(expr, out._current_root))
        
        # check duplicates in sample_pvars
        sample_pvar_set = set(sample_pvars)
        if len(sample_pvar_set) != len(sample_pvars):
            raise RDDLInvalidNumberOfArgumentsError(
                f'Sampling parameter(s) {sample_pvars} of {op} are duplicated.\n' + 
                PST(expr, out._current_root))
        
        # sample_pvars are excluded when tracing arguments (e.g., mean)
        # because they are only introduced through sampling which follows after 
        # evaluation of the arguments in a depth-first traversal
        batch_objects = [pvar for pvar in objects if pvar[0] not in sample_pvar_set]
         
        # trace all parameters
        enum_types = set()
        for (i, arg) in enumerate(args):
            
            # sample_pvars cannot be argument parameters
            name, pvars = arg.args
            if pvars is None:
                pvars = []
            bad_pvars = {pvar for pvar in pvars if pvar in sample_pvar_set}
            if bad_pvars:
                raise RDDLInvalidObjectError(
                    f'Parameter(s) {bad_pvars} of argument <{name}> at position '
                    f'{i + 1} of {op} can not be sampling parameter(s) '
                    f'{sample_pvar_set}.\n' + PST(expr, out._current_root))
            
            # check number of _ parameters is valid in argument
            underscores = [j for (j, pvar) in enumerate(pvars) if pvar == '_']            
            if len(underscores) != required_underscores[i]:
                raise RDDLInvalidNumberOfArgumentsError(
                    f'Argument <{name}> at position {i + 1} of {op} must contain '
                    f'{required_underscores[i]} sampling parameter(s) _, '
                    f'got {len(underscores)}.\n' + PST(expr, out._current_root))
            
            # trace argument
            self._trace(arg, batch_objects, out)
            
            # argument cannot be object type
            RDDLObjectsTracer._check_not_object(
                arg, expr, out, f'Argument {i + 1} of {op}') 
        
            # record types represented by _
            ptypes = self.rddl.param_types[name]
            enum_types.update({ptypes[j] for j in underscores})
            
        # types represented by _ must be the same
        if len(enum_types) != 1:
            raise RDDLTypeError(
                f'Sampling parameter(s) _ across all argument(s) of {op} must '
                f'correspond to a single type, got multiple types {enum_types}.\n' + 
                PST(expr, out._current_root))
        
        # objects of type _ must be compatible with sample dimension(s)
        enum_type, = enum_types
        sample_pvar_indices = tuple(scope_pvars[pvar] for pvar in sample_pvars)
        for index in sample_pvar_indices:
            pvar, ptype = objects[index]
            if ptype != enum_type:
                raise RDDLTypeError(
                    f'{op} sampling is performed over type <{enum_type}>, '
                    f'but destination variable <{pvar}> is of type <{ptype}>.\n' + 
                    PST(expr, out._current_root))
             
        out._append(expr, objects, None, True, sample_pvar_indices)

    # ===========================================================================
    # matrix algebra
    # ===========================================================================
    
    def _trace_matrix(self, expr, objects, out):
        _, op = expr.etype
        if op == 'det':
            self._trace_matrix_det(expr, objects, out)
        elif op == 'inverse':
            self._trace_matrix_inv(expr, objects, out, pseudo=False)
        elif op == 'pinverse':
            self._trace_matrix_inv(expr, objects, out, pseudo=True)
        elif op == 'cholesky':
            self._trace_matrix_inv(expr, objects, out, pseudo=False)
        else:
            raise RDDLNotImplementedError(
                f'Internal error: matrix operation {op} is not supported.\n' + 
                PST(expr, out._current_root))
    
    def _trace_matrix_det(self, expr, objects, out):
        * pvars, arg = expr.args
        
        # validate types according to aggregation rule
        iter_objects = [ptype for (_, ptype) in pvars]
        self._check_iteration_variables(objects, iter_objects, expr, out)
        
        # check that matrix is square
        pvar1, pvar2 = iter_objects
        _, ptype1 = pvar1
        _, ptype2 = pvar2
        n1, n2 = self.rddl.object_counts((ptype1, ptype2))      
        if n1 != n2:
            raise RDDLInvalidObjectError(
                f'Matrix in det operation must be square, '
                f'got {n1} objects of type <{ptype1}> '
                f'and {n2} objects of type <{ptype2}>.\n' + 
                PST(expr, out._current_root))
        
        # axes of new free variables in aggregation are appended to value array
        new_objects = objects + iter_objects
        reduced_axes = tuple(range(len(objects), len(new_objects)))   
        cached_sim_info = (new_objects, reduced_axes)
        
        # trace the matrix with the new objects
        self._trace(arg, new_objects, out)
        is_fluent = out.cached_is_fluent(arg)
        RDDLObjectsTracer._check_not_object(
            arg, expr, out, f'Argument of matrix det')     
        
        out._append(expr, objects, None, is_fluent, cached_sim_info)
        
        # log information about matrix operation
        if self.logger is not None:
            message = (f'[info] computing object info for matrix operation:'
                       f'\n\tmatrix operation              =det'
                       f'\n\tdimension variables(s)        ={pvars}'
                       f'\n\tfree object(s) in outer scope ={objects}'
                       f'\n\tfree object(s) in inner scope ={new_objects}'
                       f'\n\treduction axes                ={reduced_axes}\n')
            self.logger.log(message)        
        
    def _trace_matrix_inv(self, expr, objects, out, pseudo):
        _, op = expr.etype
        pvars, arg = expr.args
        
        # check that all pvars are defined in the outer scope        
        scope_pvars = {pvar: i for (i, (pvar, _)) in enumerate(objects)}
        bad_pvars = {pvar for pvar in pvars if pvar not in scope_pvars}
        if bad_pvars:
            raise RDDLInvalidObjectError(
                f'Row or column parameter(s) {bad_pvars} of {op} are not defined '
                f'in outer scope, must be one of {set(scope_pvars.keys())}.\n' + 
                PST(expr, out._current_root))
        
        # check duplicates in pvars
        pvar1, pvar2 = pvars
        if pvar1 == pvar2:
            raise RDDLInvalidNumberOfArgumentsError(
                f'Row and column parameters of {op} must differ, '
                f'got <{pvar1}> and <{pvar2}>.\n' + PST(expr, out._current_root))
            
        # for regular inverse, check that matrix is square
        index_pvar1 = scope_pvars[pvar1]
        index_pvar2 = scope_pvars[pvar2]
        _, ptype1 = objects[index_pvar1]
        _, ptype2 = objects[index_pvar2]
        if not pseudo:
            n1, n2 = self.rddl.object_counts((ptype1, ptype2))        
            if n1 != n2:
                raise RDDLInvalidObjectError(
                    f'Matrix in {op} operation must be square, '
                    f'got {n1} objects of type <{ptype1}> '
                    f'and {n2} objects of type <{ptype2}>.\n' + 
                    PST(expr, out._current_root))
        
        # move the matrix parameters to end of objects
        batch_objects = [pvar for pvar in objects if pvar[0] not in scope_pvars]
        new_objects = batch_objects + [(pvar1, ptype1), (pvar2, ptype2)]
        
        # trace the matrix expression
        self._trace(arg, new_objects, out)
        is_fluent = out.cached_is_fluent(arg)
        RDDLObjectsTracer._check_not_object(
            arg, expr, out, f'Argument of matrix {op}')     
        
        # save the location of the moved indices in objects
        pvar_indices = (index_pvar1, index_pvar2)
        out._append(expr, objects, None, is_fluent, pvar_indices)
        
        # log information about matrix operation
        if self.logger is not None:
            message = (f'[info] computing object info for matrix operation:'
                       f'\n\tmatrix operation              ={op}'
                       f'\n\tdimension variables(s)        ={pvars}'
                       f'\n\tfree object(s) in outer scope ={objects}'
                       f'\n\tfree object(s) in inner scope ={new_objects}'
                       f'\n\tindices in outer scope        ={pvar_indices}\n')
            self.logger.log(message)        
