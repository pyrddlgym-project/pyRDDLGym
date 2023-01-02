import datetime
import numpy as np
from typing import Callable, Iterable, List, Set, Tuple, Union

from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidNumberOfArgumentsError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidObjectError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLNotImplementedError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLTypeError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLUndefinedVariableError

from pyRDDLGym.Core.Compiler.RDDLDecompiler import RDDLDecompiler
from pyRDDLGym.Core.Compiler.RDDLModel import RDDLModel
from pyRDDLGym.Core.Parser.expr import Expression, Value


class RDDLObjectsTracer:
    '''Performs static/compile-time tracing of a RDDL AST representation and
    annotates nodes with useful information about objects and enums that appear
    inside expressions. Also compiles initial value tensors for all pvariables.
    '''
    
    INT = np.int64
    REAL = np.float64
        
    NUMPY_TYPES = {
        'int': INT,
        'real': REAL,
        'bool': bool
    }
    
    DEFAULT_VALUES = {
        'int': 0,
        'real': 0.0,
        'bool': False
    }
    
    VALID_SYMBOLS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    def __init__(self, rddl: RDDLModel, tensorlib=np, debug: bool=False):
        self.rddl = rddl
        self.tensorlib = tensorlib
        self.debug = debug   
             
        self.filename = f'{rddl._AST.domain.name}_{rddl._AST.instance.name}.log'
        
    # ===========================================================================
    # logging and error handling
    # ===========================================================================
    
    def _clear_log(self):
        if self.debug:
            fp = open(self.filename, 'w')
            fp.write('')
            fp.close()
        
    def _append_log(self, msg: str) -> None:
        if self.debug:
            fp = open(self.filename, 'a')
            timestamp = str(datetime.datetime.now())
            fp.write(f'{timestamp}: {msg}\n')
            fp.close()
    
    @staticmethod
    def _check_not_enum(arg, expr, msg):
        if arg.enum_type is not None:
            raise RDDLTypeError(
                f'{msg} can not be an enum type <{arg.enum_type}>.\n' + 
                RDDLObjectsTracer._print_stack_trace(expr)) 
                
    @staticmethod
    def _print_stack_trace(expr):
        if isinstance(expr, Expression):
            trace = RDDLDecompiler().decompile_expr(expr)
        else:
            trace = str(expr)
        return f'>> {trace}'
    
    # ===========================================================================
    # compilation of RDDL domain objects and initial value tensors
    # ===========================================================================
    
    def _compile_objects(self): 
        rddl = self.rddl       
        self.grounded = {var: list(rddl.grounded_names(var, types))
                         for (var, types) in rddl.param_types.items()}    
            
        self.index_of_object = {obj: i 
                                for objects in rddl.objects.values() 
                                    for (i, obj) in enumerate(objects)}
        
    def _compile_init_values(self):
        rddl = self.rddl
                
        # initial values consists of non-fluents, state and action fluents
        init_values = {}
        init_values.update(rddl.nonfluents)
        init_values.update(rddl.init_state)
        init_values.update(rddl.actions)
        init_values = {name: value 
                       for (name, value) in init_values.items() 
                       if value is not None}

        # enum literals are converted to integers
        for (name, value) in init_values.items():
            var = rddl.parse(name)[0]
            prange = rddl.variable_ranges[var]
            if prange in rddl.enum_types:
                if rddl.objects_rev.get(value, None) != prange:
                    raise RDDLInvalidObjectError(
                        f'Literal <{value}> does not belong to enum <{prange}>, '
                        f'must be one of {set(rddl.objects[prange])}.')
                init_values[name] = self.index_of_object[value]
        
        init_arrays = {}
        for var in rddl.variable_ranges.keys():
            
            # try to extract a default value if missing for a primitive type
            # enum types are treated as integers
            prange = rddl.variable_ranges[var]
            default = RDDLObjectsTracer.DEFAULT_VALUES.get(prange, None)
            if default is None:
                if prange in rddl.enum_types:
                    prange = 'int'
                    default = 0
                else:
                    raise RDDLTypeError(
                        f'Type <{prange}> of variable <{var}> is not valid, '
                        f'must be an enum type in {rddl.enum_types} '
                        f'or one of {set(RDDLObjectsTracer.DEFAULT_VALUES.keys())}.')   
            
            # create a tensor filled with init_values
            types = rddl.param_types[var]
            dtype = RDDLObjectsTracer.NUMPY_TYPES[prange]
            if rddl.is_grounded or not types:
                for name in rddl.grounded_names(var, types):
                    init_arrays[name] = dtype(init_values.get(name, default)) 
            else:
                grounded_values = [init_values.get(name, default) 
                                   for name in rddl.grounded_names(var, types)]
                array = np.asarray(grounded_values, dtype=dtype)
                array = np.reshape(array, newshape=self.shape(types), order='C')
                init_arrays[var] = array     
        self.init_values = init_arrays
        
        # log shapes of initial values
        tensor_info = '\n\t'.join((
            f'{k}{[] if rddl.is_grounded else rddl.param_types[k]}, '
            f'shape={v.shape if type(v) is np.ndarray else ()}, '
            f'dtype={v.dtype if type(v) is np.ndarray else type(v).__name__}'
        ) for k, v in init_arrays.items())
        self._append_log(f'initializing pvariable tensors:' 
                            f'\n\t{tensor_info}\n')
        
    # ===========================================================================
    # basic utility functions (e.g., shape, coordinate info)
    # ===========================================================================
    
    def coordinates(self, objects: Iterable[str], msg: str='') -> Tuple[int, ...]:
        '''Converts a list of objects into their coordinate representation.
        
        :param objects: object instances corresponding to valid types defined
        in the RDDL domain
        :param msg: an error message to print in case the conversion fails.
        '''
        index_of_obj = self.index_of_object
        try:
            return tuple(index_of_obj[obj] for obj in objects)
        except:
            for obj in objects:
                if obj not in index_of_obj:
                    raise RDDLInvalidObjectError(
                        f'Object <{obj}> is not valid, '
                        f'must be one of {set(index_of_obj.keys())}.'
                        f'\n{msg}')
    
    def shape(self, types: Iterable[str], msg: str='') -> Tuple[int, ...]:
        '''Given a list of RDDL types, returns the shape of a tensor
        that would hold all values of a pvariable with those type arguments
        
        :param types: a list of RDDL types
        :param msg: an error message to print in case the calculation fails
        '''
        objects = self.rddl.objects
        try:
            return tuple(len(objects[ptype]) for ptype in types)
        except:
            for ptype in types:
                if ptype not in objects:
                    raise RDDLInvalidObjectError(
                        f'Type <{ptype}> is not valid, '
                        f'must be one of {set(objects.keys())}.'
                        f'\n{msg}')
    
    def expand(self, var: str, values: np.ndarray) -> Iterable[Tuple[str, Value]]:
        '''Produces a grounded representation of the pvariable var from its 
        tensor representation. The output is a dict whose keys are grounded
        representations of the var, and values are read from the tensor.
        
        :param var: the pvariable
        :param values: the tensor whose values correspond to those of var(?...)        
        '''
        keys = self.grounded[var]
        values = np.ravel(values, order='C')
        if len(keys) != values.size:
            raise RDDLInvalidNumberOfArgumentsError(
                f'Variable <{var}> requires {len(keys)} values, got {values.size}.')
        return zip(keys, values)

    # ===========================================================================
    # main tracing routines
    # ===========================================================================
    
    def trace(self):
        self._clear_log()        
        self._compile_objects()
        self._compile_init_values()  
        self._cached_transforms = {}
        
        rddl = self.rddl
        for objects, expr in rddl.cpfs.values():
            self._trace(expr, objects)
        self._trace(rddl.reward, [])
        for expr in rddl.invariants + rddl.preconditions + rddl.terminals:
            self._trace(expr, [])
        
    # ===========================================================================
    # start of tracing subroutines
    # ===========================================================================
        
    def _trace(self, expr, objects):
        etype, _ = expr.etype
        if etype == 'constant':
            return self._trace_constant(expr, objects)
        elif etype == 'pvar':
            return self._trace_pvar(expr, objects)
        elif etype == 'arithmetic':
            return self._trace_arithmetic(expr, objects)
        elif etype == 'relational':
            return self._trace_relational(expr, objects)
        elif etype == 'boolean':
            return self._trace_logical(expr, objects)
        elif etype == 'aggregation':
            return self._trace_aggregation(expr, objects)
        elif etype == 'func':
            return self._trace_func(expr, objects)
        elif etype == 'control':
            return self._trace_control(expr, objects)
        elif etype == 'randomvar':
            return self._trace_random(expr, objects)
    
    # ===========================================================================
    # leaves
    # ===========================================================================
    
    def _trace_constant(self, expr, objects):
        const = expr.args 
            
        if self.rddl.is_grounded: 
            # grounded domain only returns a scalar (e.g. array with one elem)
            expr.cached_sim_info = const
        else: 
            # argument is reshaped to match the free variables "objects"
            expr.cached_sim_info = self._array_from_scalar(const, objects)
    
        expr.enum_type = None
        
    def _trace_pvar(self, expr, objects):
        var, pvars = expr.args   
             
        if var in self.rddl.enum_literals: 
            # enum literal value treated as int
            const = self.index_of_object[var]
            if self.rddl.is_grounded: 
                # grounded domain only returns a scalar
                expr.cached_sim_info = const
            else: 
                # argument reshaped to match the free variables "objects"
                expr.cached_sim_info = self._array_from_scalar(const, objects)
        
            # store the enum type info
            expr.enum_type = self.rddl.objects_rev[var]            
        else: 
            if self.rddl.is_grounded:
                expr.cached_sim_info = None
            else:
                # enum literal args converted to int and slice the array
                # then array reshaped to match the free variables "objects"
                msg = RDDLObjectsTracer._print_stack_trace(expr)
                slices, literals = self._slice(var, pvars, msg=msg)
                transform = self._map(var, pvars, objects, literals, msg=msg)
                expr.cached_sim_info = (slices, transform)
            
            # store the enum type info
            base_var = self.rddl.parse(var)[0]
            prange = self.rddl.variable_ranges.get(base_var, None)
            expr.enum_type = prange if prange in self.rddl.enum_types else None
        
    def _array_from_scalar(self, scalar, objects):
        ptypes = [ptype for (_, ptype) in objects]
        shape = self.shape(ptypes)
        return np.full(shape=shape, fill_value=scalar)
        
    def _slice(self, var: str,
               pvars: List[str],
               msg: str='') -> Tuple[Set[int], Tuple[Union[int, slice], ...]]:
        '''Given a var and a list of pvariable arguments containing both free 
        and/or enum literals, produces a tuple of slice or int objects that, 
        when indexed into an array of the appropriate shape, produces an array 
        where all literals are evaluated in the array. Also returns a set of 
        indices of pvars that are literals.
        '''
        if pvars is None:
            pvars = []
            
        types_in = self.rddl.param_types.get(var, [])
        if len(pvars) != len(types_in):
            raise RDDLInvalidNumberOfArgumentsError(
                f'Variable <{var}> requires {len(types_in)} parameters, '
                f'got {len(pvars)}.'
                f'\n{msg}')
            
        cached_slices = []
        literals = set()
        for (i, pvar) in enumerate(pvars):
            if pvar in self.rddl.enum_literals:
                enum_type = self.rddl.objects_rev[pvar]
                if types_in[i] != enum_type: 
                    raise RDDLInvalidObjectError(
                        f'Argument {i + 1} of variable <{var}> expects type '
                        f'<{types_in[i]}>, got <{pvar}> of type <{enum_type}>.'
                        f'\n{msg}')
                slice_object = self.index_of_object[pvar]
                literals.add(i)
            else:
                slice_object = slice(None)
            cached_slices.append(slice_object)
        cached_slices = tuple(cached_slices)
        return (cached_slices, literals)

    def _map(self, var: str,
             obj_in: List[str],
             sign_out: List[Tuple[str, str]],
             literals: Set[int],
             msg: str='') -> Callable[[np.ndarray], np.ndarray]:
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
        if obj_in is None:
            obj_in = []
                
        # check that the input objects match fluent type definition
        types_in = self.rddl.param_types.get(var, [])
        if len(obj_in) != len(types_in):
            raise RDDLInvalidNumberOfArgumentsError(
                f'Variable <{var}> requires {len(types_in)} parameters, '
                f'got {len(obj_in)}.'
                f'\n{msg}')
        
        # eliminate literals
        old_sign_in = tuple(zip(obj_in, types_in))
        types_in = [p for (i, p) in enumerate(types_in) if i not in literals]
        obj_in = [p for (i, p) in enumerate(obj_in) if i not in literals]
        
        # reached limit on number of valid dimensions
        valid_symbols = RDDLObjectsTracer.VALID_SYMBOLS
        n_out = len(sign_out)
        if n_out > len(valid_symbols):
            raise RDDLInvalidNumberOfArgumentsError(
                f'At most {len(valid_symbols)} parameter arguments are supported '
                f'but variable <{var}> has {n_out} arguments.'
                f'\n{msg}')
        
        # find a map permutation(a,b,c...) -> (a,b,c...) for correct transpose
        sign_in = tuple(zip(obj_in, types_in))
        permutation = [None] * len(obj_in)
        new_dims = []
        for i_out, (o_out, t_out) in enumerate(sign_out):
            new_dim = True
            for i_in, (o_in, t_in) in enumerate(sign_in):
                if o_in == o_out:
                    permutation[i_in] = i_out
                    new_dim = False
                    if t_out != t_in: 
                        raise RDDLInvalidObjectError(
                            f'Argument {i_in + 1} of variable <{var}> '
                            f'expects type <{t_in}>, got <{o_out}> of type <{t_out}>.'
                            f'\n{msg}')
            
            # need to expand the shape of the value array
            if new_dim:
                permutation.append(i_out)
                new_dims.append(len(self.rddl.objects[t_out]))
                
        # safeguard against any free remaining variables not accounted for
        free = {obj_in[i] for (i, p) in enumerate(permutation) if p is None}
        if free:
            raise RDDLInvalidNumberOfArgumentsError(
                f'Variable <{var}> has unresolved parameter(s) {free}.'
                f'\n{msg}')
        
        # compute the mapping function as follows:
        # 1. append new axes to value tensor equal to # of missing variables
        # 2. broadcast new axes to the desired shape (# of objects of each type)
        # 3. rearrange the axes as needed to match the desired variables in order
        #     3a. in most cases, it suffices to use np.transform (cheaper)
        #     3b. in cases where we have a more complex contraction like 
        #         fluent(?x) = matrix(?x, ?x), we will use np.einsum
        in_shape = self.shape(types_in)
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
        
        # check if the tensor transform with the given signature already exists
        # if yes, retrieve it from the cache; if no, create it and cache it
        _id = f'{new_axis}_{out_shape}_{use_einsum}_{use_tr}_{subscripts}'
        _transform = self._cached_transforms.get(_id, None)     
        mynp = self.tensorlib   
        if _transform is None:
            
            def _transform(arg):
                sample = arg
                if new_axis:
                    sample = mynp.expand_dims(sample, axis=new_axis)
                    sample = mynp.broadcast_to(sample, shape=out_shape)
                if use_einsum:
                    sample = mynp.einsum(subscripts, sample)
                elif use_tr:
                    sample = mynp.transpose(sample, axes=subscripts)
                return sample
            
            self._cached_transforms[_id] = _transform
        
        # log information about the new transformation
        operation = mynp.einsum if use_einsum else (
                        mynp.transpose if use_tr else 'None')
        self._append_log(
            f'computing info for filling missing pvariable arguments:' 
                f'\n\tvar           ={var}'
                f'\n\toriginal args ={old_sign_in}'
                f'\n\tliterals      ={literals}'
                f'\n\tnew args      ={sign_in}'
                f'\n\ttarget args   ={sign_out}'
                f'\n\tnew axes      ={new_axis}'
                f'\n\toperation     ={operation}, subscripts={subscripts}'
                f'\n\tunique op id  ={id(_transform)}\n'
        )
            
        return _transform
    
    # ===========================================================================
    # compound expressions
    # ===========================================================================
    
    def _trace_arithmetic(self, expr, objects):
        for i, arg in enumerate(expr.args):
            self._trace(arg, objects)
        
            # argument cannot be enum type
            RDDLObjectsTracer._check_not_enum(
                arg, expr, f'Argument {i + 1} of operator {expr.etype[1]}')
        expr.enum_type = None
    
    def _trace_relational(self, expr, objects):
        enum_types = set()
        for arg in expr.args:
            self._trace(arg, objects)            
            enum_types.add(arg.enum_type)
        
        # can not mix different enum types or primitive and enum types
        _, op = expr.etype
        if len(enum_types) != 1:
            raise RDDLTypeError(
                f'Relational operator {op} can not compare arguments '
                f'of different enum types or mix enum and non-enum types.\n' + 
                RDDLObjectsTracer._print_stack_trace(expr))
        
        # can not use operator besides == and ~= to compare enum types
        enum_type = next(iter(enum_types))
        valid_enum_types = {'==', '~='}
        if enum_type is not None and op not in valid_enum_types:
            raise RDDLNotImplementedError(
                f'Relational operator {op} is not valid for comparing enum types, '
                f'must be one of {valid_enum_types}.\n' + 
                RDDLObjectsTracer._print_stack_trace(expr))        
        expr.enum_type = None  
    
    def _trace_logical(self, expr, objects):
        for i, arg in enumerate(expr.args):
            self._trace(arg, objects)
            
            # argument cannot be enum type
            RDDLObjectsTracer._check_not_enum(
                arg, expr, f'Argument {i + 1} of operator {expr.etype[1]}')
        expr.enum_type = None
    
    def _trace_aggregation(self, expr, objects):
        if self.rddl.is_grounded:
            raise Exception(
                f'Internal error: aggregation in grounded domain.\n' + 
                RDDLObjectsTracer._print_stack_trace(expr))
            
        _, op = expr.etype
        * pargs, arg = expr.args
        
        # cache and read reduced axes tensor info for the aggregation
        new_objects = objects + [ptype for (_, ptype) in pargs]
        reduced_axes = tuple(range(len(objects), len(new_objects)))   
        expr.cached_sim_info = (new_objects, reduced_axes)
                
        self._append_log(
            f'computing object info for aggregation:'
                f'\n\toperator       ={op} {pargs}'
                f'\n\tinput objects  ={new_objects}'
                f'\n\toutput objects ={objects}'
                f'\n\toperation axes ={reduced_axes}\n'
        )        
            
        # check for undefined types
        bad_types = {ptype 
                     for (_, ptype) in new_objects 
                     if ptype not in self.rddl.objects}
        if bad_types:
            raise RDDLInvalidObjectError(
                f'Type(s) {bad_types} are not defined, '
                f'must be one of {set(self.rddl.objects.keys())}.\n' + 
                RDDLObjectsTracer._print_stack_trace(expr))
            
        # check for duplicated iteration variables
        for _, (free_new, _) in pargs:
            for (free_old, _) in objects:
                if free_new == free_old:
                    raise RDDLInvalidObjectError(
                        f'Iteration variable <{free_new}> is already defined '
                        f'in outer scope.\n' + 
                        RDDLObjectsTracer._print_stack_trace(expr))
        
        # trace the aggregated expression with the new objects
        self._trace(arg, new_objects)
        
        # argument cannot be enum type
        RDDLObjectsTracer._check_not_enum(
            arg, expr, f'Argument of aggregation {op}')        
        expr.enum_type = None
        
    def _trace_func(self, expr, objects):
        for i, arg in enumerate(expr.args):
            self._trace(arg, objects)
            
            # argument cannot be enum type
            RDDLObjectsTracer._check_not_enum(
                arg, expr, f'Argument {i + 1} of function {expr.etype[1]}') 
        expr.enum_type = None
            
    # ===========================================================================
    # control flow
    # ===========================================================================
    
    def _trace_control(self, expr, objects):
        _, op = expr.etype
        if op == 'if':
            self._trace_if(expr, objects)
        elif op == 'switch':
            self._trace_switch(expr, objects)
            
    def _trace_if(self, expr, objects):
        pred, *cases = expr.args
        self._trace(pred, objects)
        enum_types = set()
        for arg in cases:
            self._trace(arg, objects)
            enum_types.add(arg.enum_type)
        
        # can not mix different enum types or primitive and enum types
        if len(enum_types) != 1:
            raise RDDLTypeError(
                f'Branches in if then else cannot produce values '
                f'of different enum types or mix enum and non-enum types.\n' + 
                RDDLObjectsTracer._print_stack_trace(expr))            
        expr.enum_type = next(iter(enum_types))
    
    def _trace_switch(self, expr, objects):
        pred, *cases = expr.args
        
        # must be a pvar
        if not pred.is_pvariable_expression():
            raise RDDLNotImplementedError(
                f'Switch predicate is not a pvariable.\n' + 
                RDDLObjectsTracer._print_stack_trace(expr))
            
        # type in pvariables scope must be an enum
        name, _ = pred.args
        var = self.rddl.parse(name)[0]
        enum_type = self.rddl.variable_ranges[var]
        if enum_type not in self.rddl.enum_types:
            raise RDDLTypeError(
                f'Type <{enum_type}> of switch predicate <{name}> is not an '
                f'enum type, must be one of {self.rddl.enum_types}.\n' + 
                RDDLObjectsTracer._print_stack_trace(expr))
            
        # default statement becomes ("default", expr)
        case_dict = dict((cvalue if ctype == 'case' else (ctype, cvalue)) 
                          for (ctype, cvalue) in cases)
        if len(case_dict) != len(cases):
            raise RDDLInvalidNumberOfArgumentsError(
                f'Duplicated literal or default cases.\n' + 
                RDDLObjectsTracer._print_stack_trace(expr))
        
        # order enum cases by canonical ordering of literals
        expr.cached_sim_info = self._order_cases(enum_type, case_dict, expr)
        
        # trace predicate and cases
        self._trace(pred, objects)
        enum_types = set()
        for arg in case_dict.values():
            self._trace(arg, objects)
            enum_types.add(arg.enum_type)
        
        # can not mix different enum types or primitive and enum types
        if len(enum_types) != 1:
            raise RDDLTypeError(
                f'Cases in switch cannot produce values '
                f'of different enum types or mix enum and non-enum types.\n' + 
                RDDLObjectsTracer._print_stack_trace(expr))            
        expr.enum_type = next(iter(enum_types))
        
    def _order_cases(self, enum_type, case_dict, expr): 
        enum_values = self.rddl.objects[enum_type]
        
        # check that all literals belong to enum_type
        for literal in case_dict.keys():
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
                index = self.index_of_object[literal]
                expressions[index] = arg
        
        # if default statement is missing, cases must be comprehensive
        default_expr = case_dict.get('default', None)
        if default_expr is None:
            for i, arg in enumerate(expressions):
                if arg is None:
                    raise RDDLUndefinedVariableError(
                        f'Enum literal <{enum_values[i]}> of type <{enum_type}> '
                        f'is missing in case list.\n' + 
                        RDDLObjectsTracer._print_stack_trace(expr))
        
        # log cases ordering
        active_expr = [i for i, v in enumerate(expressions) if v is not None]
        self._append_log(
            f'computing case info for {expr.etype[1]}:'
                f'\n\tenum type ={enum_type}'
                f'\n\tcases     ={active_expr}'
                f'\n\tdefault   ={default_expr is not None}\n'
        )     
        
        return (expressions, default_expr)
    
    # ===========================================================================
    # random variable
    # ===========================================================================
    
    def _trace_random(self, expr, objects):
        _, name = expr.etype
        if name == 'Discrete':
            self._trace_discrete(expr, objects)
        else:
            self._trace_random_other(expr, objects)
                
    def _trace_discrete(self, expr, objects):
        (_, enum_type), *cases = expr.args
            
        # enum type must be a valid enum type
        if enum_type not in self.rddl.enum_types:
            raise RDDLTypeError(
                f'Type <{enum_type}> in Discrete distribution is not an '
                f'enum, must be one of {self.rddl.enum_types}.\n' + 
                RDDLObjectsTracer._print_stack_trace(expr))
            
        # no duplicate cases are allowed, and no default assumed
        case_dict = dict(case_tup for (_, case_tup) in cases)
        if len(case_dict) != len(cases):
            raise RDDLInvalidNumberOfArgumentsError(
                f'Duplicated literal or default cases.\n' + 
                RDDLObjectsTracer._print_stack_trace(expr))
        
        # order enum cases by canonical ordering of literals
        expr.cached_sim_info, _ = self._order_cases(enum_type, case_dict, expr)
    
        # trace each case expression
        for i, arg in enumerate(case_dict.values()):
            self._trace(arg, objects)
            
            # argument cannot be enum type
            RDDLObjectsTracer._check_not_enum(
                arg, expr, f'Expression in case {i + 1} of distribution Discrete') 
        expr.enum_type = enum_type
    
    def _trace_random_other(self, expr, objects):
        for i, arg in enumerate(expr.args):
            self._trace(arg, objects)
                
            # argument cannot be enum type
            RDDLObjectsTracer._check_not_enum(
                arg, expr, f'Argument {i + 1} of distribution {expr.etype[1]}') 
        expr.enum_type = None
        
