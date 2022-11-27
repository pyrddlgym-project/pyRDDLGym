from typing import Iterable, List, Tuple
import warnings

from pyRDDLGym.Core.Debug.decompiler import RDDLDecompiler
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidNumberOfArgumentsError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidObjectError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLNotImplementedError
from pyRDDLGym.Core.Parser.expr import Expression
from pyRDDLGym.Core.Parser.rddl import RDDL


class LiftedRDDLTypeAnalysis:
    
    VALID_SYMBOLS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    def __init__(self, rddl: RDDL, debug: bool=False):
        self.rddl = rddl
        self.debug = debug
        
        self.domain = rddl.domain
        self.instance = rddl.instance
        self.non_fluents = rddl.non_fluents
        
        self._compile_types()
        self._compile_objects()
        
    def _compile_types(self):
        self.pvar_types = {}
        for pvar in self.domain.pvariables:
            primed_name = name = pvar.name
            if pvar.is_state_fluent():
                primed_name = name + '\''
            ptypes = pvar.param_types
            if ptypes is None:
                ptypes = []
            self.pvar_types[name] = ptypes
            self.pvar_types[primed_name] = ptypes
            
        self.cpf_types = {}
        for cpf in self.domain.cpfs[1]:
            _, (name, objects) = cpf.pvar
            if objects is None:
                objects = [] 
            types = self.pvar_types[name]
            self.cpf_types[name] = [(o, types[i]) for i, o in enumerate(objects)]
        
        if self.debug:
            pvar = ''.join(f'\n\t\t{k}: {v}' for k, v in self.pvar_types.items())
            cpf = ''.join(f'\n\t\t{k}: {v}' for k, v in self.cpf_types.items())
            warnings.warn(
                f'compiling type info:'
                f'\n\tpvar types ={pvar}'
                f'\n\tcpf types  ={cpf}\n'
            )
    
    def _compile_objects(self): 
        self.objects = {}
        self.objects_to_index = {}
        self.objects_to_type = {}
        for name, ptype in self.non_fluents.objects:
            self.objects[name] = {obj: i for i, obj in enumerate(ptype)}
            for obj in ptype:
                if obj in self.objects_to_type:
                    other_name = self.objects_to_type[obj]
                    raise RDDLInvalidObjectError(
                        f'Types <{other_name}> and <{name}> '
                        f'can not share the same object <{obj}>.')
            self.objects_to_index.update(self.objects[name])
            self.objects_to_type.update({obj: name for obj in ptype})
        
        if self.debug:
            obj = ''.join(f'\n\t\t{k}: {v}' for k, v in self.objects.items())
            warnings.warn(
                f'compiling object info:'
                f'\n\tobjects ={obj}\n'
            )
    
    def coordinates(self, objects: Iterable[str], msg: str) -> Tuple[int, ...]:
        try:
            return tuple(self.objects_to_index[obj] for obj in objects)
        except:
            for obj in objects:
                if obj not in self.objects_to_index:
                    raise RDDLInvalidObjectError(
                        f'Object <{obj}> declared in {msg} is not valid.')
    
    def shape(self, types: Iterable[str]) -> Tuple[int, ...]:
        return tuple(len(self.objects[ptype]) for ptype in types)
    
    def is_compatible(self, var: str, objects: List[str]) -> bool:
        types = self.pvar_types[var]
        if objects is None:
            objects = []
        n_types = len(types)
        n_objects = len(objects)
        if n_types != n_objects:
            return False
        for ptype, obj in zip(types, objects):
            if obj not in self.objects_to_type or ptype != self.objects_to_type[obj]:
                return False
        return True
        
    @staticmethod
    def _print_stack_trace(expr):
        if isinstance(expr, Expression):
            trace = RDDLDecompiler().decompile_expr(expr)
        else:
            trace = str(expr)
        return '>> ' + trace
    
    def map(self, var: str,
            obj_in: List[str],
            sign_out: List[Tuple[str, str]],
            expr: Expression) -> Tuple[str, bool, Tuple[int, ...]]:
                
        # check that the input objects match fluent type definition
        types_in = self.pvar_types.get(var, [])
        if obj_in is None:
            obj_in = []
        n_in = len(obj_in)
        n_req = len(types_in)
        if n_in != n_req:
            raise RDDLInvalidNumberOfArgumentsError(
                f'Variable <{var}> requires {n_req} parameters, got {n_in}.\n' + 
                LiftedRDDLTypeAnalysis._print_stack_trace(expr))
            
        # reached limit on number of valid dimensions
        valid_symbols = LiftedRDDLTypeAnalysis.VALID_SYMBOLS
        n_max = len(valid_symbols)
        n_out = len(sign_out)
        if n_out > n_max:
            raise RDDLNotImplementedError(
                f'Up to {n_max}-D are supported, '
                f'but variable <{var}> is {n_out}-D.\n' + 
                LiftedRDDLTypeAnalysis._print_stack_trace(expr))
        
        # find a map permutation(a,b,c...) -> (a,b,c...) for the correct einsum
        sign_in = tuple(zip(obj_in, types_in))
        lhs = [None] * len(obj_in)
        new_dims = []
        for i_out, (o_out, t_out) in enumerate(sign_out):
            new_dim = True
            for i_in, (o_in, t_in) in enumerate(sign_in):
                if o_in == o_out:
                    lhs[i_in] = valid_symbols[i_out]
                    new_dim = False
                    if t_out != t_in: 
                        raise RDDLInvalidObjectError(
                            f'Argument <{i_in + 1}> of variable <{var}> '
                            f'expects object of type <{t_in}>, '
                            f'got <{o_out}> of type <{t_out}>.\n' + 
                            LiftedRDDLTypeAnalysis._print_stack_trace(expr))
            
            # need to expand the shape of the value array
            if new_dim:
                lhs.append(valid_symbols[i_out])
                new_dims.append(len(self.objects[t_out]))
                
        # safeguard against any free types
        free = [types_in[i] for i, p in enumerate(lhs) if p is None]
        if free:
            raise RDDLInvalidNumberOfArgumentsError(
                f'Variable <{var}> has free parameter(s) {free}.\n' + 
                LiftedRDDLTypeAnalysis._print_stack_trace(expr))
        
        # this is the necessary information for np.einsum
        lhs = ''.join(lhs)
        rhs = valid_symbols[:n_out]
        permute = lhs + ' -> ' + rhs
        identity = lhs == rhs
        new_dims = tuple(new_dims)
        
        if self.debug:
            warnings.warn(
                f'computing info for pvariable transform:' 
                f'\n\texpr     ={expr}'
                f'\n\tinputs   ={sign_in}'
                f'\n\ttargets  ={sign_out}'
                f'\n\tnew axes ={new_dims}'
                f'\n\teinsum   ={permute}\n'
            )
        
        return (permute, identity, new_dims)
        
