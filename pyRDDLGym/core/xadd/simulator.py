from typing import cast, Union, Dict, Any
import numpy as np

from pyRDDLGym.core.compiler.initializer import RDDLValueInitializer
from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.simulator import RDDLSimulator
from pyRDDLGym.core.parser.expr import Expression

from pyRDDLGym.xadd.model import RDDLModelXADD
from pyRDDLGym.xadd.levels import RDDLLevelAnalysisXADD


class RDDLSimulatorXADD(RDDLSimulator):

    def __init__(self, model: RDDLPlanningModel,
                 allow_synchronous_state: bool=True,
                 rng: np.random.Generator=np.random.default_rng()) -> None:
        self._model = cast(RDDLModelXADD, model)
        self.context = self._model._context
        self._rng = rng
        
        self._compile()
        
    def _compile(self):
        
        # compile initial values
        initializer = RDDLValueInitializer(self._model)
        self.init_values = initializer.initialize()
        
        # compute dependency graph for CPFs and sort them by evaluation order
        sorter = RDDLLevelAnalysisXADD(self._model, allow_synchronous_state)
        levels = sorter.compute_levels()   
        self.cpfs = []  
        for cpfs in levels.values():
            for cpf in cpfs:
                expr = self._model.cpfs[cpf]
                prange = rddl.variable_ranges[cpf]
                dtype = RDDLValueInitializer.NUMPY_TYPES.get(
                    prange, RDDLValueInitializer.INT)
                self.cpfs.append((cpf, expr, dtype))
        
        # no tracing
        self.traced = None
        
        # initialize all fluent and non-fluent values        
        self.subs = self.init_values.copy()
        self.state = None
        self.noop_actions = {var: values
                             for (var, values) in self.init_values.items()
                             if self._model.variable_types[var] == 'action-fluent'}
        self.grounded_noop_actions = self.noop_actions
        self.grounded_action_ranges = self._model.action_ranges
        self._pomdp = bool(self._model.observ_fluents)
        
        # cached for performance
        self.invariant_names = [f'Invariant {i}' for i in range(len(self._model.invariants))]        
        self.precond_names = [f'Precondition {i}' for i in range(len(self._model.preconditions))]
        self.terminal_names = [f'Termination {i}' for i in range(len(self._model.terminations))]       
        
    def _sample(self, expr: Union[int, Expression], subs: Dict[str, Any]):
        """Samples the current XADD node by substituting the values stored in 'subs'.

        Args:
            expr (int): The XADD node to which values are substituted
            subs (Dict[str, Any]): The dictionary containing 'var_name -> value' mappings
        """
        if isinstance(expr, Expression):
            return super()._sample(expr, subs)
        else:
            subs = {
                self.var_name_to_sympy_var[self.var_name_to_sympy_var_name.get(v, v)]: val 
                for v, val in subs.items() 
                if not v in self._model.non_fluents 
                and self.var_name_to_sympy_var.get(self.var_name_to_sympy_var_name.get(v, v))
            }
            
            # If there exists a random variable, sample it first
            var_set = self.context.collect_vars(expr)
            rv_set = {v for v in var_set if v in self.context._random_var_set}
            if len(rv_set) > 0:
                for rv in rv_set:
                    if str(rv).startswith('#_UNIFORM'):
                        subs[rv] = self._sample_uniform(expr, subs)
                    elif str(rv).startswith('#_GAUSSIAN'):
                        subs[rv] = self._sample_normal(expr, subs)
                    else:
                        raise ValueError
            cont_assign = {var: val 
                           for var, val in subs.items() 
                           if var in var_set and var in self.context._cont_var_set}
            bool_assign = {var: val 
                           for var, val in subs.items() 
                           if var in var_set and var in self.context._bool_var_set}
            res = self.context.evaluate(node_id=expr,
                                        bool_assign=bool_assign,
                                        cont_assign=cont_assign,
                                        primitive_type=True)
            return res
    
    def _sample_uniform(self, expr, subs):
        return self._rng.uniform(0, 1)
    
    def _sample_normal(self, expr, subs):
        return self._rng.standard_normal()
    
    def _sample_exponential(self, expr, subs):
        raise NotImplementedError
    
    def _sample_bernoulli(self, expr, subs):
        raise NotImplementedError
    
    def _sample_binomial(self, expr, subs):
        raise NotImplementedError
    
    def _sample_gamma(self, expr, subs):
        raise NotImplementedError
    
    def _sample_beta(self, expr, subs):
        raise NotImplementedError
    
    def _sample_weibull(self, expr, subs):
        raise NotImplementedError
    
    def _sample_dirac_delta(self, expr, subs):
        raise NotImplementedError
    
    def _sample_geometric(self, expr, subs):
        raise NotImplementedError
    
    def _sample_gumbel(self, expr, subs):
        raise NotImplementedError
    
    def _sample_kron_delta(self, expr, subs):
        raise NotImplementedError
    
    def _sample_negative_binomial(self, expr, subs):
        raise NotImplementedError
    
    def _sample_poisson(self, expr, subs):
        raise NotImplementedError
    
    def _sample_student(self, expr, subs):
        raise NotImplementedError

    @property
    def sympy_var_name_to_var_name(self):
        return self._model._sympy_var_name_to_var_name

    @property
    def var_name_to_sympy_var_name(self):
        return self._model._var_name_to_sympy_var_name
    
    @property
    def var_name_to_sympy_var(self):
        return self.context._str_var_to_var
