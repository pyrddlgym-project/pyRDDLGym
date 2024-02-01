from typing import cast, Union, Dict, Any
import numpy as np

from pyRDDLGym.Core.Simulator.RDDLSimulator import RDDLSimulatorWConstraints, RDDLSimulator
from pyRDDLGym.Core.Compiler.RDDLModel import PlanningModel
from pyRDDLGym.Core.Parser.expr import Expression
from pyRDDLGym.XADD.RDDLModelXADD import RDDLModelWXADD
from pyRDDLGym.XADD.RDDLLevelAnalysisXADD import RDDLLevelAnalysisWXADD


class RDDLSimulatorWXADD(RDDLSimulatorWConstraints):

    def __init__(self, model: PlanningModel,
                 rng: np.random.Generator=np.random.default_rng(),
                 compute_levels: bool=True,
                 max_bound: float=np.inf) -> None:
        self._model = cast(RDDLModelWXADD, model)
        self.context = self._model._context
        self._rng = rng
        
        # perform a dependency analysis and topological sort to compute levels
        dep_analysis = RDDLLevelAnalysisWXADD(self._model)
        self.cpforder = dep_analysis.compute_levels()
        self._order_cpfs = list(sorted(self.cpforder.keys()))
        
        self._action_fluents = set(self._model.actions.keys())
        self._init_actions = self._model.actions.copy()
        
        # non-fluent will never change
        self._subs = self._model.nonfluents.copy()
        
        # is a POMDP
        self._pomdp = bool(self._model.observ)

        # __init__ from 'RDDLSimulatorWConstraints'
        self.epsilon = 0.001
        
        # self.BigM = float(max_bound)
        
        self.BigM = max_bound
        self._bounds = {}
        for state in model.states:
            self._bounds[state] = [-self.BigM, self.BigM]
        for derived in model.derived:
            self._bounds[derived] = [-self.BigM, self.BigM]
        for interm in model.interm:
            self._bounds[interm] = [-self.BigM, self.BigM]
        for action in model.actions:
            self._bounds[action] = [-self.BigM, self.BigM]
        for obs in model.observ:
            self._bounds[obs] = [-self.BigM, self.BigM]

        # actions and states bounds extraction for gym's action and state spaces repots only!
        # currently supports only linear in\equality constraints
        for action_precond in model.preconditions:
            self._parse_bounds_rec(action_precond, self._model.actions)

        for state_inv in model.invariants:
            self._parse_bounds_rec(state_inv, self._model.states)

        for name in self._bounds:
            lb, ub = self._bounds[name]
            RDDLSimulator._check_bounds(
                lb, ub, f'Variable <{name}>', self._bounds[name])
        
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
                if not v in self._model.nonfluents 
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
