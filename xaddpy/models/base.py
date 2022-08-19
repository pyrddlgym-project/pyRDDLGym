from gurobipy import GRB
import gurobipy as gp
from xaddpy.utils.gurobi_util import GurobiModel
import numpy as np
import json
import sympy as sp
import torch.nn as nn
import torch
from xaddpy.utils.global_vars import REL_TYPE
import xaddpy.utils.util as util
from xaddpy.utils.milp_encoding import convert2GurobiExpr, relConverter
from xaddpy.utils.logger import logger
from tqdm import tqdm
from collections import OrderedDict


class Oracle:
    """
    Implementation of an oracle solver. Oracle should be initialized with problem-specific formulations:
    objective and constraints. Dimensionality should also be set.
    """
    def __init__(self, name, var_name_rule=None, sense=GRB.MINIMIZE, verbose=0, fname=None):
        self.g_model = GurobiModel(name)
        self.g_model.setParam('OutputFlag', verbose)
        self.sense = sense
        self._presolved = False
        self.presolved_cost = None
        self.presolved_solution = None
        # self.name_to_var = {}
        self.var_lst = []
        self.num_var = 0
        self.var_name_rule = var_name_rule
        if fname is not None:
            self.read_problem_from_file(fname)

    def read_problem_from_file(self, fname: str):
        # The format of the file can follow that of the xadd .json file.
        # We rely on sympy for converting its expressions into the gurobi model.
        try:
            with open(fname, "r") as json_file:
                prob_instance = json.load(json_file)
        except:
            print("Failed to open file!!")
            raise FileNotFoundError
        lp_fname = fname.replace('.json', '.lp')

        # Namespace to be used to define sympy symbols
        ns = {}

        # is_minimize?
        self.sense = GRB.MINIMIZE if prob_instance['is-minimize'] else GRB.MAXIMIZE

        # Create Sympy symbols for cvariables and bvariables
        if len(prob_instance['cvariables0']) == 1 and isinstance(prob_instance['cvariables0'][0], int):
            cvariables0 = sp.symbols(f"x1:{prob_instance['cvariables'][0] + 1}")
        else:
            cvariables0 = []
            for v in prob_instance['cvariables0']:
                cvariables0.append(sp.symbols(v))
        cvariables = cvariables0

        if len(prob_instance['bvariables']) == 1 and isinstance(prob_instance['bvariables'][0], int):
            bvariables = sp.symbols(f"b1:{prob_instance['bvariables'][0] + 1}", integer=True)
        else:
            bvariables = []
            for v in prob_instance['bvariables']:
                bvariables.append(sp.symbols(v, integer=True))

        bvar_dim = len(bvariables)
        bvar_dim = 2 if bvar_dim == 1 else bvar_dim  # Binary variable dimension
        cvar_dim = len(cvariables)  # Continuous variable dimension
        var_dim = cvar_dim + bvar_dim
        assert (bvar_dim == 0 and cvar_dim != 0) or (bvar_dim != 0 and cvar_dim == 0)

        variables = list(cvariables) + list(bvariables)
        ns.update({str(v): v for v in variables})

        # retrieve lower and upper bounds over decision variables
        min_vals = prob_instance['min-values']
        max_vals = prob_instance['max-values']

        if len(min_vals) == 1 and len(min_vals) == len(max_vals):  # When a single number is used for all cvariables
            min_vals = min_vals * cvar_dim
            max_vals = max_vals * cvar_dim

        assert len(min_vals) == len(max_vals) and len(min_vals) == cvar_dim, \
            "Bound information mismatch!\n cvariables: {}\tmin-values: {}\tmax-values: {}".format(
                prob_instance['cvariables'],
                prob_instance['min-values'],
                prob_instance['max-values'])

        bound_dict = {}
        for i, (lb, ub) in enumerate(zip(min_vals, max_vals)):
            lb, ub = sp.S(lb), sp.S(ub)
            bound_dict[ns[str(cvariables[i])]] = (lb, ub)

        # Update XADD attributes
        variables = [ns[str(v)] for v in variables]
        if self.var_name_rule is not None:
            variables = list(sorted(variables, key=self.var_name_rule))

        bvariables = [ns[str(bvar)] for bvar in bvariables]
        cvariables = [ns[str(cvar)] for cvar in cvariables]

        # Read constraints and link with the created Sympy symbols
        ineq_constrs = []
        for const in prob_instance['ineq-constr']:
            ineq_constrs.append(sp.sympify(const, locals=ns))

        assert prob_instance['xadd'] == "", "Can't translate an initial XADD into Gurobi model\n" \
                                            "Try providing constraints explicitly instead."

        eq_constr_dict, variables = util.compute_rref_filter_eq_constr(prob_instance['eq-constr'], variables, locals=ns)

        sympy2gurobi = self.g_model.sympy_to_gurobi
        for v in variables:
            if v in cvariables:
                lb, ub = bound_dict[v]
                gvar = self.add_variable(name=str(v), lb=lb, ub=ub, vtype=GRB.CONTINUOUS)
                sympy2gurobi[v] = gvar
            else:
                gvar = self.add_variable(name=str(v), vtype=GRB.BINARY)
                sympy2gurobi[v] = gvar

        # Convert equality constraints in sympy to gurobi constraints
        for v, rhs in eq_constr_dict.items():
            gvar: gp.Var = convert2GurobiExpr(v, self.g_model)
            rhs = convert2GurobiExpr(rhs, self.g_model)
            self.g_model.addConstr(gvar == rhs, "Equality constraint for {}".format(v))

        # Convert inequality constraints in sympy to gurobi constraints
        for i, const in enumerate(ineq_constrs):
            lhs, rhs, rel = const.lhs, const.rhs, REL_TYPE[type(const)]
            gp_lhs = convert2GurobiExpr(lhs, g_model=self.g_model)
            gp_rhs = convert2GurobiExpr(rhs, g_model=self.g_model)
            if rel == '<=' or rel == '<':
                self.g_model.addConstr(gp_lhs <= gp_rhs, f"Inequality constraint {i}")
            else:
                self.g_model.addConstr(gp_lhs >= gp_rhs, f"Inequality constraint {i}")

        self.g_model.update()
        self.g_model.write(lp_fname)

    def add_variable(self, name, lb=float('-inf'), ub=float('inf'), vtype=None):
        if vtype == GRB.CONTINUOUS:
            var = self.g_model.addVar(lb=lb, ub=ub, vtype=vtype, name=name)
        elif vtype == GRB.BINARY:
            var = self.g_model.addVar(vtype=vtype, name=name)
        else:
            raise ValueError(f"Unrecognized vtype {vtype} for variable {name}")

        # self.name_to_var[name] = var
        self.var_lst.append(var)
        self.num_var += 1
        return var

    def set_objective(self, obj):
        self.g_model.setObjective(obj, sense=self.sense)

    def optimize(self):
        status = self.g_model.optimize()
        return status

    def presolve(self, c):
        """
        Given the dataset c of true coefficients, presolve for each data instance and store the result.
        This way, per each problem, we only need to solve for the true solution once.
        Args:
            c (numpy.ndarray): the true coefficients of decision variables for a given data instance
        """
        opt_objective, opt_solution = [], []
        num_var = self.num_var
        var_lst = self.var_lst.copy()
        if self.num_var == 1 and self.var_lst[0].vtype == GRB.BINARY:
            num_var = 2
            var_lst.append(1 - var_lst[0])

        for i in tqdm(range(c.shape[0]), desc="Presolve for true solutions and cache results"):
            objval, sol = self(c[i], var_lst, num_var)
            opt_objective.append(objval)
            opt_solution.append(sol)

        self.presolved_cost = opt_objective
        self.presolved_solution = np.array(opt_solution)
        self._presolved = True

    @property
    def presolved_result(self):
        return self.presolved_cost, self.presolved_solution

    def __call__(self, coeffs, var_lst=None, num_var=None):
        """
        Solve an optimization problem given the coefficients.
        Note: the order of the coefficients must match that of the variables in the var_lst.
        Args:
            coeffs (np.ndarray): the vector of coefficients of decision variables for a data instance
        Returns:
             opt_objective (float)  The optimal objective value obtained
             opt_solution (np.ndarray)  The optimal solution obtained
        """
        if var_lst is None:
            var_lst = self.var_lst

        if num_var is None:
            num_var = self.num_var
        self.set_objective(gp.LinExpr(coeffs, var_lst))
        status = self.optimize()
        if status != GRB.OPTIMAL:
            raise gp.GurobiError("Infeasible data instance")

        opt_objective = self.g_model.objVal
        opt_solution = [v.x for v in self.var_lst]
        if self.num_var == 1 and self.var_lst[0].vtype == GRB.BINARY:
            opt_solution.append(1 - self.var_lst[0].x)

        return opt_objective, np.array(opt_solution)


class Base(nn.Module):
    def __init__(self, dataset, target_dim, input_dim, use_validation=False, linear=True, scaler=None, **kwargs):
        super().__init__()

        # Link data
        self.X_train = dataset['train'][0]
        self.y_train = dataset['train'][1]
        self.X_val = dataset['val'][0]
        self.y_val = dataset['val'][1]
        self.X_test = dataset['test'][0]
        self.y_test = dataset['test'][1]
        self.n_train = self.X_train.shape[0]
        self.n_val = self.X_val.shape[0] if use_validation else 0
        self.n_test = self.X_test.shape[0]

        self.decision_dim = target_dim
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.linear = linear
        self.use_validation = use_validation

        if linear:
            self.model = nn.Linear(input_dim, target_dim)
        else:
            assert 'embed_dim' in kwargs
            embed_dim = kwargs['embed_dim'].copy()
            assert len(embed_dim) >= 1, "A nonlinear NN should have at least one hidden layer"
            layers, act = [], []

            in_dim = input_dim
            embed_dim.append(target_dim)
            for i in range(len(embed_dim)):
                layers.append(nn.Linear(in_dim, embed_dim[i], ))
                if i < len(embed_dim) - 1:
                    act.append(nn.ReLU())
                    in_dim = embed_dim[i]

            self.layers = nn.ModuleList(layers)
            self.act = nn.ModuleList(act)
            self.model = self.layers[-1]

        self._rng: np.random.RandomState
        if kwargs.get('rng', False):
            self._rng: np.random.RandomState = kwargs['rng']
        else:
            self._rng: np.random.RandomState = np.random.RandomState(0)
        self.scaler = scaler

        assert 'fname_param' in kwargs, "`fname_param` should be passed"
        self._fname_param = kwargs['fname_param']

        self._optimal_obj = None
        self._best_obj = float('inf')
        self._best_param = None
        self._best_epoch = 0

    def forward(self, x):
        if self.linear:
            return self.model(x)
        else:
            for i in range(len(self.layers)):
                x = self.layers[i](x)
                if i < len(self.layers) - 1:
                    x = self.act[i](x)
            return x

    def test_model(self, X=None, y=None, dom='', train=True):
        if dom == 'shortest':
            actual_cost, optimal_cost, regret = self.test_shortest(X, y)
        elif dom == 'energy':
            actual_cost, optimal_cost, regret = self.test_energy(X, y)
        elif dom == 'classification':
            actual_cost, optimal_cost, regret = self.test_classification(X, y)
        else:
            raise ValueError("Unrecognized domain passed")
        return actual_cost, optimal_cost, regret

    def test_classification(self, X, y):
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float)
            c_hat = self.forward(X)

            w_ests = np.empty((0, self.target_dim))
            w_trues = np.empty((0, self.target_dim))
            for i in range(c_hat.size()[0]):
                _, w_est = self.oracle(c_hat[i].numpy())
                _, w_true = self.oracle(y[i])
                w_ests = np.vstack((w_ests, w_est.reshape(1, -1)))
                w_trues = np.vstack((w_trues, w_true.reshape(1, -1)))
            min_est = np.sum(y * w_ests, axis=1)
            min_true = np.sum(y * w_trues, axis=1)
            optimal_cost = np.mean(min_true)
            actual_cost = np.mean(min_est)
            regret = actual_cost - optimal_cost  # Extra travel time
        return actual_cost, optimal_cost, regret

    def test_shortest(self, X, y):
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float)  # shape = (test_size, feature_dim)
            c_hat = self.forward(X)                 # shape = (test_size, dec_dim)

            w_ests = np.empty((0, self.target_dim))
            w_trues = np.empty((0, self.target_dim))
            for i in range(c_hat.size()[0]):
                _, w_est = self.oracle(c_hat[i].numpy())
                _, w_true = self.oracle(y[i])
                w_ests = np.vstack((w_ests, w_est.reshape(1, -1)))
                w_trues = np.vstack((w_trues, w_true.reshape(1, -1)))
            min_est = np.sum(y * w_ests, axis=1)
            min_true = np.sum(y * w_trues, axis=1)
            optimal_cost = np.mean(min_true)
            actual_cost = np.mean(min_est)
            regret = actual_cost - optimal_cost  # Extra travel time
        return actual_cost, optimal_cost, regret

    def test_energy(self, X, y):
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float)               # shape = (num_days, num time slots, feature_dim)
            c_hat = self.forward(X).detach().numpy().squeeze()  # shape = (num_days, num time slots)

            actual_cost, optimal_cost, regret = self.oracle.compute_regret(c_hat, y)
        return np.mean(actual_cost), np.mean(optimal_cost), np.mean(regret)

    def save_params(self, fname, theta_only=True):
        state_dict = self.state_dict()
        weight = state_dict['model.weight']
        bias = state_dict['model.bias']

        if weight.squeeze().ndim == 1:
            params = torch.cat((bias, weight.squeeze())).view(-1, 1)
        else:
            params = torch.cat((bias.view(-1, 1), weight), dim=1)
        params = params.detach().numpy()
        np.save(fname, params)

    def load_params(self, f_npy):
        """
        Loads the optimized parameters to the PyTorch model.
        """
        param_array = np.load(f_npy)
        # 1-dimensional parameter vector
        if param_array.shape[1] == 1:
            bias = param_array[0]
            weight = param_array[1:].transpose()
        else:
            bias = param_array[:, 0]
            weight = param_array[:, 1:]
        state_dict = OrderedDict({'weight': torch.tensor(weight, requires_grad=False),
                                  'bias': torch.tensor(bias, requires_grad=False)})
        self.model.load_state_dict(state_dict, strict=False)