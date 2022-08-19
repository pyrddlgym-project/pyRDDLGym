"""Exact MILP reduction of Smart Predict-then-Optimize (EMSPO)"""
from xaddpy.models.base import Base
from xaddpy.utils.logger import logger
from gurobipy import GRB, LinExpr, quicksum
from xaddpy.utils.gurobi_util import GurobiModel

from torch import optim
from torch import nn
import torch
import numpy as np
from collections import OrderedDict


class EMSPO(Base):
    """
    The exact approach to solving predict-then-optimize problems.
    In the case of a linear model, the parameters of the linear model should have been obtained from solving
    a MILP model in xadd/solve.py. Then, this class simply loads the optimized parameters and computes
    the test performance.
    # If learn_feature = True, then the MSE loss is used to train the neural network parameters, which outputs
    # a feature embedding. Then, a MILP model is built from the learned features in order to optimize for the linear
    # parameters that correspond to the fully-connected layer of the neural net.
    """

    def __init__(
            self,
            domain,
            target_dim,
            input_dim,
            dataset,
            embed_dim,
            feature_dim,
            param_dim,
            theta_method=0,
            linear=True,
            scaler=None,
            oracle=None,
            l2_lamb=0,
            learn_feature=False,
            optimizer=optim.Adam,
            lr=0.001,
            use_validation=False,
            batch_size=10,
            **kwargs
    ):
        super().__init__(
            dataset,
            target_dim,
            input_dim,
            embed_dim=embed_dim,
            use_validation=use_validation,
            linear=linear,
            scaler=scaler,
            **kwargs
        )

        self.feature_dim = feature_dim
        self.param_dim = param_dim
        self.learn_feature = learn_feature
        self.lr = lr
        self.use_validation = use_validation
        self.oracle = oracle
        self.g_model = None
        self.domain = domain

        if learn_feature:
            self.optimizer = optimizer(self.parameters(), lr=self.lr, weight_decay=l2_lamb)
            self.batch_size = batch_size
            self.criterion = nn.MSELoss(reduction='mean')

        self._theta_sum = None
        if kwargs.get('theta_constr', False):
            self._theta_sum = kwargs['theta_constr']

        """
        How to handle pathological solution
        # method 0: constrain the absolute value of the sum of thetas equal to some positive constant
        # method 1: compute the mean of the data and set equality constraints
        """
        self._theta_method = theta_method

        self.prob_configs = None
        if self.domain == 'energy_scheduling':
            self.prob_configs = kwargs['prob_configs']

    def fit(self, epochs, **kwargs):
        """
        When self.learn_feature = True, then we pretrain the feature embedding using the MSE loss.
        """
        batch_size = self.batch_size
        num_batch = self.n_train // batch_size
        idx_lst = list(range(self.n_train))

        for e in range(epochs):
            self._rng.shuffle(idx_lst)
            for i in range(num_batch):
                self.optimizer.zero_grad()
                X = torch.tensor(self.X_train[idx_lst[i: i + batch_size]], dtype=torch.float)
                c_true = torch.tensor(self.y_train[idx_lst[i: i + batch_size]], dtype=torch.float)

                c_est = self.forward(X)

                loss = self.criterion(c_est, c_true)
                loss.backward()
                self.optimizer.step()

            if self.use_validation and (e % (epochs // 10) == 0):
                extra = self.test_model(self.X_val, self.y_val)
                pred_error = self.test_prediction(self.X_val, self.y_val)
                print("Epoch: {}/{}\t Extra: {}\tTrain loss: {:.5f}\tVal loss: {:.5f}".format(e + 1, epochs, extra,
                                                                                              loss.item(), pred_error))

    def test_prediction(self, X=None, y=None):
        if X is None and y is None:
            X, y = self.X_test, self.y_test

        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float)
            y = torch.tensor(y, dtype=torch.float)
            c_hat = self.forward(X)
            loss = self.criterion(c_hat, y)
        return loss.item()

    def update_gurobi_model(self, g_model: GurobiModel):
        param_array = []
        if self.param_dim == 1:
            for i in range(1, self.param_dim[0] + 1):
                param_array.append(g_model.addVar(lb=float('-inf'), ub=float('inf'),
                                                  vtype=GRB.CONTINUOUS, name=f'theta{i}'))
        else:
            for i in range(self.param_dim[0]):
                param_array.append([])
                s_id = 0
                e_id = self.param_dim[1]
                for j in range(s_id, e_id):
                    lb, ub = float('-inf'), float('inf')
                    param_array[i].append(g_model.addVar(lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name=f'theta{i + 1}{j}'))
        self._param_array = np.array(param_array)

        g_model.update()
        g_model.setAttr('_param_array', self._param_array)
        self.g_model = g_model

    def subst_data_to_milp(self, X, cost, variables):
        theta = self._param_array

        if self._theta_method == 1:  # Fix the bias to the mean of the data
            if self.domain in ['shortest_path', 'classification']:
                c_mean = np.mean(cost, axis=0)
            elif self.domain == 'energy_scheduling':
                c_mean = [np.mean(cost)]
            for i in range(len(c_mean)):
                c_avg_i = c_mean[i]
                theta_i0 = theta[i, 0]
                theta_i0.setAttr('UB', c_avg_i)
                theta_i0.setAttr('LB', c_avg_i)

        elif self._theta_method == 0:
            assert self._theta_sum is not None
            self.g_model.addConstr(np.sum(theta) == self._theta_sum, name='theta_sum_fixed')
            logger.info("'Sum of theta = constant' constraint is added")
        else:
            raise ValueError("'theta_method' should be either 0 or 1")

        obj = 0
        for i in range(len(X)):
            feature_i = X[i]
            if feature_i.ndim == 1:  # Shortest path / classification
                feature_i = np.hstack((1, feature_i))
            else:
                feature_i = np.hstack((np.ones((feature_i.shape[0], 1)), feature_i))
            cost_i = cost[i]
            if feature_i.ndim == 1:
                c_hat = theta @ feature_i
            else:
                c_hat = feature_i @ theta[0]
            c_hat_i: List[gp.Var] = [self.g_model.getVarByName(f'c{j}__{i}') for j in range(1, len(c_hat) + 1)]

            for j in range(len(c_hat_i)):  # For each decision dimension
                if c_hat_i[j] is not None:
                    self.g_model.addConstr(c_hat_i[j] == c_hat[j], name=f'c_{j + 1}__{i}_eq_theta@feature')

            # Get gurobi decision variables
            dec_vars = [self.g_model.getVarByName(f'{str(v_name)}__{i}') for v_name in variables]
            # Define the loss variable associated with this data instance; add it to the overall objective
            if self.domain in ['shortest_path', 'classification']:
                l_i = LinExpr(cost_i[:len(dec_vars)], dec_vars)
            elif self.domain == 'energy_scheduling':
                num_machines = self.prob_configs['num_machines']
                num_tasks = self.prob_configs['num_tasks']
                POWER_USE = self.prob_configs['POWER_USE']
                DURATION = self.prob_configs['DURATION']
                q = self.prob_configs['q']
                N = 1440 // q
                MACHINES = range(num_machines)
                TASKS = range(num_tasks)

                # 'x0_0_6__0' => (0, 0, 6)
                get_tuple_from_name = lambda n: tuple(map(int, n.split('__')[0][1:].split('_')))
                # (0, 0, 6) => x0_0_6__0: gurobipy.Var
                get_var_from_tuple = lambda t: self.g_model.getVarByName(f'x{"_".join(map(str, t))}__{i}')
                dec_vars_tuples = {get_tuple_from_name(v._VarName) for v in dec_vars}

                l_i = quicksum([
                    get_var_from_tuple((j, mc, t)) * np.sum(cost_i[t: t + DURATION[j]]) * POWER_USE[j] * q / 60
                    for j in TASKS for t in range(N - DURATION[j] + 1) for mc in MACHINES if
                    (j, mc, t) in dec_vars_tuples
                ])

            obj += 1 / len(X) * l_i

        # Set the objective
        self.g_model.setObjective(obj, sense=GRB.MINIMIZE)
        self.g_model.update()

    def scale_opt_params(self):
        pass

    def handle_solutions(self, verbose):
        if verbose:
            if self.g_model.status == GRB.OPTIMAL:
                for v in self.g_model.getVars():
                    logger.info(f'Objective: {self.g_model.objVal}')
            elif self.g_model.status == GRB.TIME_LIMIT:
                logger.info(f'Solver timed out... Best objective: {self.g_model._best_obj}')

        # Get the parameters of the prediction model
        logger.info('Export the optimized parameters to a numpy array')
        if self.g_model.solCount > 0:
            param_matrix = self.g_model._param_array
            param_to_return = np.zeros(param_matrix.shape, dtype=float)
            if param_matrix.ndim == 1:
                cols = 1
            else:
                cols = param_matrix.shape[1]
            for i in range(param_matrix.shape[0]):
                for j in range(cols):
                    try:
                        param_to_return[i, j] = param_matrix[i, j].x
                    except:
                        param_to_return[i, j] = var_sol[f'theta{i + 1}{j}']
            logger.info('Done!')
            logger.info(f"Model parameters: {param_to_return}")
            logger.info(f"Objective: {self.g_model.objVal}")
        else:
            raise Exception('No feasible solution found during optimization')

        # Scale the parameters by normalizing them
        abs_sum = np.abs(np.sum(param_to_return))
        param_to_return /= abs_sum
        return param_to_return