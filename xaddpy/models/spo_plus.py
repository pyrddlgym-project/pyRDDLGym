import numpy as np
import torch
import torch.optim as optim

from xaddpy.models.base import Base, Oracle
from xaddpy.utils.logger import logger

from typing import Optional, List

"""
Code imported from https://github.com/JayMan91/NeurIPSIntopt/
"""

class SPOPlus(Base):
    """
    SPO+ approach as in Elmachtoub and Grigas (2017).
    If sgd is set to True, then subgradient descent based approach is used.
    Otherwise, the reformulation approach is used.
    """
    def __init__(
            self,
            target_dim: int,
            input_dim: int,
            dataset: dict,
            embed_dim: List[int],
            linear: bool = True,
            scaler: Optional = None,
            oracle=None,
            batch_size=10,
            optimizer=optim.Adam,
            l2_lamb=0,
            lr=0.001,
            use_validation=False,
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
        self.lr = lr
        self.l2_lamb = l2_lamb
        self.batch_size = batch_size
        self.optimizer = optimizer(self.parameters(), lr=self.lr)

        # Oracle
        self.oracle = oracle
        assert oracle._presolved, "Oracle should have presolved! Exiting..."
        self.optimal_solution = oracle.presolved_solution


class SPOPlusShortest(SPOPlus):
    def __init__(
            self,
            target_dim: int,
            input_dim: int,
            dataset: dict,
            embed_dim: List[int],
            linear: bool = True,
            scaler: Optional = None,
            oracle: Oracle = None,
            batch_size=10,
            optimizer=optim.Adam,
            l2_lamb=0,
            lr=0.001,
            use_validation=False,
            **kwargs
    ):
        super().__init__(
            target_dim,
            input_dim,
            dataset,
            embed_dim,
            linear=linear,
            scaler=scaler,
            oracle=oracle,
            batch_size=batch_size,
            optimizer=optimizer,
            l2_lamb=l2_lamb,
            lr=lr,
            use_validation=use_validation,
            **kwargs
        )
        self._dom = 'shortest'

    def fit(self, epochs, **kwargs):
        batch_size = self.batch_size
        num_batch = self.n_train // batch_size
        idx_lst = list(range(self.n_train))

        for e in range(epochs):
            self._rng.shuffle(idx_lst)
            for i in range(num_batch):
                self.optimizer.zero_grad()
                X, c_true = self.X_train[idx_lst[i: i+batch_size]], self.y_train[idx_lst[i: i + batch_size]]
                w_true = self.optimal_solution[idx_lst[i: i+batch_size]]

                # Need to handle each data instance separately
                for j in range(len(X)):
                    X_j, c_true_j = torch.tensor(X[j, :], dtype=torch.float), c_true[j, :]
                    w_true_j = w_true[j]

                    c_pred_tensor = self.forward(X_j)
                    c_hat_j = c_pred_tensor.detach().numpy()
                    c_spo = (2 * c_hat_j - c_true_j)

                    _, w_spo = self.oracle(c_spo)

                    # Compute the subgradient
                    # Note: backward method takes this as argument, computing the necessary dot product
                    grad = torch.from_numpy(w_true_j - w_spo).float()
                    grad = grad.squeeze()
                    c_pred_tensor.backward(gradient=grad)
                self.optimizer.step()

            msg = f"Epoch: {e + 1}/{epochs}".center(25) + "|"
            if self.use_validation:
                actual_cost, opt_cost, regret = self.test_model(self.X_val, self.y_val, dom=self._dom)
                msg += f'Cost: {actual_cost:.10f}'.ljust(25) + f'Optimal: {opt_cost:.10f}'.ljust(25) + \
                       f'Regret: {regret:.10f}'.ljust(25)
            else:
                actual_cost, opt_cost, regret = self.test_model(self.X_train, self.y_train, dom=self._dom)
                msg += f'Cost: {actual_cost:.10f}'.ljust(25) + f'Optimal: {opt_cost:.10f}'.ljust(25) + \
                       f'Regret: {regret:.10f}'.ljust(25)
                # Update the best objective
                self._optimal_obj = opt_cost
                if actual_cost < self._best_obj:
                    self._best_obj = actual_cost
                    self._best_param = self.save_params(self._fname_param)
                    self._best_epoch = e

            if epochs >= 10 and e % (epochs // 10) == 0:
                logger.info(msg)


class SPOPlusEnergy(SPOPlus):
    def __init__(
            self,
            target_dim: int,
            input_dim: int,
            dataset: dict,
            embed_dim: List[int],
            linear: bool = True,
            scaler: Optional = None,
            oracle: Oracle = None,
            batch_size=10,
            optimizer=optim.Adam,
            l2_lamb=0,
            lr=0.001,
            use_validation=False,
            **kwargs
    ):
        super().__init__(
            target_dim,
            input_dim,
            dataset,
            embed_dim,
            linear=linear,
            scaler=scaler,
            oracle=oracle,
            batch_size=batch_size,
            optimizer=optimizer,
            l2_lamb=l2_lamb,
            lr=lr,
            use_validation=use_validation,
            **kwargs
        )
        assert 'sample_per_day' in kwargs, "`sample_per_day' should be passed..."
        self.n_items = kwargs['sample_per_day']
        self.oracle.model.Params.OutputFlag = 0
        self._dom = 'energy'

    def fit(self, epochs, **kwargs):
        batch_size = self.batch_size
        num_batch = self.n_train // batch_size
        idx_lst = list(range(self.n_train))

        for e in range(epochs):
            self._rng.shuffle(idx_lst)
            for i in range(num_batch):
                self.optimizer.zero_grad()
                X, c_true = self.X_train[idx_lst[i: i+batch_size]], self.y_train[idx_lst[i: i + batch_size]]
                w_true = self.optimal_solution[idx_lst[i: i+batch_size]]

                for j in range(len(X)):
                    X_j = torch.tensor(X[j, :], dtype=torch.float).squeeze(0)
                    c_true_j = c_true[j, :]
                    w_true_j = w_true[j]

                    c_pred_tensor = self.forward(X_j).squeeze()
                    c_hat_j = c_pred_tensor.detach().numpy()
                    c_spo = (2 * c_hat_j - c_true_j)

                    w_spo, _ = self.oracle.solve_model(c_spo)

                    # Compute the subgradient (note: backward method takes this as argument, computing the necessary
                    # dot product
                    grad = torch.from_numpy(w_true_j - w_spo).float()
                    grad = grad.squeeze()
                    c_pred_tensor.backward(gradient=grad)

                self.optimizer.step()

            msg = f"Epoch: {e + 1}/{epochs}".center(25) + "|"
            if self.use_validation:
                actual_cost, opt_cost, regret = self.test_model(self.X_val, self.y_val, dom=self._dom)
                msg += f'Cost: {actual_cost:.10f}'.ljust(25) + f'Optimal: {opt_cost:.10f}'.ljust(25) + \
                       f'Regret: {regret:.10f}'.ljust(25)
            else:
                actual_cost, opt_cost, regret = self.test_model(self.X_train, self.y_train, dom=self._dom)
                msg += f'Cost: {actual_cost:.10f}'.ljust(25) + f'Optimal: {opt_cost:.10f}'.ljust(25) + \
                       f'Regret: {regret:.10f}'.ljust(25)
                # Update the best objective
                self._optimal_obj = opt_cost
                if actual_cost < self._best_obj:
                    self._best_obj = actual_cost
                    self._best_param = self.save_params(self._fname_param)
                    self._best_epoch = e

            if epochs >= 10 and e % (epochs // 10) == 0:
                logger.info(msg)


class SPOPlusClassify(SPOPlus):
    def __init__(
            self,
            target_dim: int,
            input_dim: int,
            dataset: dict,
            embed_dim: List[int],
            linear: bool = True,
            scaler: Optional = None,
            oracle: Oracle = None,
            batch_size=10,
            optimizer=optim.Adam,
            l2_lamb=0,
            lr=0.001,
            use_validation=False,
            **kwargs
    ):
        super().__init__(
            target_dim,
            input_dim,
            dataset,
            embed_dim,
            linear=linear,
            scaler=scaler,
            oracle=oracle,
            batch_size=batch_size,
            optimizer=optimizer,
            l2_lamb=l2_lamb,
            lr=lr,
            use_validation=use_validation,
            **kwargs
        )
        self._dom = 'classification'

    def fit(self, epochs, **kwargs):
        batch_size = self.batch_size
        num_batch = self.n_train // batch_size
        idx_lst = list(range(self.n_train))

        for e in range(epochs):
            self._rng.shuffle(idx_lst)
            for i in range(num_batch):
                self.optimizer.zero_grad()
                X, c_true = self.X_train[idx_lst[i: i + batch_size]], self.y_train[idx_lst[i: i + batch_size]]
                w_true = self.optimal_solution[idx_lst[i: i + batch_size]]

                for j in range(len(X)):
                    X_j = torch.tensor(X[j, :], dtype=torch.float).squeeze(0)
                    c_true_j = c_true[j, :]
                    w_true_j = w_true[j]

                    c_pred_tensor = self.forward(X_j).squeeze()
                    c_hat_j = c_pred_tensor.detach().numpy()
                    c_spo = (2 * c_hat_j - c_true_j)

                    _, w_spo = self.oracle(c_spo)

                    # Compute the subgradient (note: backward method takes this as argument, computing the necessary
                    # dot product
                    grad = torch.from_numpy(w_true_j - w_spo).float()
                    grad = grad.squeeze()
                    c_pred_tensor.backward(gradient=grad)

                self.optimizer.step()

            msg = f"Epoch: {e + 1}/{epochs}".center(25) + "|"
            if self.use_validation:
                actual_cost, opt_cost, regret = self.test_model(self.X_val, self.y_val, dom=self._dom)
                msg += f'Cost: {actual_cost:.10f}'.ljust(25) + f'Optimal: {opt_cost:.10f}'.ljust(25) + \
                       f'Regret: {regret:.10f}'.ljust(25)
            else:
                actual_cost, opt_cost, regret = self.test_model(self.X_train, self.y_train, dom=self._dom)
                msg += f'Cost: {actual_cost:.10f}'.ljust(25) + f'Optimal: {opt_cost:.10f}'.ljust(25) + \
                       f'Regret: {regret:.10f}'.ljust(25)
                # Update the best objective
                self._optimal_obj = opt_cost
                if actual_cost < self._best_obj:
                    self._best_obj = actual_cost
                    self._best_param = self.save_params(self._fname_param)
                    self._best_epoch = e

            if epochs >= 10 and e % (epochs // 10) == 0:
                logger.info(msg)