"""
Interior-Point Optimization for Predict+Optimize by Mandi and Guns (2020)
This code adapts the implementation in https://github.com/JayMan91/NeurIPSIntopt/.
"""

import torch.optim as optim
import numpy as np
import torch
import copy
from typing import Optional, List
from scipy.linalg import LinAlgError

from xaddpy.models.util import qptl as qp
import xaddpy.models.util.intopt as ip_model
from xaddpy.models.base import Base, Oracle
import xaddpy.models.util.util as util
from xaddpy.models.util.linalg import _remove_redundant_rows
from xaddpy.experiments.predopt.energy_scheduling import util as energy_util
from xaddpy.utils.logger import logger

import warnings
warnings.filterwarnings("ignore")


class IntOpt(Base):
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

        assert 'method' in kwargs, "'method' parameter should be provided (IntOpt)"
        assert 'max_iter' in kwargs, "'max_iter' parameter should be provided (IntOpt)"
        assert 'smoothing' in kwargs, "'smoothing' parameter should be provided (IntOpt)"
        assert 'damping' in kwargs, "'damping' parameter should be provided (IntOpt)"
        assert 'thr' in kwargs, "'thr' parameter should be provided (IntOpt)"
        assert 'mu0' in kwargs, "'mu0' parameter should be provided (IntOpt)"
        self.method = kwargs['method']
        self.max_iter = kwargs['max_iter']
        self.smoothing = kwargs['smoothing']
        self.damping = kwargs['damping']
        self.thr = kwargs['thr']
        self.mu0 = kwargs['mu0']

        self.optimizer = optimizer(self.parameters(), lr=self.lr)
        self.oracle = oracle

    def fit(self, *args, **kwargs):
        raise NotImplementedError


class IntOptShortest(IntOpt):
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
            full_row_rank=True,
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
        assert 'timeout_iter' in kwargs, "Timeout per iteration should be properly set (IntOpt)"
        self.timeout_iter = kwargs['timeout_iter']
        self.full_row_rank = full_row_rank
        self._dom = 'shortest'

    def fit(self, epochs, **kwargs):
        assert 'A' in kwargs, "Node-Edge incidence matrix should be given"

        idx_lst = list(range(self.n_train))
        num_data = self.n_train
        target_dim = self.target_dim

        A = kwargs['A']

        # We only consider the path from south-west corner to the north-east corner
        src, dst = 0, A.shape[0] - 1
        b = np.zeros(A.shape[0])
        b[src] = 1
        b[dst] = -1

        if self.full_row_rank:
            rows_to_remove = _remove_redundant_rows(A)
            A = torch.from_numpy(np.delete(A, rows_to_remove, axis=0)).float()
            b = torch.from_numpy(np.delete(b, rows_to_remove)).float()
        else:
            A = torch.from_numpy(A).float()
            b = torch.from_numpy(b).float()

        for e in range(epochs):
            self._rng.shuffle(idx_lst)

            # For each data instance
            for i in range(self.n_train):
                X, c_true = self.X_train[idx_lst[i]], self.y_train[idx_lst[i]]
                X = torch.tensor(X, dtype=torch.float)
                c_true_tensor = torch.from_numpy(c_true).float()
                self.optimizer.zero_grad()

                c_pred = self(X)
                try:
                    with util.time_limit(self.timeout_iter):
                        func = ip_model.IPOfunc(A, b, torch.Tensor(), torch.Tensor(),
                                                pc=True, max_iter=self.max_iter,
                                                bounds=[(0, None)], smoothing=self.smoothing,
                                                thr=self.thr, method=self.method,
                                                mu0=self.mu0, damping=self.damping)
                        x = func(c_pred)
                        loss = (c_true_tensor * x).mean()
                        c_pred.retain_grad()
                        loss.backward()
                    forward_solved = ip_model.IPOfunc.forward_solved()
                except util.TimeoutException as msg:
                    forward_solved = False
                    # logger.info(f'Epoch[{e + 1}::{i + 1}] timeout occurred')
                except LinAlgError as msg:
                    raise Exception
                except Exception as msg:
                    forward_solved = False

                if not forward_solved:
                    pass
                    # logger.info(f"Epoch[{e + 1}/{i + 1}] fwd pass not solved")

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


class IntOptEnergy(IntOpt):
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
        assert 'prob_configs' in kwargs, "'prob_configs' should be passed..."
        assert 'timeout_iter' in kwargs, "Timeout per iteration should be properly set (IntOpt)"
        self.timeout_iter = kwargs['timeout_iter']
        self.n_items = kwargs['sample_per_day']
        self.prob_configs = kwargs['prob_configs']
        self._dom = 'energy'

    def fit(self, epochs, **kwargs):

        batch_size = self.batch_size
        num_batch = self.n_train // batch_size
        idx_lst = list(range(self.n_train))

        A, b, G, h, F = energy_util.set_up_milp_matrix(self.prob_configs, 'intopt')

        for e in range(epochs):
            self._rng.shuffle(idx_lst)
            for i in range(num_batch):
                self.optimizer.zero_grad()
                X, c_true = self.X_train[idx_lst[i: i+batch_size]], self.y_train[idx_lst[i: i + batch_size]]

                for j in range(len(X)):

                    X_j = torch.tensor(X[j, :], dtype=torch.float).squeeze(0)
                    c_true_j = c_true[j, :]
                    c_true_j = torch.mm(F, torch.tensor(c_true_j, dtype=torch.float).unsqueeze(1)).squeeze()
                    c_pred_j = torch.mm(F, self(X_j)).squeeze()

                    try:
                        with util.time_limit(self.timeout_iter):
                            func = ip_model.IPOfunc(A, b, G, h,
                                                    pc=True, max_iter=self.max_iter,
                                                    bounds=[(0., None)], smoothing=self.smoothing,
                                                    thr=self.thr, method=self.method,
                                                    mu0=self.mu0, damping=self.damping)
                            x = func(c_pred_j)
                            loss = (x * c_true_j).mean()
                            c_pred_j.retain_grad()
                            loss.backward()
                        forward_solved = ip_model.IPOfunc.forward_solved()
                    except util.TimeoutException as msg:
                        forward_solved = False
                        # logging.info("Timeout occurred")
                        # logger.info(f'Epoch[{e + 1}::{i + 1}] timeout occurred')
                    except LinAlgError as msg:
                        raise Exception
                    except Exception as msg:
                        forward_solved = False

                    if forward_solved:
                        # logging.info("backward done {} {} {}".format(e, i, j))
                        pass
                    else:
                        pass
                        # logger.info(f"Epoch[{e + 1}/{i + 1}] fwd pass not solved")

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


class IntOptClassify(IntOpt):
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
        assert 'timeout_iter' in kwargs, "Timeout per iteration should be properly set (IntOpt)"
        self.timeout_iter = kwargs['timeout_iter']
        self._dom = 'classification'

    def fit(self, epochs, **kwargs):
        idx_lst = list(range(self.n_train))

        A = np.ones(self.target_dim).reshape(1, -1)
        b = np.array([[1]])

        A = torch.from_numpy(A).float()
        b = torch.from_numpy(b).float()

        for e in range(epochs):
            self._rng.shuffle(idx_lst)

            # For each data instance
            for i in range(self.n_train):
                X, c_true = self.X_train[idx_lst[i]], self.y_train[idx_lst[i]]
                X = torch.tensor(X, dtype=torch.float)
                c_true_tensor = torch.from_numpy(c_true).float()
                self.optimizer.zero_grad()

                c_pred = self(X)
                try:
                    with util.time_limit(self.timeout_iter):
                        func = ip_model.IPOfunc(A, b, torch.Tensor(), torch.Tensor(),
                                                pc=True, max_iter=self.max_iter,
                                                bounds=[(0, None)], smoothing=self.smoothing,
                                                thr=self.thr, method=self.method,
                                                mu0=self.mu0, damping=self.damping)
                        x = func(c_pred)
                        loss = (c_true_tensor * x).mean()
                        c_pred.retain_grad()
                        loss.backward()
                    forward_solved = ip_model.IPOfunc.forward_solved()
                except util.TimeoutException as msg:
                    forward_solved = False
                    # logger.info(f'Epoch[{e + 1}::{i + 1}] timeout occurred')
                except LinAlgError as msg:
                    raise Exception
                except Exception as msg:
                    forward_solved = False

                if not forward_solved:
                    pass
                    # logger.info(f"Epoch[{e + 1}/{i + 1}] fwd pass not solved")

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