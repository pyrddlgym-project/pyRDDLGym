import torch.optim as optim
import numpy as np
import torch
import copy
from typing import Optional, List

from xaddpy.models.util import qptl as qp
import xaddpy.models.util.util as util
from xaddpy.models.base import Base, Oracle
from xaddpy.experiments.predopt.energy_scheduling import util as energy_util
from xaddpy.utils.logger import logger


class QPTL(Base):
    """
    QPTL (quadratic programming task loss) by Wilder et al. (2019).
    This code adapts the implementation in https://github.com/JayMan91/NeurIPSIntopt/.
    """
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
            tau=100000,
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
        self.tau = tau
        self.optimizer = optimizer(self.parameters(), lr=self.lr)

        # Oracle
        self.oracle = oracle

    def fit(self, *args, **kwargs):
        raise NotImplementedError


class QPTLShortest(QPTL):
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
        assert 'timeout_iter' in kwargs, "Timeout per iteration should be properly set (QPTL)"
        self.timeout_iter = kwargs['timeout_iter']
        self._dom = 'shortest'

    def fit(self, epochs, **kwargs):
        assert 'A' in kwargs, "Node-Edge incidence matrix should be given"

        idx_lst = list(range(self.n_train))
        num_data = self.n_train
        target_dim = self.target_dim

        A = torch.from_numpy(kwargs['A']).float()
        Q = torch.eye(target_dim) / self.tau

        G = torch.cat((-torch.eye(target_dim), torch.eye(target_dim)))
        h = torch.cat((torch.zeros(target_dim), torch.ones(target_dim)))

        # We only consider the path from south-west corner to the north-east corner
        src, dst = 0, A.shape[0] - 1
        b = torch.zeros(A.shape[0])
        b[src] = 1
        b[dst] = -1

        model_params = qp.make_gurobi_model(G.detach().numpy(),
                                            h.detach().numpy(),
                                            A.detach().numpy(),
                                            b.detach().numpy(),
                                            Q.detach().numpy())

        for e in range(epochs):
            self._rng.shuffle(idx_lst)

            # For each data instance
            for i in range(num_data):
                self.optimizer.zero_grad()

                X, c_true = self.X_train[idx_lst[i]], self.y_train[idx_lst[i]]
                c_true_tensor = torch.from_numpy(c_true).float()

                X_tensor = torch.tensor(X, dtype=torch.float)
                c_pred = self(X_tensor)

                if any(torch.isnan(torch.flatten(c_pred)).tolist()):
                    logger.warning("Alert! nan in param c_pred")
                if any(torch.isinf(torch.flatten(c_pred)).tolist()):
                    logger.warning("Alert! inf in param c_pred")

                func = qp.QPFunction(verbose=False, solver=qp.QPSolvers.GUROBI,
                                     model_params=model_params)
                x = func(Q.expand(1, *Q.shape), c_pred.squeeze(), G.expand(1, *G.shape),
                         h.expand(1, *h.shape), A.expand(1, *A.shape), b.expand(1, *b.shape))

                c_pred.retain_grad()
                loss = (c_true_tensor * x).mean()
                loss.backward()
                c_grad = copy.deepcopy(c_pred.grad)
                if any(torch.isnan(torch.flatten(c_grad)).tolist()):
                    print("Alert: nan in param c_grad")

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


class QPTLEnergy(QPTL):
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
        assert 'timeout_iter' in kwargs, "Timeout per iteration should be properly set (QPTL)"
        self.n_items = kwargs['sample_per_day']
        self.prob_configs = kwargs['prob_configs']
        self.timeout_iter = kwargs['timeout_iter']
        self._dom = 'energy'

    def fit(self, epochs, **kwargs):

        batch_size = self.batch_size
        num_batch = self.n_train // batch_size
        idx_lst = list(range(self.n_train))

        A, b, G, h, F = energy_util.set_up_milp_matrix(self.prob_configs, 'qptl')
        Q = torch.eye(F.shape[0]) / self.tau
        model_params = qp.make_gurobi_model(G.detach().numpy(),
                                            h.detach().numpy(),
                                            A.detach().numpy(),
                                            b.detach().numpy(),
                                            Q.detach().numpy())

        for e in range(epochs):
            self._rng.shuffle(idx_lst)
            for i in range(num_batch):
                self.optimizer.zero_grad()
                X, c_true = self.X_train[idx_lst[i: i + batch_size]], self.y_train[idx_lst[i: i + batch_size]]

                for j in range(len(X)):
                    X_j = torch.tensor(X[j, :], dtype=torch.float).squeeze(0)
                    c_true_j = c_true[j, :]
                    c_true_j = torch.mm(F, torch.tensor(c_true_j, dtype=torch.float).unsqueeze(1)).squeeze()
                    c_pred_j = torch.mm(F, self(X_j)).squeeze()
                    try:
                        with util.time_limit(self.timeout_iter):
                            solver = qp.QPFunction(verbose=False, solver=qp.QPSolvers.GUROBI, model_params=model_params)
                            x = solver(Q.expand(1, *Q.shape), c_pred_j.squeeze(), G.expand(1, *G.shape),
                                       h.expand(1, *h.shape), A.expand(1, *A.shape), b.expand(1, *b.shape))
                        forward_solved = True
                    except util.TimeoutException as msg:
                        forward_solved = False
                        # logging.info("Timeout occurred")
                        # logger.info(f'Epoch[{e+1}::{i+1}] timeout occurred')
                    except Exception as msg:
                        forward_solved = False
                        logger.error(msg)

                    if forward_solved:
                        loss = (x.squeeze() * c_true_j.squeeze()).mean()
                        loss.backward()
                    else:
                        pass
                        # logger.info(f"Epoch[{e+1}/{i+1}] fwd pass not solved")

                self.optimizer.step()

                # if forward_solved:
                #     logger.info(f'Epoch[{e+1}/{i+1}], loss(train):{loss.item():.2f}')

                # if ((i+1) % 7 == 0) | ((i+1) % num_batch == 0):
                #     if self.model_save:
                #         torch.save(self.model.state_dict(), f'{self.model_name}_Epoch{e}_{i}.pth')

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


class QPTLClassify(QPTL):
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
        assert 'timeout_iter' in kwargs, "Timeout per iteration should be properly set (QPTL)"
        self.timeout_iter = kwargs['timeout_iter']
        self._dom = 'classification'

    def fit(self, epochs, **kwargs):
        idx_lst = list(range(self.n_train))

        A = torch.ones(self.target_dim, dtype=torch.float).view(1, -1)
        b = torch.tensor([1], dtype=torch.float)
        Q = torch.eye(self.target_dim) / self.tau
        # A = torch.from_numpy(A).float()
        # b = torch.from_numpy(b).float()

        G = -torch.eye(self.target_dim) #torch.cat((-torch.eye(self.target_dim), torch.eye(self.target_dim)))
        h = torch.zeros(self.target_dim) #torch.cat((torch.zeros(self.target_dim), torch.ones(self.target_dim)))

        model_params = qp.make_gurobi_model(G.detach().numpy(),
                                            h.detach().numpy(),
                                            A.detach().numpy(),
                                            b.detach().numpy(),
                                            Q.detach().numpy())

        for e in range(epochs):
            self._rng.shuffle(idx_lst)

            # For each data instance
            for i in range(self.n_train):
                self.optimizer.zero_grad()

                X, c_true = self.X_train[idx_lst[i]], self.y_train[idx_lst[i]]
                c_true_tensor = torch.from_numpy(c_true).float()

                X_tensor = torch.tensor(X, dtype=torch.float)
                c_pred = self(X_tensor)

                if any(torch.isnan(torch.flatten(c_pred)).tolist()):
                    logger.warning("Alert! nan in param c_pred")
                if any(torch.isinf(torch.flatten(c_pred)).tolist()):
                    logger.warning("Alert! inf in param c_pred")

                func = qp.QPFunction(verbose=False, solver=qp.QPSolvers.GUROBI,
                                     model_params=model_params)
                x = func(Q.expand(1, *Q.shape), c_pred.squeeze(), G.expand(1, *G.shape),
                         h.expand(1, *h.shape), A.expand(1, *A.shape), b.expand(1, *b.shape))

                c_pred.retain_grad()
                loss = (c_true_tensor * x).mean()
                loss.backward()
                c_grad = copy.deepcopy(c_pred.grad)
                if any(torch.isnan(torch.flatten(c_grad)).tolist()):
                    print("Alert: nan in param c_grad")

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