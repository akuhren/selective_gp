#!/usr/bin/env python

from os.path import join as pjoin
from datetime import datetime
import inspect
import re
from tqdm.auto import tqdm
import sys

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn


class TorchScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("means", torch.tensor(0.))
        self.register_buffer("stds", torch.tensor(1.))

    def fit(self, X):
        self.means = X.mean(axis=0)
        self.stds = X.std(axis=0)
        self.stds[self.stds == 0] = 1.0

    def transform(self, X):
        return (X - self.means) / self.stds

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.Tensor(X)

        return X * self.stds + self.means

    def forward(self, X):
        return self.transform(X)


class BaseModel(nn.Module):
    """
    Module for handling generic tasks, such as fitting parameters and logging
    diagnostics. Interesting stuff happens in subclasses of this one.
    """
    def __init__(self):
        super().__init__()

        self.callbacks = {}
        self.stop_conditions = {}

        self._fixed_parameters = set()
        self._free_parameters = set()

        self.register_buffer("epoch", torch.tensor(0))

        self.writer = None

    def create_writer(self, log_dir="./tensorboard_logdir", name=None):
        if name is None:
            name = datetime.now().strftime("%b_%d_%Y_%H_%M_%S")
        self.writer = SummaryWriter(log_dir=pjoin(log_dir, name))

    def _log_lik(self, **observed_values):
        """
        Log likelihood to be optimized

        Keyword arguments should be name-value pairs of observed values to
        evaluate over. By default we assume that the input is named X and
        output is named Y (so should be called with `X=X, Y=Y`).
        """
        raise NotImplementedError

    def _KL(self):
        """
        Kullback-Leibler divergence, acting as regularisation term.
        """
        raise NotImplementedError

    def _log_prior(self):
        """
        Optional prior term.
        """
        return torch.tensor(0., device=self.device, dtype=self.dtype)

    def _get_optimizer(self, optimizer_arg, var_learning_rate,
                       hp_learning_rate):
        """
        Instantiate optimizer with different learning rates for variational
        and hyper-parameters.
        """
        if (
            inspect.isclass(optimizer_arg) and
            issubclass(optimizer_arg, torch.optim.Optimizer)
        ):
            optimizer_clz = optimizer_arg
        elif isinstance(optimizer_arg, str):
            clz_dict = {
                "adam": torch.optim.Adam,
                "adamax": torch.optim.Adamax,
                "sgd": torch.optim.SGD,
                "asgd": torch.optim.ASGD,
            }
            try:
                optimizer_clz = clz_dict[optimizer_arg.lower()]
            except KeyError:
                optimizers_str = ", ".join(clz_dict.keys())
                raise Exception(f"No such optimizer: '{optimizer_arg}'. "
                                f"Available optimizers: {optimizers_str}")
        else:
            raise Exception("Invalid optimizer argument.")

        def _match_free(p):
            if len(self._free_parameters) == 0:
                return True
            return any(p_ in p for p_ in self._free_parameters)

        def _match_fixed(p):
            return any(p_ in p for p_ in self._fixed_parameters)

        named_params = {n: p for n, p in self.named_parameters()
                        if _match_free(n) and not _match_fixed(n)}

        param_group_dict = {}
        for n, p in named_params.items():
            if re.match(r".*\.variational_mean.?$", n):
                pg = "means"
                lr = var_learning_rate
            elif re.match(r".*\.chol_variational_covar.?$", n):
                pg = "covars"
                lr = var_learning_rate
            else:
                pg = "hyper"
                lr = hp_learning_rate
            param_group = param_group_dict.setdefault(pg, {
                "params": [], "type": pg, "lr": lr})
            param_group["params"].append(p)

        param_groups = [pg for pg in param_group_dict.values()
                        if pg["params"] != [] and pg["lr"] > 0]
        return optimizer_clz(param_groups)

    def add_free_parameter(self, parameter_name):
        assert any(parameter_name in n for n, _ in self.named_parameters())
        self._free_parameters.add(parameter_name)

    def remove_free_parameter(self, parameter_name):
        self._free_parameters.remove(parameter_name)

    def add_fixed_parameter(self, parameter_name):
        assert any(parameter_name in n for n, _ in self.named_parameters())
        self._fixed_parameters.add(parameter_name)

    def remove_fixed_parameter(self, parameter_name):
        self._fixed_parameters.remove(parameter_name)

    def _prepare_epoch(self):
        pass

    def _finalize_epoch(self):
        pass

    def fit_scaler(self, name, tensor):
        scaler_name = name + "_scaler"
        scaler = getattr(self, scaler_name)
        return scaler.fit(tensor)

    def _transform(self, name, tensor):
        scaler_name = name + "_scaler"
        scaler = getattr(self, scaler_name)
        return scaler.transform(tensor)

    def _fit_transform(self, name, tensor):
        scaler_name = name + "_scaler"
        scaler = getattr(self, scaler_name)
        return scaler.fit_transform(tensor)

    def _inverse_transform(self, name, tensor):
        scaler_name = name + "_scaler"
        scaler = getattr(self, scaler_name)
        return scaler.inverse_transform(tensor)

    def fit(self, max_epochs=500, batch_size=None, verbose=True,
            optimizer="Adam", var_learning_rate=0.02, hp_learning_rate=0.02,
            n_mcmc_samples=1, **observed_values):
        if max_epochs == 0:
            return

        # Collect all observed data into a dictionary
        observed_scaled = {}
        for name, tensor in observed_values.items():
            if not torch.is_tensor(tensor):
                raise Exception("Unknown keyword argument: " + name)

            tensor = tensor.to(device=self.device, dtype=self.dtype)
            tensor = self._transform(name, tensor)
            observed_scaled[name] = tensor

        names, tensors = zip(*observed_scaled.items())
        n = len(tensors[0])

        # DataLoader slows iterations down quite a bit so we ignore it when
        # we are not batching
        if batch_size is None:
            data_loader = [tensors]
        else:
            dataset = TensorDataset(*tensors)
            data_loader = DataLoader(dataset, batch_size=batch_size,
                                     shuffle=True)

        self.optimizer = self._get_optimizer(
            optimizer, var_learning_rate, hp_learning_rate)

        if verbose:
            progress_bar = tqdm(total=max_epochs)

        was_training = self.training
        self.train()
        for _ in range(max_epochs):
            acc_log_lik = 0.0
            acc_log_prior = 0.0
            acc_KL = 0.0
            acc_loss = 0.0

            for value_batch in data_loader:
                data_dict = dict(zip(names, value_batch))

                self.optimizer.zero_grad()

                batch_ratio = len(value_batch[0]) / n
                log_liks = torch.zeros(n_mcmc_samples, device=self.device,
                                       dtype=self.dtype)

                # Get MCMC samples for log likelihood
                for i in range(n_mcmc_samples):
                    self._prepare_epoch()
                    log_liks[i] = self._log_lik(**data_dict)

                # Get loss terms
                log_lik = log_liks.mean()
                KL = self._KL() * batch_ratio
                log_prior = self._log_prior() * batch_ratio

                # Take gradient step
                loss = -(log_lik + log_prior - KL)
                loss.backward()
                self.optimizer.step()

                # Bookkeeping
                self._finalize_epoch()
                acc_log_lik += log_lik.item()
                acc_log_prior += log_prior.item()
                acc_KL += KL.item()
                acc_loss = loss.item()

            self.write_to_tensorboard({
                "model/log_lik": acc_log_lik,
                "model/log_prior": acc_log_prior,
                "model/KL": acc_KL,
                "model/loss": acc_loss
            })

            self.epoch += 1

            if verbose:
                progress_bar.set_postfix({"ELBO": -acc_loss})
                progress_bar.update()

            # Callbacks
            for cb, (args, kwargs, update_interval) in self.callbacks.items():
                if self.epoch % update_interval == 0:
                    cb(*args, **kwargs)

            # Stop conditions
            stop = False
            for sc, (args, kwargs) in self.stop_conditions.items():
                if sc(*args, **kwargs):
                    stop = True
                    break
            if stop:
                break

        if verbose:
            progress_bar.refresh()
            progress_bar.close()
            sys.stderr.flush()

        if not was_training:
            self.eval()

    def write_to_tensorboard(self, named_values):
        if self.writer is None:
            return

        for n, v in named_values.items():
            self.writer.add_scalar(n, v, global_step=self.epoch)
        self.writer.flush()

    @property
    def n_parameters(self):
        return sum([p.numel() for p in self.parameters()])

    def register_callback(self, method, args=(), kwargs={},
                          update_interval=20):
        self.callbacks[method] = (args, kwargs, update_interval)

    def unregister_callback(self, method):
        del self.callbacks[method]

    def register_stop_condition(self, method, args=(), kwargs={}):
        self.stop_conditions[method] = (args, kwargs)

    def unregister_stop_condition(self, method):
        del self.stop_conditions[method]

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

    def predict(self, X, return_std=False, return_covariance=False):
        raise NotImplementedError

    def get_predictive_distribution(self, X):
        raise NotImplementedError
