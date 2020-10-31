#!/usr/bin/env python

import time
import os

import torch
from torch.utils.data import TensorDataset, DataLoader

import mlflow
from mlflow.entities import ViewType

from .print_helpers import eprint, bold


def get_experiment_id(experiment_name):
    try:
        exp_id = mlflow.create_experiment(experiment_name)
        eprint(f"Created new experiment {bold(experiment_name)}")
    except mlflow.exceptions.MlflowException:
        mlflow.set_experiment(experiment_name)
        exp = mlflow.get_experiment_by_name(experiment_name)
        exp_id = exp.experiment_id
    os.environ["EXPERIMENT_ID"] = exp_id
    return exp_id


def run_exists(params, tags={}):
    return (get_run_info(params, tags=tags) is not None)


def get_run_info(params, tags={}):
    exp_id = os.environ["EXPERIMENT_ID"]

    filter_atoms = ([f"tags.{k} = '{v}'" for k, v in tags.items()] +
                    [f"params.{k} = '{v}'" for k, v in params.items()])
    filter_string = " AND ".join(filter_atoms)
    runs = mlflow.search_runs(
        experiment_ids=[exp_id], run_view_type=ViewType.ACTIVE_ONLY,
        filter_string=filter_string)

    # Filter out failed runs
    runs = runs[runs["status"].values.astype(str) == "FINISHED"]

    if len(runs) == 0:
        return
    elif len(runs) > 1:
        eprint("Warning: Multiple runs found for filter string:\n" +
               filter_string)
    return runs.iloc[0]


def get_ELBO(model, dataset, batch_size=None, reps=50):
    def _get_batch_log_lik(X_batch, Y_batch):
        with torch.no_grad():
            l = [model._log_lik(X=X_batch, Y=Y_batch) for _ in range(reps)]
        return torch.stack(l).mean()

    Y = model.Y_scaler.transform(dataset.Y_train)

    if model.latent_layer is not None:
        # If we are evaluating a GPLVM, we need to call with indices
        X = torch.arange(len(Y))
    else:
        X = model.X_scaler.transform(dataset.X_train)

    if batch_size is None:
        log_lik = _get_batch_log_lik(X, Y)
    else:
        XY = TensorDataset(X, Y)
        data_loader = DataLoader(XY, batch_size=batch_size)
        l = [_get_batch_log_lik(X_, Y_) for X_, Y_ in data_loader]
        log_lik = torch.stack(l).sum()

    with torch.no_grad():
        return log_lik.item(), model._KL().item()


def _get_loglik(model, X, Y, batch_size=None, reps=1):
    if len(X) == 0:
        return 0.0

    def _get_batch_log_lik(X_batch, Y_batch):
        log_liks = X_batch.new_zeros(reps)
        for i in range(reps):
            Y_dist = model.get_predictive_distribution(X_batch)
            Y_reshaped = Y_batch.reshape(*Y_dist.mean.shape)
            log_liks[i] = torch.sum(Y_dist.log_prob(Y_reshaped))
        return log_liks.mean().item()

    if batch_size is None:
        log_lik = _get_batch_log_lik(X, Y)
    else:
        XY = TensorDataset(X, Y)
        data_loader = DataLoader(XY, batch_size=batch_size)
        log_lik = sum(_get_batch_log_lik(X_, Y_) for X_, Y_ in data_loader)
    return log_lik / len(X)


def get_loglik(model, dataset, train=True, test=True, batch_size=None, reps=1):
    assert train or test
    XYs = []
    if train:
        XYs.append((dataset.X_train, dataset.Y_train))
    if test:
        XYs.append((dataset.X_test, dataset.Y_test))

    return [_get_loglik(model, X, Y, batch_size, reps=reps) for X, Y in XYs]


def _get_accuracy(model, X, Y, batch_size, reps=1):
    if len(X) == 0:
        return X.new_zeros([])

    def _get_batch_acc(X_batch, Y_batch):
        accs = X_batch.new_zeros(reps)
        for i in range(reps):
            Y_pred = model.get_predictive_distribution(X_batch).sample()
            Y_reshaped = Y_batch.reshape(*Y_pred.shape)
            accs[i] = (Y_pred == Y_reshaped).double().mean()
        return accs.mean().item()

    if batch_size is None:
        acc = _get_batch_acc(X, Y)
    else:
        XY = TensorDataset(X, Y)
        data_loader = DataLoader(XY, batch_size=batch_size)
        acc = (sum(_get_batch_acc(X_, Y_) for X_, Y_ in data_loader) /
               len(data_loader))
    return acc


def get_accuracy(model, dataset, train=True, test=True, batch_size=None,
                 reps=1):
    assert train or test
    XYs = []
    if train:
        XYs.append((dataset.X_train, dataset.Y_train))
    if test:
        XYs.append((dataset.X_test, dataset.Y_test))

    return [_get_accuracy(model, X, Y, batch_size, reps=reps) for X, Y in XYs]


def _get_RMSE(model, X, Y, batch_size, reps=1):
    if len(X) == 0:
        return X.new_zeros([])

    def _get_batch_RMSE(X_batch, Y_batch):
        rmses = X_batch.new_zeros(reps)
        for i in range(reps):
            Y_pred = model.get_predictive_distribution(X_batch).mean
            Y_reshaped = Y_batch.reshape(*Y_pred.shape)
            rmses[i] = ((Y_pred - Y_reshaped)**2).mean().sqrt()
        return rmses.mean().item()

    if batch_size is None:
        rmse = _get_batch_RMSE(X, Y)
    else:
        XY = TensorDataset(X, Y)
        data_loader = DataLoader(XY, batch_size=batch_size)
        rmse = sum((_get_batch_RMSE(X_, Y_) for X_, Y_ in data_loader) /
                   len(data_loader))
    return rmse


def get_RMSE(model, dataset, train=True, test=True, batch_size=None,
             reps=1):
    assert train or test
    XYs = []
    if train:
        XYs.append((dataset.X_train, dataset.Y_train))
    if test:
        XYs.append((dataset.X_test, dataset.Y_test))

    return [_get_RMSE(model, X, Y, batch_size, reps=reps) for X, Y in XYs]


def get_prediction_times(model, dataset, max_obs=1000):
    if model.latent_layer is not None:
        # If we are evaluating a GPLVM, we need to call with indices
        X = torch.arange(len(dataset.X_train))
    else:
        X = model.X_scaler.transform(dataset.X_train)

    X = X[:max_obs, None]
    N = len(X)

    tic_clock = time.perf_counter()
    tic_wall = time.process_time()

    for x in X:
        model.get_predictive_distribution(x)

    toc_clock = time.perf_counter()
    toc_wall = time.process_time()

    return (toc_clock - tic_clock) / N, (toc_wall - tic_wall) / N,
