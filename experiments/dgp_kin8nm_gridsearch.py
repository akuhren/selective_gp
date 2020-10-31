#!/usr/bin/env python

import itertools
import numpy as np
import mlflow

from selective_gp.utils import (
    load_data, get_model, get_ELBO, get_loglik, get_experiment_id,
    run_exists, eprint, bold, green, fit_layerwise,
    get_prediction_times)

import click


def run_single(M1, M2, fold, n_folds, device):
    test_size = 1 / n_folds

    # Get dataset and model
    dataset = load_data("uci_kin8nm", seed=fold, device=device,
                        test_size=test_size)
    model = get_model(
        dataset, n_inducing=[M1, M2], n_layers=2, device=device,
        add_input=True)

    # Create callback for logging status to tracking server
    def status_cb():
        mlflow.log_metric("current_epoch", model.epoch.item())
    model.register_callback(status_cb, update_interval=10)

    # Fit model
    eprint(bold("Layerwise fitting"))
    fit_layerwise(model, dataset, batch_size=4096, max_epochs=500)

    eprint(bold("\nJoint fitting"))
    model.fit(X=dataset.X_train, Y=dataset.Y_train, batch_size=4096,
              max_epochs=3000)

    # Log metrics
    eprint(bold("\nEvaluating metrics"))
    model.eval()
    log_lik, KL = get_ELBO(model, dataset, batch_size=4096)
    clock_time, wall_time = get_prediction_times(model, dataset)
    train_log_lik, test_log_lik = get_loglik(
        model, dataset, train=True, test=True, batch_size=4096)

    mlflow.log_metrics({
        "log_lik": log_lik,
        "KL": KL,
        "ELBO": train_log_lik - KL,
        "clock_time": clock_time,
        "wall_time": wall_time,
        "train_log_lik": train_log_lik,
        "test_log_lik": test_log_lik,
    })

    eprint()


@click.command()
@click.option("--device", type=click.Choice(["cpu", "cuda"]), default="cpu")
@click.option("--grid-range", nargs=3, default=(10, 101, 10))
@click.option("--index-range", nargs=2, default=None)
@click.option("--n-folds", type=int, default=5)
def run_all(grid_range, index_range, device, n_folds):
    grid_range = np.arange(*grid_range)
    K = len(grid_range)  # Number of steps along each grid dimension

    exp_id = get_experiment_id("dgp_kin8nm_gridsearch")

    if not index_range:
        index_range = (0, K**2)
    index_range = range(*index_range)

    eprint(f"Index range [{min(index_range)}, {max(index_range) + 1}) out of "
           f"{K**2} in total.")

    for i, fold in itertools.product(index_range, range(1, n_folds + 1)):

        # Get number of inducing points per dimension
        M1, M2 = [grid_range[n] for n in np.unravel_index(i, [K] * 2)]

        eprint(f"{bold('Grid index')}: {i}\n"
               f"{bold('M1')}:         {M1}\n"
               f"{bold('M2')}:         {M2}\n"
               f"{bold('Fold')}        {fold}/{n_folds}")

        # Set parameters defining this run
        params = {"M1": M1, "M2": M2, "fold": fold}

        if run_exists(params):
            eprint(green("Already exists\n"))
            continue

        with mlflow.start_run(experiment_id=exp_id):
            mlflow.log_params(params)
            run_single(device=device, n_folds=n_folds, **params)


if __name__ == "__main__":
    run_all()
