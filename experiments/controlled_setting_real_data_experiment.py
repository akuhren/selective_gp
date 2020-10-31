#!/usr/bin/env python

import itertools
import torch
import mlflow

from selective_gp.utils import (
    load_data, get_model, get_ELBO, get_experiment_id, eprint, bold, green,
    remove_points, run_exists)

import click


def run_single(device, M, noise, epochs, adaptive, prior_weight, dataset_name,
               fold, n_folds):
    # Get data and model
    test_size = 1 / n_folds if n_folds > 1 else 0
    dataset = load_data(dataset_name, seed=fold, device=device,
                        test_size=test_size)
    dataset.add_noise(noise)

    collapsed = (dataset.task_type == "regression")

    model = get_model(
        dataset, n_inducing=M, device=device, collapsed=collapsed,
        prior_weight=prior_weight)

    # Initialize hyperparameters
    gp, = model.gps
    gp.kernel.outputscale = 1.0
    gp.kernel.base_kernel.lengthscale = 1.0

    if not adaptive:
        # Fit model
        model.fit(X=dataset.X_train, Y=dataset.Y_train, max_epochs=epochs)
    else:
        # Pre-fit
        model.fit(X=dataset.X_train, Y=dataset.Y_train, max_epochs=epochs // 2)

        # Prune
        gp.prior_point_process.rate.fill_(prior_weight)
        gp.variational_point_process.probabilities = 0.5
        model.fit_score_function_estimator(
            X=dataset.X_train, Y=dataset.Y_train, learning_rate=0.3,
            max_epochs=300, n_mcmc_samples=64)
        remove_points(gp)
        eprint(f"Post pruning: {gp.n_inducing}\n")

        # Post-fit
        model.fit(X=dataset.X_train, Y=dataset.Y_train, max_epochs=epochs)

    # Log metrics
    scaled_log_lik, KL = get_ELBO(model, dataset, reps=1)
    ELBO = scaled_log_lik - KL
    mlflow.log_metrics({
        "ELBO": ELBO,
        "n_inducing": gp.n_inducing,
    })

    if adaptive:
        vpp = gp.variational_point_process
        mlflow.log_metrics({
            "mean_M": vpp.expected_points.item(),
            "var_M": vpp.expected_points_variance.item(),
        })


@click.command()
@click.option("--epochs", default=1000)
@click.option("--device", default="cpu", type=click.Choice(["cpu", "cuda"]))
@click.option("--dataset-name", default="uci_energy")
@click.option("--inducing-range", nargs=3, default=(10, 101, 10))
@click.option("--noise", default=0.0)
@click.option("--n-folds", default=5)
@click.option("--adaptive/--no-adaptive", default=False)
@click.option("--prior-weight", default=0.5)
def run_all(dataset_name, epochs, device, n_folds, adaptive, prior_weight,
            noise, inducing_range):

    dataset = load_data(dataset_name)
    eprint(f"{bold('Dataset:   ')} {dataset_name}\n"
           f"{bold('Task type: ')} {dataset.task_type}\n"
           f"{bold('N:         ')} {len(dataset)}\n"
           f"{bold('D:         ')} {dataset.input_dims}\n")

    Ms = torch.arange(*inducing_range).tolist()

    # ID of currently running experiment
    exp_id = get_experiment_id("controlled_setting_real_data")

    for M, fold in itertools.product(Ms, range(1, n_folds + 1)):
        eprint(f"{bold('Noise: ')} {noise:.3f}\n"
               f"{bold('M:     ')} {M}\n"
               f"Fold {fold}/{n_folds}")

        # Set parameters defining this run
        params = {"M": M, "noise": noise, "epochs": epochs,
                  "adaptive": adaptive, "prior_weight": prior_weight,
                  "dataset_name": dataset_name, "fold": fold}

        if run_exists(params):
            eprint(green("Already exists\n"))
            continue

        with mlflow.start_run(experiment_id=exp_id):
            mlflow.log_params(params)
            run_single(n_folds=n_folds, device=device, **params)

        eprint()


if __name__ == "__main__":
    run_all()
