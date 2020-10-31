#!/usr/bin/env python

import mlflow

from selective_gp.utils import (
    load_data, get_model, get_ELBO, get_loglik, get_experiment_id,
    eprint, bold, green, fit_layerwise, remove_points, get_prediction_times,
    run_exists)

import click


def run_single(prior_weight, device, M, fold, n_folds):
    # Get dataset and model
    test_size = 1 / n_folds
    dataset = load_data("uci_kin8nm", seed=fold, device=device,
                        test_size=test_size)
    model = get_model(dataset, n_inducing=M, n_layers=2, device=device,
                      add_input=True)

    # Create callback for logging status to tracking server
    def status_cb():
        mlflow.set_tag("current_epoch", model.epoch.item())
    model.register_callback(status_cb, update_interval=10)

    # Pre-fit model, first one layer at a time, all layers the jointly
    eprint(bold("\nLayerwise pre-fit"))
    fit_layerwise(model, dataset, batch_size=4096, max_epochs=300)

    eprint(bold("\nJoint pre-fit"))
    model.fit(X=dataset.X_train, Y=dataset.Y_train, batch_size=4096,
              max_epochs=500)

    # Infer probabilities of inclusion for all pseudo-points and sample
    # from resulting distribution to prune model
    eprint(bold("\nPruning"))
    for gp in model.gps:
        gp.variational_point_process.probabilities = 0.8

    model.fit_score_function_estimator(
        X=dataset.X_train, Y=dataset.Y_train, learning_rate=0.3, max_epochs=10,
        n_mcmc_samples=32)

    for gp in model.gps:
        remove_points(gp)

    # Post-fit model, all layers jointly
    eprint(bold("\nJoint post-fit"))
    model.fit(X=dataset.X_train, Y=dataset.Y_train, batch_size=4096,
              max_epochs=500)

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

    for layer, gp in enumerate(model.gps, 1):
        mlflow.log_param(f"M{layer}", gp.n_inducing)

    eprint()


@click.command()
@click.option("--device", type=click.Choice(["cpu", "cuda"]), default="cpu")
@click.option("--n-folds", default=5)
@click.option("--initial-inducing", "M", default=150)
@click.option("--prior-weight", type=float, default=1.0)
def run(M, device, prior_weight, n_folds):
    # ID of currently running experiment
    exp_id = get_experiment_id("dgp_kin8nm_adaptive_pruning")

    for fold in range(1, n_folds + 1):
        eprint(bold(f"Fold {fold}/{n_folds}"))

        # Set parameters and tags defining this run
        params = {
            "M": M, "prior_weight": prior_weight, "fold": fold
        }

        if run_exists(params):
            eprint(green("Already exists\n"))
            continue

        with mlflow.start_run(experiment_id=exp_id):
            mlflow.log_params(params)
            run_single(device=device, n_folds=n_folds, **params)


if __name__ == "__main__":
    run()
