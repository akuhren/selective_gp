#!/usr/bin/env python

import warnings
import numpy as np
import torch
import mlflow

from sklearn.cluster import KMeans
import gpytorch
from selective_gp.utils import (
    load_data, get_model, get_ELBO, get_experiment_id, eprint, green,
    remove_points, run_exists)
from copy import deepcopy

import click


class FullGP(gpytorch.models.ExactGP):
    """
    Helper class for calculating the exact marginal log likelihood.
    """
    def __init__(self, X, y):
        super().__init__(X, y, gpytorch.likelihoods.GaussianLikelihood())

        rbf_kernel = gpytorch.kernels.RBFKernel()
        self.kernel = gpytorch.kernels.ScaleKernel(rbf_kernel)
        self.fit()

    def forward(self, x):
        covariance = self.kernel(x)
        mvn = gpytorch.distributions.MultivariateNormal
        return mvn(torch.zeros(len(x)), covariance)

    def fit(self):
        self.train()
        optimizer = torch.optim.Adam(params=self.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        X, = self.train_inputs
        y = self.train_targets
        for i in range(50):
            optimizer.zero_grad()
            dist = self(X)
            loss = -mll(dist, y)
            loss.backward()
            optimizer.step()
        self.eval()

    @property
    @torch.no_grad()
    def log_likelihood(self):
        X, = self.train_inputs
        y = self.train_targets
        with warnings.catch_warnings():
            # Ignore GPyTorch warning for calling module on training data
            warnings.simplefilter("ignore")
            marginal_dist = self(X)
        log_lik = self.likelihood.log_marginal(y, marginal_dist)
        return log_lik.sum().item()


def get_exact_log_lik(X, Y):
    full_gp = FullGP(X, Y.flatten())
    return full_gp.log_likelihood


def initialize_inducing(model, X, Y):
    model.likelihood.noise_covar.noise = 0.1
    gp, = model.gps

    km = KMeans(n_clusters=gp.n_inducing)
    km.fit(X)
    Z_init = torch.as_tensor(km.cluster_centers_).double()
    gp.inducing_inputs.data.copy_(Z_init)


def logspace(l, u, *args, **kwargs):
    return np.exp(np.linspace(np.log(l), np.log(u), *args, **kwargs))


def load_synthetic_data(noise, lengthscale, clustering, seed=None,
                        device="cpu"):
    dataset = load_data(
        "synthetic_regression", seed=seed, device=device, n_observations=1000,
        lengthscale=1, obs_noise=noise, x_min=0, x_max=100)

    if lengthscale != 1:
        dataset.clear()
        upper = 100 / lengthscale
        dataset.get_batch(1000, input_range=(0, upper))
        dataset.X_train.mul_(lengthscale)

    if clustering:
        dataset.clear()
        sigma = torch.tensor([[1 / clustering]])
        dataset.get_batch_multimode(1000, 5, sigma=sigma)

    return dataset


def run_single(M, characteristic, char_value, adaptive, prior_weight, device,
               n_posterior_samples):

    default_values = {"noise": 0.3, "clustering": None, "lengthscale": 1.0}
    default_values[characteristic] = char_value

    dataset = load_synthetic_data(**default_values, seed=0, device=device)
    model = get_model(dataset, n_inducing=M, n_layers=1, device=device,
                      scale_X=False, scale_Y=False, collapsed=True,
                      prior_weight=prior_weight)

    # Initialize hyper-parameters
    gp, = model.gps
    gp.kernel.outputscale = 1.0
    gp.kernel.base_kernel.lengthscale = default_values["lengthscale"]

    # Initialise inducing points with k-means clustering
    initialize_inducing(model, dataset.X_train, dataset.Y_train)

    # Get log marginal of an exact GP
    exact_log_lik = get_exact_log_lik(dataset.X_train, dataset.Y_train)
    mlflow.log_metric("exact_log_lik", exact_log_lik)

    if not adaptive:
        # Fit model and record log likelihood
        model.fit(X=dataset.X_train, Y=dataset.Y_train, max_epochs=300)

        log_lik, KL = get_ELBO(model, dataset, batch_size=None, reps=1)
        mlflow.log_metrics({
            "sparse_log_lik": log_lik,
            "sparse_KL": KL,
            "sparse_ELBO": log_lik - KL,
            "n_u": gp.n_inducing,
        })
    else:
        # Pre-fit
        model.fit(X=dataset.X_train, Y=dataset.Y_train, max_epochs=300)

        # Prune model
        model.fit_score_function_estimator(
            X=dataset.X_train, Y=dataset.Y_train, max_epochs=500,
            n_mcmc_samples=16, learning_rate=0.3)

        # Record statistics
        vpp = gp.variational_point_process
        with torch.no_grad():
            p = vpp.probabilities
        expected_points = p.sum().item()
        stddev_points = (p * (1 - p)).sum().sqrt().item()

        mlflow.log_metrics({
            "expected_points": expected_points,
            "stddev_points": stddev_points,
        })

        # Draw sets from point process and record log likelihood
        state_dict = deepcopy(gp.state_dict())
        for i in range(n_posterior_samples):
            eprint(f"\nSample {i + 1:02d}/{n_posterior_samples}")

            if i > 0:
                for n, t in gp.named_parameters():
                    t.data = state_dict[n]
                    t.grad = None
                for n, t in gp.named_buffers():
                    t.data = state_dict[n]

            remove_points(gp)

            # Post-fit
            model.fit(X=dataset.X_train, Y=dataset.Y_train, max_epochs=200)

            log_lik, KL = get_ELBO(model, dataset, batch_size=None, reps=1)

            prefix = f"draw_{i:02d}__"
            mlflow.log_metrics({
                prefix + "sparse_log_lik": log_lik,
                prefix + "sparse_KL": KL,
                prefix + "sparse_ELBO": log_lik - KL,
                prefix + "n_u": gp.n_inducing
            })


@click.command()
@click.option("--characteristic", default="lengthscale",
              type=click.Choice(["lengthscale", "clustering", "noise"]))
@click.option("--device", default="cpu", type=click.Choice(["cpu", "cuda"]))
@click.option("--inducing-range", nargs=3, default=(20, 81, 5))
@click.option("--adaptive/--no-adaptive", default=False)
@click.option("--prior-weight", default=0.1)
@click.option("--n-samples", "n_posterior_samples", default=10)
def run_all(characteristic, inducing_range, adaptive, device, prior_weight,
            n_posterior_samples):
    exp_id = get_experiment_id("controlled_setting_synth_data")

    characteristic_vector = {
        "noise": logspace(0.3, 1.0, 5).tolist(),
        "clustering": logspace(0.03, 0.5, 5).tolist(),
        "lengthscale": logspace(0.8, 2.5, 5).tolist(),
    }[characteristic]

    Ms = torch.arange(*inducing_range).tolist()
    product = [(n, M) for n in characteristic_vector for M in Ms]
    for c, M in product:
        eprint(f"{characteristic}: {c:.3f}\n"
               f"M: {M}\n")

        # Set parameters defining this run
        params = {"M": M, "characteristic": characteristic, "char_value": c,
                  "adaptive": adaptive, "prior_weight": prior_weight}

        if run_exists(params):
            eprint(green("Already exists\n"))
            continue

        with mlflow.start_run(experiment_id=exp_id):
            mlflow.log_params(params)
            run_single(device=device, n_posterior_samples=n_posterior_samples,
                       **params)

        eprint()


if __name__ == "__main__":
    run_all()
