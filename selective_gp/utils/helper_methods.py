#!/usr/bin/env python

import torch

from gpytorch.likelihoods import (
    GaussianLikelihood, BernoulliLikelihood, MultitaskGaussianLikelihood,
    SoftmaxLikelihood)

from selective_gp.models import DeepGPModel, GPLVM
from selective_gp.layers import SVGP
from selective_gp.datasets import SyntheticData, SquareWave, RealData
from .print_helpers import eprint, bold


torch.set_default_dtype(torch.float64)


def _sample_layers(X, model, n_layers):
    with torch.no_grad():
        F = model.X_scaler.transform(X)
        for gp in model.gps[:n_layers]:
            if isinstance(F, torch.distributions.Distribution):
                F = F.mean.reshape(-1, gp.input_dims)
            F = gp(F)
        return F


def get_gplvm_model(dataset, n_inducing, latent_dimensions=2,
                    projection_dimensions=20, device="cpu", prior_weight=1,
                    scale_Y=True):
    N, D = dataset.Y_train.shape
    model = GPLVM(latent_dimensions, D, N, n_inducing=n_inducing,
                  projection_dimensions=projection_dimensions)
    model.cpu()

    # Flip X and Y because "input" data is now the model output
    Y = dataset.Y_train
    if scale_Y:
        model.Y_scaler.fit(Y)

    model.likelihood.noise_covar.noise = 0.01
    model.likelihood.noise_covar.raw_noise.requires_grad_(False)

    Z_init = initialize_PCA(Y, latent_dimensions)
    model.latent_layer.variational_mean.data.copy_(Z_init)
    model.initialize_inducing_points(Z_init)

    if device == "cpu":
        model.cpu()
    else:
        model.cuda()

    return model


def get_model(dataset, n_inducing=50, n_layers=1, output_dimensions=None,
              add_input=False, prior_weight=1, collapsed=False,
              scale_X=True, scale_Y=True, device="cpu"):
    task_type = dataset.task_type

    if task_type == "binary_classification":
        likelihood = BernoulliLikelihood()
        likelihood_input_dims = 1
    elif task_type == "regression":
        likelihood = GaussianLikelihood()
        likelihood_input_dims = 1
    elif task_type == "multiout_regression":
        likelihood_input_dims = dataset.Y_train.shape[-1]
        likelihood = MultitaskGaussianLikelihood(likelihood_input_dims)
    elif task_type == "multi_classification":
        # We assume that classes are enumerated 0, 1, ..., num_classes - 1
        num_classes = int(dataset.Y_train.max() + 1)
        likelihood = SoftmaxLikelihood(
            mixing_weights=False, num_classes=num_classes)
        likelihood_input_dims = num_classes
    else:
        raise Exception(f"Invalid task type: {task_type}")

    D = dataset.input_dims
    model = DeepGPModel(likelihood=likelihood, add_input=add_input)

    if isinstance(n_inducing, int):
        n_inducing = [n_inducing] * n_layers
    else:
        assert len(n_inducing) == n_layers

    if output_dimensions is None:
        # If not provided, we assume it is input dimensionality for all layers
        # except the last
        output_dimensions = [D] * (n_layers - 1) + [likelihood_input_dims]
    else:
        assert len(output_dimensions) == n_layers
        assert output_dimensions[-1] == likelihood_input_dims
        assert not add_input

    for D_out, n_u in zip(output_dimensions, n_inducing):
        D_in = model.gps[-1].output_dims if len(model.gps) > 0 else D
        gp = SVGP(
            D_in, D_out, n_inducing=n_u, collapsed=collapsed)
        gp.prior_point_process.rate.fill_(prior_weight * D_out)
        model.add_gp(gp)

    model.cpu()

    X, Y = dataset.X_train, dataset.Y_train
    if scale_X:
        model.X_scaler.fit(X)

    if "regression" in task_type and scale_Y:
        model.Y_scaler.fit(Y)

    model.initialize_inducing_points(X)

    if device == "cpu":
        model.cpu()
    else:
        model.cuda()

    return model


def load_data(dataset_name, seed=None, lvm_model=False, device="cpu",
              latent_dimensions=2, **kwargs):
    if dataset_name == "synthetic_regression":
        N = kwargs.pop("n_observations", 200)
        dataset = SyntheticData(task_type="regression", seed=seed, **kwargs)
        dataset.get_batch(batch_size=N)
    elif dataset_name == "synthetic_classification":
        N = kwargs.pop("n_observations", 200)
        dataset = SyntheticData(task_type="binary_classification", seed=seed,
                                **kwargs)
        dataset.get_batch(batch_size=N)
    elif dataset_name == "square_wave":
        dataset = SquareWave(**kwargs)
    else:
        try:
            clz = getattr(RealData, dataset_name)
        except AttributeError:
            raise Exception(f"Invalid dataset name: {dataset_name}")

        kwargs.setdefault("test_size", 0.1)
        dataset = clz(seed=seed, **kwargs)

    if lvm_model:
        # Set input as output and forget output old input
        dataset.Y_train = dataset.X_train
        dataset.Y_test = dataset.X_test
        dataset.X_train = torch.empty((len(dataset.Y_train), 0))
        dataset.X_test = torch.empty((len(dataset.Y_test), 0))

        dataset.task_type = "multiout_regression"
        dataset.input_dims = latent_dimensions

    if device == "cpu":
        dataset.cpu()
    else:
        dataset.cuda()

    return dataset


def initialize_PCA(Y, D):
    from sklearn.decomposition import PCA
    pca = PCA(D)
    Z_np = pca.fit_transform(Y.cpu())
    Z = torch.tensor(Z_np).to(Y)
    Z -= Z.mean()

    W = torch.tensor(pca.components_).to(Y)

    s = Z.std()
    s.requires_grad_(True)
    m = Z.mean()
    m.requires_grad_(True)

    opt = torch.optim.Adam([m, s], lr=1e-3)
    losses = []
    for _ in range(100):
        opt.zero_grad()
        Y_pred = ((Z - m) / s) @ W
        loss = ((Y_pred - Y)**2).sum()
        loss.backward()
        opt.step()

        losses.append(loss.item())

    Z = (Z - m) / s
    return Z.detach()


def remove_points(gp):
    b = gp.variational_point_process.sample()
    to_be_removed, = torch.where(b == 0)
    gp.remove_inducing_points(to_be_removed)
    gp.variational_point_process.probabilities = 1


def fit_layerwise(model, dataset, **kwargs):
    X, Y = dataset.X_train, dataset.Y_train

    # Remove all GP layers
    names, gps = zip(*model.named_gps)
    for n in names:
        del model._modules[n]

    # Insert layers one at a time a fit
    for name, gp in zip(names, gps):
        model.add_gp(gp)

        if len(model.gps) > 1:
            X_u = gp.inducing_inputs
            perm = torch.randperm(len(X))
            idxs = perm[:len(X_u)]

            pred_dist = _sample_layers(X[idxs], model, len(model.gps)-1)
            F_new = pred_dist.mean.reshape(-1, gp.input_dims)

            strat = gp.variational_strategy
            dist = strat._variational_distribution
            var_mean = dist.variational_mean
            var_chol_covar = dist.chol_variational_covar
            X_u.data.copy_(F_new)

            var_chol_covar.data.mul_(1e-3)
            if len(model.gps) == len(gps):
                # Last one
                F_summed = F_new.sum(axis=1)
                var_mean.data.copy_(F_summed.reshape(*var_mean.shape))
            elif var_mean.numel() == X_u.numel():
                var_mean.data.copy_(F_new.reshape(*var_mean.shape))

            gp.kernel.noise = 1e-1
            gp.kernel.base_kernel.outputscale = 1e-1
            strat.variational_params_initialized.fill_(1)

        model.fit(X=X, Y=Y, **kwargs)


def _should_stop(model, point_list):
    if len(point_list) == 1:
        return False

    ps = torch.cat([gp.variational_point_process.probabilities
                    for gp in model.gps])
    if torch.all(ps > 0.9):
        return True

    if len(point_list) >= 3 and torch.tensor(point_list[-3:]).std() < 0.5:
        return True


def prune_model(model, dataset, max_trimming_rounds=1, trimming_epochs=100,
                n_mcmc_samples=8, probability_learning_rate=0.3,
                intermittent_epochs=50):

    X, Y = dataset.X_train, dataset.Y_train

    # Main loop
    point_list = []
    for r in torch.arange(max_trimming_rounds) + 1:
        eprint(bold(f"\nRound {r:02d}"))

        model.fit_score_function_estimator(
            X=X, Y=Y, learning_rate=probability_learning_rate,
            max_epochs=trimming_epochs, n_mcmc_samples=n_mcmc_samples)
        for gp in model.gps:
            remove_points(gp)

        ep = sum(gp.variational_point_process.expected_points.item()
                 for gp in model.gps)
        point_list.append(ep)

        for name, gp in model.named_gps:
            eprint(f"{name}: {bold(gp.n_inducing)}")

        # Re-train between rounds
        if intermittent_epochs > 0 and r != max_trimming_rounds:
            model.fit(X=X, Y=Y, max_epochs=intermittent_epochs)

        if _should_stop(model, point_list):
            break
