#!/usr/bin/env python

import torch
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

from selective_gp.models.base_model import BaseModel, TorchScaler
from selective_gp.sfe import score_function_estimator

import gpytorch
from gpytorch.likelihoods import (
    GaussianLikelihood, MultitaskGaussianLikelihood)
from gpytorch.distributions import (
    MultitaskMultivariateNormal, MultivariateNormal)
from gpytorch.lazy import DiagLazyTensor, CholLazyTensor
from gpytorch import settings
from gpytorch.utils.cholesky import psd_safe_cholesky


class DeepGPModel(BaseModel):
    """
    Deep Gaussian process implemented according to:

    Salimbeni, Hugh, and Marc Deisenroth.
    "Doubly stochastic variational inference for deep Gaussian processes."
    Advances in Neural Information Processing Systems. 2017.

    The observed input, X, may optionally be added to each intermittent layer
    to avoid problems of non-injectivity as described in

    Duvenaud, David, et al.
    "Avoiding pathologies in very deep networks."
    Artificial Intelligence and Statistics. 2014.

    GP layers are added incrementally after intialisation with the `add_gp`
    method. The standard "flat" SVGP is retreived by only using one layer.
    """
    def __init__(self, likelihood=GaussianLikelihood(), add_input=False):
        super().__init__()
        self.likelihood = likelihood

        self.X_scaler = TorchScaler()
        self.Y_scaler = TorchScaler()
        self.add_input = add_input
        self.latent_layer = None

        self.add_input = add_input

    @property
    def named_gps(self):
        return [(n, m) for n, m in sorted(self._modules.items())
                if n[:3] == "gp_"]

    @property
    def gps(self):
        return [m for _, m in self.named_gps]

    def _log_lik(self, **observed_values):
        assert set(observed_values.keys()) == set(["X", "Y"])
        X = observed_values["X"]
        Y = observed_values["Y"]

        # Get output distribution
        f_dist = self(X, Y=Y)
        assert Y.numel() == f_dist.mean.numel()

        # Get log likelihood
        Y = Y.reshape(*f_dist.mean.shape)
        log_lik = self.likelihood.expected_log_prob(Y, f_dist).sum()

        return log_lik

    def _KL(self):
        # Accumulate KL divergence over all layers
        KL = torch.tensor(0., device=self.device, dtype=self.dtype)
        for gp in self.gps:
            KL.add_(gp.variational_strategy.kl_divergence().sum())

        return KL

    def initialize_inducing_points(self, X):
        """
        Initialise inducing points of top-level GP as random subset of X.
        """

        X = self.X_scaler(X)
        top_gp = self.gps[0]
        x_u = top_gp.variational_strategy.inducing_points
        if x_u.dim() == 2:
            n_u = len(x_u)
        else:
            n_u = x_u.size(1)
        perm = torch.randperm(len(X))
        idxs = perm[:n_u]

        x_u.data.copy_(X[idxs])

        assert torch.all(x_u.abs() < 1e5)

    def add_gp(self, gp, name=None):
        if not name:
            name = f"gp_{len(self.gps):02d}"
        self.add_module(name, gp)

    def forward(self, X, Y=None):

        # Propagate samples through the layers.
        F = X
        for i, gp in enumerate(self.gps):
            if isinstance(F, MultivariateNormal):
                # Sample from independent Gaussian
                m = F.mean.reshape(-1, gp.input_dims)
                s = F.stddev.reshape(-1, gp.input_dims)
                F = Normal(loc=m, scale=s).rsample()

            if i > 0 and self.add_input:
                # Add input to layers after the first one
                assert F.shape == X.shape
                F.add_(X)

            # Get posterior distribution
            F = gp(F, Y=Y, likelihood=self.likelihood)

        if isinstance(self.likelihood, MultitaskGaussianLikelihood):
            D = self.likelihood.num_tasks
        else:
            D = 1

        # If our likelihood is one-dimensional but we are outputting multiple
        # dimensions, we sum the dimensions together
        if isinstance(F, MultitaskMultivariateNormal) and D == 1:
            mean = F.mean.sum(axis=-1)
            variance = F.variance.sum(axis=-1)
            F = MultivariateNormal(mean, DiagLazyTensor(variance))

        return F

    def _get_bound_for_point_samples(self, X, Y, samples):
        """
        Helper function for getting the ELBO when only a subset of inducing
        points are present in each layer, as given by argument samples
        """

        def _update_inducing_points(gp, Z, m, S=None, L=None):
            # Setting the inducing points for a given layer
            assert (S is None) ^ (L is None)

            strat = gp.variational_strategy
            dist = strat._variational_distribution

            strat.inducing_points.data = Z
            dist.variational_mean.data = m
            if S is not None:
                L = psd_safe_cholesky(S)
            dist.chol_variational_covar.data = L
            gp.clear_caches()

        # Save current parameters
        initial_params = {}
        for n, gp in self.named_gps:
            var_dist = gp.variational_strategy.variational_distribution
            initial_params[n] = {
                "Z": gp.inducing_inputs.clone(),
                "m": var_dist.mean.clone(),
                "L": var_dist.scale_tril
            }

        # Remove points according to provided point samples
        for (n, gp), sample in zip(self.named_gps, samples):
            # Include only the selected inducing points for the given sample
            Z_ = initial_params[n]["Z"][sample]
            m_ = initial_params[n]["m"][..., sample]
            L = initial_params[n]["L"]
            S = CholLazyTensor(L).add_jitter().evaluate()
            S = S[..., sample, :][..., :, sample]
            _update_inducing_points(gp, Z_, m_, S=S)

        # Evaluate bound
        X_scaled = self.X_scaler.transform(X)
        Y_scaled = self.Y_scaler.transform(Y)
        bound = self._log_lik(X=X_scaled, Y=Y_scaled) - self._KL()

        # Restore parameters
        for n, gp in self.named_gps:
            Z = initial_params[n]["Z"]
            m = initial_params[n]["m"]
            L = initial_params[n]["L"]
            _update_inducing_points(gp, Z, m, L=L)

        return bound

    def fit_score_function_estimator(self, X=None, Y=None, learning_rate=0.3,
                                     max_epochs=500, n_mcmc_samples=8,
                                     verbose=True):
        """
        Fit the score function estimator to infer probability of inclusion for
        the inducing points across layers. The main logic is carried out in
        `selective_gp.sfe.score_function_estimator`.
        """

        pps = [gp.variational_point_process for gp in self.gps]

        # Define likelihood and KL functions (eq. (2) in the paper)
        def _likelihood_function(sample_dict):
            samples = [sample_dict[pp] for pp in pps]
            return self._get_bound_for_point_samples(X, Y, samples)

        def _KL_function():
            kl = torch.zeros([], dtype=self.dtype, device=self.device)
            for gp in self.gps:
                vpp = gp.variational_point_process
                ppp = gp.prior_point_process
                kl.add_(kl_divergence(vpp, ppp))
            return kl

        # Update callbacks to include incrementing epoch variable
        def _epoch_cb():
            self.epoch.add_(1)
        callbacks = {**self.callbacks, _epoch_cb: ((), {}, 1)}

        was_training = self.training
        self.train()

        score_function_estimator(
            _likelihood_function, _KL_function, pps, max_epochs=max_epochs,
            learning_rate=learning_rate, n_mcmc_samples=n_mcmc_samples,
            verbose=verbose, callbacks=callbacks,
            stop_conditions=self.stop_conditions)

        if not was_training:
            self.eval()

    def _finalize_epoch(self):
        """
        Tensorboard bookkeeping after each epoch.
        """
        if self.writer is None:
            return

        step = self.epoch
        for gp_name, gp in self.named_gps:
            for n, p, c in gp.named_parameters_and_constraints():
                if "kernel" not in n:
                    continue

                n = n.replace(".", "/")
                if c:
                    p = c.transform(p)
                    n = n.replace("raw_", "")
                p = torch.mean(p)

                self.writer.add_scalar(f"{gp_name}/{n}", p, global_step=step)

            e = gp.variational_point_process.probabilities.sum()
            self.write_to_tensorboard({
                f"{gp_name}/number_of_points": gp.n_inducing,
                f"{gp_name}/expected_points": e,
            })

        for n, p, c in self.likelihood.named_parameters_and_constraints():
            if c:
                p = c.transform(p)
                n = n.replace("raw_", "")
            v = torch.mean(p)
            self.writer.add_scalar("likelihood/" + n, v, global_step=step)

    def get_predictive_distribution(self, X, full_covariance=False):
        """
        Get distribution for novel input.
        """
        X = self.X_scaler(X)
        f_dist = self(X)

        with torch.no_grad(), settings.num_likelihood_samples(1):
            y_dist = self.likelihood(f_dist)

        no_scaling = (
            torch.all(self.Y_scaler.means == 0) and
            torch.all(self.Y_scaler.stds == 1)
        )

        if no_scaling:
            return y_dist
        elif isinstance(y_dist, MultivariateNormal):
            mean = self.Y_scaler.inverse_transform(y_dist.mean)
            variance = y_dist.variance * self.Y_scaler.stds**2
            if full_covariance:
                return MultivariateNormal(mean, torch.diag(variance))
            else:
                return Normal(mean, variance.sqrt())
        else:
            raise NotImplementedError

    def sample(self, X, use_covariance=False):
        """
        Sample output for novel input.
        """
        X = self.X_scaler(X)
        with torch.no_grad(), gpytorch.settings.num_likelihood_samples(1):
            f_dist = self(X)
            y_dist = self.likelihood(f_dist)
            if isinstance(y_dist, MultivariateNormal):
                mean = self.Y_scaler.inverse_transform(y_dist.mean)
                if not use_covariance:
                    stddev = y_dist.stddev * self.Y_scaler.stds
                    y_dist = Normal(mean, stddev)
                else:
                    cov = y_dist.covariance_matrix * self.Y_scaler.stds**2
                    y_dist = MultivariateNormal(mean, cov)
            return y_dist.sample().flatten()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
