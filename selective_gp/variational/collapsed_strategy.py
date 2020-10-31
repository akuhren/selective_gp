#!/usr/bin/env python

import torch
from torch.nn import Module

from gpytorch.variational import UnwhitenedVariationalStrategy
from gpytorch.distributions import MultivariateNormal
from gpytorch.utils.memoize import cached
from gpytorch.utils.cholesky import psd_safe_cholesky
from gpytorch.lazy import CholLazyTensor


class CollapsedDistribution(Module):
    def __init__(self, n_inducing):
        super().__init__()
        self.num_inducing_points = n_inducing
        self.batch_shape = torch.Size([])
        self.mean_init_std = 1e-10
        self.register_buffer("variational_mean", torch.zeros(n_inducing))
        self.register_buffer("chol_variational_covar", torch.eye(n_inducing))

    def forward(self):
        m = self.variational_mean
        L = self.chol_variational_covar
        return MultivariateNormal(m, CholLazyTensor(L))

    def update_distribution(self, model, X, Y, X_u, likelihood):
        n_u = len(X_u)
        v = likelihood.noise_covar.noise
        b = 1 / v

        full_inputs = torch.cat([X_u, X], dim=-2)
        full_output = model.forward(full_inputs)
        full_covar = full_output.lazy_covariance_matrix

        K_MM = full_covar[..., :n_u, :n_u].add_jitter()
        K_MN = full_covar[..., :n_u, n_u:].evaluate()
        K_MM_inv = torch.inverse(K_MM.evaluate())

        a = K_MM.inv_matmul(K_MN)
        inner = a.matmul(a.T)

        A = b * inner + K_MM_inv
        S = torch.inverse(A)
        m = b * S @ K_MM_inv @ K_MN @ Y

        self.variational_mean = m.flatten()
        self.chol_variational_covar = psd_safe_cholesky(S)


class CollapsedStrategy(UnwhitenedVariationalStrategy):
    def __init__(self, model, n_inducing):
        X_u = torch.rand((n_inducing, model.input_dims))
        collapsed_distribution = CollapsedDistribution(n_inducing)
        super().__init__(model, X_u, collapsed_distribution,
                         learn_inducing_locations=True)

    @property
    @cached(name="prior_distribution_memo")
    def prior_distribution(self):
        out = self.model.forward(self.inducing_points)
        cov = out.lazy_covariance_matrix.add_jitter()
        res = MultivariateNormal(out.mean, cov)
        return res

    def __call__(self, X, Y=None, likelihood=None, prior=False):
        # If we're in prior mode, then we're done!
        if prior:
            return self.model.forward(X)

        X_u = self.inducing_points
        if self.training:
            if hasattr(self, "_memoize_cache"):
                delattr(self, "_memoize_cache")
                self._memoize_cache = dict()
            assert Y is not None
            assert likelihood is not None

            self._variational_distribution.update_distribution(
                self.model, X, Y, X_u, likelihood)

        var_dist = self._variational_distribution.forward()

        return self.forward(
            X,
            X_u,
            inducing_values=var_dist.mean,
            variational_inducing_covar=var_dist.lazy_covariance_matrix
        )
