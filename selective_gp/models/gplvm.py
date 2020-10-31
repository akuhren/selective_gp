#!/usr/bin/env python

import torch
import torch.nn as nn

from .deep_gp_model import DeepGPModel
from selective_gp.layers import SVGP, LatentLayer

from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.distributions import (
    MultitaskMultivariateNormal, MultivariateNormal)
from gpytorch.lazy import DiagLazyTensor


class GPLVM(DeepGPModel):
    """
    Gaussian process Latent Variable Model.

    This model uses the logic of the `DeepGPModel` class to fit a SVGP along
    with a set of latent input parameters. For high-dimensional input, a linear
    projection may be used by setting the `projection_dimensions`. This adds
    a projection step after the SVGP, such that

    p(y | f) = N(y | f x L, I·σ),

    where L is a `projection_dims` x `output_dimensions` matrix that is learnt
    jointly with the remaining parameters.
    """
    def __init__(self, latent_dimensions, output_dimensions, n_observations,
                 projection_dimensions=None, n_inducing=50, **kwargs):
        if "likelihood" in kwargs:
            raise Exception("Likelihood should not be set for the GP-LVM")
        kwargs["likelihood"] = MultitaskGaussianLikelihood(
            num_tasks=output_dimensions)

        super().__init__(**kwargs)

        self.Q = latent_dimensions
        self.D = output_dimensions
        self.N = n_observations
        self.K = projection_dimensions or self.D

        if projection_dimensions is not None:
            L = torch.zeros(self.K, self.D) + 0.1
            self.register_parameter("L", nn.Parameter(L))
        else:
            self.L = None

        svgp = SVGP(self.Q, self.K, n_inducing=n_inducing, collapsed=False)
        self.add_gp(svgp)
        self.latent_layer = LatentLayer(self.N, self.Q)

    def fit_score_function_estimator(self, X=None, Y=None, **kwargs):
        # Set input as indices
        assert X is None
        X = torch.arange(len(Y))
        return super().fit_score_function_estimator(X=X, Y=Y, **kwargs)

    def fit(self, *args, **kwargs):
        # Set input as indices
        assert "X" not in kwargs
        kwargs["X"] = torch.arange(len(kwargs["Y"]))
        return super().fit(*args, **kwargs)

    def _KL(self):
        # Add KL divergence for latent input to the SVGP divergence
        latent_KL = self.latent_layer.kl_divergence()
        return latent_KL + super()._KL()

    def forward(self, X, **kwargs):
        # We require the provided X to be a set of indices that maps to our
        # latent input.
        assert X.dim() == 1 and torch.all(X == X.to(torch.int64))
        X = self.latent_layer(indices=X.to(torch.int64))
        mvn = super().forward(X)

        # Optionally make projection.
        # TODO covariance has not been implemented.
        if self.L is not None:
            proj_mean = (mvn.mean @ self.L).T
            proj_std = (mvn.stddev @ self.L).T
            mvn = MultivariateNormal(proj_mean, DiagLazyTensor(proj_std**2))
            mvn = MultitaskMultivariateNormal.from_batch_mvn(mvn)

        return mvn
