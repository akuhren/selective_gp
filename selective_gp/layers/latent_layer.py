#!/usr/bin/env python

import torch
from torch.nn import Parameter
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal

from gpytorch import Module
from gpytorch.constraints import Positive
from gpytorch.distributions import (
    MultivariateNormal, MultitaskMultivariateNormal)
from gpytorch.lazy import DiagLazyTensor


class LatentLayer(Module):
    """
    Latent layer for use in GP-LVM. It comprises N latent variables, each with
    a Gaussian prior and variational distribution.

    The prior is isotropic with configurable mean and variance. The
    variational distribution may be set to share variance
    """
    def __init__(self, n_elements, n_dimensions, prior_mean=0,
                 prior_variance=1, share_variational_variance=False):
        super().__init__()

        self.prior = Normal(prior_mean, prior_variance**0.5)

        mean = self.prior.sample([n_elements, n_dimensions])
        if share_variational_variance:
            raw_variance = torch.zeros((n_elements, 1))
        else:
            raw_variance = torch.zeros_like(mean)

        self.constraint = Positive()
        self.register_parameter("variational_mean", Parameter(mean))
        self.register_parameter("raw_variational_variance",
                                Parameter(raw_variance))
        self.variational_variance = torch.ones_like(self.variational_mean)

        self.input_dims = 0
        self.output_dims = n_dimensions

    @property
    def n_elements(self):
        return self.variational_mean.shape[0]

    @property
    def variational_variance(self):
        return self.constraint.transform(self.raw_variational_variance)

    @variational_variance.setter
    def variational_variance(self, value):
        param = self.raw_variational_variance
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(param)
        param.data.copy_(value.reshape(*param.shape))

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, indices=None):
        """
        Return the variational posterior for the latent variables, pertaining
        to provided indices
        """
        if indices is None:
            ms = self.variational_mean
            vs = self.variational_variance
        else:
            ms = self.variational_mean[indices]
            vs = self.variational_variance[indices]

        vs = vs.expand(len(vs), self.output_dims)

        if self.output_dims == 1:
            m, = ms
            v, = vs
            return MultivariateNormal(m, DiagLazyTensor(v))
        else:
            mvns = [MultivariateNormal(m, DiagLazyTensor(v))
                    for m, v in zip(ms.T, vs.T)]
            return MultitaskMultivariateNormal.from_independent_mvns(mvns)

    def kl_divergence(self):
        """
        KL divergence from variational to prior distribution.
        """
        flat_m = self.variational_mean.T.flatten()

        v = self.variational_variance
        flat_v = v.expand(len(v), self.output_dims).flatten()

        return torch.sum(kl_divergence(Normal(flat_m, flat_v), self.prior))
