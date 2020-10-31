#!/usr/bin/env python

import torch
from torch.distributions.kl import kl_divergence
from torch.nn import Parameter

from selective_gp.point_processes import (
    SquaredReducingPointProcess, PoissonPointProcess)
from selective_gp.variational import CollapsedStrategy

from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ZeroMean
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    UnwhitenedVariationalStrategy)
from gpytorch.distributions import (
    MultivariateNormal, MultitaskMultivariateNormal)

from gpytorch.models import ApproximateGP

from gpytorch.utils.cholesky import psd_safe_cholesky


def _reregister(module, name, tensor):
    old_p = getattr(module, name)
    assert old_p.dim() == tensor.dim()
    if isinstance(old_p, Parameter):
        module.register_parameter(name, Parameter(tensor))
    elif name in [n for n, _ in module.named_buffers()]:
        module.register_buffer(name, tensor)
    else:
        old_p.detach_()
        old_p.resize_(tensor.size())
        old_p.copy_(tensor)
        old_p.requires_grad_(True)


class SVGP(ApproximateGP):
    """
    Sparse Variational Gaussian process with associated prior and variational
    point processes. The approximate posterior is inferred either with a
    collapsed bound as in

    Titsias, Michalis.
    "Variational learning of inducing variables in sparse Gaussian processes."
    Artificial Intelligence and Statistics. 2009.

    or an uncollapsed bound as in

    Hensman, Fusi, and Lawrence.
    "Gaussian Processes for Big Data."
    "Proceedings of the Conference on Uncertainty in Artificial Intelligence. 2013.

    Multiple outputs are only allowed for the latter, in which case case all
    dimensions are assumed independent.
    """
    def __init__(self, input_dims, output_dims, n_inducing=50,
                 mean=None, kernel=None, collapsed=False):
        # Cast in case numpy-type
        self.input_dims = int(input_dims)
        self.output_dims = int(output_dims)
        n_inducing = int(n_inducing)

        if output_dims is None or output_dims == 1:
            batch_shape = torch.Size([])
        else:
            batch_shape = torch.Size([self.output_dims])
        x_u = torch.randn(n_inducing, self.input_dims)

        if collapsed:
            assert batch_shape == torch.Size([])
            strategy = CollapsedStrategy(self, n_inducing)
        else:
            variational_dist = CholeskyVariationalDistribution(
                n_inducing, batch_shape=batch_shape)
            strategy = UnwhitenedVariationalStrategy(
                self, x_u, variational_dist, learn_inducing_locations=True)

        super().__init__(strategy)

        if mean is None:
            mean = ZeroMean()
        self.mean = mean

        if kernel is None:
            rbf = RBFKernel(ard_num_dims=input_dims)
            kernel = ScaleKernel(rbf)
        self.kernel = kernel

        self.prior_point_process = SquaredReducingPointProcess(n_inducing)
        self.variational_point_process = PoissonPointProcess(n_inducing)

    @property
    def variational_distribution(self):
        return self.variational_strategy._variational_distribution

    def point_process_divergence(self):
        return kl_divergence(self.variational_point_process,
                             self.prior_point_process)

    @property
    def n_inducing(self):
        return len(self.inducing_inputs)

    @property
    def inducing_inputs(self):
        return self.variational_strategy.inducing_points

    @inducing_inputs.setter
    def inducing_inputs(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.inducing_inputs)

        param = self.variational_strategy.inducing_points
        param.data.copy_(value.reshape(*param.shape))

    @property
    def inducing_distribution(self):
        return self.variational_strategy.variational_distribution

    def remove_inducing_points(self, idxs):
        """
        Remove the inducing points corresponding to provided indices.
        """

        mask = torch.ones(len(self.inducing_inputs), dtype=bool)
        mask[idxs] = torch.tensor(False)

        strat = self.variational_strategy
        dist = strat._variational_distribution

        # Point process probabilities
        raw_p = self.variational_point_process.raw_probabilities[mask]
        _reregister(self.variational_point_process, "raw_probabilities", raw_p)

        # Inducing inputs
        x_u = strat.inducing_points.data[mask]
        _reregister(strat, "inducing_points", x_u)

        # Inducing outputs
        def _chol(L):
            return psd_safe_cholesky(L @ L.T + I*1e-8)

        I = torch.eye(len(x_u), device=x_u.device)
        m = dist.variational_mean
        L = dist.chol_variational_covar

        if m.dim() == 1:
            m = m.data[mask]
        elif m.dim() == 2:
            m = m.data[:, mask]
        else:
            raise NotImplementedError

        if L.dim() == 2:
            S = (L @ L.T)[mask][:, mask]
            L = psd_safe_cholesky(S + I*1e-8)
        elif L.dim() == 3:
            S = (L @ L.transpose(1, 2))[:, mask][:, :, mask]
            L = psd_safe_cholesky(S + I*1e-8)
        else:
            raise NotImplementedError

        _reregister(dist, "variational_mean", m)
        _reregister(dist, "chol_variational_covar", L)

        self.clear_caches()

    def clear_caches(self):
        try:
            self.variational_strategy._memoize_cache.clear()
        except AttributeError:
            pass

    def forward(self, X):
        """
        Return prior distribution
        """
        mean = self.mean(X)
        covariance_matrix = self.kernel(X)

        assert covariance_matrix.dim() == 2

        if mean.dim() == 2:
            m, k = mean.shape
            covariance_matrix = covariance_matrix.expand(m, k, k)

        return MultivariateNormal(mean, covariance_matrix)

    def __call__(self, X, Y=None, likelihood=None, prior=False):
        strat = self.variational_strategy
        if isinstance(strat, CollapsedStrategy):
            res = strat(X, Y, likelihood=likelihood, prior=prior)
        else:
            res = strat(X, prior=prior)

        N, D = len(X), self.output_dims
        if D == 1:
            mean = res.mean.reshape(N)
            assert res._covar.shape == (N, N)
            res = MultivariateNormal(mean, res._covar)
        else:
            assert res.mean.size(0) == D
            res = MultitaskMultivariateNormal.from_batch_mvn(res)

        return res
