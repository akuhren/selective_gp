#!/usr/bin/env python

import torch
import torch.nn as nn
from torch.distributions import Bernoulli
from torch.distributions.kl import register_kl
from gpytorch.constraints import Interval


eps = 1e-8


class PointProcess(nn.Module):
    def __init__(self):
        super().__init__()

    def log_pmf(self, mask):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def rsample(self):
        raise NotImplementedError

    def remove_points(self, idxs):
        raise NotImplementedError

    def add_points(self, n_points):
        raise NotImplementedError

    def cross_entropy(self, other_point_process):
        raise NotImplementedError

    @property
    def entropy(self):
        raise NotImplementedError

    @property
    def marginal_probabilities(self):
        raise NotImplementedError

    @property
    def expected_points(self):
        raise NotImplementedError

    @property
    def expected_points_variance(self):
        raise NotImplementedError


class SquaredReducingPointProcess(PointProcess):
    def __init__(self, max_inducing=None, rate=5):
        super().__init__()
        self.register_buffer("rate", torch.tensor(rate).double())

    def log_pmf(self, mask):
        # Note: Unnormalized
        return -self.rate * mask.sum()**2

    def cross_entropy(self, other_point_process):
        # Note: Unnormalized
        e = other_point_process.expected_points
        e_v = other_point_process.expected_points_variance
        return self.rate * (e_v + e**2)


class PoissonPointProcess(PointProcess):
    def __init__(self, n_points, prior=None):
        super(PoissonPointProcess, self).__init__()
        self.interval = Interval(0.0, 1.0)
        self.register_parameter(
            "raw_probabilities", nn.Parameter(torch.ones(n_points)))
        self.register_buffer("temperature", torch.tensor(0.2))
        self.prior = prior
        self.probabilities = 1

    @property
    def probabilities(self):
        return self.interval.transform(self.raw_probabilities)

    @property
    def marginal_probabilities(self):
        return self.probabilities

    @property
    def expected_points(self):
        return self.probabilities.sum()

    @property
    def expected_points_variance(self):
        p = self.probabilities
        return (p * (1 - p)).sum()

    @property
    def entropy(self):
        p = self.probabilities
        return -torch.sum(p * torch.log(p) + (1 - p) * torch.log1p(-p))

    @property
    def log_probabilities(self):
        # TODO
        raise NotImplementedError

    @probabilities.setter
    def probabilities(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_probabilities)
        l, u = self.interval.lower_bound, self.interval.upper_bound
        value = torch.clamp(value, l + eps, u - eps)
        inv_value = self.interval.inverse_transform(value)
        self.raw_probabilities.data.copy_(inv_value)

    def log_pmf(self, mask):
        assert mask.shape == self.raw_probabilities.shape
        mask = mask.to(float)
        return torch.sum(mask * torch.log(self.probabilities) +
                         (1 - mask) * torch.log(1 - self.probabilities))

    def sample(self, sample_shape=torch.Size([])):
        return Bernoulli(self.probabilities).sample(sample_shape)


@register_kl(PoissonPointProcess, PointProcess)
def kl_pp_pp(variational_point_process, prior_point_process):
    return (prior_point_process.cross_entropy(variational_point_process) -
            variational_point_process.entropy)
