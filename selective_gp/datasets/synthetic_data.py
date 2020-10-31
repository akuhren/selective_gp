#!/usr/bin/env python

import torch

from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, MaternKernel
from sklearn.datasets import make_blobs

from .dataset import Dataset


class NearestNeighborClassifier(object):
    def __init__(self, X, Y):
        from sklearn.neighbors import KNeighborsClassifier
        self.knc = KNeighborsClassifier()
        self.knc.fit(X, Y)

    def __call__(self, X):
        return torch.as_tensor(self.knc.predict(X)).double().reshape(-1, 1)


class NearestNeighborRegressor(object):
    def __init__(self, X, Y):
        from sklearn.neighbors import KNeighborsRegressor
        self.knr = KNeighborsRegressor()
        self.knr.fit(X, Y)

    def __call__(self, X):
        return torch.as_tensor(self.knr.predict(X)).double().reshape(-1, 1)


class SyntheticData(Dataset):
    def __init__(self, input_dims=1, x_min=0, x_max=100, test_size=0.75,
                 task_type="binary_classification", obs_noise=0.01,
                 lengthscale=3.0, n_points=1600, seed=None):

        self.obs_noise = obs_noise
        self.samples_per_dim = int(n_points ** (1 / input_dims))

        if not torch.is_tensor(x_min):
            x_min = torch.as_tensor(x_min).expand(input_dims)

        if not torch.is_tensor(x_max):
            x_max = torch.as_tensor(x_max).expand(input_dims)

        assert x_min.shape == x_max.shape == (input_dims,)

        if seed is not None:
            torch.manual_seed(seed & ((1 << 63) - 1))

        Xs = torch.meshgrid(*[torch.linspace(_min, _max, self.samples_per_dim)
                              for _min, _max in zip(x_min, x_max)])
        X = torch.stack(Xs).T.reshape(-1, input_dims)

        if task_type == "regression":
            Y = self._draw_gp_function(X, lengthscale)
            self.function = NearestNeighborRegressor(X, Y)
        elif task_type == "binary_classification":
            if input_dims == 1:
                Y = self._draw_gp_function(X, lengthscale)
                Y = (Y - Y.min()) / (Y.max() - Y.min())
                Y = torch.round(Y)
            else:
                n_samples = 5000
                n_clusters = 10
                X, Y = make_blobs(n_samples=n_samples, n_features=input_dims,
                                  centers=n_clusters, random_state=seed)
                Y[Y < n_clusters // 2] = 0.
                Y[Y >= n_clusters // 2] = 1.

                X, Y = torch.as_tensor(X), torch.as_tensor(Y)

                X_normed = (X - X.min(0).values)
                X_normed = X_normed / X_normed.max(0).values
                X = X_normed * (x_max - x_min) + x_min

            self.function = NearestNeighborClassifier(X, Y)
            self.obs_noise = 0.0
        else:
            raise Exception("Invalid function type: {}".format(task_type))

        self._latent_function_values = None
        self.input_range = torch.stack((x_min, x_max)).T.double()

        self.test_size = test_size
        self.X_train = torch.empty((0, input_dims))
        self.X_test = torch.empty((0, input_dims))
        self.Y_train = torch.empty((0, 1))
        self.Y_test = torch.empty((0, 1))

        super().__init__(task_type)

    def _draw_gp_function(self, X, lengthscale=10.0, kernel_str="RBF"):
        if kernel_str == "RBF":
            kernel = RBFKernel()
        elif kernel_str == "Mat":
            kernel = MaternKernel(nu=0.5)
        else:
            raise Exception("Invalid kernel string: {}".format(kernel_str))

        kernel.lengthscale = lengthscale
        with torch.no_grad():
            lazy_cov = kernel(X)
            mean = torch.zeros(lazy_cov.size(0))
            mvn = MultivariateNormal(mean, lazy_cov)
            Y = mvn.rsample()[:, None]
        return Y

    @property
    def latent_function_values(self):
        if self._latent_function_values is None:
            ranges = [torch.linspace(_min, _max, self.samples_per_dim)
                      for _min, _max in self.global_scope]
            X_grids = torch.meshgrid(*ranges)
            X = torch.stack(X_grids).T.reshape(-1, self.input_dims)
            F = self.function(X)
            F_grid = F.reshape(*X_grids[0].shape)
            self._latent_function_values = (X_grids, F_grid)
        return self._latent_function_values

    def clear(self):
        self.X_train = self.X_train.new_empty((0, *self.X_train.shape[1:]))
        self.X_test = self.X_test.new_empty((0, *self.X_test.shape[1:]))
        self.Y_train = self.Y_train.new_empty((0, *self.Y_train.shape[1:]))
        self.Y_test = self.Y_test.new_empty((0, *self.Y_test.shape[1:]))

    def get_batch(self, batch_size=512, input_range=None):
        total_size = int(batch_size / (1 - self.test_size))

        # Sample input
        l, u = self.input_range.T

        r = torch.rand((total_size, self.input_dims))
        X = r * (u - l) + l
        assert torch.all((l <= X) & (X <= u))

        # Get clean output and add noise
        F = self.function(X)
        Y = F + torch.randn_like(F) * self.obs_noise

        # Divide into training/test and extend data
        self.X_train = torch.cat((self.X_train, X[:batch_size]))
        self.Y_train = torch.cat((self.Y_train, Y[:batch_size]))
        self.X_test = torch.cat((self.X_test, X[batch_size:]))
        self.Y_test = torch.cat((self.Y_test, Y[batch_size:]))

    def get_batch_multimode(self, batch_size, n_modes, lhs=True,
                            sigma=torch.tensor([[0.1]])):
        total_size = int(batch_size / (1 - self.test_size))
        l, u = self.input_range.T

        # sample from isotropic mixture of gaussians
        if lhs and n_modes > 1:
            splits = torch.linspace(l.item(), u.item(), n_modes + 1)
            locs = []
            for l_, u_ in zip(splits[:-1], splits[1:]):
                # unif = torch.distributions.Uniform(l_, u_)
                # locs.append(unif.sample([1]))
                locs.append(torch.tensor([(l_ + u_) / 2.]))
        else:
            unif = torch.distributions.Uniform(l, u)
            locs = [unif.sample() for _ in range(n_modes)]

        mixture = [MultivariateNormal(loc, sigma) for loc in locs]
        cs = torch.randint(0, n_modes, (total_size, ))
        X = torch.stack([mixture[i].sample() for i in cs])
        F = self.function(X)
        Y = F + torch.randn_like(F) * self.obs_noise

        # Divide into training/test and extend data
        self.X_train = torch.cat((self.X_train, X[:batch_size]))
        self.Y_train = torch.cat((self.Y_train, Y[:batch_size]))
        self.X_test = torch.cat((self.X_test, X[batch_size:]))
        self.Y_test = torch.cat((self.Y_test, Y[batch_size:]))


class SquareWave(Dataset):
    def __init__(self, n_observations=500, test_size=0.1, include_gap=False):

        if include_gap:
            n1 = n_observations // 2
            n2 = n_observations - n1
            X1 = torch.linspace(0, 7.5, n1)[:, None]
            X2 = torch.linspace(12.5, 20, n2)[:, None]
            X = torch.cat((X1, X2))
        else:
            X = torch.linspace(0, 20, n_observations)[:, None]
        Y = torch.round(torch.sin(X) * 0.5 + 0.5)

        n_test = int(n_observations * test_size)
        shuffled = torch.randperm(n_observations)
        test_idxs = torch.sort(shuffled[:n_test]).values
        train_idxs = torch.sort(shuffled[n_test:]).values

        self.X_train = X[train_idxs]
        self.Y_train = Y[train_idxs]
        self.X_test = X[test_idxs]
        self.Y_test = Y[test_idxs]

        super().__init__("regression")
