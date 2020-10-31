#!/usr/bin/env python

import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_samples(model, ax=None, resolution=1000, n_samples=10, xlim=None):
    if ax is None:
        ax = plt.subplots()[1]

    if xlim is None:
        xlim = ax.get_xlim()

    x_in = torch.linspace(*xlim, resolution)
    y_samples = []
    for _ in range(n_samples):
        f = x_in
        for gp in model.gps:
            f_dist = gp(f[:, None])
            f_dist._covar = f_dist._covar.add_jitter()
            f = f_dist.sample()
        y_samples.append(f.tolist())

    ax.plot(x_in, np.transpose(y_samples), color=(0, 0, 0, 0.1))


def plot_density(model, ax, resolution=10, n_samples=50, xlim=None, ylim=None,
                 cmap=plt.cm.Blues):

    if xlim is None:
        xlim = ax.get_xlim()

    if ylim is None:
        ylim = ax.get_ylim()

    xmin, xmax = xlim
    ymin, ymax = ylim

    def _add_density(x_array, y_array):
        i = ((y_array - ymin) / (ymax - ymin) * dim_y).to(int)
        j = ((x_array - xmin) / (xmax - xmin) * dim_x).to(int)
        for i_, j_ in zip(i, j):
            try:
                M[i_, j_] += 1.0
            except IndexError:
                continue

    dim_x = int((xmax - xmin) * n_samples)
    dim_y = int((ymax - ymin) * n_samples)
    M = np.zeros((dim_y, dim_x))

    x_in = torch.linspace(xmin, xmax, dim_x)
    for _ in range(n_samples):
        f = x_in
        for gp in model.gps:
            f_dist = gp(f[:, None])
            m, s = f_dist.mean, f_dist.stddev
            f = torch.distributions.Normal(loc=m, scale=s).sample()
        _add_density(x_in, f)

    yr = np.linspace(ymin, ymax, dim_y)
    xr = np.linspace(xmin, xmax, dim_x)
    y, x = np.meshgrid(xr, yr)
    ax.pcolormesh(y, x, M, cmap=cmap, zorder=-1, vmin=0, vmax=M.max())
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))


def plot_deep_latent(model, axes=None, xlims=None, **kwargs):
    if axes is None:
        axes = plt.subplots(1, len(model.gps))[1]
        for ax in axes:
            ax.margins(x=0)

    if xlims is None:
        xlims = [None] * len(model.gps)
    else:
        assert len(xlims) == len(model.gps)

    for gp, ax, xlim in zip(model.gps, axes, xlims):
        plot_latent(gp, ax=ax, xlim=xlim, **kwargs)


def plot_latent(gp, ax=None, xlim=None, resolution=200, cmap=plt.cm.Set1):
    if ax is None:
        ax = plt.subplots()[1]

    with torch.no_grad():
        x_u = gp.inducing_inputs.clone().flatten()
        u = gp.inducing_distribution.mean.clone()

        p = gp.variational_point_process.probabilities
        color = [(0, 0, 0, p_) for p_ in p]

    ax.scatter(x_u, u, color=color, edgecolor="k")

    if xlim is None:
        xlim = ax.get_xlim()

    x_ = torch.linspace(*xlim, resolution)

    with torch.no_grad():
        f_dist = gp(x_[:, None])

    m, s = f_dist.mean, f_dist.stddev
    ax.plot(x_, m, color=cmap(1))
    ax.fill_between(x_, m - s, m + s, color=cmap(1, 0.3))


def plot_probabilities(gp, ax=None, color=None):
    if ax is None:
        ax = plt.subplots()[1]

    with torch.no_grad():
        x_u = gp.inducing_inputs.clone().flatten()
        p = gp.variational_point_process.probabilities

    if color is None:
        color = plt.cm.Set(0)[:3]
    ax.bar(x_u, p, color=color + (0.5,), width=2, edgecolor=color)
    ax.set_ylim(0, 1)
