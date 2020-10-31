#!/usr/bin/env python

import torch
from torch.distributions import Normal
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from selective_gp.utils import (
    load_data, get_model, remove_points, fit_layerwise)


plt.ion()
sns.set(
    font_scale=1.3,
    style="whitegrid",
)


def plot_data(ax):
    ax.plot(X.flatten(), Y.flatten(), "kx")


@torch.no_grad()
def plot_latent(*axes, resolution=100):
    for ax, gp in zip(axes, model.gps):
        if ax is axes[0]:
            xlim = ax.get_xlim()

        x_u = gp.inducing_inputs.clone().flatten()
        u = gp.inducing_distribution.mean.clone()
        p = gp.variational_point_process.probabilities
        color = [(0, 0, 0, p_) for p_ in p]
        ax.scatter(x_u, u, color=color, edgecolor="k")

        if ax is not axes[0]:
            xlim = ax.get_xlim()
        x_ = torch.linspace(*ax.get_xlim(), resolution)
        f_dist = gp(x_[:, None])
        m, s = f_dist.mean, f_dist.stddev
        color = sns.xkcd_rgb["slate blue"]
        ax.plot(x_, m, color=color)
        ax.fill_between(x_, m - s, m + s, color=color, alpha=0.3)

        ax.set_xlim(*xlim)


def plot_samples(ax, resolution=500, n_samples=5):
    model.eval()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_in = torch.linspace(*xlim, resolution)
    y_samples = []
    for _ in range(n_samples):
        f = x_in
        for gp in model.gps:
            f_dist = gp(f[:, None])
            f_dist._covar = f_dist._covar.add_jitter()
            f = f_dist.sample()
        y_samples.append(f.tolist())
    color = sns.xkcd_rgb["slate blue"]
    ax.plot(x_in, np.transpose(y_samples), color=color, alpha=0.3)
    ax.set(xlim=xlim, ylim=ylim)
    model.train()


def plot_density(ax, resolution=5, n_samples=20):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

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
            f = Normal(loc=m, scale=s).rsample()
        _add_density(x_in, f)

    yr = np.linspace(ymin, ymax, dim_y)
    xr = np.linspace(xmin, xmax, dim_x)
    y, x = np.meshgrid(xr, yr)
    ax.pcolormesh(y, x, M, cmap="Reds", zorder=-1)
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))


def plotted(f):
    def inner(*args, fig_title=None, **kwargs):
        fig, axes = plt.subplots(1, 5, figsize=(16, 3))
        if fig_title:
            fig.suptitle(fig_title)
        axes[0].set_xlim(-5, 25)
        for ax in axes[3:]:
            ax.set(xlim=(-5, 25), ylim=(-0.5, 1.5))
        for i, ax in enumerate(axes[:3], 1):
            ax.set_title(f"Latent function, layer {i}")
        axes[3].set_title("Posterior samples")
        axes[4].set_title("Density map")
        plt.tight_layout()
        plt.subplots_adjust(top=0.8)

        def _setup():
            for ax in axes:
                for artist in ax.lines + ax.collections:
                    artist.remove()

        def _finalize():
            fig.canvas.draw()
            plt.pause(1e-5)

        model.register_callback(_setup, update_interval=100)
        model.register_callback(plot_latent, axes[:3], update_interval=100)
        model.register_callback(plot_samples, (axes[3],), update_interval=100)
        model.register_callback(plot_density, (axes[4],), update_interval=100)
        model.register_callback(_finalize, update_interval=100)

        res = f(*args, **kwargs)

        model.callbacks.clear()

        return res
    return inner


@plotted
def pre_fit(cache_name=None):
    for gp in model.gps:
        gp.variational_point_process.probabilities = 1.0
    fit_layerwise(model, dataset, batch_size=None, max_epochs=500)
    model.fit(X=X, Y=Y, max_epochs=500, hp_learning_rate=0.005,
              var_learning_rate=0.01)


@plotted
def post_fit(cache_name=None):
    model.fit(X=X, Y=Y, max_epochs=500, hp_learning_rate=0.005,
              var_learning_rate=0.01)


@plotted
def prune(sfe=True, cache_name=None):
    for gp in model.gps:
        gp.prior_point_process.rate.fill_(0.5)
        gp.variational_point_process.probabilities = 0.5

    model.fit_score_function_estimator(
        X=X, Y=Y, learning_rate=0.3, max_epochs=1000, n_mcmc_samples=16)


# Sample clean observations from a square wave and define 3 layer model with
# 50 inducing points in each layer
dataset = load_data("square_wave", n_observations=100)
X, Y = dataset.X_train, dataset.Y_train
M = 50
model = get_model(dataset, n_inducing=M, n_layers=3, scale_X=False,
                  scale_Y=False)

# Initialise inducing points as equi-distant arrays
X_range = torch.linspace(X.min(), X.max(), M)
for i, gp in enumerate(model.gps):
    r = (X.min(), X.max()) if i == 0 else (-5, 5)
    gp.inducing_inputs = torch.linspace(*r, M)

# First we fit without point process, then we prune by learning posterior
# probability of inclusion and sampling inducing points from resulting
# distribution, and then we fit again
pre_fit(fig_title="Pre-fitting")
prune(fig_title="Pruning")
for gp in model.gps:
    remove_points(gp)
    gp.variational_point_process.probabilities = 1.0
post_fit(fig_title="Post-fitting")

plt.ioff()
plt.show()
