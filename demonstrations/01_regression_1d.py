#!/usr/bin/env python

import torch
import matplotlib.pyplot as plt
import seaborn as sns

from selective_gp.utils import load_data, get_model, remove_points


plt.ion()
plt.close("all")
sns.set(
    font_scale=1.5,
    style="whitegrid",
)


def _clear(ax):
    for artist in ax.lines + ax.collections + ax.patches:
        artist.remove()


def plot_latent(ax):
    _clear(ax)
    gp, = model.gps

    x_in = torch.linspace(xmin, xmax, 500)
    with torch.no_grad():
        x_u = gp.inducing_inputs.clone().flatten()
        u = gp.inducing_distribution.mean.clone()

        pred_dist = model.get_predictive_distribution(x_in[:, None])
        m, s = pred_dist.mean, pred_dist.stddev

        p = gp.variational_point_process.probabilities

    color = sns.xkcd_rgb["slate blue"]
    ax.plot(x_in, m, color=color)
    ax.fill_between(x_in, m - s, m + s, color=color, alpha=0.3)

    ax.plot(X.flatten(), Y.flatten(), "x", color=plt.cm.Greys(0.9, 0.5))

    color = [(0, 0, 0, p_) for p_ in p]
    ax.scatter(x_u, u, color=color, edgecolor="k")

    ax.figure.canvas.draw()
    plt.pause(1e-5)


def plot_probabilities(ax):
    _clear(ax)
    gp, = model.gps

    with torch.no_grad():
        x_u = gp.inducing_inputs.clone().flatten()
        p = gp.variational_point_process.probabilities

    color = sns.xkcd_rgb["moss green"]
    ax.bar(x_u, p, color=color, alpha=0.8, width=0.8, edgecolor=color)
    ax.set_ylim(0, 1)
    ax.figure.canvas.draw()
    plt.pause(1e-5)


def fit(epochs=500, cache_name=None):
    model.fit(X=X, Y=Y, max_epochs=epochs)


def prune(epochs=100, sfe=True, cache_name=None):
    model.fit_score_function_estimator(
        X=X, Y=Y, learning_rate=0.3, max_epochs=epochs, n_mcmc_samples=8)


# Load data and start model
dataset = load_data("synthetic_regression", input_dims=1, seed=0,
                    n_observations=200, x_min=0, x_max=50)
X, Y = dataset.X_train, dataset.Y_train
M = 50
mask = ((X < 15) | (35 < X)).flatten()
X = X[mask]
Y = Y[mask]
xmin, xmax = -10, 60
model = get_model(dataset, n_inducing=M, scale_X=False, scale_Y=False)
gp, = model.gps

# Initialise inducing inputs
gp.inducing_inputs = torch.linspace(xmin, xmax, M)

# Setup figure
fig = plt.figure(figsize=(12, 4))
gs = fig.add_gridspec(5, 3)
ax1 = fig.add_subplot(gs[:4, 0])
ax2 = fig.add_subplot(gs[:4, 1], sharey=ax1, sharex=ax1)
ax3 = fig.add_subplot(gs[4, 1], sharex=ax1)
ax4 = fig.add_subplot(gs[:4, 2], sharey=ax1, sharex=ax1)

for ax in fig.axes:
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.set_xlim(xmin, xmax)
plt.subplots_adjust(left=0.01, right=0.99, wspace=0.01, hspace=0.1,
                    bottom=0.01)
ax1.set_title("Pre-fitting")
ax2.set_title("Pruning")
ax4.set_title("Post-fitting")

# Initial fit with fixed probabilities
model.register_callback(plot_latent, (ax1,), update_interval=100)
fit(epochs=500)
gp.variational_point_process.probabilities = 1.0
plot_latent(ax1)

# Fit variational point process
gp.prior_point_process.rate.fill_(50)
gp.variational_point_process.probabilities = 0.2
model.register_callback(plot_latent, (ax2,), update_interval=50)
model.register_callback(plot_probabilities, (ax3,), update_interval=50)
prune(epochs=500)
model.unregister_callback(plot_probabilities)
plot_latent(ax2)

# Sample inducing points from point process and fit again
remove_points(gp)
gp.variational_point_process.probabilities = 1.0
model.register_callback(plot_latent, (ax4,), update_interval=100)
fit(epochs=500)
plot_latent(ax4)

plt.ioff()
plt.show()
