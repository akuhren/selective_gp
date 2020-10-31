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


def plot_prediction(ax):
    _clear(ax)

    with torch.no_grad():
        X_u = gp.inducing_inputs.clone()
        pred_dist = model.get_predictive_distribution(X)
        y_mean = pred_dist.mean

        p = gp.variational_point_process.probabilities

    palette = sns.diverging_palette(150, 20, s=80, l=55, as_cmap=True)
    ax.scatter(*X.T, c=y_mean, cmap=palette, s=30)

    color = [(0, 0, 0, p_) for p_ in p]
    ax.scatter(*X_u.T, color=color, edgecolor="k", s=40)

    ax.figure.canvas.draw()

    ax.set(xlim=(0, 100), ylim=(0, 100))
    plt.pause(1e-5)


def fit(epochs=500, cache_name=None):
    model.fit(X=X, Y=Y, max_epochs=epochs)


def prune(epochs=100, sfe=True, cache_name=None):
    model.fit_score_function_estimator(
        X=X, Y=Y, learning_rate=0.3, max_epochs=epochs, n_mcmc_samples=8)


def get_acc():
    f_dist = gp(X_test)
    y_dist = model.likelihood(f_dist)
    Y_sample = y_dist.sample()
    return (Y_sample.flatten() == Y_test.flatten()).to(float).mean()


# Load data and start model
dataset = load_data("synthetic_classification", input_dims=2, seed=1,
                    n_observations=10000, test_size=0.5)
X, Y = dataset.X_train, dataset.Y_train
X_test, Y_test = dataset.X_test, dataset.Y_test


def _remove_square(X_, Y_, max_obs=1000):
    mask = torch.all((30 < X_) & (X_ < 70), axis=1)
    return X_[~mask][:max_obs], Y_[~mask][:max_obs]


X, Y = _remove_square(X, Y)
X_test, Y_test = _remove_square(X_test, Y_test)
M_sqrt = 15
model = get_model(dataset, n_inducing=M_sqrt**2, scale_X=False, scale_Y=False)
gp, = model.gps

# Place inducing points on grid
X_range = torch.linspace(X.min(), X.max(), M_sqrt)
X_u_mesh = torch.stack(torch.meshgrid(X_range, X_range))
X_u_init = X_u_mesh.reshape(2, -1).T
gp.inducing_inputs.data.copy_(X_u_init)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 4), sharey=True,
                                    sharex=True)
ax1.set(xticklabels=[], yticklabels=[], xlim=(0, 100), ylim=(0, 100))
ax1.set_title("Pre-fitting\n")
ax2.set_title("Pruning\n")
ax3.set_title("Post-fitting\n")
plt.tight_layout()

model.register_callback(plot_prediction, (ax1,), update_interval=100)

# Initial fit with fixed probabilities
fit(epochs=500)
gp.variational_point_process.probabilities = 1.0
plot_prediction(ax1)
ax1.set_title(f"Pre-fitting\n"
              f"Accuracy: {get_acc():.2f}")

# Fit variational point process
gp.prior_point_process.rate.fill_(0.05)
gp.variational_point_process.probabilities = 0.2
model.register_callback(plot_prediction, (ax2,), update_interval=100)
prune(epochs=500)
plot_prediction(ax2)

# Sample inducing points from point process and fit again
remove_points(gp)
gp.variational_point_process.probabilities = 1.0
model.register_callback(plot_prediction, (ax3,), update_interval=100)
fit(epochs=500)
plot_prediction(ax3)
ax3.set_title(f"Post-fitting\n"
              f"Accuracy: {get_acc():.2f}")

fig.canvas.draw()
plt.pause(1e-5)
plt.ioff()
plt.show()
