#!/usr/bin/env python

import torch
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from selective_gp.utils import load_data, get_gplvm_model, remove_points


plt.ion()
plt.close("all")
sns.set(
    font_scale=1.5,
    style="whitegrid",
    context="paper"
)

# Get data for qPCR dataset
dataset = load_data("QPCR", seed=0, lvm_model=True, latent_dimensions=2,
                    test_size=0)
Y = dataset.Y_train

# Initialise 100 inducing points on a 10 by 10 grid
M_sqrt = 10
model = get_gplvm_model(dataset, n_inducing=M_sqrt**2, scale_Y=True,
                        projection_dimensions=None)
latent = model.latent_layer
gp, = model.gps
Z_init = latent.variational_mean.detach()
Z_range = torch.linspace(Z_init.min(), Z_init.max(), M_sqrt) * 1.2
X_u_mesh = torch.stack(torch.meshgrid(Z_range, Z_range))
gp.inducing_inputs = X_u_mesh.reshape(2, -1).T


def plot_latent(ax):
    for artist in ax.lines + ax.collections + ax.patches:
        artist.remove()

    with torch.no_grad():
        X_u = gp.inducing_inputs.clone()
        X_latent = latent.variational_mean.clone()
        p = gp.variational_point_process.probabilities

    M = np.hstack((X_latent.numpy(), dataset.labels_train[:, None]))
    df = pd.DataFrame(M, columns=["Z_1", "Z_2", "Label"])
    # TODO move legend outside
    legend = "brief" if ax is ax1 else False
    sns.scatterplot(x="Z_1", y="Z_2", hue="Label", data=df, ax=ax, s=30,
                    legend=legend)
    if legend:
        ax.legend(bbox_to_anchor=(0, 1), loc="upper right")

    color = [(0, 0, 0, p_) for p_ in p]
    ax.scatter(*X_u.T, color=color, edgecolor="k", s=20)
    ax.figure.canvas.draw()

    with torch.no_grad():
        X = torch.arange(len(Y))
        ELBO = model._log_lik(X=X, Y=Y) - model._KL()
    xlabel = None if ax is ax2 else f"ELBO: {ELBO:.4e}"
    ax.set(ylabel=None, xlabel=xlabel)
    plt.pause(1e-5)


# Setup figure
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4))
Z = gp.inducing_inputs.detach()
xmin, xmax = Z.min() - 2, Z.max() + 2
for ax in fig.axes:
    ticks = np.arange(-10, 10, 1)
    ax.set(xticklabels=[], yticklabels=[], xlim=(xmin, xmax),
           ylim=(xmin, xmax), aspect="equal", xticks=ticks, yticks=ticks)
ax1.set_title("Pre-fitting")
ax2.set_title("Pruning")
ax3.set_title("Post-fitting")
plt.tight_layout()
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.99)

# Pre-fitting. We first fit 200 epochs while keeping inducing inputs fixed,
# then another 500 with them being free
model.register_callback(plot_latent, (ax1,), update_interval=10)
gp.variational_point_process.probabilities = 1.0

model.add_fixed_parameter("inducing_points")
model.fit(Y=Y, max_epochs=200)
model.remove_fixed_parameter("inducing_points")
model.fit(Y=Y, max_epochs=800)
plot_latent(ax1)

# Pruning. We use score function estimation to fit the variational point
# process over 1000 epochs
model.register_callback(plot_latent, (ax2,), update_interval=5)
gp.prior_point_process.rate.fill_(5.0)
gp.variational_point_process.probabilities = 0.1
model.fit_score_function_estimator(
    Y=Y, learning_rate=0.5, max_epochs=200, n_mcmc_samples=64)
plot_latent(ax2)

# Post-fit after taking sample of inducing points
remove_points(gp)
model.register_callback(plot_latent, (ax3,), update_interval=10)
gp.variational_point_process.probabilities = 1.0
model.fit(Y=Y, max_epochs=500)
plot_latent(ax3)

plt.ioff()
plt.show()
