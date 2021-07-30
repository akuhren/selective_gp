Code accompanying the paper [Probabilistic Selection of Inducing Points in Sparse Gaussian Processes](https://arxiv.org/abs/2010.09370) by Uhrenholt, Charvet, and Jensen.

Published at The 37th Conference on Uncertainty in Artificial Intelligence (UAI), 2021.

## Paper synopsis
The motivating problem of this paper is that of deciding how many inducing points to rely on when fitting a sparse Gaussian process. This is ultimately a trade-off between model complexity and capacity, and the appropriate number of inducing points depends on the available computing power and the required fidelity of the model predictions. While gridsearch over the number of points is certainly possible, it will for large ranges be very time-consuming, especially in the context of deep Gaussian processes where different layers may have different fidelities and thus require different number of inducing points for an optimal trade-off.

We present a probabilistic approach towards tackling this problem in which each inducing points is associated with a probability of inclusion that is set by the model. Meanwhile, the model is penalised for the total amount of probability assigned and so must choose to include only those points that best explain the observed data.

[![1D regression](https://github.com/akuhren/selective_gp/blob/master/img/reg_1d.gif "1D regression")](https://github.com/akuhren/selective_gp/blob/master/img/reg_1d.gif "1D regression")

This is achieved by expanding the hierarchical model with a point process prior over inducing inputs that favours smaller sets of points. To accomodate this expansion, we also introduce a variational point process that assigns probabilities to individual points. The KL divergence between these two point processes is closed-form computable, and so the new expected likelihood can be optimised straightforwardly through score function estimation.

The new model is agnostic to number of input/output dimensions and is readily applicable for regression, classification, latent variable modelling, and even deep Gaussian processes:

[![DGP regression](https://github.com/akuhren/selective_gp/blob/master/img/reg_dgp.gif "DGP regression")](https://github.com/akuhren/selective_gp/blob/master/img/reg_dgp.gif "DGP regression")

In the above example, note that the learnt functions in layer 2 and 3 are less complex than in layer 1. Consequently, the model can prune away more points in those layers without significant deterioration to the final posterior.

Finally, we can also learn the point process jointly along with the remaining model parameters. This is illustrated below in the context of Gaussian process Latent Variable Modelling (GP-LVM) where we learn a low-dimensional representation along with the function. In this example we use a single-cell gene expression dataset where each observation has 48 dimensions (qPCR values) in the original space.

[![GPLVM](https://github.com/akuhren/selective_gp/blob/master/img/gplvm.gif "GPLVM")](https://github.com/akuhren/selective_gp/blob/master/img/gplvm.gif "GPLVM")

Squares are observed embeddings in the latent space, coloured according to cell-stage (not known by the model). Circles are inducing inputs which are filled according to the probability of inclusion. Note that even though we initialise with a fairly high number of inducing points, the model only ever relies on a small, informative subset.

## Installation
The code is developed and tested for Python 3.7.

### Install locally
To install locally, first setup a virtual environment through e.g. Anaconda:
```
conda create -n selective_gp python=3.7
source activate selective_gp
```

Install module and dependencies:
```
pip install -r requirements.txt
pip install -e .
```

### Run notebooks via docker
Alternatively, the notebooks can be run interactively through a Docker-container as such:

```
docker build -t username/selective-gp .
docker run -p 8888:8888 -e JUPYTER_ALLOW_INSECURE_WRITES=true username/selective-gp
```

Now copy one of the listed URL's to a browser to access the notebook.

### MLFlow
Experiments are run through [MLFlow](https://mlflow.org/). By default, all results are saved directly in a subfolder of `experiments`, but this can be changed by setting the environment variable `MLFLOW_TRACKING_URI`.
