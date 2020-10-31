#!/usr/bin/env python

import torch
from torch.optim import Adam
from tqdm.auto import tqdm


def _zero_baseline(*args, **kwargs):
    return 0


def _decaying_average_baseline(bound_mean_trace, lookback=200,
                               decay_rate=0.95):
    # Construct weight tensor
    t = torch.tensor(bound_mean_trace[-lookback:])
    w = torch.ones_like(t)
    w_ = decay_rate
    for i in torch.arange(len(w)) + 1:
        w[-i] = w_
        w_ *= decay_rate
    w.div_(w.sum())

    return (t * w).sum()


def _get_baseline(bound_mean_trace, *args, method="decaying_average",
                  **kwargs):
    if len(bound_mean_trace) == 0:
        return 1.0

    try:
        fn = {
            "zero_baseline": _zero_baseline,
            "decaying_average": _decaying_average_baseline,
        }[method]
    except KeyError:
        raise NotImplementedError(f"Unknown baseline function: {method}")

    baseline = fn(bound_mean_trace, *args, **kwargs)

    return baseline


def _sample(point_process):
    for _ in range(20):
        mask = point_process.sample().to(bool)
        if torch.any(mask):
            return mask
    raise RuntimeError("Could not sample mask.")


def score_function_estimator(objective_function, KL_function,
                             point_processes, learning_rate=0.3,
                             max_epochs=500, n_mcmc_samples=16, verbose=True,
                             callbacks={}, stop_conditions={}):
    if verbose:
        iterator = tqdm(range(max_epochs))
    else:
        iterator = range(max_epochs)

    parameters = []
    for pp in point_processes:
        parameters += list(pp.parameters())
        pp.train()
    dtype = parameters[0].dtype
    device = parameters[0].device
    optimizer = Adam(parameters, lr=learning_rate)

    # Baseline logic
    bound_mean_trace = []

    # Main loop
    for epoch in iterator:
        optimizer.zero_grad()

        # Prepare gradient dictionary
        grad_dict = {}
        for pp in point_processes:
            grad_dict[pp] = {}
            for name, parameter in pp.named_parameters():
                grad_tensor = torch.zeros(
                    (n_mcmc_samples, *parameter.shape), device=device,
                    dtype=dtype)
                grad_dict[pp][name] = grad_tensor

        # Sample pseudo-points and record bounds
        bounds = []
        samples = []
        with torch.no_grad():
            for k in range(n_mcmc_samples):
                sample_dict = {pp: _sample(pp) for pp in point_processes}
                bound = objective_function(sample_dict)

                bounds.append(bound)
                samples.append(sample_dict)

        # Calculate the baseline for gradient stabilization
        baseline = _get_baseline(bound_mean_trace, method="decaying_average")

        # Gradient estimation
        for k in range(n_mcmc_samples):
            optimizer.zero_grad()

            log_q = torch.zeros([], device=device, dtype=dtype)

            # Get log pmf for samples
            for pp in point_processes:
                mask = samples[k][pp]
                log_q.add_(pp.log_pmf(mask))

            surr_obj = log_q * (bounds[k] - baseline)
            surr_obj.backward()

            # Save to gradient dictionary
            for pp in point_processes:
                for name, parameter in pp.named_parameters():
                    grad_dict[pp][name][k] = parameter.grad.clone()

        # Baseline logic
        bound_mean_trace.append(sum(bounds) / n_mcmc_samples)

        # Get KL grads
        optimizer.zero_grad()
        kl = KL_function()
        kl.backward()

        # Subtract estimated objective gradients
        for pp in point_processes:
            for name, parameter in pp.named_parameters():
                all_grads = grad_dict[pp][name]
                parameter.grad -= all_grads.mean(axis=0)

        if verbose:
            iterator.set_postfix(ELBO=(bound_mean_trace[-1] - kl).item())

        # Let Adam optimizer take step
        optimizer.step()

        # Callbacks
        for cb, (args, kwargs, update_interval) in callbacks.items():
            if epoch % update_interval == 0:
                cb(*args, **kwargs)

        # Stop conditions
        stop = False
        for sc, (args, kwargs) in stop_conditions.items():
            if sc(*args, **kwargs):
                stop = True
                break
        if stop:
            break

    if verbose:
        iterator.refresh()
        iterator.close()
