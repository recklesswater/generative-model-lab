"""Diagnostics that turn "is this chain mixing?" into numbers and pictures.

Autocorrelation, effective sample size and mode-hopping counts are the operational versions of
the operator language (transition kernel, spectral gap, mixing time): they measure exactly the
quantities those words refer to.
"""

from __future__ import annotations

import numpy as np


def autocorrelation(x: np.ndarray, max_lag: int = 500) -> np.ndarray:
    """Normalised autocorrelation of a 1D series, lags 0..max_lag."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom == 0.0:
        return np.ones(max_lag + 1)
    lags = np.arange(max_lag + 1)
    return np.array([float(np.dot(x[: len(x) - k], x[k:])) / denom for k in lags])


def integrated_autocorrelation_time(x: np.ndarray, max_lag: int = 500) -> float:
    """Geyer's initial positive sequence estimator of the integrated autocorrelation time."""
    rho = autocorrelation(x, max_lag)
    tau = 1.0
    for k in range(1, max_lag, 2):
        pair = rho[k] + rho[k + 1] if k + 1 <= max_lag else rho[k]
        if pair <= 0.0:
            break
        tau += 2.0 * pair
    return float(tau)


def effective_sample_size(x: np.ndarray, max_lag: int = 500) -> float:
    """ESS = n / tau for a single scalar chain."""
    n = len(x)
    tau = integrated_autocorrelation_time(x, max_lag)
    return float(n / max(tau, 1.0))


def count_mode_hops(x0: np.ndarray, threshold: float = 0.0) -> int:
    """Number of times a trajectory crosses ``threshold`` (i.e. changes well)."""
    side = (np.asarray(x0) > threshold).astype(int)
    return int(np.count_nonzero(np.diff(side) != 0))


def longest_well_residence(x0: np.ndarray, threshold: float = 0.0) -> int:
    """Longest run of consecutive samples spent in the same well - the visible symptom of a
    chain that is technically converging but practically stuck."""
    side = (np.asarray(x0) > threshold).astype(int)
    best = run = 1
    for a, b in zip(side[:-1], side[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return int(best)


def summarize(chains, target) -> list[dict]:
    """Compact summary table for a list of chains on a double-well target."""
    rows = []
    for chain in chains:
        x0 = chain.samples[:, 0]
        ess = effective_sample_size(x0)
        total_cost = chain.cost_per_step * chain.n_steps
        hops = count_mode_hops(x0)
        rows.append({
            "sampler": chain.name,
            "steps": chain.n_steps,
            "acceptance": chain.acceptance_rate,
            "ESS(x0)": ess,
            "ESS per 1k steps": 1000.0 * effective_sample_size(x0) / chain.n_steps,
            "cost per step": chain.cost_per_step,
            "ESS per 1k evals": 1000.0 * ess / total_cost if total_cost else float("nan"),
            "mode hops": hops,
            # Absolute hop counts are not comparable across samplers: the three runs use
            # different iteration budgets, and different cost per iteration. Rates are.
            "hops per 1k steps": 1000.0 * hops / chain.n_steps,
            "hops per 1k evals": 1000.0 * hops / total_cost if total_cost else float("nan"),
            "longest dwell": longest_well_residence(x0),
            "mean x0": float(x0.mean()),
            "std x0": float(x0.std(ddof=1)),
        })
    return rows
