"""Small measurement helpers: free-energy profiles along a coordinate, and sampler tuning.

Nothing here is specific to Krylov. These are the two operations the demos repeat - histogram a
1D projection of a chain and read a barrier off it, and pick a step size that a sampler will
actually accept - factored out so the scripts read as experiments rather than bookkeeping.
"""

from __future__ import annotations

import numpy as np


def free_energy_profile(z: np.ndarray, n_bins: int = 80, lo: float = 1.0, hi: float = 99.0,
                        min_counts: float = 25.0):
    """Histogram a coordinate and return ``(F, centres, counts)`` with ``min(F) = 0``.

    ``F = -log(counts)`` is the free energy along ``z`` up to an additive constant; the profile
    is shifted so its minimum is zero, because only differences are meaningful. Bins are equal
    width and the range is the ``lo``-``hi`` percentile of the data, so that empty tails do not
    manufacture spikes. Bins with fewer than ``min_counts`` samples are marked by ``nan`` so that
    a caller cannot accidentally read a barrier off five samples.
    """
    z = np.asarray(z, dtype=float).ravel()
    edges = np.linspace(*np.percentile(z, [lo, hi]), n_bins + 1)
    counts, edges = np.histogram(z, bins=edges)
    centres = 0.5 * (edges[:-1] + edges[1:])
    with np.errstate(divide="ignore"):
        F = -np.log(counts.astype(float))
    F -= np.nanmin(F)
    F[counts < min_counts] = np.nan
    return F, centres, counts


def apparent_barrier(F: np.ndarray, centres: np.ndarray) -> float:
    """``F`` at the coordinate origin minus the deepest well: the barrier *that coordinate sees*.

    For the true reaction coordinate this is the barrier height. For any other direction it is
    not, and the point of the experiment is that the two numbers differ by a lot. Reported as a
    difference, so the additive constant in the free energy cancels.
    """
    i0 = int(np.argmin(np.abs(centres)))
    if not np.isfinite(F[i0]) or not np.isfinite(F).any():
        return float("nan")
    return float(F[i0] - np.nanmin(F))


def profile_deviation(F: np.ndarray, centres: np.ndarray,
                      reference: np.ndarray) -> float:
    """RMS deviation between a measured profile and a reference profile on the same grid.

    Both are shifted to their own minimum first, so the additive constant drops out. Bins with
    no data (``nan``) are ignored; the caller is told how many were used via the return value of
    ``profile_overlap`` if it needs that.
    """
    mask = np.isfinite(F)
    if mask.sum() < 3:
        return float("nan")
    ref = reference - np.nanmin(reference[mask])
    return float(np.sqrt(np.mean((F[mask] - ref[mask]) ** 2)))


def tune_step_size(run, target, x0: np.ndarray, target_accept: float = 0.55,
                   lo: float = 0.02, hi: float = 1.0, n_pilot: int = 2000,
                   seed: int = 0, tol: float = 0.03, max_iter: int = 14) -> tuple[float, float]:
    """Bisect the step size until the pilot acceptance rate is near ``target_accept``.

    Acceptance falls monotonically with step size for all three samplers in module 01, so
    bisection is enough. Returns ``(step_size, acceptance)``. Tuning on a short pilot and then
    freezing it is the convention module 01 uses; it keeps every reported number reproducible but
    it means the step size is *not* re-tuned per dimension, which the scan notes as a caveat.
    """
    best = (lo, float("nan"))
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        chain = run(target, n_steps=n_pilot, step_size=mid, x0=x0, seed=seed)
        acc = chain.acceptance_rate
        best = (mid, acc)
        if abs(acc - target_accept) <= tol:
            break
        if acc > target_accept:
            lo = mid
        else:
            hi = mid
    return best


def integrated_autocorrelation_time(series: np.ndarray, max_lag: int = 400) -> float:
    """Geyer's initial-positive-sequence estimator, same convention as module 01."""
    x = np.asarray(series, dtype=float).ravel()
    x = x - x.mean()
    denom = float(x @ x)
    if denom == 0.0:
        return 1.0
    lags = np.arange(max_lag + 1)
    rho = np.array([float(x[: len(x) - k] @ x[k:]) / denom for k in lags])
    tau = 1.0
    for k in range(1, max_lag, 2):
        pair = rho[k] + rho[k + 1] if k + 1 <= max_lag else rho[k]
        if pair <= 0.0:
            break
        tau += 2.0 * pair
    return float(tau)


def timescale_from_eigenvalue(mu: float, tau: int) -> float:
    """Implied timescale ``-tau / log mu`` of a transfer-operator eigenvalue. ``inf`` at mu = 1."""
    if not np.isfinite(mu) or mu <= 0.0:
        return float("nan")
    if mu >= 1.0:
        return float("inf")
    return float(-tau / np.log(mu))


def barrier_with_error(X: np.ndarray, direction: np.ndarray, n_bins: int = 80,
                       n_boot: int = 40, seed: int = 0):
    """``(barrier, standard_error)`` along a direction, with a bootstrap error bar.

    The barrier is read off a histogram, so a single number hides how noisy it is: at the top of
    a 2.5 kT barrier there are few samples, and the minimum over 80 bins is itself biased down by
    the noise. Reporting the bootstrap spread is the difference between "the barrier is 2.8" and
    "the barrier is 2.8 +- 0.1", and only the second one can be compared with a prediction.
    """
    z = np.asarray(X, dtype=float) @ np.asarray(direction, dtype=float).ravel()
    rng = np.random.default_rng(seed)
    F, centres, _ = free_energy_profile(z, n_bins=n_bins)
    point = apparent_barrier(F, centres)
    n = z.size
    boots = np.empty(n_boot)
    for b in range(n_boot):
        z_b = z[rng.integers(0, n, n)]
        F_b, c_b, _ = free_energy_profile(z_b, n_bins=n_bins)
        boots[b] = apparent_barrier(F_b, c_b)
    return point, float(np.nanstd(boots, ddof=1))


def print_table(headers, rows, floatfmt: str = "{:.3f}") -> None:
    """Print an aligned text table (the same shape of output module 01 and 05 use)."""
    for row in rows:
        if len(row) != len(headers):
            raise ValueError(f"row has {len(row)} cells, header has {len(headers)}: {row}")
    cells = []
    for row in rows:
        out = []
        for value in row:
            if isinstance(value, str):
                out.append(value)
            elif value is None or (isinstance(value, float) and not np.isfinite(value)):
                out.append("   -")
            elif float(value).is_integer() and abs(value) < 1e12:
                out.append(str(int(value)))
            else:
                out.append(floatfmt.format(value))
        cells.append(out)
    widths = [max(len(str(headers[i])), *(len(r[i]) for r in cells)) if cells
              else len(str(headers[i])) for i in range(len(headers))]
    print("  " + "  ".join(str(headers[i]).rjust(widths[i]) for i in range(len(headers))))
    print("  " + "  ".join("-" * widths[i] for i in range(len(headers))))
    for r in cells:
        print("  " + "  ".join(r[i].rjust(widths[i]) for i in range(len(headers))))
