"""The MMSE denoiser is an orthogonal projection - verified numerically.

Claim: E[x0 | x_t] is the L2 projection of x0 onto the space of functions that are measurable
with respect to x_t - i.e. onto everything you are allowed to know at time t. The defining
property of a projection is orthogonality of the residual:

    E[ (x0 - E[x0|x_t]) * g(x_t) ] = 0     for every function g of x_t.

For a Gaussian-mixture prior both sides are available in closed form, so this can be measured
rather than argued. This is the linear-algebra picture behind "the denoiser is not magic": it
is a projection, and the score is the direction of the residual (Tweedie's formula).

Usage:  python 05_spectral_view/projection_demo.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIGDIR = Path(__file__).resolve().parent.parent / "figures"
RNG = np.random.default_rng(0)

WEIGHTS = np.array([0.40, 0.35, 0.25])
MEANS = np.array([[-1.6, 0.4], [1.7, -0.5], [0.3, 1.8]])
COVS = np.array([[[0.22, 0.05], [0.05, 0.16]],
                 [[0.18, -0.06], [-0.06, 0.26]],
                 [[0.30, 0.02], [0.02, 0.12]]])


def responsibilities(x: np.ndarray, sigma2: float) -> np.ndarray:
    """Posterior component weights, shape (n, n_components)."""
    logs = []
    for w, m, c in zip(WEIGHTS, MEANS, COVS):
        cov = c + sigma2 * np.eye(2)
        diff = x - m
        inv = np.linalg.inv(cov)
        quad = np.einsum("ni,ij,nj->n", diff, inv, diff)
        logs.append(np.log(w) - 0.5 * (np.log(np.linalg.det(cov)) + quad))
    logs = np.array(logs)
    logs -= logs.max(axis=0, keepdims=True)
    w = np.exp(logs)
    return (w / w.sum(axis=0, keepdims=True)).T


def mmse_denoiser(x: np.ndarray, sigma2: float) -> np.ndarray:
    """E[x0 | x_t = x] in closed form for the Gaussian mixture."""
    r = responsibilities(x, sigma2)
    out = np.zeros_like(x)
    for i, (m, c) in enumerate(zip(MEANS, COVS)):
        cov = c + sigma2 * np.eye(2)
        # E[x0 | component i] = m_i + Sigma_i (Sigma_i + sigma^2 I)^-1 (x - m_i)
        contrib = m[None, :] + ((c @ np.linalg.inv(cov)) @ (x - m).T).T
        out += r[:, [i]] * contrib
    return out


def sample_prior(n: int) -> np.ndarray:
    idx = RNG.choice(len(WEIGHTS), size=n, p=WEIGHTS)
    return np.array([RNG.multivariate_normal(MEANS[i], COVS[i]) for i in idx])


def main() -> None:
    FIGDIR.mkdir(exist_ok=True)
    sigma2 = 0.35
    n = 200_000

    x0 = sample_prior(n)
    xt = x0 + np.sqrt(sigma2) * RNG.standard_normal((n, 2))
    xhat = mmse_denoiser(xt, sigma2)
    resid = x0 - xhat

    tests = {
        "1": np.ones(n),
        "x": xt[:, 0],
        "y": xt[:, 1],
        "x^2": xt[:, 0] ** 2,
        "xy": xt[:, 0] * xt[:, 1],
        "y^2": xt[:, 1] ** 2,
        "cos x": np.cos(xt[:, 0]),
        "sin y": np.sin(xt[:, 1]),
    }
    scale = float(np.abs(resid).mean())
    print(f"orthogonality check  (sigma^2 = {sigma2}, n = {n:,} samples)")
    print(f"  mean |residual| = {scale:.4f}")
    worst = 0.0
    for name, g in tests.items():
        val = abs(float(np.mean(resid[:, 0] * g)))
        worst = max(worst, val)
        if name in ("1", "x", "x^2", "cos x"):
            print(f"    E[ residual_x * {name:>6} ] = {val:+.2e}")
    for name, g in tests.items():
        worst = max(worst, abs(float(np.mean(resid[:, 1] * g))))
    print(f"  worst |E[residual * g(x_t)]| over every test function = {worst:.2e}")
    print("  that is what 'orthogonal projection' means, measured")

    grid = np.linspace(-3.2, 3.2, 24)
    gx, gy = np.meshgrid(grid, grid)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    field = mmse_denoiser(pts, sigma2)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    ax = axes[0]
    cloud = sample_prior(4000)
    ax.scatter(cloud[:, 0], cloud[:, 1], s=4, alpha=0.22, color="#a0aec0",
               label="prior samples")
    ax.quiver(pts[:, 0], pts[:, 1], field[:, 0] - pts[:, 0], field[:, 1] - pts[:, 1],
              color="#2b6cb0", angles="xy", scale_units="xy", scale=14, width=0.004)
    for m in MEANS:
        ax.plot(m[0], m[1], marker="x", color="#c53030", ms=9)
    ax.set_title(f"MMSE denoiser at sigma^2 = {sigma2}\n"
                 "arrows: from x_t to the expected clean sample", fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    names = list(tests)
    vals = [abs(float(np.mean(resid[:, 0] * tests[k]))) for k in names]
    ax.barh(names, vals, color="#2f855a")
    ax.axvline(worst, color="#c53030", ls="--", lw=1, label=f"worst = {worst:.1e}")
    ax.set_xlabel("| E[ residual_x * g(x_t) ] |")
    ax.set_title("The residual is orthogonal to every function of x_t\n"
                 f"(bars ~1e-3 against a residual scale of {scale:.2f})", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGDIR / "05_projection_demo.png", dpi=150)
    plt.close(fig)
    print(f"figures written to {FIGDIR}")


if __name__ == "__main__":
    main()
