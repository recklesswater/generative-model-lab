"""Conditional generation = adding the gradient of a functional to the score.

A property ("end-to-end distance", "radius", "a direction") is a functional f(x) of the
configuration. Guiding a sampler towards that property means adding grad f to the score:

    score_guided(x) = grad log p_sigma(x) + gamma * grad f(x)

What that does depends on the *kind* of functional, and the difference is exactly the linear
versus quadratic distinction from linear algebra:

  * linear functional  f(x) = a . x          -> grad f is constant -> the distribution is
                                                TRANSLATED by gamma * a
  * radial functional  f(x) = (|x - c| - r)^2 -> grad f pulls radially -> the distribution
                                                CONTRACTS onto a shell of radius r

A distance guide in the SE(3) module (04) is the same mechanism with a richer f: a property of
the clean sample, differentiated, and pushed through the reverse process. The score here is
analytic (Gaussian mixture), so no training is involved and the conclusions are exact.

Usage:  python 05_spectral_view/guidance_functionals.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIGDIR = Path(__file__).resolve().parent.parent / "figures"
RNG = np.random.default_rng(1)

WEIGHTS = np.array([0.55, 0.45])
MEANS = np.array([[-1.2, 0.0], [1.2, 0.0]])
COVS = np.array([[[0.30, 0.0], [0.0, 0.30]],
                 [[0.30, 0.0], [0.0, 0.30]]])


def score(x: np.ndarray, sigma2: float) -> np.ndarray:
    """grad log p_sigma(x) for the Gaussian mixture (closed form)."""
    return (denoiser(x, sigma2) - x) / sigma2      # Tweedie's formula


def denoiser(x: np.ndarray, sigma2: float) -> np.ndarray:
    """E[x0 | x_sigma = x]: the minimum-mean-square-error clean estimate."""
    logs, means = [], []
    for w, m, c in zip(WEIGHTS, MEANS, COVS):
        cov = c + sigma2 * np.eye(2)
        inv = np.linalg.inv(cov)
        diff = x - m
        quad = np.einsum("ni,ij,nj->n", diff, inv, diff)
        logs.append(np.log(w) - 0.5 * (np.log(np.linalg.det(cov)) + quad))
        means.append(m[None, :] + (c @ inv @ diff.T).T)
    logs = np.array(logs)
    logs -= logs.max(axis=0, keepdims=True)
    r = np.exp(logs)
    r /= r.sum(axis=0, keepdims=True)
    # r: (K, n) responsibilities, means: (K, n, 2) per-component clean estimates
    return np.einsum("kn,knj->nj", r, np.array(means))


def grad_linear(x: np.ndarray) -> np.ndarray:
    """grad f for f(x) = a . x with a = (1, 0): a constant vector field."""
    return np.tile(np.array([1.0, 0.0]), (x.shape[0], 1))


def grad_radial(x: np.ndarray, r0: float, g_max: float = 4.0) -> np.ndarray:
    """grad f for f(x) = (|x| - r0)^2: a radial pull towards the shell |x| = r0.

    The magnitude is clipped at ``g_max``. Without the clip the pull grows like 2|x| and the
    Langevin step overshoots: at gamma = 6 the guided chain diverged to NaN. Clipping bounds
    the Lipschitz constant of the guided score, which is the standard numerical guard for
    guided sampling - the constraint keeps acting, just not without limit.
    """
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-9)
    grad = 2.0 * (norm - r0) * x / norm
    magnitude = np.linalg.norm(grad, axis=1, keepdims=True)
    scale = np.minimum(1.0, g_max / np.maximum(magnitude, 1e-12))
    return grad * scale


def annealed_langevin(n: int, n_levels: int = 60, steps_per_level: int = 40,
                      sigma_max: float = 1.6, sigma_min: float = 0.05,
                      gamma: float = 0.0, functional: str = "none") -> np.ndarray:
    """Score-based sampling with guidance from a functional.

    Guidance adds an *energy* to the target density:

        p_guided(x)  ~  p(x) * exp( -gamma * f(x) )        score_guided = score - gamma * grad f

    a penalty on the functional, not a reward - the same sign convention as the distance guide
    in module 04, where a violated constraint costs energy. Three variants were tried before
    this one; the first two are worth keeping in the record because they are the natural
    mistakes:

      * ``score + gamma * grad f / sigma^2``: diverges as sigma -> 0 (effective weight
        gamma/sigma^2 ~ 240 at sigma = 0.05). The linear case drifted by -41 and the radial
        case overflowed.
      * the same thing written through the denoised estimate (``xhat - gamma grad f(xhat)``)
        inherits the same 1/sigma^2, because Tweedie's formula puts it back.
      * ``score + gamma * grad f`` (a *reward* rather than a penalty) makes a radial
        constraint run away: for |x| > r the gradient points outward, so the density is pushed
        off the shell instead of onto it (observed: values ~1e26 before the overflow guard).

    With the penalty form the linear functional has a closed-form prediction: tilting by
    ``exp(-gamma a.x)`` shifts the mean by ``-gamma * Var(x)``. That prediction is printed next
    to the measurement.
    """
    x = RNG.standard_normal((n, 2)) * sigma_max
    for sigma in np.geomspace(sigma_max, sigma_min, n_levels):
        sigma2 = float(sigma ** 2)
        eps = 0.25 * sigma2
        for _ in range(steps_per_level):
            drift = score(x, sigma2)
            if gamma and functional == "linear":
                drift = drift - gamma * grad_linear(x)
            elif gamma and functional == "radial":
                drift = drift - gamma * grad_radial(x, r0=1.0)
            x = x + eps * drift + np.sqrt(2 * eps) * RNG.standard_normal(x.shape)
    return x


def main() -> None:
    FIGDIR.mkdir(exist_ok=True)
    n = 15_000
    unguided = annealed_langevin(n)
    linear = annealed_langevin(n, gamma=0.6, functional="linear")
    gammas = [0.6, 2.0, 6.0]
    radial = {g: annealed_langevin(n, gamma=g, functional="radial") for g in gammas}

    print("guidance through a functional: what each one does")
    print(f"  unguided : mean = ({unguided[:, 0].mean():+.3f}, {unguided[:, 1].mean():+.3f})"
          f"   sd = ({unguided[:, 0].std(ddof=1):.3f}, {unguided[:, 1].std(ddof=1):.3f})")
    print(f"  linear   : mean = ({linear[:, 0].mean():+.3f}, {linear[:, 1].mean():+.3f})"
          f"   sd = ({linear[:, 0].std(ddof=1):.3f}, {linear[:, 1].std(ddof=1):.3f})")
    shift = float(linear[:, 0].mean() - unguided[:, 0].mean())
    variance = float(unguided[:, 0].var(ddof=1))
    predicted = -0.6 * variance
    print(f"  linear functional with gamma = 0.6:")
    print(f"    prediction (penalty exp(-gamma a.x)) : shift = -gamma * Var(x) = {predicted:+.3f}")
    print(f"    measured mean shift in x            : {shift:+.3f}")
    print("    a constant gradient tilts the distribution; the tilt does not grow with the "
          "number of steps")

    base_radius = np.linalg.norm(unguided, axis=1)
    within_base = float((np.abs(base_radius - 1.0) < 0.15).mean())
    print("  radial functional (|x| - 1)^2 - constraint strength is a knob:")
    print(f"    {'gamma':>6} {'predicted sd':>13} {'measured radius mean':>21}"
          f" {'measured sd':>12} {'within 0.15':>12}")
    print(f"    {'-':>6} {'-':>13} {base_radius.mean():>21.3f} {base_radius.std(ddof=1):>12.3f}"
          f" {within_base:>12.3f}   <- no guidance")
    radial_stats = {}
    for g in gammas:
        r = np.linalg.norm(radial[g], axis=1)
        predicted_sd = float(np.sqrt(1.0 / (2.0 * g)))
        within = float((np.abs(r - 1.0) < 0.15).mean())
        radial_stats[g] = (r, within, predicted_sd)
        print(f"    {g:>6.1f} {predicted_sd:>13.3f} {r.mean():>21.3f} {r.std(ddof=1):>12.3f}"
              f" {within:>12.3f}")
    print("    predicted sd = sqrt(1 / 2 gamma): the local width of the penalty, ignoring the"
          " prior. The measured width is smaller because the prior is also acting.")

    grid = np.linspace(-3.0, 3.0, 300)
    gx, gy = np.meshgrid(grid, grid)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    energy = np.min(np.stack([
        np.einsum("ni,ij,nj->n", pts - m, np.linalg.inv(c + 0.02 * np.eye(2)), pts - m)
        for m, c in zip(MEANS, COVS)]), axis=0).reshape(gx.shape)
    dens = np.exp(-0.5 * energy)

    panels = [
        ("no guidance", unguided, None),
        (f"linear functional, gamma = 0.6\npredicted shift {predicted:+.2f} / measured {shift:+.2f}",
         linear, None),
        ("radial functional, gamma = 2\n(penalty pulls onto the shell)", radial[2.0], 1.0),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True, sharey=True)
    for ax, (title, s, target_r) in zip(axes, panels):
        ax.contour(gx, gy, dens, levels=6, colors="#c53030", linewidths=0.8)
        ax.scatter(s[:4000, 0], s[:4000, 1], s=3, alpha=0.25, color="#2c5282")
        if target_r is not None:
            t = np.linspace(0, 2 * np.pi, 200)
            ax.plot(target_r * np.cos(t), target_r * np.sin(t),
                    color="#2f855a", lw=1.4, ls="--",
                    label="target shell r = 1")
            ax.legend(frameon=False, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.suptitle("Same sampler, same budget - only the functional's gradient differs",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGDIR / "05_guidance_functionals.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.hist(base_radius, bins=80, density=True, color="#cbd5e0", alpha=0.9,
            label="no guidance")
    for g, colour in zip(gammas, ["#90cdf4", "#2b6cb0", "#1a365d"]):
        r, within, _ = radial_stats[g]
        ax.hist(r, bins=80, density=True, histtype="step", lw=1.6, color=colour,
                label=f"radial gamma = {g:g}   (sd {r.std(ddof=1):.2f}, {within:.0%} within 0.15)")
    ax.axvline(1.0, color="#2f855a", ls="--", lw=1.4, label="target radius")
    ax.set_xlabel("radius |x|")
    ax.set_ylabel("density")
    ax.set_xlim(0, 3.0)
    ax.set_title("A quadratic functional contracts the distribution onto a shell\n"
                 "and the constraint strength is a knob you can measure", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGDIR / "05_guidance_radius.png", dpi=150)
    plt.close(fig)
    print(f"figures written to {FIGDIR}")


if __name__ == "__main__":
    main()
