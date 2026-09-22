"""Run the three samplers on the double well and write the diagnostics to figures/.

Usage:  python 01_mcmc_toy/run_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from diagnostics import autocorrelation, summarize  # noqa: E402
from samplers import hamiltonian_monte_carlo, mala, random_walk_metropolis  # noqa: E402
from targets import DoubleWell  # noqa: E402

FIGDIR = Path(__file__).resolve().parent.parent / "figures"


def main() -> None:
    FIGDIR.mkdir(exist_ok=True)
    target = DoubleWell()
    print(f"target: {target.name}, barrier height = {target.barrier:.2f}")

    # --- pilot: tune the step size of each sampler on a short run -------------
    # Fair comparisons require tuned samplers, so each one gets a small budget to pick its
    # step size by ESS per target evaluation (not per iteration - the costs differ).
    def tune(fn, grid, n_pilot=20_000, **kw):
        best, best_score = None, -np.inf
        for step in grid:
            chain = fn(target, n_steps=n_pilot, step_size=step, seed=11, **kw)
            score = summarize([chain], target)[0]["ESS per 1k evals"]
            if score > best_score:
                best, best_score = step, score
        return best, best_score

    step_rwm, _ = tune(random_walk_metropolis, [0.2, 0.35, 0.5, 0.7, 1.0])
    step_mala, _ = tune(mala, [0.3, 0.45, 0.6, 0.8, 1.0])
    step_hmc, _ = tune(hamiltonian_monte_carlo, [0.15, 0.2, 0.25], n_leapfrog=10)
    print(f"tuned step sizes: RWM {step_rwm}, MALA {step_mala}, HMC {step_hmc} (L = 10)")

    chains = [
        random_walk_metropolis(target, n_steps=200_000, step_size=step_rwm, seed=1),
        mala(target, n_steps=200_000, step_size=step_mala, seed=2),
        hamiltonian_monte_carlo(target, n_steps=50_000, step_size=step_hmc, n_leapfrog=10, seed=3),
    ]

    rows = summarize(chains, target)
    width = max(len(r["sampler"]) for r in rows)
    print("\n" + f"{'sampler':<{width}}  {'steps':>7}  {'accept':>7}  {'ESS(x0)':>8}  "
          f"{'ESS/1k evals':>12}  {'hops':>5}  {'dwell':>6}  {'mean x0':>8}  {'sd x0':>7}")
    for r in rows:
        print(f"{r['sampler']:<{width}}  {r['steps']:>7}  {r['acceptance']:>7.3f}  "
              f"{r['ESS(x0)']:>8.0f}  {r['ESS per 1k evals']:>12.1f}  "
              f"{r['mode hops']:>5}  {r['longest dwell']:>6}  "
              f"{r['mean x0']:>8.3f}  {r['std x0']:>7.3f}")
    print("\nESS per 1k evals counts one log-density OR one gradient as one unit: RWM spends 1 per\n"
          "iteration, MALA 3, HMC n_leapfrog + 2.")

    # --- figure 1: trace plots -------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=False)
    for ax, chain in zip(axes, chains):
        x0 = chain.samples[:, 0]
        window = min(5_000, len(x0))
        step_axis = np.arange(window)
        ax.plot(step_axis, x0[:window], lw=0.6, color="#2c5282")
        ax.axhline(0.0, color="#a0aec0", lw=0.8, ls="--")
        ax.set_ylabel("$x_0$")
        ax.set_title(f"{chain.name}: acceptance {chain.acceptance_rate:.2f}, "
                     f"{summarize([chain], target)[0]['mode hops']} barrier crossings in "
                     f"{chain.n_steps:,} steps (first {window:,} shown)",
                     fontsize=10, loc="left")
    axes[-1].set_xlabel("iteration")
    fig.suptitle("Trace plots: how a chain actually explores a bimodal target", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGDIR / "01_trace.png", dpi=150)
    plt.close(fig)

    # --- figure 2: autocorrelation --------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for chain, colour in zip(chains, ["#c53030", "#2b6cb0", "#2f855a"]):
        rho = autocorrelation(chain.samples[:20_000, 0], max_lag=200)
        ax.plot(np.arange(len(rho)), rho, lw=1.4, color=colour, label=chain.name)
    ax.axhline(0.0, color="#a0aec0", lw=0.8)
    ax.set_xlabel("lag")
    ax.set_ylabel("autocorrelation of $x_0$")
    ax.set_title("Slow ACF decay = large integrated autocorrelation time = small ESS")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "01_acf.png", dpi=150)
    plt.close(fig)

    # --- figure 3: sample clouds over the target ------------------------------
    grid = np.linspace(-2.6, 2.6, 220)
    gx, gy = np.meshgrid(grid, grid)
    energy = target.h * (gx ** 2 - target.a ** 2) ** 2 + 0.5 * target.omega ** 2 * gy ** 2
    density = np.exp(-energy)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, chain in zip(axes, chains):
        s = chain.samples[:: max(1, chain.n_steps // 6000)]
        ax.scatter(s[:, 0], s[:, 1], s=2, alpha=0.25, color="#2c5282")
        ax.contour(gx, gy, density, levels=8, colors="#c53030", linewidths=0.8)
        ax.set_title(chain.name)
        ax.set_xlim(-2.6, 2.6)
        ax.set_ylim(-2.6, 2.6)
        ax.set_xlabel("$x_0$")
    axes[0].set_ylabel("$x_1$")
    fig.suptitle("Samples (blue) against the true unnormalised density (red contours)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGDIR / "01_samples.png", dpi=150)
    plt.close(fig)

    # --- figure 4: what happens when the barrier grows ------------------------
    barriers = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for colour, (label, fn, kw, n) in zip(
        ["#c53030", "#2b6cb0", "#2f855a"],
        [("RWM", random_walk_metropolis, dict(step_size=step_rwm), 50_000),
         ("MALA", mala, dict(step_size=step_mala), 50_000),
         ("HMC", hamiltonian_monte_carlo, dict(step_size=step_hmc, n_leapfrog=10), 20_000)],
    ):
        hops = []
        for h in barriers:
            t = DoubleWell(a=target.a, h=h, omega=target.omega)
            chain = fn(t, n_steps=n, seed=7, **kw)
            hops.append(summarize([chain], t)[0]["mode hops"])
        ax.plot([DoubleWell(a=target.a, h=h).barrier for h in barriers], hops,
                marker="o", color=colour, label=label)
    ax.set_xlabel("barrier height (energy units)")
    ax.set_ylabel("barrier crossings in a fixed budget")
    ax.set_title("Raise the barrier and the cheap samplers stop exploring:\n"
                 "the spectral gap closing, made measurable")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "01_barrier_scan.png", dpi=150)
    plt.close(fig)

    print(f"\nfigures written to {FIGDIR}")


if __name__ == "__main__":
    main()
