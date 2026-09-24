"""Run the three samplers on the double well and write the diagnostics to figures/.

Usage:
python 01_mcmc_toy/run_demo.py                # one seed: tables + figures (~1 min)
    python 01_mcmc_toy/run_demo.py --seeds 5      # repeat the protocol, report mean +- sd
    python 01_mcmc_toy/run_demo.py --seeds 5 --skip-figures

Everything printed here is reproducible: the step sizes are tuned on one fixed pilot seed, and
each replication re-uses a fixed seed layout. ``--seeds`` is the honest version of the table -
a single seed has no error bar, and some of the differences below are smaller than the spread
between seeds.
"""

from __future__ import annotations

import argparse
import csv
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

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGDIR = REPO_ROOT / "figures"
CSV_PATH = REPO_ROOT / "outputs_mcmc_summary.csv"

# --------------------------------------------------------------------------------------
# Budgets and the cost convention. These are *choices*, not facts - see the README
# section "The cost convention". One log-density and one gradient both count as 1 unit.
# --------------------------------------------------------------------------------------
BUDGET = {"RWM": 200_000, "MALA": 200_000, "HMC": 50_000}
N_LEAPFROG = 10
COST_PER_STEP = {"RWM": 1.0, "MALA": 3.0, "HMC": float(N_LEAPFROG + 2)}

STEP_GRID = {
    "RWM": [0.2, 0.35, 0.5, 0.7, 1.0],
    "MALA": [0.3, 0.45, 0.6, 0.8, 1.0],
    "HMC": [0.15, 0.2, 0.25],
}

_SAMPLERS = {
    "RWM": (random_walk_metropolis, {}),
    "MALA": (mala, {}),
    "HMC": (hamiltonian_monte_carlo, {"n_leapfrog": N_LEAPFROG}),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the three samplers on the double well.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--seeds", type=int, default=1,
                        help="how many replications of the whole protocol (default 1)")
    parser.add_argument("--skip-figures", action="store_true",
                        help="only print the tables and write the CSV")
    return parser.parse_args()


def tune_step_sizes(target: DoubleWell, n_pilot: int = 20_000, pilot_seed: int = 11) -> dict:
    """Pick each sampler's step size by ESS per target evaluation on a short pilot run.

    Comparing untuned samplers says nothing, so every sampler gets the same kind of small
    budget. The pilot seed is fixed and shared, so the tuned step sizes do not depend on the
    replication seed - the spread across replications then measures sampling noise only.
    """
    tuned = {}
    for kind, (sampler, extra) in _SAMPLERS.items():
        best, best_score = None, -np.inf
        for step in STEP_GRID[kind]:
            chain = sampler(target, n_steps=n_pilot, step_size=step, seed=pilot_seed, **extra)
            score = summarize([chain], target)[0]["ESS per 1k evals"]
            if score > best_score:
                best, best_score = step, score
        tuned[kind] = best
    return tuned


def run_replicate(target: DoubleWell, steps: dict, rep: int):
    """One full set of three chains, with a fixed and reproducible seed layout."""
    base = 10 * rep
    return [
        random_walk_metropolis(target, n_steps=BUDGET["RWM"], step_size=steps["RWM"],
                               seed=base + 1),
        mala(target, n_steps=BUDGET["MALA"], step_size=steps["MALA"], seed=base + 2),
        hamiltonian_monte_carlo(target, n_steps=BUDGET["HMC"], step_size=steps["HMC"],
                                n_leapfrog=N_LEAPFROG, seed=base + 3),
    ]


def print_efficiency_table(rows) -> None:
    print("\nMixing (cost-aware):")
    print(f"{'sampler':<8}{'steps':>9}{'cost/step':>11}{'accept':>9}"
          f"{'ESS(x0)':>10}{'ESS/1k steps':>14}{'ESS/1k evals':>14}")
    for r in rows:
        print(f"{r['sampler']:<8}{r['steps']:>9,}{r['cost per step']:>11.1f}"
              f"{r['acceptance']:>9.3f}{r['ESS(x0)']:>10.0f}"
              f"{r['ESS per 1k steps']:>14.1f}{r['ESS per 1k evals']:>14.1f}")


def print_exploration_table(rows) -> None:
    print("\nExploration of the two wells "
          "(absolute counts are not comparable - the budgets differ):")
    print(f"{'sampler':<8}{'steps':>9}{'hops':>8}{'hops/1k steps':>15}"
          f"{'hops/1k evals':>15}{'longest dwell':>15}")
    for r in rows:
        print(f"{r['sampler']:<8}{r['steps']:>9,}{r['mode hops']:>8,}"
              f"{r['hops per 1k steps']:>15.1f}{r['hops per 1k evals']:>15.2f}"
              f"{r['longest dwell']:>15,}")


def print_seed_summary(all_rows) -> None:
    """Mean +- sd across replications, for the numbers a reader is most likely to quote."""
    print(f"\nAcross {len(all_rows)} seeds (mean +- sd over replications):")
    print(f"{'sampler':<8}{'ESS(x0)':>16}{'ESS/1k evals':>18}"
          f"{'hops/1k steps':>17}{'longest dwell':>17}")
    for sampler in BUDGET:
        group = [r for rep in all_rows for r in rep if r["sampler"] == sampler]

        def ms(key, digits=1):
            vals = np.array([r[key] for r in group], dtype=float)
            sd = vals.std(ddof=1) if vals.size > 1 else 0.0
            return f"{vals.mean():.{digits}f} +- {sd:.{digits}f}"

        print(f"{sampler:<8}{ms('ESS(x0)', 0):>16}{ms('ESS per 1k evals'):>18}"
              f"{ms('hops per 1k steps'):>17}{ms('longest dwell', 0):>17}")


def write_csv(all_rows, steps) -> None:
    columns = ["sampler", "seed", "steps", "cost_per_step", "tuned_step_size", "acceptance",
               "ESS_x0", "ESS_per_1k_steps", "ESS_per_1k_evals", "hops", "hops_per_1k_steps",
               "hops_per_1k_evals", "longest_dwell", "mean_x0", "std_x0"]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for rep_index, rows in enumerate(all_rows):
            for r in rows:
                writer.writerow([
                    r["sampler"], rep_index, r["steps"], r["cost per step"], steps[r["sampler"]],
                    f"{r['acceptance']:.6f}", f"{r['ESS(x0)']:.3f}",
                    f"{r['ESS per 1k steps']:.3f}", f"{r['ESS per 1k evals']:.3f}",
                    r["mode hops"], f"{r['hops per 1k steps']:.4f}",
                    f"{r['hops per 1k evals']:.4f}", r["longest dwell"],
                    f"{r['mean x0']:.6f}", f"{r['std x0']:.6f}",
                ])
    print(f"\nper-seed results written to {CSV_PATH}")


def write_figures(target: DoubleWell, chains, rows, steps) -> None:
    by_name = {r["sampler"]: r for r in rows}

    # --- figure 1: trace plots -------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=False)
    for ax, chain in zip(axes, chains):
        x0 = chain.samples[:, 0]
        window = min(5_000, len(x0))
        r = by_name[chain.name]
        ax.plot(np.arange(window), x0[:window], lw=0.6, color="#2c5282")
        ax.axhline(0.0, color="#a0aec0", lw=0.8, ls="--")
        ax.set_ylabel("$x_0$")
        ax.set_title(f"{chain.name}: acceptance {r['acceptance']:.2f}, "
                     f"{r['hops per 1k steps']:.1f} barrier crossings per 1k steps "
                     f"({r['mode hops']:,} in {r['steps']:,} steps, first {window:,} shown)",
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
    fig.suptitle("Samples (blue) against the true unnormalised density (red contours)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGDIR / "01_samples.png", dpi=150)
    plt.close(fig)

    # --- figure 4: what happens when the barrier grows ------------------------
    barriers = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for colour, (label, fn, kw, n) in zip(
        ["#c53030", "#2b6cb0", "#2f855a"],
        [("RWM", random_walk_metropolis, dict(step_size=steps["RWM"]), 50_000),
         ("MALA", mala, dict(step_size=steps["MALA"]), 50_000),
         ("HMC", hamiltonian_monte_carlo,
          dict(step_size=steps["HMC"], n_leapfrog=N_LEAPFROG), 20_000)],
    ):
        rates = []
        for h in barriers:
            t = DoubleWell(a=target.a, h=h, omega=target.omega)
            chain = fn(t, n_steps=n, seed=7, **kw)
            rates.append(summarize([chain], t)[0]["hops per 1k steps"])
        ax.plot([DoubleWell(a=target.a, h=h).barrier for h in barriers], rates,
                marker="o", color=colour, label=label)
    ax.set_xlabel("barrier height (energy units)")
    ax.set_ylabel("barrier crossings per 1k steps")
    ax.set_title("Raise the barrier and the cheap samplers stop exploring:\n"
                 "the spectral gap closing, made measurable")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "01_barrier_scan.png", dpi=150)
    plt.close(fig)

    print(f"figures written to {FIGDIR}")


def main() -> None:
    args = parse_args()
    FIGDIR.mkdir(exist_ok=True)
    target = DoubleWell()
    print(f"target: {target.name}, barrier height = {target.barrier:.2f}")
    print("cost convention: one log-density and one gradient both cost 1 unit "
          f"(RWM 1, MALA 3, HMC {N_LEAPFROG + 2} per iteration)")

    steps = tune_step_sizes(target)
    print(f"tuned step sizes: RWM {steps['RWM']}, MALA {steps['MALA']}, "
          f"HMC {steps['HMC']} (L = {N_LEAPFROG})")

    reps = max(1, args.seeds)
    all_rows = []
    first_chains = None
    for rep in range(reps):
        chains = run_replicate(target, steps, rep)
        all_rows.append(summarize(chains, target))
        if first_chains is None:
            first_chains = chains

    print_efficiency_table(all_rows[0])
    print_exploration_table(all_rows[0])
    if reps > 1:
        print_seed_summary(all_rows)

    write_csv(all_rows, steps)

    if not args.skip_figures:
        write_figures(target, first_chains, all_rows[0], steps)


if __name__ == "__main__":
    main()
