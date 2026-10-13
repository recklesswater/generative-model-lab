"""How far the Krylov collective variable goes, and what stops it.

The question this script answers is the one that follows ``cv_discovery.py``: the method worked
in eight dimensions on one target - is that a property of the method or of the target? Three
scans, each cheap, each with the exact answer available:

1. **Dimension.** The same problem in d = 2 ... 20, with a dense random reaction coordinate.
   Dimension is *not* the wall: the direction is recovered just as well at d = 20 as at d = 3.
   What changes with d is the amount of statistics needed, and that shows up as a widening
   error bar rather than a bias.
2. **The spectral gap.** The transverse directions are made softer and slower, so that the
   separation between the slow mode and the rest shrinks. This is the knob that actually
   destroys the method, and the measured timescale ratio predicts where it breaks.
3. **More than one slow coordinate.** With two barriers of different heights, a single CV
   recovers only the slower one - the Ritz spectrum says so before you look at the answer. And
   on this target the variance-based direction cannot tell the two apart at all, which is the
   clearest case of "variance is not the property being asked for".

The negative control is an anisotropic Gaussian: no barrier, no slow mode, no gap. The method
still returns a direction. It has nothing in it.

Usage:  python 06_krylov_mcmc/dimension_scan.py            # ~40 s
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "01_mcmc_toy"))

from cv_analysis import (  # noqa: E402
    barrier_with_error,
    integrated_autocorrelation_time,
    print_table,
    timescale_from_eigenvalue,
    tune_step_size,
)
from krylov import angle_degrees, slow_subspace  # noqa: E402
from samplers import mala  # noqa: E402
from targets_nd import AnisotropicGaussian, AnisotropicWells, random_directions  # noqa: E402

FIGDIR = REPO_ROOT / "figures"
STEPS = 30000
BURN_IN = 3000
KAPPA = 0.5
A = 1.5
H = 0.5
M = 4


def section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def span_angle(vectors: np.ndarray, direction: np.ndarray) -> float:
    """Smallest angle between ``direction`` and the span of the columns of ``vectors``.

    With one Krylov vector this is the plain angle between two directions; with two it is the
    principal angle to the subspace, i.e. "is this direction somewhere inside what the method
    found", which is the honest question when the slow subspace is more than one dimensional.
    """
    v = np.asarray(vectors, dtype=float)
    norm = np.linalg.norm(v, axis=0)
    cos = np.abs(v.T @ direction) / np.maximum(norm, 1e-300) / np.linalg.norm(direction)
    return float(np.degrees(np.arccos(min(1.0, cos.max()))))


def analyse(target, direction_rows, seed: int = 1, steps: int = STEPS, m: int = M):
    """Run one chain on ``target`` and return the numbers every scan below reports.

    The lag is chosen as a fixed fraction of the slow direction's own autocorrelation time rather
    than as a fixed number of steps: the tuned step size changes with the dimension and with
    ``kappa``, so a fixed lag in steps would silently compare different physical lags.
    """
    rng = np.random.default_rng(seed)
    x0 = np.zeros(target.dim)
    if hasattr(target, "well_centre"):
        x0 = target.well_centre() + 0.1 * rng.standard_normal(target.dim)
    step, pilot_acc = tune_step_size(mala, target, x0, target_accept=0.55, n_pilot=1500, seed=seed)
    chain = mala(target, n_steps=steps, step_size=step, x0=x0, seed=seed)
    X = chain.samples[BURN_IN:]

    slow_probe = direction_rows[0][1]
    u = X @ slow_probe
    tau = int(max(4, round(integrated_autocorrelation_time(u) / 15.0)))
    sub = slow_subspace(X, tau, m, seed=0)
    t1 = timescale_from_eigenvalue(sub.eigenvalues[0], tau)
    t2 = timescale_from_eigenvalue(sub.eigenvalues[1], tau) if m > 1 else np.nan

    Xc = X - X.mean(axis=0)
    C0 = Xc.T @ Xc / (X.shape[0] - 1)
    pca = np.linalg.eigh(C0)[1][:, -1]

    row = {
        "dim": target.dim,
        "step": step,
        "acceptance": chain.acceptance_rate,
        "tau": tau,
        "iat": integrated_autocorrelation_time(u),
        "angle_cv": span_angle(sub.vectors[:, :1], direction_rows[0][1]),
        "angle_pc": span_angle(pca[:, None], direction_rows[0][1]),
        "gap": (t1 / t2) if np.isfinite(t2) and t2 > 0 else np.inf,
        "cond": float(np.linalg.cond(sub.C0)),
        "eigenvalues": sub.eigenvalues.copy(),
        "sub": sub,
        "X": X,
    }
    if hasattr(target, "max_barrier"):
        row["barrier"], row["barrier_se"] = barrier_with_error(X, sub.vectors[:, 0])
        row["barrier_true"] = target.max_barrier
    if len(direction_rows) > 1:
        row["angles_true"] = [span_angle(sub.vectors, d) for _, d in direction_rows]
        row["angles_pc_true"] = [span_angle(pca[:, None], d) for _, d in direction_rows]
    return row


def main() -> None:
    FIGDIR.mkdir(exist_ok=True)

    section("1. Dimension scan: does d itself break anything?")
    dims = (2, 3, 5, 8, 12, 20)
    rows = []
    for d in dims:
        target = AnisotropicWells(random_directions(d, 1, seed=0), a=A, h=H, kappa=KAPPA)
        r = analyse(target, [("true", target.directions[0])])
        rows.append(r)
    print_table(["d", "MALA step", "acceptance", "lag tau", "IACT(u)", "angle CV (deg)",
                 "angle PC (deg)", "t1/t2", "cond C0"],
                [[r["dim"], r["step"], r["acceptance"], r["tau"], r["iat"], r["angle_cv"],
                  r["angle_pc"], r["gap"], r["cond"]] for r in rows], floatfmt="{:.3f}")
    print("  the CV angle is flat in d; the variance-based angle is at ~90 degrees in every row")
    print("  the lag grows with d because the tuned step size shrinks, which is why the lag is")
    print("  set as a fraction of the measured autocorrelation time rather than in steps")

    section("2. The barrier recovered by the CV, per dimension")
    print_table(["d", "barrier from the CV", "exact barrier", "recovered"],
                [[r["dim"], f"{r['barrier']:.2f} +- {r['barrier_se']:.2f}",
                  r["barrier_true"], r["barrier"] / r["barrier_true"]] for r in rows],
                floatfmt="{:.3f}")

    section("3. Where it breaks: making the transverse modes slow (kappa sweep)")
    kappas = (1.0, 0.5, 0.35, 0.25, 0.2)
    ksweep = []
    for kappa in kappas:
        target = AnisotropicWells(random_directions(8, 1, seed=0), a=A, h=H, kappa=kappa)
        r = analyse(target, [("true", target.directions[0])])
        r["kappa"] = kappa
        ksweep.append(r)
    print_table(["kappa", "angle CV (deg)", "angle PC (deg)", "t1/t2", "barrier from CV",
                 "IACT(u)"],
                [[r["kappa"], r["angle_cv"], r["angle_pc"], r["gap"],
                  f"{r['barrier']:.2f} +- {r['barrier_se']:.2f}", r["iat"]] for r in ksweep],
                floatfmt="{:.3f}")
    print("  as kappa falls the transverse modes get wider *and* slower: the direction is lost")
    print("  once the timescale ratio t1/t2 approaches 1, and the ratio is visible in the Ritz")
    print("  spectrum before any comparison with the truth is made")

    section("4. Two slow coordinates: how many CVs are needed, and what says so")
    # Five dimensions rather than eight, and stiffer transverse modes, so that the two barriers
    # are the two slowest things in the problem and the failure below is about the Krylov
    # dimension rather than about the transverse modes crowding the top of the spectrum.
    dim2, kappa2 = 5, 4.0
    directions2 = random_directions(dim2, 2, seed=4)
    target2 = AnisotropicWells(directions2, a=A, h=(0.5, 0.25), kappa=kappa2)
    rng2 = np.random.default_rng(1)
    x0_2 = target2.well_centre() + 0.1 * rng2.standard_normal(dim2)
    step2, _ = tune_step_size(mala, target2, x0_2, target_accept=0.55, n_pilot=1500, seed=1)
    chain2 = mala(target2, n_steps=80000, step_size=step2, x0=x0_2, seed=1)
    X2 = chain2.samples[BURN_IN:]
    u1, u2 = X2 @ directions2[0], X2 @ directions2[1]
    tau2 = int(max(4, round(integrated_autocorrelation_time(u1) / 15.0)))
    pc2 = np.linalg.eigh(np.cov(X2.T))[1][:, -1]
    print(f"  two barriers of {target2.barriers[0]:.2f} and {target2.barriers[1]:.2f} kT along two")
    print(f"  dense directions in d = {dim2}; lag {tau2}, {X2.shape[0]:,} samples kept")
    print(f"  measured autocorrelation time: u1 {integrated_autocorrelation_time(u1):.0f} steps,"
          f" u2 {integrated_autocorrelation_time(u2):.0f} steps")
    print(f"  the leading principal component points {angle_degrees(pc2, directions2[0]):.1f} deg"
          f" from u1 and {angle_degrees(pc2, directions2[1]):.1f} deg from u2")
    m_rows, spectra = [], {}
    for m_try in range(1, dim2 + 1):
        s = slow_subspace(X2, tau2, m_try, seed=0)
        spectra[m_try] = s.eigenvalues.copy()
        m_rows.append([m_try, s.eigenvalues[0],
                     s.eigenvalues[1] if m_try > 1 else np.nan,
                     angle_degrees(s.vectors[:, 0], directions2[0]),
                     angle_degrees(s.vectors[:, 0], directions2[1]),
                     angle_degrees(s.vectors[:, 1], directions2[1]) if m_try > 1 else np.nan,
                     s.residuals[1] if m_try > 1 else np.nan])
    print_table(["Krylov dim m", "mu_1", "mu_2", "v1 -> u1 (deg)", "v1 -> u2 (deg)",
                 "v2 -> u2 (deg)", "residual of pair 2"],
                m_rows, floatfmt="{:.3f}")
    full = spectra[dim2]
    print(f"  at m = {dim2} (the whole space) the two leading Ritz values are"
          f" {full[0]:.3f} and {full[1]:.3f}, then a drop to {full[2]:.3f}:")
    print("  the number of Ritz values standing above the rest is the number of collective")
    print("  variables the problem has, and it is readable without knowing the answer")
    print("  with m = 2 the second vector is a transverse mode, not u2 - the residual column is")
    print("  the warning, and at m = 4 and above both coordinates are found to about a degree")

    section("5. Negative control: no barrier, no slow mode, no gap")
    d = 8
    gauss = AnisotropicGaussian(np.linspace(0.9, 1.1, d))     # a nearly flat spectrum
    rg = analyse(gauss, [("arbitrary", np.eye(d)[0])], seed=1)
    direction8 = random_directions(8, 1, seed=0)
    rw = analyse(AnisotropicWells(direction8, a=A, h=H, kappa=KAPPA), [("true", direction8[0])])
    print_table(["target", "t1/t2 (timescale ratio)", "leading Ritz value"],
                [["double well (d=8)", rw["gap"], rw["eigenvalues"][0]],
                 ["anisotropic gaussian", rg["gap"], rg["eigenvalues"][0]]], floatfmt="{:.3f}")
    print("  the Gaussian has no separation worth the name: the leading Ritz value is close to")
    print("  its neighbours, so the 'slow direction' it returns is a number without a meaning")
    print("  the ratio is the diagnostic to check first on a real system, where the truth is")
    print("  unknown - if there is no gap, there is no collective variable to find")

    figure(rows, ksweep, spectra, rw, rg)
    print(f"\nfigures written to {FIGDIR}")


def figure(rows, ksweep, spectra, rw, rg) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(19.5, 4.6))

    ax = axes[0]
    d = [r["dim"] for r in rows]
    ax.plot(d, [r["angle_cv"] for r in rows], marker="o", color="#2b6cb0", label="Krylov CV")
    ax.plot(d, [r["angle_pc"] for r in rows], marker="s", color="#c53030",
            label="top principal component")
    ax.set_xlabel("dimension d")
    ax.set_ylabel("angle to the true direction (deg)")
    ax.set_ylim(0, 100)
    ax.set_title("Dimension is not the wall:\nthe CV angle is flat in d", fontsize=10)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    ax.errorbar(d, [r["barrier"] for r in rows], yerr=[r["barrier_se"] for r in rows],
                marker="o", color="#2f855a", capsize=3, label="barrier from the CV")
    ax.axhline(rows[0]["barrier_true"], color="#c53030", ls="--", lw=1.2,
               label=f"exact barrier = {rows[0]['barrier_true']:.2f} kT")
    ax.set_xlabel("dimension d")
    ax.set_ylabel("free-energy barrier (kT)")
    ax.set_ylim(0, 3.4)
    ax.set_title("And the physics survives with it:\nthe barrier is recovered", fontsize=10)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[2]
    ax.plot([r["kappa"] for r in ksweep], [r["angle_cv"] for r in ksweep], marker="o",
            color="#2b6cb0", label="Krylov CV")
    ax.plot([r["kappa"] for r in ksweep], [r["angle_pc"] for r in ksweep], marker="s",
            color="#c53030", label="top principal component")
    ax.set_xlabel(r"transverse stiffness $\kappa$  (smaller = slower)")
    ax.set_ylabel("angle to the true direction (deg)")
    ax.set_ylim(0, 100)
    ax2 = ax.twinx()
    ax2.plot([r["kappa"] for r in ksweep], [r["gap"] for r in ksweep], marker="^", ls=":",
             color="#2f855a", label="timescale ratio t1/t2")
    ax2.set_ylabel("t1 / t2")
    ax2.set_ylim(0, 4)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], frameon=False, fontsize=8, loc="upper left")
    ax.set_title("The knob that actually breaks it:\nwhen the gap closes, the direction is lost",
                 fontsize=10)

    ax = axes[3]
    full = max(spectra)
    values = spectra[full]
    idx = np.arange(len(values))
    ax.bar(idx, values, color="#2b6cb0")
    ax.set_xticks(idx, [f"{i + 1}" for i in idx])
    ax.set_xlabel("Ritz index")
    ax.set_ylabel(r"$\mu$")
    ax.set_ylim(0, 1.0)
    for i, mu in enumerate(values):
        ax.annotate(f"{mu:.3f}", (i, mu), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=8)
    ax.set_title(f"Two slow coordinates in d = {full}:\ntwo leading Ritz values, then a drop",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGDIR / "06_krylov_dimension_scan.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
