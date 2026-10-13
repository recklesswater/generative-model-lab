"""Let the chain pick the reaction coordinate, instead of picking it by hand.

The problem this script is about: to describe a high-dimensional density you want a few
coordinates that carry the *slow* behaviour, and in a real system nobody hands you them. The
recipe used here is the standard one - estimate a transfer operator from the trajectory, take
its leading eigenvector, and call that the collective variable (CV) - and the only part that is
special is that the operator is never formed: a Krylov (Lanczos) run applies it to a vector and
returns the leading eigenpairs.

Concretely, with features ``x`` (the coordinates themselves, centred):

    C0[i,j]   = <x_i(t) x_j(t)>            equal-time covariance
    Ctau[i,j] = <x_i(t) x_j(t+tau)>        lag-tau covariance
    Ctau v = mu C0 v                       the generalised eigenproblem; v is the CV

and ``C0^-1 C_tau`` is applied matrix-free inside the Lanczos iteration, in a whitened inner
product that makes it symmetric (see ``krylov.slow_subspace``). Both matrices and the eigenvalue
problem are textbook material; the references are listed in the README under "Provenance".

The target is a d-dimensional double well whose barrier sits along a *dense random* direction, so
no single coordinate is the reaction coordinate and its components are not readable off a plot.
Everything about the target is known exactly, so the CV can be checked three ways rather than
eyeballed: against the direction that generated the data, against an independent i.i.d. sampler
of the same density, and against the analytic free-energy profile.

Usage:
    python 06_krylov_mcmc/cv_discovery.py                     # tables + figures, ~15 s
    python 06_krylov_mcmc/cv_discovery.py --dim 12 --steps 60000
    python 06_krylov_mcmc/cv_discovery.py --skip-figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "01_mcmc_toy"))     # module 01's samplers and diagnostics

from cv_analysis import (  # noqa: E402
    barrier_with_error,
    free_energy_profile,
    integrated_autocorrelation_time,
    print_table,
    timescale_from_eigenvalue,
    tune_step_size,
)
from diagnostics import autocorrelation, count_mode_hops  # noqa: E402  (module 01)
from krylov import angle_degrees, slow_subspace, true_residual_norms  # noqa: E402
from samplers import mala  # noqa: E402  (module 01)
from targets_nd import AnisotropicWells, random_directions  # noqa: E402

FIGDIR = REPO_ROOT / "figures"
LAGS = (2, 4, 8, 16, 32, 64)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dim", type=int, default=8, help="dimension of the target (default 8)")
    p.add_argument("--steps", type=int, default=60000, help="MCMC steps (default 60000)")
    p.add_argument("--burn-in", type=int, default=5000, help="discarded initial steps (default 5000)")
    p.add_argument("--lag", type=int, default=0,
                   help="lag tau for the operator; 0 = pick it from the lag scan (default)")
    p.add_argument("--m", type=int, default=6, help="Krylov dimension (default 6)")
    p.add_argument("--kappa", type=float, default=0.5,
                   help="stiffness of the non-slow directions (default 0.5; see README)")
    p.add_argument("--a", type=float, default=1.5, help="well position (default 1.5)")
    p.add_argument("--h", type=float, default=0.5, help="barrier scale, barrier = h a^4 (default 0.5)")
    p.add_argument("--seed", type=int, default=1, help="chain seed (default 1)")
    p.add_argument("--skip-figures", action="store_true")
    return p.parse_args()


def section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def run_chain(target: AnisotropicWells, steps: int, burn_in: int, seed: int):
    """Tune MALA on a pilot, run the chain, return ``(chain, samples after burn-in, step size)``."""
    x0 = target.well_centre() + 0.1 * np.random.default_rng(seed + 100).standard_normal(target.dim)
    step, pilot_acc = tune_step_size(mala, target, x0, target_accept=0.55, seed=seed)
    chain = mala(target, n_steps=steps, step_size=step, x0=x0, seed=seed)
    return chain, step, pilot_acc, chain.samples[burn_in:]


def lag_scan(X: np.ndarray, truth: np.ndarray, m: int = 4):
    """Slow subspace at several lags: ``(lags, eigenvalues, angles)``.

    The angles are between the leading Ritz vector at that lag and the direction that actually
    generated the data. Reporting them next to the timescales is the point: a longer lag makes
    the *timescale* estimate cleaner and the *direction* estimate noisier, because long lags
    throw away pairs.
    """
    eigenvalues = np.empty((len(LAGS), m))
    angles = np.empty(len(LAGS))
    for i, tau in enumerate(LAGS):
        sub = slow_subspace(X, tau, m, seed=0)
        eigenvalues[i] = sub.eigenvalues
        angles[i] = angle_degrees(sub.vectors[:, 0], truth)
    return list(LAGS), eigenvalues, angles


def pick_lag(eigenvalues: np.ndarray, lags, fraction: float = 0.7) -> int:
    """Smallest lag whose leading implied timescale has reached ``fraction`` of the plateau.

    The rule is a choice, and the script prints the whole scan so the choice can be judged: the
    lag has to be long enough that the transverse modes have decayed, and short enough that the
    slow subspace is still estimated from many independent pairs.
    """
    times = np.array([timescale_from_eigenvalue(mu, tau)
                      for mu, tau in zip(eigenvalues[:, 0], lags)])
    best = fraction * float(np.max(times))
    for i, t in enumerate(times):
        if t >= best:
            return int(lags[i])
    return int(lags[-1])


def coordinate_table(X, X_exact, direction_rows, true_direction):
    """Barrier and profile comparison for a list of candidate coordinates."""
    rows = []
    for name, v in direction_rows:
        angle = angle_degrees(v, true_direction)
        bar_chain, bar_se = barrier_with_error(X, v)
        bar_exact, _ = barrier_with_error(X_exact, v)
        F_c, centres, _ = free_energy_profile(X @ v)
        F_e, _, _ = free_energy_profile(X_exact @ v)
        mask = np.isfinite(F_c) & np.isfinite(F_e)
        rms = float(np.sqrt(np.mean((F_c[mask] - F_e[mask]) ** 2))) if mask.sum() > 3 else np.nan
        rows.append(dict(name=name, angle=angle, barrier=bar_chain, barrier_se=bar_se,
                         barrier_exact=bar_exact, rms=rms,
                         iact=integrated_autocorrelation_time(X @ v)))
    return rows


def main() -> None:
    args = parse_args()
    FIGDIR.mkdir(exist_ok=True)
    np.set_printoptions(precision=4, suppress=True)

    directions = random_directions(args.dim, 1, seed=0)
    target = AnisotropicWells(directions, a=args.a, h=args.h, kappa=args.kappa)
    true_direction = target.directions[0]
    rng = np.random.default_rng(0)

    section("1. The target: a barrier along a dense direction")
    print(f"  {target.describe()}")
    print(f"  exact barrier along the slow direction   = {target.max_barrier:.2f} kT")
    print(f"  direction components                     = {np.array2string(true_direction, precision=3)}")
    print(f"  largest single component |d_i|           = {np.abs(true_direction).max():.3f}"
          f"  -> best single coordinate is {angle_degrees(np.eye(args.dim)[np.argmax(np.abs(true_direction))], true_direction):.0f} deg off")
    print(f"  variance per transverse direction        = 1/kappa = {1 / args.kappa:.2f}"
          f"   ({args.dim - 1} of them, {((args.dim - 1) / args.kappa):.1f} in total)")
    print("  a variance-based method is a contest between *single* directions, and any one of")
    print("  those transverse directions has more variance than the slow one - see section 2")

    section("2. The chain")
    chain, step, pilot_acc, X = run_chain(target, args.steps, args.burn_in, args.seed)
    print(f"  MALA step size tuned on a pilot          = {step:.3f} (pilot acceptance {pilot_acc:.2f})")
    print(f"  steps kept                               = {X.shape[0]:,} x {args.dim}")
    print(f"  acceptance rate                          = {chain.acceptance_rate:.3f}")
    u = X @ true_direction
    hops = count_mode_hops(u, 0.0)
    tau_u = integrated_autocorrelation_time(u)
    print(f"  crossings of the true barrier (x{u.shape[0]:,} steps) = {hops:,}"
          f"  ({1000 * hops / u.size:.1f} per 1k steps)")
    print(f"  integrated autocorrelation time of u     = {tau_u:.0f} steps")

    # the variance-vs-slowness contrast: what a variance-based method sees
    Xc = X - X.mean(axis=0)
    C0 = Xc.T @ Xc / (X.shape[0] - 1)
    pca_values, pca_vectors = np.linalg.eigh(C0)
    top_pc = pca_vectors[:, -1]
    print(f"  variance along u / along top PC          = {u.var():.2f} / {(X @ top_pc).var():.2f}"
          f"   (per direction; the transverse plane holds {((args.dim - 1) / args.kappa):.1f} in total)")
    print(f"  autocorrelation time u / top PC          = {tau_u:.0f} / {integrated_autocorrelation_time(X @ top_pc):.0f} steps")
    print("  -> the loudest direction in the sample cloud is not the slowest one")

    section("3. Krylov at a range of lags: is there a timescale plateau?")
    # three Krylov vectors is enough for a timescale plateau: the point of this table is the top
    # of the spectrum, not its tail
    lags, eigenvalues, lag_angles = lag_scan(X, true_direction, m=3)
    rows = []
    for i, tau in enumerate(lags):
        rows.append([tau] + [timescale_from_eigenvalue(mu, tau) for mu in eigenvalues[i]]
                    + [lag_angles[i]])
    print_table(["lag tau", "t1 (steps)", "t2 (steps)", "t3 (steps)", "angle to truth (deg)"],
                rows, floatfmt="{:.1f}")
    tau_star = args.lag if args.lag > 0 else pick_lag(eigenvalues, lags)
    print(f"  chosen lag                               = {tau_star} steps")
    print(f"  direct check: autocorrelation time / 2   = {tau_u / 2:.1f} steps"
          f"   (should be close to t1 above)")
    print("  note the trade-off: t1 keeps climbing with lag, but the direction stops improving")
    print("  once long lags start throwing away pairs")

    section(f"4. The slow subspace at lag tau = {tau_star}")
    sub = slow_subspace(X, tau_star, args.m, seed=0)
    # dense solve, ordered the same way as the Ritz values
    dense_mu = np.sort(np.linalg.eigvals(np.linalg.solve(sub.C0, sub.Ctau)).real)[::-1]
    gap = dense_mu[0] - dense_mu[1]
    print(f"  Ritz values mu_1..mu_{args.m}                    = {np.array2string(sub.eigenvalues, precision=4)}")
    print(f"  spectral gap mu_1 - mu_2                 = {gap:.4f}"
          f"  (implied timescale {timescale_from_eigenvalue(sub.eigenvalues[0], tau_star):.1f} vs"
          f" {timescale_from_eigenvalue(sub.eigenvalues[1], tau_star):.1f} steps)")
    print(f"  dense (non-Krylov) leading values        = {np.array2string(dense_mu[:4], precision=4)}")
    print(f"  leading value: Krylov vs dense           = {sub.eigenvalues[0]:.4f} vs {dense_mu[0]:.4f}"
          f"   (difference {abs(sub.eigenvalues[0] - dense_mu[0]):.1e})")
    print(f"  the values after the first are a near-degenerate cluster"
          f" ({dense_mu[1]:.4f} vs {dense_mu[2]:.4f}):")
    print("  those are the transverse modes, and individual members of a degenerate cluster are")
    print("  not separately identified by either method - only the subspace is")
    # is the cheap closed-form residual bound honest on this operator? check it literally
    e = np.linalg.solve(sub.whitener,
                        sub.Ctau @ sub.vectors - sub.eigenvalues * (sub.C0 @ sub.vectors))
    literal = np.linalg.norm(e, axis=0) / np.linalg.norm(sub.whitener.T @ sub.vectors, axis=0)
    ratios = literal / np.maximum(sub.bounds, 1e-300)
    print(f"  literal residual, whitened norm          = {np.array2string(literal, precision=2)}")
    print(f"  closed-form Krylov bound                 = {np.array2string(sub.bounds, precision=2)}")
    print(f"  bound held in every case (max ratio)     = {ratios.max():.3f}  (<= 1 means the bound never lied)")

    print("\n  how much of the error is the Krylov truncation, and how much is the estimator?")
    m_rows = []
    for m_try in range(2, args.dim + 1):
        s = slow_subspace(X, tau_star, m_try, seed=0)
        m_rows.append([m_try, angle_degrees(s.vectors[:, 0], true_direction),
                       s.eigenvalues[0], s.residuals[0]])
    print_table(["Krylov dim m", "angle to truth (deg)", "mu_1", "residual"],
                m_rows, floatfmt="{:.4f}")
    print(f"  at m = {args.dim} = the dimension of the problem the Krylov truncation is gone")
    print("  entirely, so whatever angle is left is the *estimator's* error, not Krylov's")
    print(f"  measured asymmetry of the raw lag-covariance (tau = {tau_star}): "
          f"{sub.asymmetry:.3f}   (see krylov.slow_subspace on why this is symmetrised)")

    angles = []
    control_rng = np.random.default_rng(12345)   # NOT the rng that built the direction - see README
    for name, v in [("true (generating) direction", true_direction),
                    ("top principal component", top_pc),
                    ("best single coordinate", np.eye(args.dim)[np.argmax(np.abs(true_direction))]),
                    ("random direction", control_rng.standard_normal(args.dim))]:
        angles.append([name, angle_degrees(v, true_direction)])
    angles.append(["Krylov CV (1st Ritz vector)", angle_degrees(sub.vectors[:, 0], true_direction)])
    angles.append(["Krylov, 2nd Ritz vector", angle_degrees(sub.vectors[:, 1], true_direction)])
    print()
    print_table(["direction", "angle to the true slow direction (deg)"],
                [[n, a] for n, a in angles], floatfmt="{:.2f}")

    print("\n  start-vector independence (5 random Krylov starts):")
    starts = [slow_subspace(X, tau_star, args.m, seed=s) for s in range(5)]
    a1 = np.array([angle_degrees(s.vectors[:, 0], true_direction) for s in starts])
    m1 = np.array([s.eigenvalues[0] for s in starts])
    print(f"    angle to truth  {a1.mean():.2f} +- {a1.std(ddof=1):.2f} deg"
          f"   (spread {a1.max() - a1.min():.2f} deg)")
    print(f"    mu_1            {m1.mean():.4f} +- {m1.std(ddof=1):.4f}")
    print("    the start vector is arbitrary; the answer is not")

    section("5. Does the CV actually carry the physics? Free energy along each coordinate")
    X_exact = target.sample_exact(400_000, seed=3)
    rows = coordinate_table(X, X_exact, [
        ("true direction", true_direction),
        ("Krylov CV", sub.vectors[:, 0]),
        ("top principal component", top_pc),
        ("best single coordinate", np.eye(args.dim)[np.argmax(np.abs(true_direction))]),
        ("random direction", control_rng.standard_normal(args.dim)),
    ], true_direction)
    print(f"  reference: {X_exact.shape[0]:,} i.i.d. samples from the target (exact, no MCMC)")
    print()
    print_table(["coordinate", "angle (deg)", "barrier (chain)", "barrier (exact)",
                 "RMS(chain,exact)", "IACT (steps)"],
                [[r["name"], r["angle"], f"{r['barrier']:.2f} +- {r['barrier_se']:.2f}",
                  r["barrier_exact"], r["rms"], r["iact"]]
                 for r in rows], floatfmt="{:.2f}")
    print(f"  barrier is F(0) - min F on the coordinate, in kT; the exact value along the slow"
          f" direction is {target.max_barrier:.2f}")
    cv_row = rows[1]
    print(f"  the Krylov CV recovers {100 * cv_row['barrier'] / target.max_barrier:.0f}% of the"
          f" barrier; the top PC keeps {100 * rows[2]['barrier'] / target.max_barrier:.0f}%"
          f" of it (a well that looks single at that coordinate)")

    section("6. Markov consistency of the CV (Chapman-Kolmogorov in one dimension)")
    z = X @ sub.vectors[:, 0]
    rho = autocorrelation(z, max_lag=4 * tau_star)
    mu1 = sub.eigenvalues[0]
    print(f"  measured autocorrelation of z at lag tau  = {rho[tau_star]:.4f}"
          f"   (Ritz value mu_1 = {mu1:.4f})")
    print(f"  measured at lag 2 tau                     = {rho[2 * tau_star]:.4f}"
          f"   predicted by the 1D model mu_1^2 = {mu1 ** 2:.4f}"
          f"   (ratio {rho[2 * tau_star] / mu1 ** 2:.3f})")
    print(f"  self-consistency only: rho(tau)^2        = {rho[tau_star] ** 2:.4f}"
          f"   (ratio {rho[2 * tau_star] / rho[tau_star] ** 2:.3f})")
    print("  a ratio near 1 means the projected dynamics is Markovian at this resolution;")
    print("  deviations are the fast modes leaking back in, and they shrink as lag grows")

    if not args.skip_figures:
        figure_lag_scan(lags, eigenvalues, lag_angles, tau_u / 2, tau_star, sub)
        figure_cv(target, X, X_exact, sub, top_pc, true_direction, u, tau_star)
        figure_3d(args)
        print(f"\nfigures written to {FIGDIR}")


def figure_lag_scan(lags, eigenvalues, lag_angles, tau_direct, tau_star, sub) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    ax = axes[0]
    for k in range(eigenvalues.shape[1]):
        times = np.array([timescale_from_eigenvalue(mu, tau) for mu, tau in zip(eigenvalues[:, k], lags)])
        ax.plot(lags, times, marker="o", lw=1.4, label=f"Ritz mode {k + 1}")
    ax.axhline(tau_direct, color="#c53030", ls="--", lw=1.2,
               label=r"independent estimate: $\tau_{u}/2$")
    ax.axvline(tau_star, color="#a0aec0", ls=":", lw=1.2)
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"lag $\tau$ (MCMC steps)")
    ax.set_ylabel("implied timescale (steps)")
    ax.set_title("Timescale spectrum vs lag: the plateau is the answer,\n"
                 "and the fastest modes fall off the bottom", fontsize=10)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    ax.plot(lags, lag_angles, marker="s", lw=1.4, color="#2f855a")
    ax.axvline(tau_star, color="#a0aec0", ls=":", lw=1.2)
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"lag $\tau$ (MCMC steps)")
    ax.set_ylabel("angle to the true direction (deg)")
    ax.set_title("A longer lag does not buy a better direction:\n"
                 "it buys a better eigenvalue from fewer pairs", fontsize=10)

    ax = axes[2]
    idx = np.arange(len(sub.eigenvalues))
    ax.bar(idx, sub.eigenvalues, color="#2b6cb0")
    ax.set_xticks(idx, [f"{i + 1}" for i in idx])
    ax.set_xlabel("Ritz index")
    ax.set_ylabel(r"$\mu$ (closer to 1 = slower)")
    ax.set_ylim(0, 1.05)
    for i, (mu, res) in enumerate(zip(sub.eigenvalues, sub.residuals)):
        ax.annotate(f"{mu:.3f}", (i, mu), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=8)
    ax.set_title(r"Krylov spectrum at the chosen lag $\tau^*$" "\n"
                 "the separation between index 1 and index 2 is what makes a 1D coordinate\n"
                 "defensible; the rest are the fast transverse modes", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGDIR / "06_krylov_lag_scan.png", dpi=150)
    plt.close(fig)


def figure_cv(target, X, X_exact, sub, top_pc, true_direction, u, tau_star) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    ax = axes[0]
    for series, colour, label in [(u, "#2f855a", "true slow coordinate $u$"),
                                  (X @ top_pc, "#c53030", "top principal component"),
                                  (X @ sub.vectors[:, 0], "#2b6cb0", "Krylov CV $z_1$")]:
        rho = autocorrelation(series, max_lag=300)
        ax.plot(np.arange(len(rho)), rho, lw=1.4, color=colour, label=label)
    ax.set_xlabel("lag (steps)")
    ax.set_ylabel("autocorrelation")
    ax.set_title("Slowness is the property being asked for:\n"
                 "the CV tracks the slow coordinate, the PC does not", fontsize=10)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    for v, colour, label in [(sub.vectors[:, 0], "#2b6cb0", "Krylov CV"),
                             (top_pc, "#c53030", "top principal component")]:
        F, centres, _ = free_energy_profile(X @ v)
        ax.plot(centres, F, marker="o", ms=3, lw=1.2, color=colour, label=f"{label} (chain)")
        F_e, c_e, _ = free_energy_profile(X_exact @ v)
        ax.plot(c_e, F_e, lw=1.0, ls="--", color=colour, alpha=0.7, label=f"{label} (exact)")
    grid = np.linspace(-3, 3, 200)
    ax.plot(grid, target.free_energy(grid), color="#2f855a", lw=1.6,
            label="analytic $h(u^2-a^2)^2$")
    ax.axhline(target.max_barrier, color="#a0aec0", ls=":", lw=1.0)
    ax.set_ylim(-0.5, 4.2)
    ax.set_xlabel("coordinate value")
    ax.set_ylabel("free energy (kT)")
    ax.set_title("Free energy along a coordinate is a *property of that coordinate*:\n"
                 "a badly chosen one flattens the barrier", fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    names = ["Krylov CV", "top PC", "best single\ncoordinate", "random\ndirection"]
    vals = [angle_degrees(sub.vectors[:, 0], true_direction), angle_degrees(top_pc, true_direction),
            angle_degrees(np.eye(target.dim)[np.argmax(np.abs(true_direction))], true_direction),
            angle_degrees(np.random.default_rng(0).standard_normal(target.dim), true_direction)]
    ax.bar(names, vals, color=["#2b6cb0", "#c53030", "#d69e2e", "#718096"])
    ax.axhline(90.0, color="#a0aec0", lw=0.8)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.1f}", (i, v), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=9)
    ax.set_ylabel("angle to the true slow direction (deg)")
    ax.set_ylim(0, 100)
    ax.set_title("Which direction did each method find?\n"
                 "0 = the barrier direction, 90 = orthogonal to it", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGDIR / "06_krylov_cv.png", dpi=150)
    plt.close(fig)


def figure_3d(args) -> None:
    """The same experiment in three dimensions, where the whole picture fits on a plot.

    Three dimensions is the largest case that can be *looked at*, and the user's question was
    whether the method still means anything there. It does: the cloud is genuinely 3D, the
    reaction coordinate is still not aligned with any axis, and the two arrows (the direction
    that generated the data and the one Krylov recovered from the chain) land on top of each
    other.
    """
    target = AnisotropicWells(random_directions(3, 1, seed=2), a=args.a, h=args.h, kappa=args.kappa)
    chain, step, _, X = run_chain(target, max(20000, args.steps // 2), 3000, seed=5)
    sub = slow_subspace(X, 16, 4, seed=0)
    true_direction = target.directions[0]
    cv = sub.vectors[:, 0]
    angle = angle_degrees(cv, true_direction)
    print(f"\n  3D cross-check: d = 3, MALA step {step:.2f}, acceptance {chain.acceptance_rate:.2f},"
          f" angle to truth {angle:.2f} deg")

    fig = plt.figure(figsize=(12.4, 5.0))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    s = X[:: max(1, X.shape[0] // 4000)]
    ax.scatter(s[:, 0], s[:, 1], s[:, 2], c=s @ cv, s=3, cmap="coolwarm", alpha=0.5)
    anchor = target.well_centre()
    for vec, col, lw, label in [(true_direction, "#1a202c", 2.0, "generating direction"),
                                (cv, "#2b6cb0", 4.0, "Krylov CV from the chain")]:
        ax.quiver(*anchor, *(2.4 * vec), color=col, lw=lw, arrow_length_ratio=0.16, label=label)
    ax.set_title(f"3D samples coloured by the discovered CV\n"
                 f"black: true direction, blue: Krylov (angle {angle:.1f} deg)", fontsize=10)
    ax.set_xlabel("$x_0$")
    ax.set_ylabel("$x_1$")
    ax.set_zlabel("$x_2$")
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    ax2 = fig.add_subplot(1, 2, 2)
    u = X @ true_direction
    pc3 = np.linalg.eigh(np.cov(X.T))[1][:, -1]
    # an eigenvector has no preferred sign; flip both candidates to correlate positively with u
    # so that the figure reads as "same coordinate, possibly reflected" instead of "negative slope"
    cv_signed = cv if float(u @ (X @ cv)) > 0 else -cv
    pc_signed = pc3 if float(u @ (X @ pc3)) > 0 else -pc3
    ax2.scatter(u, X @ cv_signed, s=3, alpha=0.25, color="#2b6cb0", label="Krylov CV")
    ax2.scatter(u, X @ pc_signed, s=3, alpha=0.25, color="#c53030",
                label="top principal component")
    lim = 2.6
    ax2.plot([-lim, lim], [-lim, lim], color="#1a202c", lw=1.0, ls="--", label="identity")
    ax2.set_xlim(-lim, lim)
    ax2.set_ylim(-lim, lim)
    ax2.set_xlabel("true slow coordinate $u$")
    ax2.set_ylabel("candidate coordinate")
    ax2.set_title("The CV is a linear function of the true coordinate;\n"
                  "the variance-based one is a wide cloud", fontsize=10)
    ax2.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGDIR / "06_krylov_cv_3d.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
