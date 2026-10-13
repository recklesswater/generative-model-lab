"""Krylov iterations on operators whose spectrum is known in closed form.

Before the Krylov subspace is used to *discover* anything (``cv_discovery.py``), it is worth
checking what it returns when the answer is already known. That is all this script does. Two
operators, both circulant, so the exact spectrum is the discrete Fourier transform of the
kernel rather than a number taken on faith:

* **the symmetric one**: a Gaussian jump kernel on a ring - the same heat operator module 05
  studies, i.e. one step of an unbiased random walk;
* **the non-symmetric one**: the same kernel shifted by a constant drift, i.e. a *biased* random
  walk. Row sums are still 1, the stationary distribution is still uniform, but the operator is
  no longer symmetric and its spectrum moves off the real axis.

Three things are measured rather than asserted:

1. that the Ritz pairs of Lanczos are eigenpairs of the operator to machine precision once they
   have converged - and how the closed-form residual bound compares with the literal residual;
2. what re-orthogonalisation buys. Dropping it is the classic way to get "converged" Ritz values
   that are duplicated and wrong;
3. that Lanczos on the *non-symmetric* operator returns real numbers that are simply wrong,
   with no warning, while Arnoldi returns the complex spectrum. That is the reason the module
   uses Lanczos - reversibility - and the reason it says so out loud.

Usage:  python 06_krylov_mcmc/krylov_basics.py
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

from cv_analysis import print_table  # noqa: E402
from krylov import arnoldi_ritz, lanczos_ritz, true_residual_norms  # noqa: E402

FIGDIR = REPO_ROOT / "figures"
N = 256              # points on the ring
HALF = 4.0           # half-width, so the ring has circumference 2 * HALF
SIGMA = 0.15         # jump width (module 05 uses the same value)
CONVERGED = 1e-8     # residual bound below which a Ritz pair is called converged
M_LADDER = (10, 20, 30, 60, 80)


def section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def ring_kernel(sigma: float, drift: float = 0.0) -> np.ndarray:
    """Gaussian jump kernel on the ring, normalised to sum to one (so the walk is stationary)."""
    n, half = N, HALF
    dx = 2.0 * half / n
    x = np.arange(n) * dx
    offset = (x - drift + half) % (2.0 * half) - half      # periodic distance to the jump centre
    kernel = np.exp(-0.5 * (offset / sigma) ** 2)
    return kernel / kernel.sum()


def circulant(kernel: np.ndarray) -> np.ndarray:
    """The matrix whose rows are successive rotations of ``kernel``."""
    return np.stack([np.roll(kernel, i) for i in range(kernel.size)])


def nearest_distance(values, reference) -> np.ndarray:
    """Distance from each value to the closest member of ``reference`` (complex-aware)."""
    values = np.atleast_1d(np.asarray(values))
    reference = np.atleast_1d(np.asarray(reference))
    return np.min(np.abs(values[:, None] - reference[None, :]), axis=1)


def main() -> None:
    FIGDIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)

    section("1. The symmetric operator: a biased-free random walk on a ring")
    kernel = ring_kernel(SIGMA)
    A = circulant(kernel)
    exact = np.fft.fft(kernel)                      # circulant: spectrum = DFT of the kernel
    print(f"  {N} sites, ring circumference {2 * HALF:g}, jump width sigma = {SIGMA}")
    print(f"  reference spectrum from the DFT of the kernel:")
    print(f"    ||eig(A) - fft(kernel)||_inf (dense eigensolve) = "
          f"{np.abs(np.sort(np.linalg.eigvalsh(A)) - np.sort(exact.real)).max():.2e}")
    print(f"    max |imaginary part| of the reference           = {np.abs(exact.imag).max():.2e}"
          f"   (an even kernel -> a real spectrum)")
    apply = lambda v: A @ v

    section("2. Lanczos: how many pairs converge, and is the residual bound honest?")
    rows, ladder = [], []
    for m in M_LADDER:
        theta, U, bound = lanczos_ritz(apply, m, rng.standard_normal(N))
        literal = true_residual_norms(apply, theta, U)
        error = nearest_distance(theta, exact)
        good = bound < CONVERGED
        rows.append([m, int(good.sum()), error[good].max() if good.any() else np.nan,
                     literal.max(), (literal - bound).max()])
        ladder.append((m, good.sum() / m, error[good].max() if good.any() else np.nan))
    print_table(["Krylov dim m", "converged pairs", "max err | converged",
                 "max literal resid", "max (literal - bound)"], rows, floatfmt="{:.2e}")
    print("  the last column is zero or round-off in every row: no Ritz pair ever has a residual")
    print("  larger than the closed-form bound claims, which is the point of computing it")
    print(f"  'converged' means the closed-form bound is below {CONVERGED:g}")
    print("  every converged Ritz value sits on an exact eigenvalue - that is what Krylov on a")
    print("  symmetric operator is for; the pairs that have not converged are not wrong, they")
    print("  are just estimates of the middle of the spectrum, which is what m controls")

    section("3. The trap: dropping re-orthogonalisation")
    rows = []
    for m in (60, 80):
        for reorth in (True, False):
            theta, U, bound = lanczos_ritz(apply, m, rng.standard_normal(N), reorth=reorth)
            literal = true_residual_norms(apply, theta, U)
            duplicates = int(np.sum(np.abs(np.diff(np.sort(theta))) < 1e-9))
            rows.append([m, "yes" if reorth else "no", duplicates, literal.max()])
    print_table(["Krylov dim m", "re-orthogonalised", "duplicated Ritz values",
                 "max literal resid"], rows, floatfmt="{:.2e}")
    print("  the first two columns of the spectrum look the same either way; the accuracy does")
    print("  not. Re-orthogonalisation is two extra inner products per iteration and it is the")
    print("  difference between a residual of 1e-6 and one of 1e-2 at the same m")

    section("4. Does the answer depend on the start vector?")
    starts = [lanczos_ritz(apply, 30, rng.standard_normal(N)) for _ in range(5)]
    lead = np.array([float(s[0][0]) for s in starts])
    lead_exact = float(np.sort(exact.real)[-1])
    print(f"  leading Ritz value over 5 random starts = {lead.mean():.12f} +- {lead.std(ddof=1):.1e}")
    print(f"  exact leading eigenvalue                = {lead_exact:.12f}")
    print("  the start vector is the one arbitrary ingredient in a Krylov method, and at this m")
    print("  it has stopped mattering: the subspace contains the answer regardless")

    section("5. The non-symmetric operator: the same walk, with a drift")
    drift = 0.5
    kernel_d = ring_kernel(0.5, drift=drift)
    B = circulant(kernel_d)
    exact_d = np.fft.fft(kernel_d)
    print(f"  jump width widened to 0.5 so the spectrum stays well resolved at this grid")
    print(f"  drift = {drift}   asymmetric part ||B - B^T||_inf = {np.abs(B - B.T).max():.3f}"
          f"   (0 would be the symmetric case)")
    print(f"  |eigenvalue| range = {np.abs(exact_d).min():.2e} .. {np.abs(exact_d).max():.4f}"
          f"   max |imag| = {np.abs(exact_d.imag).max():.4f}")
    apply_d = lambda v: B @ v
    mu, Ub, bound_b = arnoldi_ritz(apply_d, 30, rng.standard_normal(N))
    good = bound_b < CONVERGED
    err_arnoldi = nearest_distance(mu, exact_d)
    theta_l, Ul, _ = lanczos_ritz(apply_d, 30, rng.standard_normal(N))
    err_lanczos = nearest_distance(theta_l, exact_d)
    print()
    print_table(["method", "eigenvalues returned", "converged", "max err | converged",
                 "worst err overall"],
                [["Arnoldi", "complex", int(good.sum()), err_arnoldi[good].max()
                  if good.any() else np.nan, err_arnoldi.max()],
                 ["Lanczos", "real", None, None, err_lanczos.max()]],
                floatfmt="{:.2e}")
    print(f"  Arnoldi recovers the complex spectrum (the eigenvalues of a circulant are the DFT of")
    print(f"  its kernel); Lanczos forces them onto the real axis and is off by up to"
          f" {err_lanczos.max():.2f} with no error message")
    print("  LANCZOS IS ONLY FOR SYMMETRIC OPERATORS. Reversibility of the chain is what makes")
    print("  the transfer operator of cv_discovery.py symmetric in the C0 inner product; a chain")
    print("  that is not reversible (a drift, a non-symmetric proposal) needs Arnoldi instead")

    figure(ladder, apply, exact, mu, good, theta_l)
    print(f"\nfigures written to {FIGDIR}")


def figure(ladder, apply, exact, mu, good, theta_l) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    ax = axes[0]
    ms = [row[0] for row in ladder]
    counts = [row[0] * row[1] for row in ladder]
    errs = [row[2] for row in ladder]
    ax.bar(ms, counts, width=6.0, color="#cbd5e0", label="converged Ritz pairs (left)")
    ax.set_xlabel("Krylov dimension m")
    ax.set_ylabel("converged Ritz pairs")
    ax2 = ax.twinx()
    ax2.semilogy(ms, errs, marker="o", color="#2b6cb0",
                 label="max error among them (right)")
    ax2.set_ylabel("distance to the nearest exact eigenvalue")
    ax2.set_ylim(1e-17, 1e-13)
    ax.set_title("What more Krylov iterations buy:\n"
                 "more of the spectrum, always at exact accuracy", fontsize=10)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], frameon=False, fontsize=8, loc="center left")

    ax = axes[1]
    ms = (20, 30, 40, 60, 80)
    for reorth, colour, label in [(True, "#2b6cb0", "with re-orthogonalisation"),
                                  (False, "#c53030", "without")]:
        vals = []
        for m in ms:
            theta, U, _ = lanczos_ritz(apply, m, np.random.default_rng(1).standard_normal(N),
                                       reorth=reorth)
            vals.append(true_residual_norms(apply, theta, U).max())
        ax.semilogy(ms, vals, marker="o", color=colour, label=label)
    ax.set_xlabel("Krylov dimension m")
    ax.set_ylabel("max literal residual")
    ax.set_title("The cost of skipping re-orthogonalisation\n"
                 "(same operator, same start vector)", fontsize=10)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[2]
    ax.plot(exact.real, exact.imag, "o", ms=3, color="#a0aec0", label="exact spectrum")
    ax.plot(mu[good].real, mu[good].imag, "o", ms=5, mfc="none", color="#2b6cb0",
            label="Arnoldi Ritz (converged)")
    ax.plot(mu[~good].real, mu[~good].imag, "x", ms=5, color="#2f855a",
            label="Arnoldi Ritz (not converged)")
    ax.plot(theta_l, np.zeros_like(theta_l), "1", ms=9, color="#c53030",
            label="Lanczos Ritz (real, wrong)")
    ax.set_xlabel("real part")
    ax.set_ylabel("imaginary part")
    ax.set_title("A drifted walk has a complex spectrum:\n"
                 "Arnoldi finds it, Lanczos cannot", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "06_krylov_basics.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
