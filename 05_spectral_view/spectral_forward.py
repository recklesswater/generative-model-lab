"""The forward noising process as a linear operator - and what its spectrum says.

A noising step is a convolution, i.e. a *linear operator* on the space of densities (a heat
semigroup). Everything the forward process destroys over time is therefore encoded in that
operator's spectrum: the eigenfunctions are Fourier modes, the eigenvalues decay like
exp(-t k^2), and high frequencies die first.

This script *computes* that spectrum instead of asserting it. It compares the eigenvalues
obtained by dense diagonalisation with the analytic Gaussian prediction, and reports the gap
instead of hiding it.

Idealisation, stated up front: the operator is treated as a pure heat semigroup with variance
t. In the variance-preserving parameterisation the signal is additionally scaled by
sqrt(abar_t); that is a diagonal rescaling of the state, not of the spectrum, so it does not
change the ordering of what dies first. See README.md.

Usage:  python 05_spectral_view/spectral_forward.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIGDIR = Path(__file__).resolve().parent.parent / "figures"
# The kernel must be sampled well above the grid spacing, otherwise the sampled Gaussian is
# barely resolved (a couple of points per sigma) and its discrete spectrum aliases badly at
# high frequencies.  N = 257 over a half-width of 4 gives dx ~ 0.031, i.e. ~5 points per sigma
# at sigma = 0.15 - enough for the numerical spectrum to match the analytic one in the band
# that actually carries signal.  The residual disagreement at the tail is reported, not hidden.
N = 257
L = 4.0
DX = 2 * L / N


def periodic_gaussian_kernel(sigma: float, n: int = N, dx: float = DX) -> np.ndarray:
    """Gaussian kernel sampled on a periodic grid, normalised to sum to one."""
    r = np.arange(n)
    r = np.minimum(r, n - r)
    k = np.exp(-0.5 * (r * dx) ** 2 / sigma ** 2)
    return k / k.sum()


def circulant_matrix(kernel: np.ndarray) -> np.ndarray:
    return np.array([np.roll(kernel, i) for i in range(len(kernel))])


def wavenumbers(n: int = N, dx: float = DX) -> np.ndarray:
    """Frequencies in cycles per unit length, ordered as numpy.fft.fftfreq."""
    return np.fft.fftfreq(n, d=dx)


def analytic_eigenvalues(sigma: float) -> np.ndarray:
    """Eigenvalues of convolution with N(0, sigma^2): exp(-2 pi^2 sigma^2 xi^2)."""
    xi = wavenumbers()
    return np.exp(-2 * np.pi ** 2 * sigma ** 2 * xi ** 2)


def main() -> None:
    FIGDIR.mkdir(exist_ok=True)
    xi = wavenumbers()
    order = np.argsort(xi)

    sigma0 = 0.15
    kernel = periodic_gaussian_kernel(sigma0)
    P = circulant_matrix(kernel)
    # For a circulant matrix the spectrum is exactly the DFT of the generating kernel, and the
    # mode order then matches numpy.fft.fftfreq. (Eigenvalues from a dense eigensolver come back
    # in an arbitrary order, so pairing them with fftfreq by sorting is wrong - that mistake was
    # made and caught here.)
    numeric = np.real(np.fft.fft(kernel))
    analytic = analytic_eigenvalues(sigma0)
    # residual proof that these really are the eigenvalues of P
    residual = 0.0
    for m in (1, 7, 33, N // 2):
        v = np.exp(2j * np.pi * m * np.arange(N) / N)
        residual = max(residual, float(np.max(np.abs(P @ v - numeric[m] * v))))
    abs_err = float(np.max(np.abs(numeric - analytic)))
    band = analytic > 1e-6          # where the comparison is meaningful
    rel_err = float(np.max(np.abs(numeric[band] - analytic[band]) / analytic[band]))
    print(f"operator: {N} x {N} circulant built from a Gaussian with sigma = {sigma0}")
    print(f"  eigenvalue residual ||P v - lambda v||  = {residual:.2e}   (v = Fourier modes)")
    print(f"  discrete spectrum (DFT of the sampled kernel) vs continuum formula:")
    print(f"    max absolute error            = {abs_err:.2e}")
    print(f"    max relative error, |lambda|>1e-6 = {rel_err:.2e}   "
          f"({int(band.sum())} of {N} modes)")
    print(f"    the sampled kernel needs ~5 points per sigma: at dx = {DX:.4f} and "
          f"sigma = {sigma0} that is {sigma0 / DX:.1f}")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for sigma, colour in zip([0.05, 0.15, 0.4], ["#2b6cb0", "#2f855a", "#c53030"]):
        lam = analytic_eigenvalues(sigma)[order]
        ax.semilogy(xi[order], np.maximum(lam, 1e-16), lw=1.6, color=colour,
                    label=f"sigma = {sigma:.2f}   (t = {sigma ** 2:.3f})")
        cutoff = 1.0 / (np.sqrt(2) * np.pi * sigma)
        ax.axvline(cutoff, color=colour, lw=0.8, ls=":")
    ax.set_xlabel("spatial frequency (cycles per unit length)")
    ax.set_ylabel("eigenvalue of the noising operator")
    ax.set_title("What the forward process destroys, per frequency\n"
                 "eigenvalue = exp(-2 pi^2 sigma^2 xi^2)")
    ax.set_xlim(0, 4.0)
    ax.set_ylim(1e-16, 1.5)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGDIR / "05_spectral_eigenvalues.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(P, cmap="magma")
    ax.set_title(f"The forward operator is circulant (sigma = {sigma0})\n"
                 "each row: how one bin spreads over all bins", fontsize=10)
    ax.set_xlabel("target bin")
    ax.set_ylabel("source bin")
    fig.colorbar(im, ax=ax, fraction=0.046, label="weight")
    fig.tight_layout()
    fig.savefig(FIGDIR / "05_spectral_operator.png", dpi=150)
    plt.close(fig)

    x = np.arange(N) * DX - L
    sigmas = np.linspace(0.001, 0.6, 60)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for k, colour in zip([1, 4, 16, 40], ["#2c5282", "#2b6cb0", "#38a169", "#c53030"]):
        mode = np.cos(2 * np.pi * k * x / (2 * L))
        amps = []
        for sigma in sigmas:
            out = circulant_matrix(periodic_gaussian_kernel(sigma)) @ mode
            amps.append(float(out @ mode / (mode @ mode)))
        ax.semilogy(sigmas ** 2, np.maximum(amps, 1e-16), lw=1.6, color=colour,
                    label=f"mode k = {k}")
    ax.set_xlabel("noise variance t = sigma^2")
    ax.set_ylabel("amplitude left in that mode")
    ax.set_title("High-frequency modes are destroyed first\n"
                 "same operator, four Fourier modes", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGDIR / "05_spectral_mode_decay.png", dpi=150)
    plt.close(fig)

    print("  mode amplitude remaining at sigma = 0.15:")
    for k in (1, 4, 16, 40):
        lam = float(np.exp(-2 * np.pi ** 2 * sigma0 ** 2 * (k / (2 * L)) ** 2))
        print(f"    k = {k:>2}  ->  {lam:.3e}")
    print("  noise variance t at which each mode drops to half:")
    for k in (1, 4, 16, 40):
        xi_k = k / (2 * L)
        t_half = np.log(2) / (2 * np.pi ** 2 * xi_k ** 2)
        print(f"    k = {k:>2}  ->  t = {t_half:.3f}")
    print(f"figures written to {FIGDIR}")


if __name__ == "__main__":
    main()
