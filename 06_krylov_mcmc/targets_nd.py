"""Targets in d dimensions whose slow direction is known exactly, and is not a coordinate axis.

The whole point of a collective variable is that you do not get to read it off the sample
cloud. So the toy here is built the other way round: the reaction coordinate is *chosen first*
(a random orthonormal set of directions), the density is written in terms of it, and the task
is to recover that direction from an MCMC chain alone.

    U(x) = sum_k h_k (u_k^2 - a_k^2)^2 + kappa/2 * |x - sum_k u_k d_k|^2,      u_k = d_k . x

Properties that make this a usable test bed (all exact, no approximation error):

* the wells sit at ``u_k = +-a_k`` with the transverse coordinates zero, so there are
  ``2^n_slow`` modes;
* the barrier separating the two wells along ``u_k`` is ``h_k a_k^4``;
* the marginal of ``u`` factorises, ``p(u) ~ exp(-sum_k h_k (u_k^2 - a_k^2)^2)``, because the
  transverse terms do not depend on ``u`` and the rotation ``x -> (u, x_perp)`` has unit
  Jacobian. The free-energy profile along the reaction coordinate is therefore known in closed
  form and can be compared against whatever an MCMC chain produces.

The one free choice that makes the problem non-trivial is the ratio of **variance to slowness**:
``kappa`` is set low enough that the transverse directions carry *more* variance than the
reaction coordinate, while relaxing much faster. A variance-based method (PCA) therefore points
somewhere else entirely, and only a method that looks at how fast a direction *changes* finds
the barrier. That contrast is the experiment; the module README states it with numbers.
"""

from __future__ import annotations

import numpy as np


def random_directions(dim: int, n_slow: int, seed: int = 0) -> np.ndarray:
    """``n_slow`` orthonormal, dense directions in R^dim - deliberately not coordinate axes."""
    if n_slow > dim:
        raise ValueError("cannot have more directions than dimensions")
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((dim, n_slow)))
    return Q.T                       # (n_slow, dim), rows orthonormal


class AnisotropicWells:
    """``U(x)`` above: wells along a set of orthonormal directions, stiff everywhere else.

    Interface matches ``01_mcmc_toy/targets.py`` (``log_prob``, ``grad_log_prob``), with ``d``
    free to be any dimension, so the samplers from module 01 run on it unchanged.
    """

    name = "anisotropic wells"

    def __init__(self, directions: np.ndarray, a=1.5, h=0.5, kappa: float = 1.0) -> None:
        D = np.atleast_2d(np.asarray(directions, dtype=float))
        if D.shape[0] > D.shape[1]:
            raise ValueError("need n_slow <= dim")
        Q, _ = np.linalg.qr(D.T)                       # orthonormalise the rows
        self.directions = Q[:, : D.shape[0]].T         # (n_slow, dim)
        self.n_slow, self.dim = self.directions.shape
        self.a = np.broadcast_to(np.asarray(a, dtype=float), (self.n_slow,)).astype(float).copy()
        self.h = np.broadcast_to(np.asarray(h, dtype=float), (self.n_slow,)).astype(float).copy()
        self.kappa = float(kappa)
        self.barriers = self.h * self.a ** 4

    # -- the two functions the samplers are allowed to know --------------------------------
    def project(self, x: np.ndarray) -> np.ndarray:
        """``u = D x``: the coordinates along the slow directions (shape (n_slow,))."""
        return self.directions @ np.asarray(x, dtype=float).ravel()

    def potential(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float).ravel()
        u = self.project(x)
        x_perp = x - self.directions.T @ u
        return float(np.sum(self.h * (u ** 2 - self.a ** 2) ** 2)
                     + 0.5 * self.kappa * float(x_perp @ x_perp))

    def log_prob(self, x: np.ndarray) -> float:
        return -self.potential(x)

    def grad_log_prob(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).ravel()
        u = self.project(x)
        x_perp = x - self.directions.T @ u
        du = 4.0 * self.h * u * (u ** 2 - self.a ** 2)     # dU/du_k
        return -(self.directions.T @ du + self.kappa * x_perp)

    # -- exact references, used to *check* the estimates, never to produce them -------------
    def free_energy(self, u: np.ndarray) -> np.ndarray:
        """Exact ``F(u) = -log p(u) + const`` up to an additive constant: ``sum_k h_k (u_k^2-a_k^2)^2``."""
        u = np.asarray(u, dtype=float)
        if u.ndim == 1 and self.n_slow > 1:
            u = u.reshape(-1, self.n_slow)
        if u.shape[-1] != self.n_slow:          # a flat 1D array with a single slow direction
            u = u.reshape(*u.shape, 1)
        return np.sum(self.h * (u ** 2 - self.a ** 2) ** 2, axis=-1)

    @property
    def max_barrier(self) -> float:
        return float(np.max(self.barriers))

    def well_centre(self, signs=None) -> np.ndarray:
        """The centre of one well: ``sum_k +- a_k d_k``."""
        if signs is None:
            signs = np.ones(self.n_slow)
        return self.directions.T @ (np.asarray(signs, dtype=float) * self.a)

    def describe(self) -> str:
        barriers = ", ".join(f"{b:.2f}" for b in self.barriers)
        return (f"{self.name}: dim = {self.dim}, slow directions = {self.n_slow}, "
                f"barrier(s) = {barriers} kT, kappa = {self.kappa:g}")

    # -- ground truth: independent samples, no MCMC ----------------------------------------
    def sample_exact(self, n: int, seed: int = 0) -> np.ndarray:
        """``n`` independent samples from the target, by construction rather than by sampling.

        The density factorises into one 1D double well per slow direction and an isotropic
        Gaussian in the remaining ``dim - n_slow`` directions, so both parts can be drawn
        directly: the 1D marginal by inverse-CDF on a fine grid (the same trick that makes the
        analytic profile exact), the transverse part as ``N(0, I/kappa)`` mapped into the
        orthogonal complement. The result is unbiased and i.i.d., which gives the demos a
        reference that is independent of the chain being tested - for *any* direction, since the
        samples can be projected onto whatever coordinate is being judged.
        """
        rng = np.random.default_rng(seed)
        us = np.empty((n, self.n_slow))
        for k in range(self.n_slow):
            width = 1.0 / np.sqrt(8.0 * self.h[k] * self.a[k] ** 2)   # local well width
            hi = self.a[k] + 8.0 * width
            grid = np.linspace(-hi, hi, 40001)
            dens = np.exp(-self.h[k] * (grid ** 2 - self.a[k] ** 2) ** 2)
            cdf = np.cumsum(dens)
            cdf /= cdf[-1]
            us[:, k] = np.interp(rng.random(n), cdf, grid)

        x = self.directions.T @ us.T                       # (dim, n) part along the slow directions
        n_perp = self.dim - self.n_slow
        if n_perp > 0:
            A = rng.standard_normal((self.dim, n_perp))
            A -= self.directions.T @ (self.directions @ A)  # project onto the orthogonal complement
            basis, _ = np.linalg.qr(A)
            x = x + basis @ (rng.standard_normal((n_perp, n)) / np.sqrt(self.kappa))
        return x.T


class AnisotropicGaussian:
    """``U(x) = 0.5 * sum_i lam_i x_i^2`` - the *negative control* for slow-subspace hunting.

    Its generator has eigenvalues ``-lam_i``; if the spectrum is nearly flat there is no slow
    mode to find, so any direction a Krylov method returns is noise. Used in
    ``dimension_scan.py`` to show what "no spectral gap" looks like as a number.
    """

    name = "anisotropic gaussian"

    def __init__(self, spectrum: np.ndarray) -> None:
        self.lam = np.asarray(spectrum, dtype=float).ravel()
        self.dim = self.lam.size

    def potential(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float).ravel()
        return 0.5 * float(np.sum(self.lam * x ** 2))

    def log_prob(self, x: np.ndarray) -> float:
        return -self.potential(x)

    def grad_log_prob(self, x: np.ndarray) -> np.ndarray:
        return -self.lam * np.asarray(x, dtype=float).ravel()
