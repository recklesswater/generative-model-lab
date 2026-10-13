"""Krylov subspace iterations, written out: Lanczos, Arnoldi, and the Ritz pairs they return.

The routines here are the textbook ones, not new ones. Sources, so that the provenance is not
ambiguous:

* ``lanczos``  - the symmetric Lanczos process, Golub & Van Loan, *Matrix Computations*, 4th ed.,
  section 10.1 (Algorithm 10.1.1), with the re-orthogonalisation of section 5.2.8.
* ``arnoldi``  - the Arnoldi process, Golub & Van Loan Algorithm 10.5.1 (and Trefethen & Bau,
  *Numerical Linear Algebra*, Lectures 33-36 for the Ritz-pair reading of what comes out).

Two things are deliberate:

1. **Everything is matrix-free.** The routines only ever *apply* the operator to a vector, so
   the operator never has to be written down as a matrix. That is the reason Krylov methods are
   used on problems that are too big to assemble - and it is also why the same code below runs
   unchanged on a 256x256 grid operator and on a d-by-d sample covariance.
2. **The residuals are computed in closed form** from the last sub-diagonal entry of the
   tridiagonal (Lanczos) or Hessenberg (Arnoldi) matrix. The demos then compare that cheap
   formula against the literal ``||A u - theta u||``, so the number on screen is checked rather
   than trusted.

Nothing in this file is a contribution. If it disagrees with the books above, the books win.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def as_matvec(A):
    """Accept either a callable ``v -> A v`` or a square array, and return a callable."""
    if callable(A):
        return A
    M = np.asarray(A, dtype=float)
    return lambda v: M @ v


def reorthogonalise(w: np.ndarray, V: np.ndarray, passes: int = 2) -> np.ndarray:
    """Classical Gram-Schmidt against the columns of ``V``, applied ``passes`` times.

    One pass is enough in exact arithmetic and is *not* enough in floating point once the
    Krylov basis starts to converge: the loss of orthogonality is silent, and its symptom is a
    duplicated ("ghost") Ritz value that looks converged and is not. Two passes are the standard
    guard (Golub & Van Loan, section 5.2.8). ``krylov_basics.py`` measures what happens without
    it instead of asserting it.
    """
    for _ in range(passes):
        w = w - V @ (V.T @ w)
    return w


@dataclass
class LanczosRun:
    """One Lanczos run: the tridiagonal ``T`` and the basis it was built on."""

    alpha: np.ndarray          # (k,)   diagonal of T
    beta: np.ndarray           # (k-1,) first sub/super-diagonal of T
    basis: np.ndarray          # (n, k) orthonormal basis of the Krylov subspace
    tail: float = 0.0          # ||w|| after the final step: the beta_k of the residual bound
    breakdown: bool = False    # True if the Krylov space closed early (invariant subspace)

    @property
    def k(self) -> int:
        return int(self.alpha.size)

    @property
    def T(self) -> np.ndarray:
        return (np.diag(self.alpha)
                + np.diag(self.beta, 1)
                + np.diag(self.beta, -1))


def lanczos(A, m: int, v0: np.ndarray, reorth: bool = True,
            breakdown_tol: float = 1e-11) -> LanczosRun:
    """Symmetric Lanczos with start vector ``v0`` and at most ``m`` iterations.

    Requires an operator that is symmetric in the standard inner product (or, more generally,
    self-adjoint in whatever inner product the operator is *supposed* to be studied in - the
    generalised case is handled by ``generalized_slow_subspace`` below). Passing a non-symmetric
    operator here is a mistake: it does not raise, it returns plausible-looking wrong numbers.
    ``krylov_basics.py`` section 2 shows exactly that.
    """
    apply = as_matvec(A)
    v0 = np.asarray(v0, dtype=float).ravel()
    n = v0.size
    norm0 = float(np.linalg.norm(v0))
    if norm0 == 0.0:
        raise ValueError("start vector is zero")

    basis = np.zeros((n, m))
    alpha = np.empty(m)
    beta = np.empty(max(m - 1, 0))
    v = v0 / norm0
    k = m
    breakdown = False
    tail = 0.0

    for j in range(m):
        basis[:, j] = v
        w = apply(v)
        alpha[j] = float(v @ w)
        w = w - alpha[j] * v
        if j > 0:
            w = w - beta[j - 1] * basis[:, j - 1]
        if reorth:
            w = reorthogonalise(w, basis[:, : j + 1])
        b = float(np.linalg.norm(w))
        if b <= breakdown_tol:
            k = j + 1
            breakdown = True
            break
        if j < m - 1:
            beta[j] = b
            v = w / b
        else:
            # The last norm is *not* an entry of T: T has only k-1 subdiagonals, while the
            # residual bound needs ||w|| after step k. Storing it separately is the whole point of
            # this variable - an earlier version of this file used beta[-1] here and produced a
            # bound that was wrong by a constant factor (see the trap in the README).
            tail = b

    return LanczosRun(alpha[:k], beta[: k - 1], basis[:, :k], tail, breakdown)


def lanczos_ritz(A, m: int, v0: np.ndarray, reorth: bool = True,
                 order: str = "descending"):
    """Ritz values, Ritz vectors and the cheap residual bound from one Lanczos run.

    Returns ``(theta, U, resid_bound)`` with ``U[:, i]`` the Ritz vector belonging to
    ``theta[i]``. The residual bound ``|beta_k * s_k[i]|`` is the textbook one: it costs nothing
    extra because ``s`` (the eigenvector of ``T``) is already computed.
    """
    run = lanczos(A, m, v0, reorth=reorth)
    theta, S = np.linalg.eigh(run.T)
    U = run.basis @ S
    resid = np.abs(run.tail * S[-1, :])
    idx = np.argsort(-theta) if order == "descending" else np.argsort(theta)
    return theta[idx], U[:, idx], resid[idx]


@dataclass
class ArnoldiRun:
    """One Arnoldi run: the upper Hessenberg ``H`` and its basis."""

    H: np.ndarray              # (k+1, k) upper Hessenberg
    basis: np.ndarray          # (n, k)
    breakdown: bool = False

    @property
    def k(self) -> int:
        return int(self.H.shape[1])

    @property
    def square(self) -> np.ndarray:
        return self.H[: self.k, : self.k]


def arnoldi(A, m: int, v0: np.ndarray, reorth: bool = True,
            breakdown_tol: float = 1e-11) -> ArnoldiRun:
    """Arnoldi with start vector ``v0`` and at most ``m`` iterations. No symmetry assumed.

    Same Gram-Schmidt-with-re-orthogonalisation as ``lanczos``; the difference is that every
    previous basis vector is orthogonalised against (not just the last two), because a general
    operator does not produce a tridiagonal matrix.
    """
    apply = as_matvec(A)
    v0 = np.asarray(v0, dtype=float).ravel()
    n = v0.size
    norm0 = float(np.linalg.norm(v0))
    if norm0 == 0.0:
        raise ValueError("start vector is zero")

    basis = np.zeros((n, m))
    H = np.zeros((m + 1, m))
    v = v0 / norm0
    k = m
    breakdown = False

    for j in range(m):
        basis[:, j] = v
        w = apply(v)
        for i in range(j + 1):
            H[i, j] = float(basis[:, i] @ w)
            w = w - H[i, j] * basis[:, i]
        if reorth:
            for _ in range(2):
                for i in range(j + 1):
                    c = float(basis[:, i] @ w)
                    H[i, j] += c
                    w = w - c * basis[:, i]
        h = float(np.linalg.norm(w))
        H[j + 1, j] = h
        if h <= breakdown_tol:
            k = j + 1
            breakdown = True
            break
        v = w / h

    return ArnoldiRun(H[:, :k], basis[:, :k], breakdown)


def arnoldi_ritz(A, m: int, v0: np.ndarray, reorth: bool = True,
                 order: str = "descending"):
    """Ritz values (complex, in general), Ritz vectors and the cheap residual bound.

    Sorted by real part, descending - the convention that matters when the operator is a
    transition operator and the interesting eigenvalues are the ones near 1.
    """
    run = arnoldi(A, m, v0, reorth=reorth)
    mu, S = np.linalg.eig(run.square)
    U = run.basis @ S
    if run.k > 1:
        resid = np.abs(run.H[run.k, run.k - 1] * S[-1, :])
    else:
        resid = np.zeros(run.k)
    idx = np.argsort(-mu.real) if order == "descending" else np.argsort(mu.real)
    return mu[idx], U[:, idx], resid[idx]


def true_residual_norms(A, theta: np.ndarray, U: np.ndarray) -> np.ndarray:
    """The literal ``||A u - theta u||`` for every Ritz pair - the check on the closed form."""
    apply = as_matvec(A)
    out = np.empty(len(theta))
    for i in range(len(theta)):
        out[i] = float(np.linalg.norm(apply(U[:, i]) - theta[i] * U[:, i]))
    return out


def angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between two vectors in degrees, in [0, 90] (sign of a direction is not meaningful)."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    c = abs(float(a @ b)) / (float(np.linalg.norm(a)) * float(np.linalg.norm(b)))
    return float(np.degrees(np.arccos(min(1.0, max(0.0, c)))))


# ----------------------------------------------------------------------------------------
# From a chain to a symmetric operator: the only part of this file that knows about MCMC.
# ----------------------------------------------------------------------------------------


def lagged_covariances(X: np.ndarray, tau: int):
    """``(C0, C_tau)``: the equal-time and lag-``tau`` covariance of the features ``X``.

    ``C0[i,j] = <x_i(t) x_j(t)>`` and ``C_tau[i,j] = <x_i(t) x_j(t+tau)>``, both with the sample
    mean subtracted, so that the constant function (the stationary mode, eigenvalue 1) is
    projected out. Estimating the transfer operator from a trajectory through these two matrices
    is standard practice - see the provenance note in the README.
    """
    Xc = np.asarray(X, dtype=float)
    Xc = Xc - Xc.mean(axis=0)
    n = Xc.shape[0]
    C0 = Xc.T @ Xc / (n - 1)
    Ctau = Xc[:-tau].T @ Xc[tau:] / (n - tau)
    return C0, Ctau


@dataclass
class SlowSubspace:
    """A Krylov estimate of the slow subspace of a transfer operator, in the sample coordinates.

    ``vectors[:, i]`` is the i-th slow coordinate direction in R^d: projecting a sample gives the
    collective variable ``z_i = vectors[:, i] . x``. Eigenvalues come out in (0, 1] for a
    reversible chain, with values near 1 meaning slow.
    """

    eigenvalues: np.ndarray       # (k,)
    vectors: np.ndarray           # (d, k)
    residuals: np.ndarray         # (k,) relative residual of the generalised problem
    bounds: np.ndarray            # (k,) closed-form Lanczos residual bound in the whitened space
    C0: np.ndarray
    Ctau: np.ndarray
    whitener: np.ndarray          # (d, d) Cholesky factor L of C0, kept so residuals can be checked
    asymmetry: float              # ||Ctau - Ctau^T||_F / ||Ctau||_F of the raw estimate
    tau: int

    def project(self, X: np.ndarray) -> np.ndarray:
        """Collective variables ``z = X @ vectors`` for an array of samples, shape (n, k)."""
        return np.asarray(X, dtype=float) @ self.vectors


def slow_subspace(X: np.ndarray, tau: int, m: int, v0: np.ndarray | None = None,
                  seed: int = 0, reorth: bool = True, symmetrize: bool = True) -> SlowSubspace:
    """Slow subspace of the transfer operator estimated from a chain, by Lanczos.

    The generalised eigenproblem is ``C_tau v = mu C0 v``. It is solved *without* forming
    ``C0^-1 C_tau``: ``C0`` is positive definite, so write ``C0 = L L^T`` (Cholesky) and apply
    Lanczos to the symmetric operator

        A = L^-1 C_tau L^-T

    which has the same eigenvalues, and whose eigenvectors are related to the generalised ones by
    ``v = L^-T w``. That is the whole trick: reversibility of the chain is what makes ``C_tau``
    symmetric, symmetry is what makes the *symmetric* Lanczos process valid here, and whitening by
    ``L`` is what makes the problem look symmetric in the standard inner product. If the chain is
    not reversible, ``C_tau`` is not symmetric, this function's answer is not trustworthy, and
    ``arnoldi`` on ``C0^-1 C_tau`` is the right tool instead (``krylov_basics.py`` section 3).

    ``seed`` only sets the random start vector - it is the one arbitrary ingredient, and the
    demos check that the answer does not depend on it.

    ``symmetrize`` replaces the raw sample lag-covariance by its symmetric part
    ``(C_tau + C_tau^T)/2``. This matters and is not cosmetic. For a reversible chain the *true*
    ``C_tau`` is symmetric, which is exactly what licenses Lanczos here; the *sample* estimate is
    not, and it is not approximately symmetric either once the lag is a good fraction of the
    correlation time (the script prints the measured asymmetry). Feeding the raw estimate into
    Lanczos applies a symmetric algorithm to a slightly non-symmetric operator: the eigenvalues
    survive, the residual bound stops meaning anything, and the eigenvectors get worse. The
    returned ``asymmetry`` is what that costs, measured.
    """
    C0, Ctau = lagged_covariances(X, tau)
    asymmetry = float(np.linalg.norm(Ctau - Ctau.T) / max(np.linalg.norm(Ctau), 1e-300))
    if symmetrize:
        Ctau = 0.5 * (Ctau + Ctau.T)
    try:
        L = np.linalg.cholesky(C0)
    except np.linalg.LinAlgError as exc:                       # pragma: no cover - reported, not hidden
        cond = float(np.linalg.cond(C0))
        raise np.linalg.LinAlgError(
            f"C0 is not positive definite (condition number {cond:.2e}). The sample covariance "
            "of a short chain in a high dimension can be singular, and then there is no "
            "generalised eigenproblem to solve. Lengthen the chain or shrink the feature set."
        ) from exc

    def operator(w: np.ndarray) -> np.ndarray:
        y = np.linalg.solve(L.T, w)          # L^-T w
        return np.linalg.solve(L, Ctau @ y)  # L^-1 C_tau y

    if v0 is None:
        v0 = np.random.default_rng(seed).standard_normal(C0.shape[0])
    eigenvalues, Utilde, bounds = lanczos_ritz(operator, m, v0, reorth=reorth)
    vectors = np.linalg.solve(L.T, Utilde)                     # back to sample coordinates
    # unit-norm directions, so that a collective variable z = v . x is on the same scale as any
    # hand-picked coordinate; only the direction matters, so this is free.
    vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=0), 1e-300)

    # Residual measured in the inner product the Lanczos bound is valid in. The generalised
    # residual ``C_tau v - mu C0 v`` becomes ``L^T (A w - mu w)`` after whitening, so its
    # ``C0^-1``-norm is exactly ``||A w - mu w||`` - the quantity the closed-form bound bounds.
    # Checking it in the *Euclidean* norm of the sample coordinates instead is a different
    # number (it is scaled by however much L stretches), and comparing the two is a mistake this
    # module made once and now computes explicitly.
    e = np.linalg.solve(L, Ctau @ vectors - eigenvalues * (C0 @ vectors))
    w = L.T @ vectors
    residuals = np.linalg.norm(e, axis=0) / np.maximum(np.linalg.norm(w, axis=0), 1e-300)
    return SlowSubspace(eigenvalues, vectors, residuals, bounds, C0, Ctau, L, asymmetry, tau)
