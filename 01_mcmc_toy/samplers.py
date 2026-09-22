"""Three samplers on the same target: random-walk Metropolis, MALA and Hamiltonian MC.

The three differ only in what they are allowed to know about the target and how far they
move per iteration:

===========  ==================  =========================
sampler      uses gradient?      proposal
===========  ==================  =========================
RWM          no                  isotropic Gaussian walk
MALA         yes                 gradient drift + Gaussian noise
HMC          yes                 simulate Hamiltonian dynamics, then accept
===========  ==================  =========================

All three are Metropolis-corrected, so all three are exact in the limit - they differ in how
many *iterations* that takes, which is what the diagnostics measure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from targets import Target


@dataclass
class Chain:
    """Result of a sampler run."""

    samples: np.ndarray
    accepted: np.ndarray = field(repr=False)
    name: str = "chain"
    grad_evals: int = 0
    cost_per_step: float = 1.0  # target evaluations per iteration (log-density or gradient)

    @property
    def acceptance_rate(self) -> float:
        return float(self.accepted.mean())

    @property
    def n_steps(self) -> int:
        return int(self.samples.shape[0])

    @property
    def grad_evals_per_step(self) -> float:
        return self.grad_evals / max(self.n_steps, 1)


def random_walk_metropolis(
    target: Target,
    n_steps: int = 200_000,
    step_size: float = 0.35,
    x0: np.ndarray | None = None,
    seed: int = 0,
) -> Chain:
    """Symmetric random-walk Metropolis. No gradient information is used."""
    rng = np.random.default_rng(seed)
    x = np.array([-1.6, 0.0]) if x0 is None else np.asarray(x0, dtype=float).copy()
    logp = target.log_prob(x)

    samples = np.empty((n_steps, x.size))
    accepted = np.zeros(n_steps, dtype=bool)
    for i in range(n_steps):
        proposal = x + step_size * rng.standard_normal(x.size)
        logp_new = target.log_prob(proposal)
        if np.isfinite(logp_new) and np.log(rng.random()) < logp_new - logp:
            x, logp = proposal, logp_new
            accepted[i] = True
        samples[i] = x
    return Chain(samples, accepted, name="RWM", grad_evals=0, cost_per_step=1.0)


def mala(
    target: Target,
    n_steps: int = 200_000,
    step_size: float = 0.30,
    x0: np.ndarray | None = None,
    seed: int = 0,
) -> Chain:
    """Metropolis-adjusted Langevin algorithm.

    The proposal is ``x' = x + step^2/2 * grad log p(x) + step * N(0, I)``. The drift term is
    evaluated at the *current* point, so the proposal is not symmetric and the Gaussian
    proposal densities must be carried through the acceptance ratio.
    """
    rng = np.random.default_rng(seed)
    x = np.array([-1.6, 0.0]) if x0 is None else np.asarray(x0, dtype=float).copy()
    logp = target.log_prob(x)
    scale = step_size ** 2

    samples = np.empty((n_steps, x.size))
    accepted = np.zeros(n_steps, dtype=bool)
    n_grad = 0
    for i in range(n_steps):
        grad = target.grad_log_prob(x)
        n_grad += 1
        mean = x + 0.5 * scale * grad
        proposal = mean + step_size * rng.standard_normal(x.size)

        grad_new = target.grad_log_prob(proposal)
        n_grad += 1
        mean_back = proposal + 0.5 * scale * grad_new
        # log q(x | x') - log q(x' | x) for isotropic Gaussians of variance step^2
        log_q_back = -0.5 / scale * float(np.sum((x - mean_back) ** 2))
        log_q_forward = -0.5 / scale * float(np.sum((proposal - mean) ** 2))

        logp_new = target.log_prob(proposal)
        log_alpha = logp_new - logp + log_q_back - log_q_forward
        if np.isfinite(logp_new) and np.log(rng.random()) < log_alpha:
            x, logp = proposal, logp_new
            accepted[i] = True
        samples[i] = x
    return Chain(samples, accepted, name="MALA", grad_evals=n_grad, cost_per_step=3.0)


def hamiltonian_monte_carlo(
    target: Target,
    n_steps: int = 50_000,
    step_size: float = 0.20,
    n_leapfrog: int = 10,
    x0: np.ndarray | None = None,
    seed: int = 0,
) -> Chain:
    """HMC with identity mass matrix and a standard leapfrog integrator.

    One iteration simulates ``n_leapfrog`` steps of Hamiltonian dynamics (which travel far
    across the space) and then applies a single Metropolis correction; the energy error of the
    integrator is what the acceptance rate measures.
    """
    rng = np.random.default_rng(seed)
    x = np.array([-1.6, 0.0]) if x0 is None else np.asarray(x0, dtype=float).copy()
    logp = target.log_prob(x)

    samples = np.empty((n_steps, x.size))
    accepted = np.zeros(n_steps, dtype=bool)
    n_grad = 0
    for i in range(n_steps):
        p = rng.standard_normal(x.size)
        x_new, p_new = x.copy(), p.copy()

        p_new = p_new + 0.5 * step_size * target.grad_log_prob(x_new); n_grad += 1
        for _ in range(n_leapfrog):
            x_new = x_new + step_size * p_new
            with np.errstate(over="ignore", invalid="ignore"):
                p_new = p_new + step_size * target.grad_log_prob(x_new)
            n_grad += 1
        p_new = p_new - 0.5 * step_size * target.grad_log_prob(x_new); n_grad += 1

        logp_new = target.log_prob(x_new)
        h_old = -logp + 0.5 * float(np.dot(p, p))
        h_new = -logp_new + 0.5 * float(np.dot(p_new, p_new))
        if np.isfinite(h_new) and np.log(rng.random()) < h_old - h_new:
            x, logp = x_new, logp_new
            accepted[i] = True
        samples[i] = x
    return Chain(samples, accepted, name="HMC", grad_evals=n_grad,
                 cost_per_step=float(n_leapfrog + 2))
