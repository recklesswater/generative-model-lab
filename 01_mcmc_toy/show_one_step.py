"""One step of each sampler, printed in full - the hand-checkable companion to ``samplers.py``.

Every number the acceptance ratio is built from ends up on screen: the two log-densities, the
proposal density in both directions, the acceptance probability, the uniform draw that decides,
and the resulting accept/reject. Nothing is hidden behind a library call, so the output can be
reproduced with a calculator.

Usage:
    python 01_mcmc_toy/show_one_step.py
    python 01_mcmc_toy/show_one_step.py --step-rwm 0.5 --step-mala 0.5 --step-hmc 0.2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from targets import DoubleWell  # noqa: E402

# A fixed starting point inside the left well, deliberately off the x1 = 0 axis so that the
# gradient has two non-zero components and the printed numbers are not accidentally tidy.
X_START = np.array([-1.20, 0.35])
RNG_SEED = 20260923


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print one full step of each sampler.")
    parser.add_argument("--step-rwm", type=float, default=0.5, help="RWM step size (default 0.5)")
    parser.add_argument("--step-mala", type=float, default=0.5, help="MALA step size (default 0.5)")
    parser.add_argument("--step-hmc", type=float, default=0.2, help="HMC step size (default 0.2)")
    return parser.parse_args()


def rule(title: str) -> None:
    print("\n" + title)
    print("-" * len(title))


def gradient_check(target: DoubleWell, x: np.ndarray, h: float = 1e-6) -> float:
    """Largest absolute difference between the analytic and the central-difference gradient."""
    analytic = target.grad_log_prob(x)
    numeric = np.empty_like(x)
    for i in range(x.size):
        e = np.zeros_like(x)
        e[i] = h
        numeric[i] = (target.log_prob(x + e) - target.log_prob(x - e)) / (2.0 * h)
    return float(np.max(np.abs(analytic - numeric)))


def show_target(target: DoubleWell, x: np.ndarray) -> None:
    rule(f"Target: {target.name}")
    print(f"  parameters          a = {target.a}, h = {target.h}, omega = {target.omega}")
    print(f"  barrier height      h * a^4 = {target.barrier:.4f} energy units")
    print(f"  minima              x0 = +-{target.a}, x1 = 0")
    print(f"  start point         x  = [{x[0]:.4f}, {x[1]:.4f}]  (left well)")
    print(f"  U(x)                = {target.potential(x):.6f}")
    print(f"  log pi(x) = -U(x)   = {target.log_prob(x):.6f}")
    print(f"  grad log pi(x)      = [{target.grad_log_prob(x)[0]:.6f}, "
          f"{target.grad_log_prob(x)[1]:.6f}]")
    print(f"  normalising constant Z is never computed - every acceptance ratio below")
    print(f"  is a ratio of two un-normalised densities, so Z cancels.")


def show_rwm(target: DoubleWell, x: np.ndarray, step: float, rng: np.random.Generator) -> None:
    rule("RWM: symmetric random-walk proposal")
    noise = rng.standard_normal(x.size)
    proposal = x + step * noise
    logp, logp_new = target.log_prob(x), target.log_prob(proposal)
    delta_u = target.potential(proposal) - target.potential(x)
    log_alpha = logp_new - logp
    alpha = min(1.0, float(np.exp(log_alpha)))
    uniform = float(rng.random())
    accepted = np.log(uniform) < log_alpha

    print(f"  step size           {step}")
    print(f"  noise xi            [{noise[0]:+.6f}, {noise[1]:+.6f}]")
    print(f"  proposal x'         [{proposal[0]:.6f}, {proposal[1]:.6f}]")
    print(f"  U(x')               {target.potential(proposal):.6f}   (U(x) = "
          f"{target.potential(x):.6f}, delta U = {delta_u:+.6f})")
    print(f"  log pi(x')          {logp_new:.6f}")
    print(f"  log q(x'|x)         = log q(x|x')  ->  cancels (symmetric proposal)")
    print(f"  log alpha           = log pi(x') - log pi(x) = {log_alpha:+.6f}")
    print(f"  alpha               = {alpha:.6f}")
    print(f"  u ~ U(0,1)          = {uniform:.6f}   (accept iff u < alpha)  ->  "
          f"{'ACCEPT' if accepted else 'REJECT'}")


def show_mala(target: DoubleWell, x: np.ndarray, step: float, rng: np.random.Generator) -> None:
    rule("MALA: gradient drift + noise, Metropolis corrected")
    scale = step ** 2
    grad = target.grad_log_prob(x)
    mean = x + 0.5 * scale * grad
    noise = rng.standard_normal(x.size)
    proposal = mean + step * noise

    grad_new = target.grad_log_prob(proposal)
    mean_back = proposal + 0.5 * scale * grad_new
    log_q_forward = -0.5 / scale * float(np.sum((proposal - mean) ** 2))
    log_q_back = -0.5 / scale * float(np.sum((x - mean_back) ** 2))

    logp, logp_new = target.log_prob(x), target.log_prob(proposal)
    log_alpha = logp_new - logp + log_q_back - log_q_forward
    alpha = min(1.0, float(np.exp(log_alpha)))
    uniform = float(rng.random())
    accepted = np.log(uniform) < log_alpha

    print(f"  step size           {step}   (proposal variance step^2 = {scale:.6f})")
    print(f"  grad log pi(x)      [{grad[0]:+.6f}, {grad[1]:+.6f}]")
    print(f"  drift 0.5 s^2 grad  [{0.5 * scale * grad[0]:+.6f}, {0.5 * scale * grad[1]:+.6f}]")
    print(f"  proposal mean       [{mean[0]:.6f}, {mean[1]:.6f}]   <- x + drift")
    print(f"  noise xi            [{noise[0]:+.6f}, {noise[1]:+.6f}]")
    print(f"  proposal x'         [{proposal[0]:.6f}, {proposal[1]:.6f}]")
    print(f"  log q(x'|x)         {log_q_forward:.6f}")
    print(f"  log q(x|x')         {log_q_back:.6f}   <- not equal: the proposal is asymmetric")
    print(f"  log pi(x') - log pi(x)   {logp_new - logp:+.6f}")
    print(f"  log alpha           = {logp_new - logp:+.6f} + "
          f"({log_q_back:.6f}) - ({log_q_forward:.6f}) = {log_alpha:+.6f}")
    print(f"  alpha               = {alpha:.6f}")
    print(f"  u ~ U(0,1)          = {uniform:.6f}   (accept iff u < alpha)  ->  "
          f"{'ACCEPT' if accepted else 'REJECT'}")


def show_hmc(target: DoubleWell, x: np.ndarray, step: float, rng: np.random.Generator) -> None:
    rule("HMC: one leapfrog step (L = 1) and one Metropolis correction")
    p = rng.standard_normal(x.size)
    x_new, p_new = x.copy(), p.copy()

    p_new = p_new + 0.5 * step * target.grad_log_prob(x_new)
    x_new = x_new + step * p_new
    p_new = p_new + step * target.grad_log_prob(x_new)
    p_new = p_new - 0.5 * step * target.grad_log_prob(x_new)

    u_old, u_new = target.potential(x), target.potential(x_new)
    h_old = u_old + 0.5 * float(np.dot(p, p))
    h_new = u_new + 0.5 * float(np.dot(p_new, p_new))
    delta_h = h_new - h_old
    log_alpha = -delta_h
    alpha = min(1.0, float(np.exp(log_alpha)))
    uniform = float(rng.random())
    accepted = np.log(uniform) < log_alpha

    print(f"  step size           {step}")
    print(f"  momentum p ~ N(0,I) [{p[0]:+.6f}, {p[1]:+.6f}]")
    print(f"  position x          [{x[0]:.6f}, {x[1]:.6f}]")
    print(f"  position x'         [{x_new[0]:.6f}, {x_new[1]:.6f}]"
          f"   (moved |x'-x| = {float(np.linalg.norm(x_new - x)):.6f})")
    print(f"  momentum p'         [{p_new[0]:+.6f}, {p_new[1]:+.6f}]"
          f"   (|p| changed by {float(np.linalg.norm(p_new) - np.linalg.norm(p)):+.6f})")
    print(f"  U(x)                {u_old:.6f}")
    print(f"  U(x')               {u_new:.6f}")
    print(f"  H(x, p)             {h_old:.6f}   = U + 0.5|p|^2")
    print(f"  H(x', p')           {h_new:.6f}")
    print(f"  delta H             {delta_h:+.6f}   <- integrator error, the *only* reason to reject")
    print(f"  log alpha = -delta H {log_alpha:+.6f}")
    print(f"  alpha               = {alpha:.6f}")
    print(f"  u ~ U(0,1)          = {uniform:.6f}   (accept iff u < alpha)  ->  "
          f"{'ACCEPT' if accepted else 'REJECT'}")


def show_self_checks(target: DoubleWell) -> list[str]:
    """Print three invariants and return the names of any that fail.

    These are exact properties of the target and of the analytic gradient, so they double as a
    smoke test: if the gradient had a typo, MALA and HMC would satisfy the wrong detailed-balance
    relation and *still look fine*, and only a check like this would catch it.
    """
    failures: list[str] = []
    rule("Self-checks you can redo by hand")

    minimum = np.array([-target.a, 0.0])
    grad_at_minimum = target.grad_log_prob(minimum)
    if float(np.max(np.abs(grad_at_minimum))) > 1e-12:
        failures.append("gradient at the minimum is not zero")
    print(f"  1. gradient at the minimum x = [{minimum[0]:.4f}, 0]: "
          f"grad log pi = [{grad_at_minimum[0]:+.3e}, {grad_at_minimum[1]:+.3e}]")
    print(f"     -> the MALA drift vanishes exactly at the bottom of a well, so there MALA")
    print(f"        degenerates to RWM at the same step size. Gradient methods buy you")
    print(f"        geometry in the tails, not at the mode.")

    d = 0.4
    left = np.array([-target.a + d, 0.25])
    right = np.array([target.a - d, 0.25])
    if abs(target.potential(left) - target.potential(right)) > 1e-12:
        failures.append("the two wells are not symmetric")
    print(f"\n  2. symmetry: U([{left[0]:.2f}, {left[1]:.2f}]) = {target.potential(left):.6f}, "
          f"U([{right[0]:.2f}, {right[1]:.2f}]) = {target.potential(right):.6f}")
    print(f"     -> the two wells hold *exactly* half the probability each, whatever the")
    print(f"        barrier height is. 'Which well is the chain in' is therefore a fair")
    print(f"        coin flip that any sampler must reproduce given enough crossings.")

    err = gradient_check(target, X_START)
    if err > 1e-6:
        failures.append(f"analytic gradient disagrees with central differences ({err:.2e})")
    print(f"\n  3. analytic gradient vs central differences at the start point:")
    print(f"     max |analytic - numeric| = {err:.3e}")
    print(f"     -> if the analytic gradient were wrong, MALA and HMC would silently")
    print(f"        satisfy the wrong detailed-balance relation, and only this check")
    print(f"        (or a toy target with a known answer) would catch it.")

    print("\n  What the three blocks above should teach:")
    print("    * RWM's acceptance depends on delta U alone: the proposal density cancels")
    print("      because q is symmetric.")
    print("    * MALA adds two explicit log-proposal terms; they are what keeps the method")
    print("      exact even though the proposal is biased uphill.")
    print("    * HMC's acceptance is exp(-delta H): the integrator's energy error is the")
    print("      only thing that costs you acceptance, and step size controls it directly.")
    return failures


def main() -> None:
    args = parse_args()
    target = DoubleWell()
    x = X_START.copy()
    rng = np.random.default_rng(RNG_SEED)

    rule("One step of each sampler on the same target and the same start point")
    print(f"  rng seed            {RNG_SEED}   (fixed, so this output is reproducible)")
    show_target(target, x)
    show_rwm(target, x, args.step_rwm, rng)
    show_mala(target, x, args.step_mala, rng)
    show_hmc(target, x, args.step_hmc, rng)
    failures = show_self_checks(target)

    rule("Self-check result")
    if failures:
        for failure in failures:
            print(f"  FAILED: {failure}")
        raise SystemExit(1)
    print("  all three invariants hold (gradient at the minimum, well symmetry, gradient check)")


if __name__ == "__main__":
    main()
