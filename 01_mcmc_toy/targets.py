"""Target distributions used by the MCMC demo.

Everything is written as an unnormalised log-density plus its gradient, because that is
exactly what the samplers are allowed to know. The normalising constant is never used.
"""

from __future__ import annotations

import numpy as np


class Target:
    """Minimal interface: ``log_prob(x)`` and ``grad_log_prob(x)`` for ``x`` of shape (2,)."""

    name = "target"

    def log_prob(self, x: np.ndarray) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def grad_log_prob(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class StandardNormal2D(Target):
    """N(0, I) in 2D - the sanity check that every sampler works at all."""

    name = "standard normal"

    def log_prob(self, x: np.ndarray) -> float:
        return -0.5 * float(np.dot(x, x))

    def grad_log_prob(self, x: np.ndarray) -> np.ndarray:
        return -x


class DoubleWell(Target):
    """A symmetric double well: two modes separated by a barrier.

    .. math::

        U(x_0, x_1) = h (x_0^2 - a^2)^2 + \\tfrac{1}{2} \\omega^2 x_1^2

    The barrier height is ``h * a**4`` and the two minima sit at ``x_0 = +-a``, ``x_1 = 0``.
    With the default parameters the barrier is high enough that a badly tuned sampler will
    spend thousands of steps in one well - which is the whole point of the demo.
    """

    name = "double well"

    def __init__(self, a: float = 1.5, h: float = 0.5, omega: float = 2.0) -> None:
        self.a = float(a)
        self.h = float(h)
        self.omega = float(omega)
        self.barrier = self.h * self.a ** 4

    def potential(self, x: np.ndarray) -> float:
        with np.errstate(over="ignore", invalid="ignore"):
            return float(self.h * (x[0] ** 2 - self.a ** 2) ** 2
                         + 0.5 * self.omega ** 2 * x[1] ** 2)

    def log_prob(self, x: np.ndarray) -> float:
        return -self.potential(x)

    def grad_log_prob(self, x: np.ndarray) -> np.ndarray:
        with np.errstate(over="ignore", invalid="ignore"):
            d0 = -4.0 * self.h * x[0] * (x[0] ** 2 - self.a ** 2)
            d1 = -(self.omega ** 2) * x[1]
        return np.array([d0, d1])

    def mode_of(self, x: np.ndarray) -> int:
        """Which well a point belongs to (0 = left, 1 = right)."""
        return int(x[0] > 0.0)
