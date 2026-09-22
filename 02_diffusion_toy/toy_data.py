"""Two 2D densities to learn: two-moons and eight-gaussians.

They are chosen because they break different things. Two-moons is a curved, low-dimensional
manifold in a 2D space - a model that is too smooth smears the crescent into a blob.
Eight-gaussians has eight separated modes - a model can score well on average while dropping
modes entirely, which is exactly the failure that looks fine in a loss curve.
"""

from __future__ import annotations

import numpy as np


def two_moons(n: int, noise: float = 0.06, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = np.random.default_rng(0) if rng is None else rng
    half = n // 2
    theta = rng.uniform(0.0, np.pi, half)
    upper = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    lower = np.stack([1.0 - np.cos(theta), -np.sin(theta)], axis=1)
    x = np.concatenate([upper, lower], axis=0)[:n]
    return x + noise * rng.standard_normal(x.shape)


def eight_gaussians(n: int, radius: float = 2.0, sd: float = 0.22,
                    rng: np.random.Generator | None = None) -> np.ndarray:
    rng = np.random.default_rng(0) if rng is None else rng
    centres = np.stack([[radius * np.cos(k * np.pi / 4), radius * np.sin(k * np.pi / 4)]
                        for k in range(8)])
    idx = rng.integers(0, len(centres), n)
    return centres[idx] + sd * rng.standard_normal((n, 2))


DATASETS = {"two_moons": two_moons, "eight_gaussians": eight_gaussians}


def get(name: str):
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; choose from {sorted(DATASETS)}")
    return DATASETS[name]
