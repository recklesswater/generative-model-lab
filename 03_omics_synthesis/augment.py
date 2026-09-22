"""Three ways to synthesise patients, from dumbest to slightly less dumb.

All of them are deliberately simple, and all of them are **class-conditional**: samples are
generated within a class, never across classes, because a generator that mixes classes is simply
label noise with extra steps.

* ``none`` - the honest baseline (no augmentation at all).
* ``smote`` - interpolate between a minority-class sample and one of its minority-class
  neighbours. Cheap, and it stays inside the cloud of the training batch, which is exactly why it
  can amplify a batch effect.
* ``gaussian_copula`` - fit a multivariate Gaussian per class and sample from it. Slightly more
  global than SMOTE, but still a single mode per class.
"""

from __future__ import annotations

import numpy as np


def _class_counts(y: np.ndarray) -> dict[int, int]:
    return {int(c): int((y == c).sum()) for c in np.unique(y)}


def smote(x: np.ndarray, y: np.ndarray, target_per_class: int, k: int = 5,
          rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0) if rng is None else rng
    new_x, new_y = [x], [y]
    for cls, count in _class_counts(y).items():
        need = target_per_class - count
        if need <= 0:
            continue
        pool = x[y == cls]
        k_eff = min(k, len(pool) - 1)
        distances = np.linalg.norm(pool[:, None] - pool[None, :], axis=-1)
        np.fill_diagonal(distances, np.inf)
        neighbours = np.argsort(distances, axis=1)[:, :k_eff]
        picks = rng.integers(0, len(pool), need)
        partner = neighbours[picks, rng.integers(0, k_eff, need)]
        weight = rng.random((need, 1))
        synthetic = pool[picks] + weight * (pool[partner] - pool[picks])
        new_x.append(synthetic)
        new_y.append(np.full(need, cls))
    return np.vstack(new_x), np.concatenate(new_y)


def gaussian_copula(x: np.ndarray, y: np.ndarray, target_per_class: int,
                    rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0) if rng is None else rng
    new_x, new_y = [x], [y]
    for cls, count in _class_counts(y).items():
        need = target_per_class - count
        if need <= 0:
            continue
        pool = x[y == cls]
        mean = pool.mean(axis=0)
        cov = np.cov(pool, rowvar=False)
        cov += 1e-6 * np.eye(cov.shape[0])
        synthetic = rng.multivariate_normal(mean, cov, size=need)
        new_x.append(synthetic)
        new_y.append(np.full(need, cls))
    return np.vstack(new_x), np.concatenate(new_y)


METHODS = {"none": None, "smote": smote, "gaussian_copula": gaussian_copula}


def augment(method: str, x: np.ndarray, y: np.ndarray, ratio: float,
            rng: np.random.Generator | None = None):
    """Apply ``method`` so that both classes reach ``ratio`` x the original majority size."""
    if method == "none":
        return x, y
    if method not in METHODS:
        raise KeyError(f"unknown method {method!r}; choose from {sorted(METHODS)}")
    target = int(round(max(_class_counts(y).values()) * ratio))
    return METHODS[method](x, y, target, rng=rng)
