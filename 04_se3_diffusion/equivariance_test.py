"""The test that matters: rotate the input, and the prediction must rotate with it.

Two independent checks are run:

1. **Group utilities.** ``exp``/``log`` are inverses on SO(3), and corrupted frames stay valid
   rotation matrices (``R^T R = I``, ``det R = 1``). This is what fails if you add Gaussian
   noise to the nine entries of a rotation matrix instead of using the tangent space.
2. **The denoiser.** For a random global rotation ``Q``:

   .. math::

       \\hat\\epsilon(Q \\cdot T) = Q\\,\\hat\\epsilon(T), \\qquad
       \\hat\\xi(Q \\cdot T) = \\hat\\xi(T)

   i.e. the translation noise rotates with the molecule, and the rotation noise (a body-frame
   quantity) does not change at all. The check uses **untrained** weights on purpose: this is a
   property of the architecture, so it must hold before any training happens.

Run:  python 04_se3_diffusion/equivariance_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from frames import invariant_features, invariant_features_torch, make_dataset  # noqa: E402
from model import LocalFrameDenoiser, to_world  # noqa: E402
from so3 import exp_map, geodesic_noise, log_map, random_rotation  # noqa: E402


def check_group_utilities(rng: np.random.Generator) -> list[tuple[str, float, float]]:
    results = []

    # log(exp(w)) == w only for |w| < pi (the logarithm returns the principal value)
    worst = 0.0
    for _ in range(200):
        w = rng.standard_normal(3)
        w = w / max(np.linalg.norm(w), 1e-12) * rng.uniform(0.0, np.pi * 0.98)
        worst = max(worst, float(np.abs(log_map(exp_map(w)) - w).max()))
    results.append(("log(exp(w)) == w for |w| < pi", worst, 1e-8))

    # and the group-level round trip, which holds for any w
    worst = 0.0
    for _ in range(200):
        R = exp_map(rng.standard_normal(3) * 3.0)
        worst = max(worst, float(np.abs(exp_map(log_map(R)) - R).max()))
    results.append(("exp(log(exp(w))) == exp(w)", worst, 1e-8))

    worst_orth = worst_det = 0.0
    for _ in range(200):
        R = geodesic_noise(np.eye(3), sigma=0.7, rng=rng)
        worst_orth = max(worst_orth, float(np.abs(R.T @ R - np.eye(3)).max()))
        worst_det = max(worst_det, float(abs(np.linalg.det(R) - 1.0)))
    results.append(("R^T R == I after tangent-space noise", worst_orth, 1e-12))
    results.append(("det R == 1 after tangent-space noise", worst_det, 1e-12))

    worst = 0.0
    for _ in range(200):
        R1, R2 = random_rotation(rng), random_rotation(rng)
        R = R1.T @ R2
        worst = max(worst, float(np.abs(exp_map(log_map(R)) - R).max()))
    results.append(("exp(log(R)) == R", worst, 1e-8))
    return results


def check_model_equivariance(seed: int = 0, n_res: int = 12, n_samples: int = 8):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    R_np, t_np = make_dataset(n_samples, n_res, rng)
    R = torch.tensor(R_np, dtype=torch.float32)
    t = torch.tensor(t_np, dtype=torch.float32)

    model = LocalFrameDenoiser(n_res=n_res)          # deliberately untrained
    step = torch.full((n_samples,), 37, dtype=torch.long)

    feats = invariant_features_torch(R, t)
    v_hat, xi_hat = model(feats, step)
    eps_hat = to_world(v_hat, R)

    # numpy cross-check of the invariant features themselves
    feats_np = invariant_features(R_np[0], t_np[0])
    feats_torch = feats[0].detach().numpy()
    feature_gap = float(np.abs(feats_np - feats_torch).max())

    Q_np = random_rotation(rng)
    Q = torch.tensor(Q_np, dtype=torch.float32)
    R_q = Q @ R
    t_q = t @ Q.T
    v_q, xi_q = model(invariant_features_torch(R_q, t_q), step)
    eps_q = to_world(v_q, R_q)

    eps_expected = eps_hat @ Q.T          # world-frame vectors rotate with the molecule
    err_translation = float((eps_q - eps_expected).abs().max().detach())
    err_rotation = float((xi_q - xi_hat).abs().max().detach())
    return feature_gap, err_translation, err_rotation


def main() -> None:
    rng = np.random.default_rng(0)
    print("group utilities")
    ok = True
    for name, value, tol in check_group_utilities(rng):
        flag = "ok" if value <= tol else "FAIL"
        ok &= value <= tol
        print(f"  {name:<44} max error {value:.3e}  [{flag}]")

    gap, err_t, err_r = check_model_equivariance()
    print("denoiser equivariance (untrained weights)")
    print(f"  {'invariant features: numpy vs torch':<44} max error {gap:.3e}")
    print(f"  {'eps_hat(Q.T) == Q eps_hat(T)':<44} max error {err_t:.3e}")
    print(f"  {'xi_hat(Q.T) == xi_hat(T)':<44} max error {err_r:.3e}")
    ok &= gap < 1e-5 and err_t < 1e-5 and err_r < 1e-5

    if not ok:
        raise SystemExit("equivariance check failed")
    print("\nequivariance holds to float32 precision - it is architectural, not learned")


if __name__ == "__main__":
    main()
