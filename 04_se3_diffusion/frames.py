"""Rigid frames, global transforms, and the invariant features the denoiser is allowed to see.

A frame is ``(R, t)`` with ``R in SO(3)``. A **global** rotation ``Q`` acts as
``(R, t) -> (Q R, Q t)``: the physics of a protein does not change when the whole molecule is
rotated in space, so the model must be equivariant to exactly this action.

The construction used here is the cheapest way to get exact equivariance: build features that
are *invariant* under a global rotation by expressing everything in a **local frame**, then
express predictions in that same local frame and rotate back at the end. Tensor-product
networks (e3nn and friends) do the same thing with learned Clebsch-Gordan coefficients instead
of a hard-coded local basis.
"""

from __future__ import annotations

import numpy as np

from so3 import exp_map, geodesic_noise, random_rotation


# --------------------------------------------------------------------------- data
def helix_frames(n_res: int = 12, radius: float = 1.8, pitch: float = 0.75,
                 jitter: float = 0.12, rng: np.random.Generator | None = None):
    """A toy backbone: residues on a helix, each frame oriented along the local tangent."""
    rng = np.random.default_rng(0) if rng is None else rng
    theta = np.arange(n_res) * 2.0 * np.pi / 6.0
    t = np.stack([radius * np.cos(theta), radius * np.sin(theta), pitch * theta], axis=1)
    t = t + jitter * rng.standard_normal(t.shape)

    R = np.empty((n_res, 3, 3))
    for i in range(n_res):
        nxt = t[(i + 1) % n_res] - t[i - 1]
        z = nxt / max(np.linalg.norm(nxt), 1e-8)
        helper = np.array([0.0, 1.0, 0.0]) if abs(z[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        x = np.cross(helper, z)
        x /= max(np.linalg.norm(x), 1e-8)
        y = np.cross(z, x)
        R[i] = np.stack([x, y, z], axis=1)
    return R, t


def make_dataset(n_samples: int = 512, n_res: int = 12, rng: np.random.Generator | None = None):
    """A batch of noisy helices, randomly rotated (so the model cannot memorise an orientation)."""
    rng = np.random.default_rng(0) if rng is None else rng
    R = np.empty((n_samples, n_res, 3, 3))
    t = np.empty((n_samples, n_res, 3))
    for s in range(n_samples):
        Rs, ts = helix_frames(n_res, jitter=0.18, rng=rng)
        Q = random_rotation(rng)
        R[s], t[s] = Q[None] @ Rs, ts @ Q.T
    return R, t


# --------------------------------------------------------------------------- group action
def apply_global(R: np.ndarray, t: np.ndarray, Q: np.ndarray, shift: np.ndarray | None = None):
    """Act with a global rotation (and optional translation) on a batch of frames."""
    R2 = Q[None, None] @ R
    t2 = t @ Q.T
    if shift is not None:
        t2 = t2 + shift[None, None]
    return R2, t2


# --------------------------------------------------------------------------- features
def invariant_features(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Per-residue features that are unchanged by any global rotation.

    For residue ``i`` we express every other residue's translation in the local frame of ``i``,
    and encode every relative rotation ``R_i^T R_j`` by its 9 entries. Under ``(R, t) -> (QR, Qt)``
    both blocks are invariant, so any function of them is invariant as well.

    Returns an array of shape ``(n_res, n_res * 12)``.
    """
    n = R.shape[0]
    feats = np.empty((n, n * 12))
    for i in range(n):
        Ri = R[i]
        rel_t = (t - t[i]) @ Ri                     # (n, 3): R_i^T (t_j - t_i)
        rel_R = np.einsum("ji,njk->nik", Ri, R)     # (n, 3, 3): R_i^T R_j
        feats[i] = np.concatenate([rel_t.reshape(-1), rel_R.reshape(-1)])
    return feats


def local_from_global(v: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Express a world-frame vector at residue ``i`` in the local frame: ``R_i^T v_i``."""
    return np.einsum("nij,nj->ni", R.transpose(0, 2, 1), v)


# --------------------------------------------------------------------------- torch version
def invariant_features_torch(R, t):
    """Differentiable version of :func:`invariant_features`, batched over samples.

    ``R``: ``(B, N, 3, 3)``; ``t``: ``(B, N, 3)`` -> ``(B, N, N * 12)``.
    """
    import torch  # imported lazily so the numpy utilities stay importable without torch

    delta = t[:, None, :, :] - t[:, :, None, :]                  # (B, N_i, N_j, 3)
    rel_t = torch.einsum("bnkl,bnmk->bnml", R, delta)            # R_i^T (t_j - t_i)
    rel_R = R.transpose(-1, -2)[:, :, None] @ R[:, None]         # R_i^T R_j
    return torch.cat([rel_t.reshape(R.shape[0], R.shape[1], -1),
                      rel_R.reshape(R.shape[0], R.shape[1], -1)], dim=-1)
