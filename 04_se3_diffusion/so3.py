"""SO(3) and SE(3) utilities: the maps that make group-aware noise possible.

The only mathematical content here is that ``SO(3)`` is a curved manifold, so "adding noise"
means moving along the tangent space at the current point and then mapping back with the
exponential map. Adding independent Gaussians to the nine entries of a rotation matrix leaves
the manifold immediately; this is exactly the mistake that equivariance tests catch.
"""

from __future__ import annotations

import numpy as np


def hat(w: np.ndarray) -> np.ndarray:
    """``R^3 -> so(3)``: skew-symmetric matrix of an angular velocity vector."""
    w = np.asarray(w, dtype=float)
    return np.array([[0.0, -w[2], w[1]],
                     [w[2], 0.0, -w[0]],
                     [-w[1], w[0], 0.0]])


def vee(W: np.ndarray) -> np.ndarray:
    """``so(3) -> R^3``: inverse of :func:`hat`."""
    W = np.asarray(W, dtype=float)
    return np.array([W[2, 1], W[0, 2], W[1, 0]])


def exp_map(w: np.ndarray) -> np.ndarray:
    """Rodrigues' formula: rotation matrix for the axis-angle vector ``w``."""
    w = np.asarray(w, dtype=float)
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.eye(3)
    axis = w / theta
    K = hat(axis)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def log_map(R: np.ndarray) -> np.ndarray:
    """Inverse of :func:`exp_map`; returns the axis-angle vector with ``|w| <= pi``."""
    R = np.asarray(R, dtype=float)
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    if theta < 1e-8:
        return np.zeros(3)
    if abs(np.pi - theta) < 1e-6:  # near pi: use the symmetric part
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(A), 0.0, None))
        if R[2, 1] < 0:
            axis[1] = -axis[1]
        if R[0, 2] < 0:
            axis[2] = -axis[2]
        if R[1, 0] < 0:
            axis[0] = -axis[0]
        axis = axis / max(np.linalg.norm(axis), 1e-12)
        return theta * axis
    return theta / (2.0 * np.sin(theta)) * vee(R - R.T)


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniform (Haar) random rotation, via quaternion normalisation."""
    q = rng.standard_normal(4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def project_to_so3(M: np.ndarray) -> np.ndarray:
    """Nearest rotation matrix (orthogonal Procrustes / SVD projection)."""
    U, _, Vt = np.linalg.svd(np.asarray(M, dtype=float))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def geodesic_angle(R1: np.ndarray, R2: np.ndarray) -> float:
    """Rotation angle between two orientations, in radians."""
    return float(np.linalg.norm(log_map(np.asarray(R1).T @ np.asarray(R2))))


def geodesic_noise(R: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Rotate ``R`` by a random tangent vector of standard deviation ``sigma`` (body frame)."""
    return R @ exp_map(sigma * rng.standard_normal(3))
