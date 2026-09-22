"""Training and sampling for frames.

Corruption process (one shot, indexed by a noise level ``sigma_t``):

.. math::

    t_i^{\\text{noisy}} = \\sqrt{\\bar\\alpha_t}\\, t_i + \\sqrt{1-\\bar\\alpha_t}\\, \\epsilon_i
    \\qquad
    R_i^{\\text{noisy}} = R_i \\exp(\\sigma_t \\xi_i)

with ``sigma_t = sqrt(1 - alpha_bar_t)`` and ``xi_i ~ N(0, I_3)`` in the body frame. Translations
therefore follow a standard variance-preserving diffusion, and rotations receive a growing
isotropic tangent-space perturbation.

Honest caveat, stated here and in the module README: this is **not** a mathematically exact
SE(3) diffusion (which interpolates between frames on the group and has a proper forward
kernel). It is a simplified corruption that keeps the mechanics - tangent-space noise, an
equivariant denoiser, guidance at sampling time - visible and testable in a few hundred lines.
Replacing it with a geodesic interpolation is on the roadmap.

Sampling re-uses the DDIM trick: predict the clean state, then re-corrupt it to the next noise
level. For translations this is exactly DDIM; for rotations it is the same idea applied to the
body-frame tangent vector.
"""

from __future__ import annotations

import numpy as np
import torch

from model import LocalFrameDenoiser, to_body, to_world


class FrameDiffusion:
    def __init__(self, n_res: int, n_steps: int = 100, beta_start: float = 1e-4,
                 beta_end: float = 0.05, device: str = "cpu") -> None:
        self.n_res = n_res
        self.n_steps = n_steps
        self.device = torch.device(device)
        betas = torch.linspace(beta_start, beta_end, n_steps, device=self.device)
        self.alpha_bars = torch.cumprod(1.0 - betas, dim=0)

    # ------------------------------------------------------------------ helpers
    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return (1.0 - self.alpha_bars[t]).sqrt()

    def corrupt(self, R: torch.Tensor, t_xyz: torch.Tensor, step: torch.Tensor):
        """Add noise at level ``step``; returns noisy frames plus the targets to predict."""
        b, n, _ = t_xyz.shape
        ab = self.alpha_bars[step].view(b, 1, 1)
        eps = torch.randn_like(t_xyz)
        t_noisy = ab.sqrt() * t_xyz + (1.0 - ab).sqrt() * eps

        xi = torch.randn(b, n, 3, device=t_xyz.device)
        sig = self.sigma(step).view(b, 1, 1)
        R_noisy = R @ _exp_map(sig * xi)
        return R_noisy, t_noisy, eps, xi

    # ------------------------------------------------------------------ training
    def loss(self, model: LocalFrameDenoiser, feats_fn, R: torch.Tensor, t_xyz: torch.Tensor):
        b = t_xyz.shape[0]
        step = torch.randint(0, self.n_steps, (b,), device=self.device)
        R_noisy, t_noisy, eps, xi = self.corrupt(R, t_xyz, step)
        feats = feats_fn(R_noisy, t_noisy)
        v_hat, xi_hat = model(feats, step)
        target_v = to_body(eps, R_noisy)
        return (torch.mean((v_hat - target_v) ** 2) + torch.mean((xi_hat - xi) ** 2))

    # ------------------------------------------------------------------ sampling
    @torch.no_grad()
    def sample(self, model: LocalFrameDenoiser, feats_fn, n_samples: int,
               steps: int | None = None, guidance=None, guide_scale: float = 0.0,
               seed: int = 0):
        """Ancestral-free (DDIM-style) sampling, optionally with property guidance.

        ``guidance`` is a callable ``(R, t) -> loss`` evaluated on the clean estimate; its
        gradient with respect to the translations is used to steer each step.
        """
        torch.manual_seed(seed)
        steps = self.n_steps if steps is None else steps
        schedule = torch.linspace(self.n_steps - 1, 0, steps, device=self.device).long()

        # start from the stationary noise: N(0, I) translations, Haar-random rotations
        t_xyz = torch.randn(n_samples, self.n_res, 3, device=self.device)
        R = _random_rotations(n_samples, self.n_res, self.device)

        for i, s in enumerate(schedule):
            step = torch.full((n_samples,), int(s), device=self.device, dtype=torch.long)
            ab = self.alpha_bars[step].view(-1, 1, 1)
            sig = self.sigma(step).view(-1, 1, 1)

            feats = feats_fn(R, t_xyz)
            v_hat, xi_hat = model(feats, step)
            eps_hat = to_world(v_hat, R)

            t_clean = ((t_xyz - (1.0 - ab).sqrt() * eps_hat) / ab.sqrt()).clamp(-8.0, 8.0)
            R_clean = R @ _exp_map(-sig * xi_hat)

            if guidance is not None and guide_scale > 0.0:
                t_clean = guidance.step(t_clean, guide_scale)

            if i == len(schedule) - 1:
                t_xyz, R = t_clean, R_clean
                break
            ab_next = self.alpha_bars[schedule[i + 1]].view(-1, 1, 1)
            sig_next = self.sigma(schedule[i + 1]).view(-1, 1, 1)
            t_xyz = ab_next.sqrt() * t_clean + (1.0 - ab_next).sqrt() * eps_hat
            R = R_clean @ _exp_map(sig_next * xi_hat)
        return R, t_xyz


# --------------------------------------------------------------------------- SO(3) in torch
def _exp_map(w: torch.Tensor) -> torch.Tensor:
    """Batched Rodrigues formula; ``w`` has shape (..., 3), returns (..., 3, 3)."""
    theta = w.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    axis = w / theta
    K = _hat(axis)
    eye = torch.eye(3, device=w.device).expand(K.shape)
    return eye + torch.sin(theta)[..., None] * K + (1.0 - torch.cos(theta))[..., None] * (K @ K)


def _hat(w: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(w[..., 0])
    rows = [torch.stack([z, -w[..., 2], w[..., 1]], -1),
            torch.stack([w[..., 2], z, -w[..., 0]], -1),
            torch.stack([-w[..., 1], w[..., 0], z], -1)]
    return torch.stack(rows, -2)


def _random_rotations(n_samples: int, n_res: int, device) -> torch.Tensor:
    q = torch.randn(n_samples, n_res, 4, device=device)
    q = q / q.norm(dim=-1, keepdim=True)
    return _quat_to_matrix(q)


def _quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    rows = [
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ]
    return torch.stack(rows, -2)
