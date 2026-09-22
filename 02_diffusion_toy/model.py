"""A deliberately small denoising network: MLP + sinusoidal time embedding.

Nothing here is meant to be competitive. The point is that the denoiser is a function of
``(x_t, t)`` and nothing else - the generative behaviour lives in the diffusion process and in
how this function is queried at sampling time.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int = 64) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10_000.0)
                          * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class Denoiser(nn.Module):
    """Predicts the noise that was added to ``x_0`` to obtain ``x_t``."""

    def __init__(self, data_dim: int = 2, hidden: int = 128, time_dim: int = 64) -> None:
        super().__init__()
        self.time = SinusoidalTimeEmbedding(time_dim)
        self.net = nn.Sequential(
            nn.Linear(data_dim + time_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, data_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, self.time(t)], dim=-1))
