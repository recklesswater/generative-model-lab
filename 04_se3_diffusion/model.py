"""An equivariant-by-construction denoiser for rigid frames.

Design (the important part, not the architecture):

* **Input**: for each residue, the *invariant* features built by
  :func:`frames.invariant_features` - the relative translations expressed in the residue's own
  frame, and the relative rotations. A global rotation leaves all of them unchanged.
* **Output**: for each residue, the translation noise expressed **in that residue's local frame**
  and the rotation noise as a **body-frame tangent vector**. A global rotation leaves both
  unchanged as well.

Because the network only ever sees invariants and only ever outputs body-frame quantities, the
mapping to the world frame at the end (``eps = R_i @ v_i``) makes the whole model exactly
equivariant: rotate the input, and the output rotates with it. ``equivariance_test.py`` asserts
this numerically; the error should be at float precision, not "approximately small".

This is the same principle as an ``e3nn`` tensor-product layer, with the Clebsch-Gordan
coefficients replaced by a hard-coded local basis.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int = 64) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10_000.0)
                          * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class LocalFrameDenoiser(nn.Module):
    """Shared per-residue MLP over invariant features.

    The same weights are applied to every residue (so the model is permutation-equivariant over
    residues as well) and the time step enters through a sinusoidal embedding.
    """

    def __init__(self, n_res: int, hidden: int = 256, time_dim: int = 64) -> None:
        super().__init__()
        self.n_res = n_res
        self.time = TimeEmbedding(time_dim)
        feat_dim = n_res * 12
        self.net = nn.Sequential(
            nn.Linear(feat_dim + time_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 6),
        )

    def forward(self, feats: torch.Tensor, t: torch.Tensor):
        """``feats``: ``(B, n_res, n_res * 12)``; ``t``: ``(B,)``.

        Returns ``(v_hat, xi_hat)``, both ``(B, n_res, 3)``, in the local / body frame.
        """
        b, n, _ = feats.shape
        h = feats.reshape(b * n, -1)
        t_emb = self.time(t)[:, None, :].expand(b, n, -1).reshape(b * n, -1)
        out = self.net(torch.cat([h, t_emb], dim=-1)).reshape(b, n, 6)
        return out[..., :3], out[..., 3:]


def to_world(v_local: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Map body-frame vectors to the world frame: ``eps_i = R_i @ v_i``."""
    return torch.einsum("bnij,bnj->bni", R, v_local)


def to_body(v_world: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`to_world`: ``v_i = R_i^T eps_i``."""
    return torch.einsum("bnji,bnj->bni", R, v_world)
