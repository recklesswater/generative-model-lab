"""A minimal DDPM with a DDIM sampler.

The forward process is fixed (a variance schedule), the network only has to learn to predict the
noise. Sampling then walks the same schedule backwards, and the DDIM formulation lets the same
trained network be queried on a shorter, deterministic trajectory.
"""

from __future__ import annotations

import torch


class Diffusion:
    def __init__(self, n_steps: int = 400, beta_start: float = 1e-4,
                 beta_end: float = 0.02, device: str = "cpu") -> None:
        self.n_steps = n_steps
        self.device = torch.device(device)
        betas = torch.linspace(beta_start, beta_end, n_steps, device=self.device)
        alphas = 1.0 - betas
        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = torch.cumprod(alphas, dim=0)

    # ---- forward process ----------------------------------------------------
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor | None = None) -> torch.Tensor:
        """``x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) eps``."""
        if noise is None:
            noise = torch.randn_like(x0)
        ab = self.alpha_bars[t].unsqueeze(-1)
        return ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise

    def loss(self, model, x0: torch.Tensor) -> torch.Tensor:
        batch = x0.shape[0]
        t = torch.randint(0, self.n_steps, (batch,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        return torch.mean((model(x_t, t) - noise) ** 2)

    # ---- sampling -----------------------------------------------------------
    @torch.no_grad()
    def sample_ddpm(self, model, n: int, data_dim: int = 2) -> torch.Tensor:
        x = torch.randn(n, data_dim, device=self.device)
        for step in reversed(range(self.n_steps)):
            t = torch.full((n,), step, device=self.device, dtype=torch.long)
            eps = model(x, t)
            mean = (x - self.betas[step] / (1.0 - self.alpha_bars[step]).sqrt() * eps) \
                / self.alphas[step].sqrt()
            x = mean if step == 0 else mean + self.betas[step].sqrt() * torch.randn_like(x)
        return x

    @torch.no_grad()
    def sample_ddim(self, model, n: int, data_dim: int = 2, steps: int = 20,
                    eta: float = 0.0) -> torch.Tensor:
        """Deterministic (eta = 0) accelerated sampling on a strided sub-schedule."""
        times = torch.linspace(self.n_steps - 1, 0, steps, device=self.device).long()
        x = torch.randn(n, data_dim, device=self.device)
        for i, step in enumerate(times):
            t = torch.full((n,), int(step), device=self.device, dtype=torch.long)
            eps = model(x, t)
            ab = self.alpha_bars[step]
            x0_hat = ((x - (1.0 - ab).sqrt() * eps) / ab.sqrt()).clamp(-6.0, 6.0)
            if i == len(times) - 1:
                x = x0_hat
                break
            ab_next = self.alpha_bars[times[i + 1]]
            sigma = (eta * ((1 - ab_next) / (1 - ab)).sqrt()
                     * (1 - ab / ab_next).sqrt())
            x = ab_next.sqrt() * x0_hat + (1 - ab_next - sigma ** 2).clamp(min=0).sqrt() * eps
            if eta > 0:
                x = x + sigma * torch.randn_like(x)
        return x
