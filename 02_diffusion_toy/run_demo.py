"""Train the 2D diffusion model on a toy density and write the figures.

Usage:
    python 02_diffusion_toy/run_demo.py                     # two-moons, 3000 steps
    python 02_diffusion_toy/run_demo.py --dataset eight_gaussians --steps 6000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ddpm import Diffusion  # noqa: E402
from model import Denoiser  # noqa: E402
from toy_data import get as get_dataset  # noqa: E402

FIGDIR = Path(__file__).resolve().parent.parent / "figures"


def train(dataset, n_steps: int, batch: int, lr: float, seed: int, device: str):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    diffusion = Diffusion(n_steps=400, device=device)
    model = Denoiser().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    pool = torch.tensor(dataset(20_000, rng=rng), dtype=torch.float32, device=device)
    losses = []
    for step in range(1, n_steps + 1):
        idx = torch.randint(0, pool.shape[0], (batch,), device=device)
        loss = diffusion.loss(model, pool[idx])
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        losses.append(loss.detach().item())
        if step % max(1, n_steps // 10) == 0:
            print(f"  step {step:>5}/{n_steps}  loss {np.mean(losses[-200:]):.4f}")
    return diffusion, model, np.array(losses)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="two_moons", choices=["two_moons", "eight_gaussians"])
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--ddim-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    FIGDIR.mkdir(exist_ok=True)
    dataset = get_dataset(args.dataset)
    print(f"dataset: {args.dataset}, training for {args.steps} steps on {args.device}")
    diffusion, model, losses = train(dataset, args.steps, args.batch, args.lr,
                                     args.seed, args.device)

    rng = np.random.default_rng(args.seed)
    truth = dataset(4000, rng=rng)
    ddpm = diffusion.sample_ddpm(model, 4000).cpu().numpy()
    ddim = diffusion.sample_ddim(model, 4000, steps=args.ddim_steps).cpu().numpy()

    # ---- a coarse coverage check: does each mode get samples? ---------------
    def coverage(samples: np.ndarray, n_modes: int = 8) -> np.ndarray:
        angles = np.arctan2(samples[:, 1], samples[:, 0])
        slots = ((angles + np.pi) / (2 * np.pi) * n_modes).astype(int) % n_modes
        return np.bincount(slots, minlength=n_modes) / len(samples)

    print("\ncoverage of the 8 angular sectors (truth vs DDPM vs DDIM-"
          f"{args.ddim_steps}):")
    for name, s in [("truth", truth), ("DDPM", ddpm), (f"DDIM-{args.ddim_steps}", ddim)]:
        cov = coverage(s)
        print(f"  {name:<8} " + " ".join(f"{v:5.3f}" for v in cov))

    # ---- figure --------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    panels = [("training data", truth, "#2d3748"),
              (f"DDPM, 400 steps", ddpm, "#2b6cb0"),
              (f"DDIM, {args.ddim_steps} steps", ddim, "#2f855a"),
              ("loss", None, "#c53030")]
    for ax, (title, samples, colour) in zip(axes, panels):
        if samples is not None:
            ax.scatter(samples[:, 0], samples[:, 1], s=2, alpha=0.25, color=colour)
            ax.set_xlim(-3.2, 3.2)
            ax.set_ylim(-3.2, 3.2)
            ax.set_aspect("equal")
            ax.set_title(title)
        else:
            smooth = np.convolve(losses, np.ones(50) / 50, mode="valid")
            ax.plot(smooth, color=colour)
            ax.set_title("denoising loss (moving average)")
            ax.set_xlabel("training step")
    fig.suptitle(f"2D diffusion on {args.dataset}: same network, two samplers", fontsize=12)
    fig.tight_layout()
    out = FIGDIR / f"02_diffusion_{args.dataset}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    # ---- figure: the reverse trajectory of a few particles -------------------
    with torch.no_grad():
        x = torch.randn(6, 2)
        path = [x.numpy().copy()]
        for step in reversed(range(diffusion.n_steps)):
            t = torch.full((6,), step, dtype=torch.long)
            eps = model(x, t)
            mean = (x - diffusion.betas[step] / (1.0 - diffusion.alpha_bars[step]).sqrt() * eps) \
                / diffusion.alphas[step].sqrt()
            x = mean if step == 0 else mean + diffusion.betas[step].sqrt() * torch.randn_like(x)
            if step % 20 == 0:
                path.append(x.numpy().copy())
    path = np.array(path)

    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    ax.scatter(truth[:, 0], truth[:, 1], s=1, alpha=0.15, color="#a0aec0")
    for i in range(path.shape[1]):
        ax.plot(path[:, i, 0], path[:, i, 1], lw=1.1, alpha=0.85)
    ax.set_aspect("equal")
    ax.set_title("Reverse diffusion: noise turning into a crescent")
    fig.tight_layout()
    fig.savefig(FIGDIR / f"02_trajectories_{args.dataset}.png", dpi=150)
    plt.close(fig)

    print(f"\nfigures written to {FIGDIR}")


if __name__ == "__main__":
    main()
