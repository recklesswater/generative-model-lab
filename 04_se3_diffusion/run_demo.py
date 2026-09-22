"""Train the frame diffusion model, check equivariance, generate, and steer with a property.

Usage:  python 04_se3_diffusion/run_demo.py --steps 2000 --n-res 12
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

from diffusion_se3 import FrameDiffusion  # noqa: E402
from equivariance_test import check_model_equivariance  # noqa: E402
from frames import invariant_features_torch, make_dataset  # noqa: E402
from model import LocalFrameDenoiser  # noqa: E402

FIGDIR = Path(__file__).resolve().parent.parent / "figures"


class DistanceGuide:
    """A property: pull two residues towards a target separation.

    This is the toy version of what conditional structure generation does with a motif, a
    binding interface or a symmetry constraint: define a differentiable score on the clean
    estimate, and use its gradient to steer the trajectory.
    """

    def __init__(self, i: int, j: int, target: float) -> None:
        self.i, self.j, self.target = i, j, target

    def step(self, t_clean: torch.Tensor, scale: float) -> torch.Tensor:
        # sampling runs under torch.no_grad(), so the guidance gradient needs its own context
        with torch.enable_grad():
            t = t_clean.detach().clone().requires_grad_(True)
            distance = torch.linalg.norm(t[:, self.i] - t[:, self.j], dim=-1)
            loss = ((distance - self.target) ** 2).mean()
            (grad,) = torch.autograd.grad(loss, t)
        return (t - scale * grad).detach()


def distances(t: np.ndarray, pairs) -> np.ndarray:
    return np.stack([np.linalg.norm(t[:, i] - t[:, j], axis=-1) for i, j in pairs], axis=1)


def backbone_length(t: np.ndarray) -> float:
    return float(np.linalg.norm(t[1:] - t[:-1], axis=-1).sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-res", type=int, default=12)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--n-samples", type=int, default=128)
    parser.add_argument("--diffusion-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    FIGDIR.mkdir(exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    R_np, t_np = make_dataset(512, args.n_res, rng)
    R = torch.tensor(R_np, dtype=torch.float32)
    t = torch.tensor(t_np, dtype=torch.float32)
    print(f"data: {R.shape[0]} structures x {args.n_res} residues")

    gap, err_t, err_r = check_model_equivariance(seed=args.seed, n_res=args.n_res)
    print(f"equivariance before training: features {gap:.2e}, "
          f"translation noise {err_t:.2e}, rotation noise {err_r:.2e}")

    diffusion = FrameDiffusion(n_res=args.n_res, n_steps=100, device=args.device)
    model = LocalFrameDenoiser(n_res=args.n_res).to(args.device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

    losses = []
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, R.shape[0], (args.batch,))
        loss = diffusion.loss(model, invariant_features_torch, R[idx], t[idx])
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        losses.append(loss.detach().item())
        if step % max(1, args.steps // 5) == 0:
            print(f"  step {step:>5}/{args.steps}  loss {np.mean(losses[-100:]):.4f}")

    R_gen, t_gen = diffusion.sample(model, invariant_features_torch, args.n_samples,
                                    steps=args.diffusion_steps, seed=args.seed)
    R_gen, t_gen = R_gen.numpy(), t_gen.numpy()
    print(f"generated {t_gen.shape[0]} structures in {args.diffusion_steps} sampling steps")

    # ---- summary of the geometry --------------------------------------------
    pairs = [(i, i + 1) for i in range(args.n_res - 1)]
    d_data = distances(t_np[: args.n_samples], pairs)
    d_gen = distances(t_gen, pairs)
    e2e = [(0, args.n_res - 1)]
    e2e_data = distances(t_np[: args.n_samples], e2e).ravel()
    e2e_gen = distances(t_gen, e2e).ravel()
    print("\n                     consecutive distance        end-to-end distance")
    print(f"  training data      {d_data.mean():6.3f} +- {d_data.std():.3f}"
          f"          {e2e_data.mean():6.3f} +- {e2e_data.std():.3f}")
    print(f"  generated          {d_gen.mean():6.3f} +- {d_gen.std():.3f}"
          f"          {e2e_gen.mean():6.3f} +- {e2e_gen.std():.3f}")

    # ---- guidance ------------------------------------------------------------
    target = float(np.median(e2e_data) * 0.6)
    guide = DistanceGuide(0, args.n_res - 1, target)
    _, t_guided = diffusion.sample(model, invariant_features_torch, args.n_samples,
                                   steps=args.diffusion_steps, guidance=guide,
                                   guide_scale=2.0, seed=args.seed)
    t_guided = t_guided.numpy()
    e2e_guided = distances(t_guided, e2e).ravel()
    print(f"\nguidance target for the end-to-end distance: {target:.3f}")
    print(f"  unguided  {e2e_gen.mean():6.3f} +- {e2e_gen.std():.3f}")
    print(f"  guided    {e2e_guided.mean():6.3f} +- {e2e_guided.std():.3f}")

    # ---- figures -------------------------------------------------------------
    fig = plt.figure(figsize=(16, 4.6))
    ax1 = fig.add_subplot(1, 4, 1, projection="3d")
    for k in range(4):
        ax1.plot(t_np[k, :, 0], t_np[k, :, 1], t_np[k, :, 2], "-o", ms=3, lw=1.2,
                 label="training" if k == 0 else None)
    for k in range(4):
        ax1.plot(t_gen[k, :, 0], t_gen[k, :, 1], t_gen[k, :, 2], "-o", ms=3, lw=1.2,
                 alpha=0.85, label="generated" if k == 0 else None)
    ax1.legend(frameon=False, fontsize=8)
    ax1.set_title("training vs generated backbones")

    ax2 = fig.add_subplot(1, 4, 2)
    ax2.hist(d_data.ravel(), bins=40, density=True, alpha=0.6, label="training")
    ax2.hist(d_gen.ravel(), bins=40, density=True, alpha=0.6, label="generated")
    ax2.set_title("consecutive distance")
    ax2.legend(frameon=False, fontsize=8)

    ax3 = fig.add_subplot(1, 4, 3)
    ax3.hist(e2e_data, bins=40, density=True, alpha=0.6, label="training")
    ax3.hist(e2e_gen, bins=40, density=True, alpha=0.6, label="generated")
    ax3.axvline(target, color="k", ls="--", lw=1, label="guide target")
    ax3.hist(e2e_guided, bins=40, density=True, histtype="step", lw=2, label="guided")
    ax3.set_title("end-to-end distance")
    ax3.legend(frameon=False, fontsize=8)

    ax4 = fig.add_subplot(1, 4, 4)
    smooth = np.convolve(losses, np.ones(50) / 50, mode="valid")
    ax4.plot(smooth)
    ax4.set_title("training loss")
    ax4.set_xlabel("step")

    fig.tight_layout()
    fig.savefig(FIGDIR / "04_frames.png", dpi=150)
    plt.close(fig)
    print(f"\nfigures written to {FIGDIR}")


if __name__ == "__main__":
    main()
