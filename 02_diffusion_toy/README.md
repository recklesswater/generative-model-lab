# 02 - Diffusion toy: a from-scratch 2D denoising model

```bash
python 02_diffusion_toy/run_demo.py                                  # two-moons
python 02_diffusion_toy/run_demo.py --dataset eight_gaussians --steps 6000
```

## What is here

| File | Content |
| --- | --- |
| `toy_data.py` | `two_moons` (a curved manifold) and `eight_gaussians` (eight separated modes) |
| `model.py` | MLP denoiser with a sinusoidal time embedding |
| `ddpm.py` | forward variance schedule, training loss, DDPM sampler, DDIM sampler |
| `run_demo.py` | trains, samples with both samplers, prints a coverage table, writes figures |

## What to look at

1. **Coverage, not just the loss.** `run_demo.py` prints how the samples distribute over eight
   angular sectors. On `eight_gaussians` a model can reach a healthy loss while quietly dropping
   one or two modes - the sector table shows it, the loss curve does not.
2. **The DDIM trade-off.** The same trained network is queried on a 20-step deterministic
   trajectory. Fewer steps is much cheaper; what it costs is mode coverage and smoothness, and
   that difference is visible by comparing the two scatter panels.
3. **The reverse trajectory.** `02_trajectories_*.png` draws six particles from noise to data -
   the clearest picture of what "iterative denoising" means in practice.

## Notes from the first run

* `two_moons` is easy to fit and hard to fit *crisply*: with a plain MLP the crescent is slightly
  blurred, which is the standard bias of a Gaussian-kernel diffusion model at low dimension.
* `eight_gaussians` is the interesting failure case. Coverage depends strongly on the number of
  training steps and on the `--seed`; this is why the coverage table is printed rather than only
  plotted.

### Update — `eight_gaussians`, 6000 steps (2026-09-22 evening run)

Run: `python 02_diffusion_toy/run_demo.py --dataset eight_gaussians --steps 6000`
Figures: `figures/02_diffusion_eight_gaussians.png`, `figures/02_trajectories_eight_gaussians.png`.

| | widest sectors (truth 0.221 / 0.255) | rarest sectors (truth 0.008 / 0.018) |
| --- | --- | --- |
| DDPM, 6000 steps | 0.218 / 0.218 — near exact | 0.017 / 0.029 — still about 2x over-covered |

* **No mode was dropped.** At this budget the failure is over-coverage of the tails, not the mode
  collapse seen at 300 steps.
* **Only one seed was run**, so "no mode loss" is not yet separated from seed luck; a multi-seed
  repeat is the first item on the roadmap.
* The tail bias does not shrink with longer training — it is the bias of a Gaussian-kernel
  diffusion model at low dimension, not a training-time problem.

Same lesson as the trace plots in module 01: **a healthy-looking loss curve is not evidence that
the distribution is right.**

## Next

* Add a small classifier-free guidance demo (`--skip` probability + guidance weight) on a
  *conditional* two-moons (e.g. moons shifted by a class label), so that the difference between
  unconditional and conditional sampling is visible.
* Replace the MLP with a small UNet-style residual network and compare sample quality at equal
  wall-clock time.
