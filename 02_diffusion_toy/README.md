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

## Next

* Add a small classifier-free guidance demo (`--skip` probability + guidance weight) on a
  *conditional* two-moons (e.g. moons shifted by a class label), so that the difference between
  unconditional and conditional sampling is visible.
* Replace the MLP with a small UNet-style residual network and compare sample quality at equal
  wall-clock time.
