# generative-model-lab

![ci](https://github.com/recklesswater/generative-model-lab/actions/workflows/ci.yml/badge.svg)

A working notebook of generative modelling, **built in public**: small, runnable modules that trace
one idea from sampling to structure generation, and that report failures as carefully as successes.

The through-line is a single question:

> **When a conditional generative model is asked to produce something new, how do we know it is
> still trustworthy once the data moves off the training distribution?**

That question is asked three times, at three different levels of structure:

| Module | Object being generated | Dimensionality | Status |
| --- | --- | --- | --- |
| [`01_mcmc_toy`](01_mcmc_toy/) | samples from a 2D double-well distribution | 2D, sampling | ✅ runnable |
| [`02_diffusion_toy`](02_diffusion_toy/) | samples from 2D toy densities (two-moons, 8-gaussians) | 2D, diffusion | ✅ runnable |
| [`03_omics_synthesis`](03_omics_synthesis/) | **synthetic patients** for small-sample omics cohorts | tabular, ~10²–10³ | 🚧 v0 demo |
| [`04_se3_diffusion`](04_se3_diffusion/) | **protein-backbone-like frames** on SE(3), with property guidance | 3D rigid-body | 🚧 v0 demo |

Nothing here is a research contribution, and nothing here is a wrapper around someone else's
checkpoint. The modules are deliberately small enough to be read end to end — the point is to make
the mechanics visible: the noise process, the score, the equivariance, the operating point, and the
external validation.

## Quickstart

```bash
pip install -r requirements.txt

# 01: sampling, and what the operator language actually looks like as a plot
python 01_mcmc_toy/run_demo.py

# 02: a from-scratch 2D diffusion model (MLP + DDPM/DDIM)
python 02_diffusion_toy/run_demo.py

# 04: SO(3)/SE(3) frames, an equivariance test, and property-guided sampling
python 04_se3_diffusion/run_demo.py

# 03: small-sample omics augmentation, demo mode (no download needed)
python 03_omics_synthesis/run_experiment.py --demo
```

Each script writes figures to `figures/` and prints a short summary table, so the whole repo can be
checked in a couple of minutes on a laptop CPU.

## What each module is actually about

### 01 — MCMC: getting the operator language out of the head and onto the screen

Random-walk Metropolis, MALA and HMC on the same symmetric double-well potential: trace plots,
autocorrelation, ESS, acceptance rate, and barrier-crossing **rates** rather than raw counts, with
error bars over 10 seeds. This is the "spectral gap" story told with pictures — mixing time is
visible as slow ACF decay, and mode hopping is what the gap is about. The headline result is
deliberately anti-textbook: with a cheap 2D target, tuned RWM is the most cost-effective of the
three (18.8 ± 1.0 ESS per 1k evaluations, against 3.6 and 3.5), while HMC crosses the barrier twice
as often *per step* but pays 12 units of cost per iteration for the privilege. `show_one_step.py`
prints one complete step of each sampler - every term of the acceptance ratio - and checks three
invariants of the target, exiting non-zero if any of them fails.

### 02 — Diffusion: the same sampling problem, solved by learning a score

A tiny MLP trained to denoise 2D toy densities, sampled with DDPM and DDIM. The interesting part is
not that it works, but what changes when it does not: mode collapse on the 8-gaussians target,
over-smoothed two-moons, and the trade-off between fewer DDIM steps and sample quality.

### 03 — Omics synthesis: does augmenting a small cohort actually help the *external* validation?

The honest version of this experiment is the one that is usually not reported. Synthetic samples
usually improve the *internal* metrics and do nothing — or worse — for a cohort collected elsewhere,
because the generator is happy to learn the batch effect instead of the biology. This module is
built around that failure mode: the same classifier is evaluated on held-out internal data **and** on
a study collected separately, and both numbers are reported.

Data: [MetaboLights MTBLS1](https://www.ebi.ac.uk/metabolights/MTBLS1) (small, public, tabular).
`run_experiment.py --demo` runs the whole pipeline on a synthetic cohort with the same shape, so the
repo is runnable without any download.

### 04 — SE(3) diffusion: the same idea, one symmetry group later

Protein backbones are not vectors, they are rigid frames: $T_i = (R_i, t_i) \in SE(3)$, one per
residue. Adding Gaussian noise to the entries of a rotation matrix breaks the symmetry the problem
has, which is why this module implements the noise in the tangent space and maps it back with the
exponential map. The test that matters is written as an assertion, not a picture:

$$f(Q \cdot T) = Q \cdot f(T) \quad \text{for a random global rotation } Q$$

and a small property-guided sampler (push two residues together, avoid clashes) — which is the
mechanism that conditional structure generation (motif scaffolding, binder design) is built on.

## Evaluation discipline (the part I care most about)

Whatever the module, the same three rules apply:

1. **Report the operating point, not just the AUC.** A ranking metric can look fine while the
   threshold that a clinic would actually use is useless.
2. **Split by source, not by row.** If the external cohort exists, it is the only number that
   matters; internal cross-validation is a sanity check.
3. **Say what the generator learned that it should not have.** Batch, site, plate, sampling date —
   if the synthetic data carries them, the metric will too.

## Roadmap

See [`ROADMAP.md`](ROADMAP.md). Next up: real MetaboLights data in module 03 (replacing the demo
cohort), a proper SE(3) transformer denoiser in module 04, and a molecule-level module
(property-guided generation on a MoleculeNet property) as the AIDD-facing counterpart.

## License

MIT — see [`LICENSE`](LICENSE). No proprietary or patient-identifiable data is used anywhere in this
repository; module 03 works on public datasets or on synthetic cohorts generated by the code itself.
