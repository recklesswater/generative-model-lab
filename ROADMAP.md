# Roadmap

The repository is developed in the open, module by module: each module is pushed in a small but
**runnable** state, then improved.

## Done

- [x] Repo skeleton, single evaluation philosophy across modules.
- [x] `01_mcmc_toy`: RWM + MALA + diagnostics (trace, ACF, ESS, mode-hopping).

## In progress

- [ ] `02_diffusion_toy`: DDPM/DDIM on two-moons and 8-gaussians, with the failure cases documented.
- [ ] `04_se3_diffusion`: SO(3)/SE(3) noise, equivariance test, property-guided sampling.
- [ ] `03_omics_synthesis`: replace the demo cohort with MetaboLights MTBLS1 behind a
      `--dataset` flag.

## Planned

- [ ] `04`: replace the MLP denoiser with a small SE(3)-equivariant (tensor-product) network and
      keep the MLP as an ablation.
- [ ] `05_molecule_guidance`: property-guided generation on a MoleculeNet task (BBBP / ClinTox /
      TOX21), reported the same way as module 03 — internal vs scaffold split.
- [ ] `06_spatial_omics` (long term): the same augmentation question on spatial transcriptomics
      (10x Visium), where the batch structure is spatial rather than tabular.
- [ ] Reproducibility: pin the environment, add a `make all` that regenerates every figure.

## Design notes

- Every module must run on CPU in a couple of minutes; if it cannot, it is split or shrunk.
- Every module must produce its own figures into `figures/` and print a summary table.
- Negative results are kept in the README of the module, not deleted.
