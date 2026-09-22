# 03 - Omics synthesis: does augmenting a small cohort help the *external* validation?

The honest version of the synthesis experiment is the one that is usually not reported:
synthetic patients often improve internal metrics and do nothing - or worse - for a cohort
collected somewhere else, because the generator is happy to learn the batch effect instead of
the biology.

```bash
python 03_omics_synthesis/run_experiment.py --demo           # synthetic two-batch cohort
python 03_omics_synthesis/run_experiment.py --data mtbls1.csv --methods none smote
```

## Design

Same features, same classifier, same decision rule; the only thing that changes is whether the
training set was augmented and by what:

| Method | Idea | Failure mode it invites |
| --- | --- | --- |
| `none` | no augmentation | loses small-sample signal |
| `smote` | interpolate minority samples with minority neighbours | stays inside the training batch, so it can amplify the batch effect |
| `gaussian_copula` | sample from a per-class multivariate Gaussian | one mode per class; ignores nonlinear structure |

Two evaluation decisions matter more than the generator:

1. the threshold is chosen on **out-of-fold training scores** at a specificity target of 85%, and
   then frozen for both held-out cohorts;
2. the external cohort is scored with that same frozen threshold - no re-calibration per cohort.

The demo cohort (`data.synthetic_cohort`) has an internal and an external batch, and the batch
offset is deliberately **entangled with the label**: a batch effect that ignores the label is
mostly harmless, one entangled with it is what makes synthetic patients dangerous.

## What the first run shows (synthetic demo, 5 seeds, 320 samples, 60 features)

| Method | Model | Internal AUC | External AUC | External sensitivity | External specificity |
| --- | --- | --- | --- | --- | --- |
| none | logreg | 0.975 (0.016) | **0.973 (0.007)** | 0.887 (0.034) | 0.923 (0.027) |
| none | gbdt | 0.977 (0.012) | **0.968 (0.015)** | 0.851 (0.056) | 0.946 (0.014) |
| smote | logreg | 0.976 (0.016) | 0.972 (0.007) | 0.887 (0.034) | 0.916 (0.027) |
| smote | gbdt | 0.975 (0.020) | 0.964 (0.014) | 0.841 (0.049) | 0.946 (0.026) |
| gaussian_copula | logreg | 0.973 (0.016) | 0.972 (0.005) | 0.903 (0.042) | 0.906 (0.027) |
| gaussian_copula | gbdt | 0.963 (0.025) | 0.962 (0.016) | 0.867 (0.038) | 0.956 (0.026) |

Mean with the standard deviation over seeds in brackets; base rates are 50% internal / 40%
external.

**Result: a null result, and it is reported as one.** On this demo cohort the augmentation does
not move the external validation beyond seed noise (all three methods overlap within about one
standard deviation), and it slightly *reduces* external sensitivity for the gradient-boosted
model. Raising the batch shift to 2.5 does not change that conclusion.

Two caveats that keep this from being a finished experiment:

* the demo cohort is synthetic, and the signal is strong enough that the classifier is robust
  whatever we feed it - the real test needs real data;
* `smote` and `gaussian_copula` are the weak baselines. A conditional diffusion model (the
  sequence of modules 02 and 04) is the interesting generator to put in this harness, and that is
  the next step.

## Files

| File | Content |
| --- | --- |
| `data.py` | synthetic two-batch cohort + a loader for a prepared MTBLS1 table |
| `augment.py` | `none`, `smote`, `gaussian_copula` (all class-conditional, hand-rolled) |
| `baselines.py` | classifiers, out-of-fold threshold selection, metrics |
| `run_experiment.py` | the loop over methods/seeds/models, the summary table, the figures |

Outputs: `figures/03_omics_augmentation.png` (internal vs external across methods),
`figures/03_omics_pca.png` (where the synthetic samples land relative to the external batch),
and `outputs_omics_augmentation.csv` (raw per-seed results).

## Getting the real data

MetaboLights MTBLS1 is small, public and tabular:

1. open <https://www.ebi.ac.uk/metabolights/MTBLS1> and download the metabolite table;
2. reshape it to one row per sample: columns `f0...fp` for the metabolites, plus `label` (0/1)
   and `batch` (use `"internal"` / `"external"` if the study has a natural two-part structure);
3. run `python 03_omics_synthesis/run_experiment.py --data that_file.csv`.

The loader is in `data.py`; there is no automatic download in the repository on purpose.

## Next

* Swap the baselines for a conditional diffusion / flow-matching generator and keep `smote` as
  the reference point.
* Add a **privacy check** (nearest-neighbour distance between synthetic and real patients) and a
  **distribution check** (MMD / energy distance) so that "the synthetic data looks like the real
  data" is measured rather than asserted.
* Add a study with a genuine cross-study external cohort, so the external number is not
  determined by a simulated batch shift.
