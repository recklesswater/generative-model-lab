# 05 - A spectral view of diffusion: operators, projections, functionals

Three runnable experiments that replace three common hand-waves about diffusion with numbers
you can check:

| script | the hand-wave | what it actually computes | runtime |
| --- | --- | --- | --- |
| `spectral_forward.py` | "the forward process destroys information" | the noising step as a **linear operator**, its eigenvectors and its eigenvalues | ~2 s |
| `projection_demo.py` | "the denoiser is not magic" | the MMSE denoiser as an **orthogonal projection**, verified by measuring the residual's inner products | ~13 s |
| `guidance_functionals.py` | "guidance steers the sampler" | guidance as the **gradient of a functional**, with the linear-vs-radial distinction measured | ~33 s |

```bash
python 05_spectral_view/spectral_forward.py
python 05_spectral_view/projection_demo.py
python 05_spectral_view/guidance_functionals.py
```

## Scope - what this module is not

**None of this is new mathematics.** Heat semigroups, conditional expectation as projection and
tilted densities are textbook material. What is here is the *computation* of those objects on
objects small enough to check, wired to the same questions the other modules ask. If a claim
below disagrees with your textbook, the textbook wins and this script has a bug - the printed
residuals are there so that statement can be tested rather than trusted.

Three deliberate idealisations:

* the forward operator is treated as a **pure heat semigroup with variance t**. In the
  variance-preserving parameterisation the signal is additionally scaled by `sqrt(abar_t)`;
  that is a diagonal rescaling of the state, not of the spectrum, so the *ordering* of what
  dies first is unchanged.
* `projection_demo.py` and `guidance_functionals.py` use an **analytic Gaussian-mixture target**
  instead of a trained network. That buys exactness at the cost of realism: there is no
  approximation error to study, only the mechanism. `02_diffusion_toy` is where the learned
  version lives.
* the sampler is annealed Langevin with an analytic score. It is not a DDPM sampler; the
  statements below are about the *target distribution* the guidance produces, not about a
  particular discretisation.

## 1. The forward process is a linear operator (`spectral_forward.py`)

A noising step is convolution with a Gaussian kernel, i.e. a linear operator on the space of
densities. On a periodic grid it becomes a **circulant matrix**, so its spectrum can be read
directly - and the eigenfunctions turn out to be Fourier modes with eigenvalues
`exp(-2 pi^2 sigma^2 xi^2)`.

What the script checks and prints (`N = 257`, `dx = 0.0311`, `sigma = 0.15`):

```
eigenvalue residual ||P v - lambda v||        = 1.3e-14      (v = Fourier modes)
discrete spectrum vs continuum formula:
  max absolute error                          = 2.2e-16
  max relative error, |lambda| > 1e-6         = 2.1e-11      (89 of 257 modes)
```

So the operator picture is exact here, not approximate. The decay numbers are the payoff:

```
mode amplitude remaining at sigma = 0.15:    k = 1 -> 0.993 | k = 4 -> 0.895
                                             k = 16 -> 0.169 | k = 40 -> 1.5e-5
noise variance at which a mode halves:       k = 1 -> 2.25  | k = 4 -> 0.140
                                             k = 16 -> 0.009 | k = 40 -> 0.0010
```

**The one sentence worth keeping:** the order in which the forward process destroys structure is
not a modelling choice - it is the spectrum of the transition operator, and high frequencies
die first (cutoff at `xi ~ 1/(sqrt(2) pi sigma)`). That is why the last steps of a reverse
process can only affect low-frequency structure.

Figures: `figures/05_spectral_eigenvalues.png` (decay per frequency), `05_spectral_operator.png`
(the circulant operator), `05_spectral_mode_decay.png` (four modes dying at different times).

### Two traps this script documents

* **Eigenvalue ordering.** Eigenvalues from a dense eigensolver come back in an arbitrary order;
  pairing them with `fftfreq` by sorting produces nonsense (the first version reported a
  relative error of 2e4). For a circulant matrix the spectrum *is* the DFT of the generating
  kernel, and the mode order matches `fftfreq` - use that, then verify with a residual.
* **Sampling the kernel.** The Gaussian must be resolved by the grid. At three points per sigma
  the discrete spectrum aliases badly at high frequencies; `N = 257` over a half-width of 4
  gives ~5 points per sigma at `sigma = 0.15`.

## 2. The MMSE denoiser is an orthogonal projection (`projection_demo.py`)

`E[x0 | x_t]` is the L2 projection of `x0` onto everything measurable with respect to `x_t`.
The defining property of a projection is that the residual is orthogonal to that subspace:

```
E[ (x0 - E[x0|x_t]) * g(x_t) ] = 0     for every function g
```

For a Gaussian mixture both sides are closed-form, so this is measured rather than argued
(200,000 samples, `sigma^2 = 0.35`, mean `|residual| = 0.328`):

```
E[ residual_x * 1     ] = 3.3e-04
E[ residual_x * x     ] = 1.3e-03
E[ residual_x * x^2   ] = 5.3e-04
E[ residual_x * cos x ] = 3.8e-05
worst over all 8 test functions x 2 coordinates = 3.4e-03
```

Three parts in a thousand of the residual scale, on finite samples - i.e. zero within sampling
noise. Together with Tweedie's formula (the score is the direction of that residual) this is the
linear-algebra reason a denoiser works: it is not a heuristic, it is a projection.

Figure: `figures/05_projection_demo.png` (the denoiser field, and the orthogonality bars).

## 3. Guidance is the gradient of a functional (`guidance_functionals.py`)

Guidance adds an **energy** to the target density - the same sign convention as the distance
guide in module 04:

```
p_guided(x) ~ p(x) * exp( -gamma * f(x) )        score_guided = score - gamma * grad f
```

and the *kind* of functional decides what happens:

| functional | effect | prediction | measured |
| --- | --- | --- | --- |
| linear `f = a.x` | **tilt**: the distribution is translated | `shift = -gamma * Var(x) = -1.049` | **-0.889** |
| radial `f = (\|x\| - r)^2` | **contraction** onto the shell | see the sweep below | see the sweep below |

The radial constraint behaves like a knob, which is the more useful result - a soft constraint
has a measurable width, and its strength is a choice:

```
  gamma   predicted sd   radius mean   measured sd   within 0.15 of r = 1
      -              -         1.343         0.510                   0.187   <- no guidance
    0.6          0.913         1.252         0.445                   0.229
    2.0          0.500         1.150         0.360                   0.299
    6.0          0.289         1.071         0.255                   0.432
```

`predicted sd = sqrt(1 / 2 gamma)` is the local width of the penalty ignoring the prior; the
measured width is smaller because the prior pushes back as well. The trend - monotone
contraction towards the target shell - is what "conditional generation by a property" means in
practice, and it is also why a distance guide alone does not give you a hard constraint.

Figures: `figures/05_guidance_functionals.png` (three samplers, one knob changed),
`figures/05_guidance_radius.png` (the dose-response).

### Three traps this script documents

All three were hit while building it, and all three are the natural first guesses:

1. `score + gamma * grad f / sigma^2` - **diverges** as sigma -> 0 (the effective weight becomes
   gamma/sigma^2, about 240 at sigma = 0.05). The linear case drifted by -41 instead of -1.
2. Writing the same thing through the denoised estimate (`xhat - gamma grad f(xhat)`) does not
   help: 1/sigma^2 re-enters through Tweedie's formula.
3. Using **+** instead of **-** for a radial constraint rewards deviation, so the distribution is
   pushed *off* the shell (observed: values of order 1e26 before the guard). And with no cap on
   the gradient, large gamma makes the Langevin step overshoot: gamma = 6 diverged to NaN until
   the pull was clipped at `|grad f| <= 4`.

## Expected results (all three scripts)

| quantity | value | interpretation |
| --- | --- | --- |
| eigenvalue residual of the circulant operator | ~1e-14 | the Fourier modes really are eigenvectors |
| discrete vs continuum spectrum | 2.2e-16 absolute | the operator picture is exact on this grid |
| worst orthogonality violation of the denoiser | 3.4e-03 | the denoiser is a projection, to within sampling noise |
| linear guidance shift | -0.889 vs predicted -1.049 | a linear functional tilts; the shift is `gamma Var(x)` |
| radial guidance width, gamma = 0.6 / 2 / 6 | sd 0.445 / 0.360 / 0.255 | a soft constraint has a measurable, tunable width |

If you see numbers far from these, something is wrong with the environment rather than with the
conclusions - these are deterministic given the seeds in the scripts.

## Where this connects

* **module 02** (`02_diffusion_toy`): the learned version of the same objects - there the score
  is a network, here it is exact.
* **module 04** (`04_se3_diffusion`): the `DistanceGuide` there is section 3's radial functional
  with a richer f and a real backbone.
* **the BD / block-dependence project**: the same spectral language on the structure side.
  A covariance spectrum answers "which direction carries the most variance"; the diffusion
  spectrum answers "which direction loses information first". Used together the question becomes:
  **first check that the representation holds, then check that the generative process keeps what
  it should.**

## Next

* The 2D version of section 1: form the operator on a 2D grid and show that the eigenfunctions
  are the 2D Fourier modes, with the decay depending on `|k|` rather than on the axis.
* Replace the analytic mixture in section 3 with the trained model from module 02, and check
  whether the measured shift still matches `gamma Var(x)`.
* A hard-constraint baseline (projection onto the shell at each step) to show what the soft
  penalty above does *not* achieve.

## Dependencies

`numpy` and `matplotlib` only. No training, no GPU, no downloads.
