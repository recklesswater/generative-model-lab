# 06 - Let the chain pick the reaction coordinate (MCMC + Krylov)

A reaction coordinate is normally chosen by hand - a distance, a dihedral, an RMSD - and the whole
family of enhanced-sampling methods (metadynamics, umbrella sampling, adaptive biasing) inherits that
choice. This module asks the other question: **can the dynamics hand the coordinate over instead?**
And how far does that survive as the dimension grows?

The recipe is the standard one from the Markov-state-model literature: estimate a transfer operator
from the trajectory, take its leading eigenvector, and call that the collective variable (CV). The one
implementation choice worth naming is that the operator is **never formed** - a Lanczos (Krylov) run
only ever applies it to a vector - which is why the same code runs on an 8x8 sample covariance and on
a 256x256 grid operator without changes.

| script | the hand-wave | what it actually computes | runtime |
| --- | --- | --- | --- |
| `krylov_basics.py` | "Krylov returns eigenpairs" | the Ritz pairs of two operators whose spectrum is known in closed form, the closed-form residual bound against the literal residual, and what re-orthogonalisation buys | ~2 s |
| `cv_discovery.py` | "you have to pick the reaction coordinate" | the leading eigenvector of the lag-tau covariance, checked three ways (against the direction that generated the data, against an independent exact sampler, against the analytic free energy), plus a 3D picture | ~15 s |
| `dimension_scan.py` | "this cannot work in high dimensions" | the same problem in d = 2 ... 20, the sweep in which it *does* break, and a negative control | ~40 s |

`krylov.py` (the iterations), `targets_nd.py` (the test target) and `cv_analysis.py` (free-energy and
tuning helpers) are the library. The samplers and diagnostics are imported from `01_mcmc_toy`
unchanged - the chain is the same chain.

```bash
python 06_krylov_mcmc/krylov_basics.py      # ~2 s
python 06_krylov_mcmc/cv_discovery.py       # ~15 s, writes 3 figures
python 06_krylov_mcmc/dimension_scan.py     # ~40 s, writes 1 figure
```

`cv_discovery.py` accepts `--dim`, `--steps`, `--burn-in`, `--lag`, `--m`, `--kappa`, `--a`, `--h`,
`--seed` and `--skip-figures` (the last one also skips the 3D cross-check chain).

## Scope - what this module is not

**None of this is new mathematics, and none of it is offered as such.** The two ingredients are
borrowed outright, and the provenance is meant to be unambiguous:

- `krylov.lanczos` is the symmetric Lanczos process - Golub & Van Loan, *Matrix Computations*, 4th
  ed., Algorithm 10.1.1, with the full re-orthogonalisation of section 5.2.8.
- `krylov.arnoldi` is the Arnoldi process - Golub & Van Loan Algorithm 10.5.1. The reading of what
  comes out (Ritz values, Ritz vectors, the residual bound) is Trefethen & Bau, *Numerical Linear
  Algebra*, Lectures 33-36.
- the CV estimator - equal-time covariance `C0`, lag-tau covariance `C_tau`, the generalised
  eigenproblem `C_tau v = mu C0 v`, leading eigenvector as the collective variable - is the standard
  time-lagged / transfer-operator construction; it is exactly what time-lagged independent component
  analysis (TICA) computes.

If a claim below disagrees with those books, the books win and there is a bug in the code - which is
why almost every number the scripts print is printed next to the thing it is meant to check.

Three deliberate idealisations:

- **the features are the coordinates themselves**, so the CV is a *linear* function of `x`. A real
  system would use distances, contacts or an RBF basis; that extension is listed under **Next** and
  is not claimed here.
- **the target is analytic** (`targets_nd.AnisotropicWells`), so the true reaction coordinate, the
  true barrier and the true marginal are known exactly. There is no approximation error to hide
  behind, and none to blame.
- **the trajectory is MALA from module 01**, with the step size tuned on a pilot run. It is not an MD
  integrator and there is no thermostat. Everything below is a statement about the transfer operator
  of the chain that was actually run.

## The target: a barrier along a direction you cannot read off a plot

```
U(x) = sum_k h_k (u_k^2 - a_k^2)^2 + (kappa/2) * |x_perp|^2,      u_k = d_k . x
```

with `d_k` a random orthonormal set (drawn by QR, `targets_nd.random_directions`), so no coordinate
axis is the reaction coordinate. The properties that make it a usable test bed are all exact:

- the wells sit at `u_k = +-a_k` with the transverse coordinates zero, so there are `2^n_slow` modes;
- the barrier separating them along `u_k` is `h_k a_k^4`;
- the marginal of `u` factorises, `p(u) ~ exp(-sum_k h_k (u_k^2 - a_k^2)^2)`, because the transverse
  terms do not depend on `u` and the rotation `x -> (u, x_perp)` has unit Jacobian.

So the free-energy profile along the reaction coordinate is known in closed form and can be compared
against whatever the chain produces.

The one free choice that makes the problem non-trivial is the ratio of **variance to slowness**.
`kappa` is set low enough that each transverse direction carries *more* variance than the reaction
coordinate (with `kappa = 0.5`: 2.00 per transverse direction versus 1.95 along the barrier) while
relaxing much faster (autocorrelation time 55 versus 122 steps). A variance-based method is a contest
between *single* directions, and it loses that contest: the top principal component lands 64.9 degrees
away from the barrier, while the Krylov CV lands 3.45 degrees away. That contrast is the experiment.

## The estimator, and why Lanczos rather than Arnoldi

With centred features `x` (the coordinates) and a lag `tau`:

```
C0[i,j]   = <x_i(t) x_j(t)>            equal-time covariance
Ctau[i,j] = <x_i(t) x_j(t+tau)>        lag-tau covariance, means removed
Ctau v = mu C0 v                       generalised eigenproblem; v is the CV
```

`C0` is symmetric positive definite, so it has a Cholesky factor `C0 = L L^T`. Substituting
`v = L^-T w` turns the generalised problem into an ordinary symmetric one:

```
A = L^-1 Ctau L^-T,     A w = mu w
```

and `slow_subspace` runs Lanczos on `A` (matrix-free: it only ever applies `L^-1 Ctau L^-T`), then
maps the vectors back with `v = L^-T w`. The whitened inner product is also what makes the angle
between two CVs meaningful - in the unwhitened coordinates a direction's "size" is its variance, and
the comparison would be measuring `C0`, not the dynamics.

**Reversibility is where the symmetry comes from.** A reversible chain has
`p(x) T(x -> y) = p(y) T(y -> x)`, which makes the transfer operator self-adjoint in the `C0` inner
product, which is what entitles us to use Lanczos (real, symmetric) rather than Arnoldi (general).
`krylov_basics.py` section 5 demonstrates the failure mode rather than asserting it: the same random
walk with a drift added has `||B - B^T||_inf = 0.022`, its spectrum moves off the real axis (max
`|imag| = 0.519`), Arnoldi recovers it to 2.5e-09, and Lanczos returns real numbers that are wrong by
up to **0.46 with no error message at all**.

One more detail, because it is easy to get wrong in a way that has no symptom: the sample estimate of
`Ctau` is **not symmetric** (`<x_i(t) x_j(t+tau)>` is not `<x_j(t) x_i(t+tau)>` at finite sample
size). `slow_subspace(..., symmetrize=True)` averages it with its transpose and reports the measured
asymmetry, which here is 0.047 at `tau = 8` and 0.54 at `tau = 64`. Feeding the raw estimate to
Lanczos would silently invalidate the residual bound; symmetrising improved the measured angle from
4.40 to 3.45 degrees as well.

## 1. The chain (`cv_discovery.py`, section 2)

`dim = 8`, `60000` steps, `5000` discarded, `kappa = 0.5`, `a = 1.5`, `h = 0.5`, `seed = 1`:

```
MALA step size (tuned on a pilot)        0.694     acceptance 0.567
samples kept                             55,000 x 8
crossings of the true barrier            911  (16.6 per 1k steps)
integrated autocorrelation time of u     122 steps
variance along u / along top PC          1.95 / 2.15
autocorrelation time u / top PC          122 / 55 steps
```

**The loudest direction in the sample cloud is not the slowest one.** That single line is the reason
the module exists.

## 2. Choosing the lag: is there a timescale plateau? (section 3)

`tau` is not a free parameter to be tuned until the answer looks good, so the script scans it and
prints what happens:

```
  lag tau  t1 (steps)  t2 (steps)  t3 (steps)  angle to truth (deg)
        2        25.2        13.9        12.9                   4.8
        4        32.5        14.2        13.2                   3.4
        8        40.9        14.1        13.1                   3.4
       16        48.4        14.5        12.8                   3.4
       32        53.8        15.2        12.2                   5.1
       64        58.1        16.4           -                   6.3
chosen lag = 8 steps      (direct check: ACF time / 2 = 60.8 steps)
```

The selection rule is mechanical: the smallest lag at which `t1` has reached 70 percent of its
plateau. Two things in that table are worth keeping rather than smoothing over. `t1` never actually
plateaus - it keeps climbing, because a longer lag biases the estimate of a slow mode upward. And the
*direction* is best in the middle and degrades again at large `tau`, because long lags throw away
sample pairs. Choosing the lag to make the timescale look nicest would pick `tau = 64` and the worst
direction.

## 3. The slow subspace, and how much of the error is Krylov's (section 4)

At the chosen `tau = 8`:

```
Ritz values mu_1..mu_6            0.8222  0.5726  0.5596  0.5507  0.5440  0.5263
spectral gap mu_1 - mu_2          0.2495     (implied timescales 40.9 vs 14.3 steps)
dense (non-Krylov) leading values 0.8222  0.5726  0.5712  0.5638
mu_1, Krylov vs dense             0.8222 vs 0.8222   (difference 1.1e-16)
closed-form bound held             max ratio 1.000    (<= 1 means the bound never lied)
```

Two honest caveats are printed with it. The values after the first are a near-degenerate cluster
(0.5726 versus 0.5712) - those are the transverse modes, and *individual members of a degenerate
cluster are not identified by either method, only the subspace is*. And the fourth residual is 0.01
while the first three are at machine zero.

Then the question that matters for the "high-dimensional approximation" worry - is the truncation
the error, or the estimator? Sweeping the Krylov dimension `m`:

```
  Krylov dim m  angle to truth (deg)    mu_1  residual
             2               3.4498   0.8222    0.0011
             4               3.4496   0.8222    0.0000
             8               3.4495   0.8222    0.0000
```

At `m = 8`, which is the full dimension of the problem, the Krylov truncation is gone *entirely* -
and the angle is 3.4495 degrees instead of 3.4498. **The truncation is not the error. The finite
sample estimate of the operator is the error.** That is the whole answer to "can a low-dimensional
subspace approximation of a high-dimensional problem be trusted": yes, and the thing that limits it
is statistics, not the subspace.

### The comparison table (section 4, second half)

| direction | angle to the true slow direction (deg) |
| --- | --- |
| true (generating) direction | 0 |
| top principal component | 64.90 |
| best single coordinate | 45.57 |
| random direction | 62.45 |
| **Krylov CV (1st Ritz vector)** | **3.45** |
| Krylov, 2nd Ritz vector | 86.55 |

The largest single component of the true direction is 0.700, i.e. the best coordinate axis is 46
degrees off. There is no coordinate to eyeball here.

Start-vector independence, 5 random Krylov starts: angle **3.45 +/- 0.00 degrees**, `mu_1 = 0.8222
+/- 0.0000`. The start vector is the one arbitrary ingredient in a Krylov method, and at this `m` it
has stopped mattering.

## 4. Does the CV carry the physics? (section 5)

The honest way to ask whether a CV is the *right* coordinate is to check whether the physics built on
it comes out right. Three references are used, and they are all independent of the Krylov run: the
generating direction, a 400,000-sample i.i.d. draw from the target (exact, no MCMC), and the analytic
marginal.

| coordinate | angle (deg) | barrier, chain | barrier, exact reference | RMS(chain, reference) | IACT (steps) |
| --- | --- | --- | --- | --- | --- |
| true direction | 0 | 2.66 +/- 0.11 | 2.48 | 0.17 | 121.6 |
| **Krylov CV** | **3.45** | **2.44 +/- 0.10** | **2.45** | **0.14** | 121.1 |
| top principal component | 64.90 | 0.01 +/- 0.03 | 0.02 | 0.09 | 54.6 |
| best single coordinate | 45.57 | 0.07 +/- 0.06 | 0.02 | 0.11 | 75.0 |
| random direction | 72.18 | 0.06 +/- 0.03 | 0.00 | 0.10 | 39.7 |

The analytic barrier is 2.53 kT. The Krylov CV recovers 96 percent of it; the top principal component
recovers 0 percent, and the small RMS column shows that it is *precisely* zero rather than noisy
zero - along that coordinate the double well looks like a single well, which is exactly what a
variance-based method is blind to.

Section 6 closes the loop by asking whether the projected dynamics is Markovian - the
Chapman-Kolmogorov test in one dimension:

```
measured autocorrelation of z at lag tau   0.8221   (Ritz value mu_1 = 0.8222)
measured at lag 2 tau                      0.7180   predicted by the 1D model mu_1^2 = 0.6760
                                                     ratio 1.062
```

A ratio near 1 says the one-dimensional model is self-consistent at this resolution; the 6 percent
excess is the fast modes leaking back in, and it shrinks as the lag grows.

## 5. Three dimensions, and the picture (section 5, and `figures/06_krylov_cv_3d.png`)

The visualisation is deliberately the easy case: `d = 3`, where the sample cloud *can* be drawn. Left
panel: the 3D scatter coloured by the CV, with an arrow for the generating direction and an arrow for
the Krylov direction. Right panel: the CV against the true coordinate, next to the top principal
component against the true coordinate (signs flipped so both read as positive slope).

```
3D cross-check: d = 3, MALA step 0.69, acceptance 0.57, angle to truth 1.31 deg
```

The picture is a sanity check, not the result - the result is that dimension is not the wall.
Section 3 of `dimension_scan.py` is where the method actually breaks, and it breaks on the spectral
gap, which you can see in the Ritz spectrum before you know the answer.

## 6. Where it breaks, and the negative control (`dimension_scan.py`)

**Dimension (sections 1-2).** The same problem in d = 2, 3, 5, 8, 12, 20:

```
   d  angle CV (deg)  angle PC (deg)  t1/t2  barrier from CV  exact barrier  recovered
   2           0.681          67.967  2.682      2.81 +/- 0.20          2.531      1.108
   3           1.178          60.823  2.785      2.45 +/- 0.11          2.531      0.968
   5           2.610          71.340  2.964      2.52 +/- 0.11          2.531      0.995
   8           4.191          61.474  2.860      2.65 +/- 0.15          2.531      1.048
  12           2.301          83.992  2.930      2.82 +/- 0.14          2.531      1.112
  20           4.455          80.128  2.581      2.61 +/- 0.16          2.531      1.030
```

The CV angle is flat in `d` (and non-monotone: it is a sampling-noise effect, not a trend). The
variance-based angle sits at 60-85 degrees in every row. What does grow with `d` is the amount of
statistics needed, and it shows up as a slightly wider error bar rather than as a bias. The lag is
chosen as a fraction of the measured autocorrelation time rather than as a fixed number of steps,
because the tuned step size changes with dimension and a fixed lag would silently compare different
physical times.

**The spectral gap is the wall (section 3).** `kappa` is the knob that makes the transverse
directions wider *and* slower at the same time:

```
kappa  angle CV (deg)  angle PC (deg)  t1/t2  barrier from CV
    1           2.231           5.645  6.187     3.08 +/- 0.15
0.500           4.191          61.474  2.860     2.65 +/- 0.15
0.350           6.781          88.772  1.940     2.80 +/- 0.21
0.250          10.747          86.704  1.354     2.33 +/- 0.14
0.200          33.843          86.835  1.186     0.10 +/- 0.05
```

The direction is lost once the timescale ratio `t1/t2` approaches 1 - and that ratio is readable in
the Ritz spectrum *before* any comparison with the truth is made. On a real system, where the truth
is unknown, it is the first thing to check.

**More than one slow coordinate (section 4).** Two barriers of 2.53 and 1.27 kT along two dense
directions in `d = 5`:

```
Krylov dim m   mu_1   mu_2  v1 -> u1 (deg)  v1 -> u2 (deg)  v2 -> u2 (deg)  residual of pair 2
           1  0.223      -          78.331          88.506               -                   -
           2  0.852  0.003           6.793          83.504          89.842               0.009
           3  0.852  0.494           4.930          85.511          63.287               0.359
           4  0.853  0.752           0.750          89.327           0.852               0.006
           5  0.853  0.752           0.793          89.324           0.804               0.000
```

At `m = 5` the whole space is spanned, the two leading Ritz values are 0.853 and 0.752, and then
there is a drop to 0.013: **the number of Ritz values standing above the rest is the number of
collective variables the problem has, and it is readable without knowing the answer.** At `m = 2` a
single CV catches only `u1` and the second vector is a transverse mode - the residual column (0.009
versus 0.359 versus 0.006) is the warning that says so. On this target the leading principal
component points 4.5 degrees from `u1` and 85.5 degrees from `u2`: it cannot tell the two apart at
all, which is the cleanest case of "variance is not the property being asked for".

**Negative control (section 5).** An anisotropic Gaussian has no barrier, no slow mode and no gap.
The method still returns a direction:

```
target                     t1/t2   leading Ritz value
double well (d=8)          2.860   0.829
anisotropic gaussian       1.081   0.131
```

The leading value sits close to its neighbours, and the direction it returns is a number without a
meaning. If there is no gap, there is no collective variable to find - and the method will not tell
you that by refusing to answer.

## 7. `krylov_basics.py`: the operator, before it is used on anything unknown

Two circulant operators on a ring of 256 sites (circumference 8), so the exact spectrum is the DFT of
the kernel rather than a number taken on faith: an unbiased Gaussian jump kernel (`sigma = 0.15`), and
the same kernel with a drift.

```
||eig(A) - fft(kernel)||_inf                1.78e-15   (dense eigensolve)
max |imaginary part| of the reference       6.94e-17   (an even kernel has a real spectrum)
```

Convergence and the honesty of the cheap residual bound:

```
Krylov dim m  converged pairs  max err | converged  max literal resid  max (literal - bound)
          10                0                    -           1.33e-01               4.34e-18
          30               17             4.44e-16           2.56e-03               6.32e-16
          60               46             2.44e-15           4.22e-03               7.57e-16
          80               71             7.03e-16           1.05e-05               6.57e-16
```

where "converged" means the closed-form bound is below 1e-08. The last column being round-off in
every row is the point: no Ritz pair ever has a residual larger than the bound claims. The pairs that
have *not* converged are not wrong, they are estimates of the middle of the spectrum, which is what
`m` controls.

The cost of skipping re-orthogonalisation - the classic way to get "converged" Ritz values that are
duplicated and wrong:

```
Krylov dim m  re-orthogonalised  duplicated Ritz values  max literal resid
          60                yes                      18           6.10e-03
          60                 no                      17           3.53e-01
          80                yes                      32           8.10e-06
          80                 no                      34           8.74e-02
```

The first two columns of the spectrum look the same either way; the accuracy does not. Two extra
inner products per iteration is the difference between 1e-06 and 1e-02 at the same `m`.

## Traps this module documents

Four bugs were hit while building it. None of them announced itself; all four produced plausible
numbers.

1. **The control experiment was secretly the treatment.** `random_directions(seed=0)` builds its
   direction from `default_rng(0).standard_normal((d, 1))`, and the main script drew its "random
   direction" control from `default_rng(0)` as well. The first eight normal draws are identical, so
   the control *was* the truth and the table reported an angle of 0 degrees for a random direction.
   A control that shares a seed with the thing it controls for is not a control. Fixed by an
   independent `default_rng(12345)`.
2. **The Lanczos residual bound used the wrong beta.** The tridiagonal `T` has `k-1` sub-diagonal
   entries, and the norm of the final residual vector - the true `beta_k` - was never stored, so the
   bound substituted `beta[-1]` and came out wrong by a constant factor. It looked *suspiciously*
   consistent: the printed ratio of literal to bound was 2.774 in every single row. A constant ratio
   across rows is the tell. `LanczosRun` now carries a `tail` field; after the fix the literal
   residual never exceeds the bound by more than round-off.
3. **`print_table` silently dropped columns.** When a row was longer than the header it was
   truncated, so the column labelled "angle" was actually printing the fourth timescale. Nothing
   crashed and the numbers looked like numbers. It now raises `ValueError` on a length mismatch.
4. **`free_energy` crashed on one-dimensional input.** With `n_slow = 1` the projection is a 1D
   array and the reshape raised `IndexError`. Fixed with `u.reshape(*u.shape, 1)`.

A fifth issue was conceptual rather than a bug, and is worth recording because the first version of
the "two slow coordinates" section was unreadable for it: the example was originally run at
`dim = 8, kappa = 0.5`, where a transverse mode is still slower than `u2` at the chosen lag - so the
second Ritz vector at `m = 2` was a transverse mode and the demonstration, while correct, made no
pedagogical point. It is now `dim = 5, kappa = 4.0`, `h = (0.5, 0.25)`, where `m = 1` gives 78
degrees, `m = 2` gives 6.8, and `m = 4` recovers both coordinates to under a degree.

## Expected results

All three scripts are deterministic given their seeds.

| quantity | value | interpretation |
| --- | --- | --- |
| angle of the Krylov CV to the true slow direction (d=8) | 3.45 deg | the dynamics hands over the coordinate |
| angle of the top principal component | 64.90 deg | the loudest direction is not the slowest |
| angle of the best single coordinate axis | 45.57 deg | there is no coordinate to eyeball |
| angle at Krylov dim m=2 vs m=8 | 3.4498 vs 3.4495 deg | the truncation is not the error, the estimator is |
| barrier recovered by the CV | 2.44 +/- 0.10 kT vs 2.53 analytic | 96 percent, checked against an i.i.d. sampler |
| barrier recovered by the top principal component | 0.01 +/- 0.03 kT | zero, and precisely zero |
| Krylov vs dense leading Ritz value | 0.8222 vs 0.8222 (1.1e-16) | the matrix-free iteration is exact on this problem |
| CV angle across d = 2 ... 20 | 0.68 ... 4.46 deg | dimension is not the wall |
| CV angle vs the timescale ratio t1/t2 | 2.2 deg at 6.19, 33.8 deg at 1.19 | the gap is the wall |
| angle to truth in 3D | 1.31 deg | the picture is drawn where it can be checked |
| closed-form vs literal Lanczos residual | literal - bound <= 7.6e-16 | the cheap bound is honest here |
| Lanczos on a non-symmetric operator (drift 0.5) | worst error 0.46, no warning | reversibility is not optional |

If you see numbers far from these, something is wrong with the environment rather than with the
conclusions.

## Where this connects

- **module 01** (`01_mcmc_toy`): the chain, its ACF and its mode-hopping diagnostics are reused
  verbatim. The ACF that module 01 plots for one coordinate is the diagonal of the operator that this
  module diagonalises.
- **module 05** (`05_spectral_view`): the same spectral language, on a different object. There the
  spectrum belongs to the *noising operator* of a diffusion model and answers "which direction loses
  information first"; here it belongs to the *transfer operator* of an MCMC chain and answers "which
  direction the dynamics is slow in". Used together the question becomes: first check that the
  representation holds, then check that the generative process keeps what it should.
- **enhanced sampling**: metadynamics, umbrella sampling and adaptive biasing all take a CV as input.
  This module is only the other half - deriving it from the dynamics instead of assuming it - and the
  dimension scan is the honest statement of when that derivation can be trusted.
- **protein folding / drug design**: RMSD, distances and dihedrals are *choices*; the claim under
  test is that the leading Ritz vector of an estimated transfer operator is the coordinate the
  dynamics actually has, and not the one a human found convenient.

## Next

- **Nonlinear features.** Replace `x` with distances, contacts or an RBF basis so that the CV is a
  nonlinear function `z = T(x)`. Everything else in the pipeline is unchanged, and the same three
  checks still apply.
- **A non-reversible chain.** `krylov_basics.py` shows what Lanczos does to a drifted operator; the
  CV pipeline should be rerun with Arnoldi so that the failure mode is handled rather than avoided.
- **A VAMP-2 score.** The estimator here is TICA; a variational score would let different feature
  sets be compared without the truth being known, which is the situation on a real system.
- **A second collective variable with a slow transverse mode**, to see how the two-CV selection rule
  ("count the Ritz values above the drop") behaves when the drop is not clean.

## Dependencies

`numpy` and `matplotlib` only - the 3D axes come from `mpl_toolkits`, which ships with matplotlib. No
training, no GPU, no downloads. Module 01 must be present (it is imported for the samplers and
diagnostics).
