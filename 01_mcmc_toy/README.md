# 01 - MCMC toy: mixing, in numbers and pictures

Three samplers, one bimodal target, no hidden machinery.

```bash
python 01_mcmc_toy/run_demo.py      # ~20 s on a laptop CPU
```

## What is here

| File | Content |
| --- | --- |
| `targets.py` | `StandardNormal2D` and `DoubleWell` (un-normalised log density + gradient only) |
| `samplers.py` | random-walk Metropolis, MALA, HMC (with leapfrog) |
| `diagnostics.py` | autocorrelation, integrated autocorrelation time, ESS, mode-hopping count, dwell time |
| `run_demo.py` | runs all three, prints a summary table, writes three figures |

## How to read the output

The double well has two minima at $x_0 = \pm a$ and a barrier of height $h a^4$
(≈ 19.7 in energy units with the default parameters, i.e. $\exp(-19.7) \approx 3\times10^{-9}$
of the probability mass sits on the barrier). A sampler that only proposes tiny steps will look
perfectly stable and be completely wrong: it will sit in one well for the whole run.

The summary table therefore reports, per sampler:

* **acceptance rate** - how often a proposal is kept;
* **ESS($x_0$)** - effective sample size, i.e. how many *independent* draws the chain is worth;
* **mode hops** - how many times the chain crossed the barrier;
* **longest dwell** - the longest stretch spent in a single well.

Each sampler is tuned first on a short pilot run (the step size that maximises ESS per target
evaluation), because comparing untuned samplers says nothing. The numbers from the current run:

| sampler | step | acceptance | ESS($x_0$) | ESS / 1k evaluations | barrier crossings | longest dwell |
| --- | --- | --- | --- | --- | --- | --- |
| RWM | 1.00 | 0.283 | 3586 | **17.9** | 6510 | 474 |
| MALA | 1.00 | 0.195 | 2248 | 3.7 | 3122 | 687 |
| HMC (L = 10) | 0.20 | 0.974 | 2160 | 3.6 | 3185 | **144** |

Read that honestly, because it is not the textbook ordering:

* **In 2D, with a cheap target, tuned RWM is hard to beat per target evaluation.** A gradient
  costs roughly what a log-density costs here, MALA spends three of them per iteration, and the
  bottleneck is not local mixing but getting over the barrier. The gradient methods are paying
  for information that this problem does not reward.
* **HMC still wins where it matters most in practice**: it reaches the same ESS with far shorter
  dwell times (144 vs 474), i.e. it actually explores both wells instead of relying on lucky
  diffusion over the barrier - and it does so at a much higher acceptance rate (0.97 vs 0.28).
* **MALA sits in between and, at equal cost, loses to both**: its high acceptance rate buys
  locally better moves, but the barrier crossing remains a rare event. This is the concrete
  version of "a bigger spectral gap is not the same as a faster mixing time on this observable".

The barrier scan (`figures/01_barrier_scan.png`) shows what happens as the barrier grows: the
cheap samplers stop crossing entirely, and the failure is silent - the traces still look like
samples, they are simply all from one well.

That is the practical content of "the transition operator has a spectral gap": the gap shows up
as the ACF decay rate in `figures/01_acf.png`, the mode-hopping count is what the gap is about,
and the ladder RWM -> MALA -> HMC is the ladder of how much of the geometry each method is
allowed to see.

## Outputs

* `figures/01_trace.png` - trace plots, annotated with barrier crossings.
* `figures/01_acf.png` - autocorrelation decay for the three samplers.
* `figures/01_samples.png` - sample clouds against the true density contours.

## Next

* **Dimension scan.** The conclusion above is a 2D conclusion. Adding a `d`-dimensional Gaussian
  and a `d`-dimensional funnel, and plotting ESS per evaluation against `d`, is the honest way to
  show where the gradient-based advantage actually appears - and it is the first thing a
  reviewer would ask for.
* Add replica exchange (parallel tempering) and show that it fixes the barrier-crossing failure
  at a measurable cost in wall-clock time.
