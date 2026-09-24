# 01 - MCMC toy: how a chain mixes, in numbers and pictures

Three samplers, one bimodal target, no hidden machinery. Every number in this file is produced
by the scripts in this directory and can be reproduced by running them.

```bash
python 01_mcmc_toy/run_demo.py                            # one seed: tables + figures (~45 s)
python 01_mcmc_toy/run_demo.py --seeds 10 --skip-figures  # mean +- sd, writes the CSV (~100 s)
python 01_mcmc_toy/show_one_step.py                       # one step of each sampler, in full
```

## Files

| File | Content |
| --- | --- |
| `targets.py` | `StandardNormal2D` and `DoubleWell` (un-normalised log density + gradient only) |
| `samplers.py` | random-walk Metropolis, MALA, HMC (with leapfrog) |
| `diagnostics.py` | autocorrelation, integrated autocorrelation time, ESS, mode-hopping **count and rate**, dwell time |
| `run_demo.py` | tunes the step sizes, runs the samplers, prints two tables, writes the figures and `outputs_mcmc_summary.csv` |
| `show_one_step.py` | prints every number inside one step of each sampler, then checks three invariants and exits non-zero if any fails |

## The target, in numbers

$U(x_0,x_1)=h(x_0^2-a^2)^2+\tfrac12\omega^2x_1^2$ with the defaults $a=1.5$, $h=0.5$,
$\omega=2$.

* two minima at $x_0=\pm a$, $x_1=0$;
* barrier height $\Delta U = h a^4 = 2.53$ energy units;
* the density at the saddle is $e^{-2.53}\approx 0.080$ of the density at the well bottom, and
  $1.8\%$ of the probability sits in the region $|x_0| < 0.2$;
* the two wells hold **exactly 50%** each, whatever the barrier height is. "Which well is the
  chain in" is therefore a fair coin; the only question is how fast the chain flips it.

> **Correction (2026-09-23).** An earlier version of this file said the barrier was ≈ 19.7 energy
> units and that $\exp(-19.7)\approx 3\times10^{-9}$ of the probability mass sits on the barrier.
> Both statements were wrong: 19.7 was left over from a different parameter set, and a barrier
> *height* is not the fraction of probability mass sitting on the barrier. $\Delta U = 2.53$ is a
> modest barrier, which is why every sampler here crosses it regularly - the dramatic
> "stuck in one well" behaviour shows up only in the barrier scan at the bottom of this file.

## The cost convention (read this before comparing anything)

One log-density evaluation and one gradient evaluation both count as **1 unit** of cost. That
makes RWM cost 1 per iteration, MALA 3, and HMC $L+2 = 12$.

This is a *choice*, not a measurement: in a real codebase a gradient is often 2-3x the price of a
log-density. Change the weights and the absolute numbers change. What does not change is the
point of the table: this target is 2D and cheap, so the gradient methods are paying for geometry
that this problem does not reward.

## What is reported

Mixing, per sampler (single seed, reproducible seed layout):

| sampler | steps | cost / step | acceptance | ESS($x_0$) | ESS / 1k steps | ESS / 1k evals |
| --- | --- | --- | --- | --- | --- | --- |
| RWM | 200,000 | 1 | 0.283 | 3586 | 17.9 | **17.9** |
| MALA | 200,000 | 3 | 0.195 | 2248 | 11.2 | 3.7 |
| HMC ($L=10$) | 50,000 | 12 | 0.974 | 2160 | 43.2 | 3.6 |

Exploration of the two wells, same runs:

| sampler | steps | hops | hops / 1k steps | hops / 1k evals | longest dwell |
| --- | --- | --- | --- | --- | --- |
| RWM | 200,000 | 6,510 | 32.5 | **32.6** | 474 |
| MALA | 200,000 | 3,122 | 15.6 | 5.2 | 687 |
| HMC ($L=10$) | 50,000 | 3,185 | **63.7** | 5.3 | **144** |

### Why the absolute hop counts are misleading

The three runs do not have the same number of iterations (200k / 200k / 50k), and one iteration
does not cost the same. Comparing "6510 crossings vs 3185" is therefore comparing two different
budgets, and it answers the wrong question: it makes RWM the most mobile sampler, when per step
HMC crosses **twice** as often.

The comparable quantities are the rates:

* HMC crosses 63.7 times per 1k steps vs RWM 33.0 and MALA 15.6 - the "momentum carries you over
  the barrier" intuition is real, but it shows up in the **rate**, not in the raw count.
* Once you divide by cost as well, HMC's advantage disappears on this problem: 5.3 crossings per
  1k evaluations vs RWM's 32.6.

Both rates are now printed by `run_demo.py` and stored per seed in `outputs_mcmc_summary.csv`.

## Results across 10 seeds (mean +- sd)

A single seed has no error bar, and some of the differences above are the same size as seed-to-seed
noise. Repeating the whole protocol 10 times:

| sampler | ESS($x_0$) | ESS / 1k evals | hops / 1k steps | longest dwell |
| --- | --- | --- | --- | --- |
| RWM | 3753 +- 207 | **18.8 +- 1.0** | 33.0 +- 0.5 | 368 +- 52 |
| MALA | 2142 +- 113 | 3.6 +- 0.2 | 15.6 +- 0.5 | 739 +- 161 |
| HMC ($L=10$) | 2082 +- 234 | 3.5 +- 0.4 | **64.8 +- 1.5** | **166 +- 32** |

## What the numbers say

1. **On this problem, tuned RWM wins per unit cost, and not by a little**: 18.8 +- 1.0 against
   3.6 +- 0.2 and 3.5 +- 0.4. The gap is ~15 sd, so it is not seed noise. If you run this and
   find that the dumbest sampler is the most cost-effective one, **that is the expected result,
   not a bug** - it is what a 2D, cheap target does to gradient methods.
2. **MALA and HMC tie on cost-efficiency (3.6 +- 0.2 vs 3.5 +- 0.4) but differ everywhere else.**
   MALA has the lowest crossing rate (15.6 +- 0.5 per 1k steps) and the longest dwell
   (739 +- 161); HMC has the highest crossing rate and the shortest dwell. Same ESS per unit
   cost, two very different mechanisms.
3. **MALA is dominated here**: RWM beats it on ESS per evaluation *and* on crossings per
   evaluation. Its extra gradient evaluations buy locally better moves, and that is not the
   currency this observable pays in.
4. **Acceptance rate and mixing are different things** - the same lesson as the barrier scan
   below. HMC has acceptance 0.974 and the lowest ESS per evaluation; RWM has acceptance 0.283
   and the highest.
5. **These are 2D conclusions.** The honest statement is "with a cheap 2D target, a gradient is
   not worth 3-12 log-densities". Where that flips - higher dimension, a target whose density is
   concentrated in a thin shell, an expensive-to-evaluate likelihood - is the next experiment,
   not a conclusion of this one.

### Barrier scan: who stops exploring, and when

`figures/01_barrier_scan.png` raises the barrier and plots crossings per 1k steps (rate, not
count). Numbers behind the figure (seed 7, budgets 50k / 50k / 20k, step sizes held at the values
tuned at $\Delta U = 2.53$):

| barrier $\Delta U$ | 1.27 | 2.53 | 3.80 | 5.06 | 6.33 | 7.59 |
| --- | --- | --- | --- | --- | --- | --- |
| RWM | 58.5 | 33.0 | 18.5 | 10.8 | 6.6 | 4.6 |
| MALA | 38.6 | 15.1 | 9.7 | 7.2 | 7.8 | 8.0 |
| HMC | **164.4** | 65.2 | 23.0 | 7.2 | 1.5 | **0.3** |

The interesting result is not the one the old version of this file claimed ("the cheap samplers
stop crossing"): it is that **HMC's crossing rate collapses fastest** - 164 down to 0.3 per 1k
steps - while the random walk keeps crossing, slowly. The reason is mechanical: HMC's momentum is
drawn from $N(0,I)$, so the kinetic energy it can spend on the barrier is capped, and a barrier
far above that cap is crossed almost never, no matter how long the trajectory. A random walk with
a large step size keeps injecting energy through the proposal.

Caveat worth stating: the step sizes are held at the values tuned at $\Delta U = 2.53$, so part
of each curve's decay is mis-tuning rather than method. Re-tuning per barrier is on the to-do
list below.

## Teaching path (how to use this as a module)

Each lesson is a command plus what the learner should see, so it can be checked rather than
believed:

1. **Run it as-is** (`run_demo.py`). Expected: RWM has the highest ESS per evaluation. The
   misconception it removes is the textbook ladder "RWM -> MALA -> HMC is monotonically
   better".
2. **Open one step** (`show_one_step.py`). Every term of the acceptance ratio is printed, and the
   script checks three invariants: the gradient vanishes at the minimum, the wells are symmetric,
   and the analytic gradient matches central differences. The misconception it removes is that
   MALA/HMC accept "because the proposal is better" - they accept on an explicit detailed-balance
   ratio, and MALA's two log-proposal terms are what make an uphill-biased proposal exact.
3. **Read the ACF figure against the table**: acceptance rate and ESS are different quantities.
4. **Re-run with `--seeds 10`** and decide which gaps are real. Expected: RWM's cost-efficiency
   gap is real (~15 sd), MALA vs HMC on ESS per evaluation is a tie (error bars overlap).
5. **Raise the barrier** and watch the *rate* change, not just the count. Expected: HMC's
   crossing rate collapses first as the barrier grows.

## Outputs

* `figures/01_trace.png` - trace plots, annotated with crossings per 1k steps.
* `figures/01_acf.png` - autocorrelation decay for the three samplers.
* `figures/01_samples.png` - sample clouds against the true density contours.
* `figures/01_barrier_scan.png` - crossing rate versus barrier height.
* `outputs_mcmc_summary.csv` - one row per sampler per seed (ESS, both rates, dwell, acceptance).

## Next

* **Dimension scan.** The conclusion above is a 2D conclusion. Add a $d$-dimensional Gaussian and
  a $d$-dimensional funnel, plot ESS per evaluation against $d$, and find where the gradient-based
  advantage actually appears. This is the first thing a reviewer asks for.
* **Re-tune per barrier height** in the scan, so the curves separate "method" from "step size".
* **Replica exchange (parallel tempering)**: show it fixes the high-barrier crossing rate at a
  measurable cost in wall-clock time.
* **A deliberately broken sampler.** Drop MALA's reverse-proposal term, rerun, and show that the
  bias is invisible in the trace plot but visible in the sample cloud. Same lesson as module 02's
  "loss converged, distribution wrong", one level down.
