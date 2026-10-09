# §F — Uncertainty: variance, quantiles, and the Poisson assumption

*Rendered by `08_uncertainty.py`. Figure: **#13** mean–variance dispersion.*

Panel: departures per (cell, hour, day) for the top 30 L6 cells over all
84 days — 720 cell-hours — i.e. exactly the panel §H1 predicts.

## F1 — Sample variance and sample quantiles

HW §1 asks for **sample variance and sample quantiles** by name. They are the
columns below, for the top 10 cells each at its own busiest hour:

| cell | peak hour | mean | **sample variance** | std | CV | **p50** | **p90** |
|---|---:|---:|---:|---:|---:|---:|---:|
| `wx4eqt` | 08:00 | 257.9 | **22,284.8** | 149.3 | 0.58 | **332** | **401** |
| `wx4eqw` | 18:00 | 209.2 | **11,138.6** | 105.5 | 0.50 | **246** | **324** |
| `wx4eqy` | 08:00 | 97.5 | **2,418.1** | 49.2 | 0.50 | **114** | **146** |
| `wx4erj` | 08:00 | 92.7 | **2,469.0** | 49.7 | 0.54 | **110** | **146** |
| `wx4eqv` | 08:00 | 89.9 | **2,212.9** | 47.0 | 0.52 | **109** | **142** |
| `wx4ern` | 18:00 | 53.9 | **683.9** | 26.2 | 0.48 | **66** | **81** |
| `wx4equ` | 17:00 | 39.8 | **205.6** | 14.3 | 0.36 | **40** | **58** |
| `wx4eqe` | 08:00 | 35.4 | **175.7** | 13.3 | 0.37 | **36** | **52** |
| `wx4erh` | 07:00 | 34.8 | **238.4** | 15.4 | 0.44 | **40** | **53** |
| `wx4eqs` | 17:00 | 25.1 | **99.8** | 10.0 | 0.40 | **25** | **36** |

**Read the p90 column operationally.** The mean is the wrong number to plan
against: a cell whose mean is 250 trips in its peak hour still has a 1-in-10
busy day at a much higher count, and it is the busy day that empties the
station. p90 is the inventory that keeps the stockout rate near one day in ten;
the mean would leave the cell short about half the time.

**Read the variance column as the modelling constraint.** The variance is
several times the mean everywhere in this table, so a method that assumes
variance = mean — a plain Poisson GLM — would be over-confident by construction.
F4 quantifies this.

### The five most variable cell-hours

| cell | hour | mean | variance | var / mean | CV |
|---|---:|---:|---:|---:|---:|
| `wx4eqt` | 08:00 | 257.9 | 22,284.8 | 86.42 | 0.58 |
| `wx4eqt` | 09:00 | 193.2 | 12,177.7 | 63.03 | 0.57 |
| `wx4eqw` | 18:00 | 209.2 | 11,138.6 | 53.24 | 0.50 |
| `wx4eqw` | 17:00 | 164.5 | 6,098.1 | 37.08 | 0.47 |
| `wx4erj` | 08:00 | 92.7 | 2,469.0 | 26.63 | 0.54 |

## F4 — Dispersion check

For each (cell, hour) with a mean of at least 0.5 trips/day
(**574 pairs**), the variance across the 84 days is plotted against the
mean on log-log axes, with the Poisson reference line `variance = mean` drawn:

| | |
|---|---|
| Pairs above the Poisson line | **88.9%** |
| Median variance / mean | **1.47** |
| Fitted log-log slope | **var ∝ mean^1.40** (Poisson = 1.00) |
| Robustness — **weekdays only**, 572 pairs | slope **1.33** |
| Robustness — **every cell**, 724 pairs | slope **1.37** |

**Two robustness checks, because the headline slope is not innocent.** The panel
mixes weekdays with weekends, and that mixture is itself a source of variance:
restricting to weekdays drops the slope to **1.33**. That is the honest
number for what a model must still explain *after* its day-of-week term is in —
some of the over-dispersion is a feature the model can absorb, and the rest is
not. Widening the population to all 563 cells gives **1.37**, so the
hotspot subset is not a special case.

**Conclusion: the data is over-dispersed, and the answer is **a negative-binomial (or quasi-Poisson) model, not a plain Poisson**.** A slope
of 1.40 against the Poisson value of 1.00 means the variance grows
faster than the mean, which is what a mixture of days with different latent
rates looks like — the extra variation is real and it is not sampling noise.

This is why §H4 fits the first of its two required methods as a **negative binomial** GLM rather than a Poisson GLM: the assumption has been tested here, before any model was fitted, and it failed. The decision is recorded in §H4 with this number as its justification.

**Why the zero-mean cell-hours are excluded from the fit.** Below a mean of
0.5 trips/day the variance is decided by whether a single trip happened;
those points sit on the reference line by construction and would dilute the
test rather than inform it. The exclusion is stated because it changes the
sample.
